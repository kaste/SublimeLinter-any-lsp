from __future__ import annotations

import sys

# kiss-reloader:
prefix = __package__ + "."  # don't clear the base package
for module_name in [
    module_name
    for module_name in sys.modules
    if module_name.startswith(prefix) and module_name != __name__
]:
    del sys.modules[module_name]


from collections import defaultdict, deque
from concurrent.futures import Future
import copy
from dataclasses import dataclass, field
from functools import partial, wraps
import inspect
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
from typing import (
    IO,
    Callable,
    Generic,
    Iterator,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
)

from typing_extensions import NotRequired, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')
Callback = Callable[["Message"], None]


import sublime
import sublime_plugin
from SublimeLinter import sublime_linter
from SublimeLinter.lint import (
    Linter,
    PermanentError,
    TransientError,
    persist,
    style,
    util,
)
from SublimeLinter.lint.backend import make_error_uid
from SublimeLinter.lint.quick_fix import (
    QuickAction,
    TextRange,
    add_at_eol,
    extend_existing_comment,
    ignore_rules_inline,
    line_error_is_on,
    merge_actions_by_code_and_line,
    quick_actions_for,
)

from .core.utils import Counter, run_on_new_thread, try_kill_proc, unflatten

logger = logging.getLogger('SublimeLinter.plugin.lsp')
JSON_RPC_MESSAGE = "Content-Length: {}\r\n\r\n{}"
DEFAULT_ERROR_TYPE = "error"
CLIENT_INFO = unflatten({
    "processId": os.getpid(),
    "clientInfo.name": "SublimeLinter",
    "clientInfo.version": "4",
})
MINIMAL_CAPABILITIES = unflatten({
    "textDocument.publishDiagnostics.codeDescriptionSupport": True,
    "textDocument.publishDiagnostics.dataSupport": True,
    "textDocument.publishDiagnostics.relatedInformation": True,
    "textDocument.publishDiagnostics.tagSupport.valueSet": [1, 2],
    "textDocument.publishDiagnostics.versionSupport": True,
    "textDocument.synchronization.didSave": True,
    "workspace.configuration": True,
    "workspace.didChangeConfiguration.dynamicRegistration": True,
    "workspace.workspaceFolders": True,
    "window.workDoneProgress": True,
})
max_capabilities_per_service: dict[str, dict] = {}
_counter = Counter()


class Message(TypedDict, total=False):
    jsonrpc: str
    method: str
    params: NotRequired[dict]


class Notification(Message):
    ...

class Request(Message):
    id: NotRequired[int]


def encode_message(msg: Message) -> bytes:
    _message = json.dumps(msg)
    return JSON_RPC_MESSAGE.format(len(_message), _message).encode()


def parse_for_message(stream: IO[bytes]) -> Optional[Message]:
    for line in stream:
        if line.startswith(b"Content-Length: "):
            content_length = int(line[16:].rstrip())
            break
    else:
        return None

    # Ignore every other possible header, just go for the next empty line
    for line in stream:
        if not line.strip():
            break

    content = stream.read(content_length)
    try:
        return json.loads(content)
    except json.decoder.JSONDecodeError:
        return None


def parse_for_messages(stream: IO):
    while msg := parse_for_message(stream):
        yield msg


ServerStates = Literal[
    "INIT", "INITIALIZE_REQUESTED", "READY", "SHUTDOWN_REQUESTED", "EXIT_REQUESTED", "DEAD"]


@dataclass
class ServerConfig:
    name: str
    cmd: list[str]
    root_dir: str | None
    initialization_options: dict[str, object] = field(default_factory=dict)
    settings: dict[str, object] = field(default_factory=dict)
    capabilities: dict[str, object] = field(default_factory=lambda: MINIMAL_CAPABILITIES)

    def __post_init__(self):
        if self.capabilities is not MINIMAL_CAPABILITIES:
            current_capabilities = max_capabilities_per_service.get(self.name, MINIMAL_CAPABILITIES)
            next_capabilities = merge_dicts(current_capabilities, self.capabilities)
            self.capabilities = max_capabilities_per_service[self.name] = next_capabilities

    def identity(self) -> tuple[str, str | None]:
        return (self.name, self.root_dir)


@dataclass
class Server:
    config: ServerConfig
    reader: IO[bytes]
    writer: IO[bytes]
    killer: Callable[[], None] = lambda: None
    handlers: dict[str, Callback] = field(default_factory=dict)

    state: ServerStates = field(init=False, default="INIT")
    messages_out_queue: deque[Message] = field(init=False, default_factory=deque)
    pending_request_ids: dict[int, Future[Message]] = field(init=False, default_factory=dict)
    capabilities: dict[str, object] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.handlers = self.handlers.copy()
        reader_thread = run_on_new_thread(self.reader_loop)
        self.wait: Callable[[float], bool] = partial(join_thread, reader_thread)
        self._writer_lock = threading.Lock()
        self.logger = logging.getLogger(f"SublimeLinter.plugin.{self.name}")

    @property
    def name(self) -> str:
        return self.config.name

    def add_listener(self, handler: Callback | dict[str, Callback]) -> None:
        if isinstance(handler, dict):
            self.handlers.update(handler)
        else:
            for frame in inspect.stack()[1:]:
                key = ":".join((frame.filename, str(frame.lineno), frame.function))
                break
            else:
                print(f"can't determine a key for {handler}")
                return
            self.handlers[key] = handler

    def kill(self):
        self.logger.info("SIGKILL")
        self.killer()

    def reader_loop(self):
        while msg := parse_for_message(self.reader):
            # print(f"<- {msg}")
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"<- {msg}")

            try:
                id = msg["id"]  # type: ignore[typeddict-item]
                fut = self.pending_request_ids.pop(id)
            except KeyError:
                pass
            else:
                fut.set_result(msg)

            for handler in self.handlers.values():
                try:
                    handler(msg)
                except Exception:
                    traceback.print_exc()

        self.state = "DEAD"
        self.logger.info(f"`{self.name}`> is now dead.")


    def request(self, method: str, params: dict = {}) -> OkFuture[Message]:
        fut: Future[Message]
        id = next_id()
        msg: Request = {"id": id, "method": method, "params": params.copy()}
        self.pending_request_ids[id] = fut = OkFuture()
        self.send(msg)
        return fut

    def notify(self, method: str, params: dict = {}) -> None:
        self.send({"method": method, "params": params.copy()})

    def write_message(self, message: Message) -> None:
        msg: Message = {"jsonrpc": "2.0", **message}

        if self.logger.isEnabledFor(logging.DEBUG):
            sanitized = copy.deepcopy(msg)
            try:
                msg["params"]["contentChanges"][0]["text"]
            except KeyError:
                pass
            else:
                sanitized["params"]["contentChanges"][0]["text"] = "..."
            try:
                msg["params"]["textDocument"]["text"]
            except KeyError:
                pass
            else:
                sanitized["params"]["textDocument"]["text"] = "..."
            self.logger.debug(f"-> {sanitized}")

        try:
            id = msg["id"]  # type: ignore[typeddict-item]
            fut = self.pending_request_ids[id]
        except KeyError:
            pass
        else:
            if not fut.set_running_or_notify_cancel():
                return

        msg_ = encode_message(msg)
        with self._writer_lock:
            self.writer.write(msg_)
            self.writer.flush()

    def in_shutdown_phase(self) -> bool:
        return self.state in ("SHUTDOWN_REQUESTED", "EXIT_REQUESTED", "DEAD")

    def send(self, message: Message) -> None:
        # print("send message:", message)
        if self.state == "DEAD":
            self.logger.warn("Server is already dead.")
            return

        if self.state == "EXIT_REQUESTED":
            self.logger.warn("Server `exit` has already been requested")
            return

        if message["method"] == "initialize":
            if self.state == "INIT":
                self.write_message(message)
                self.state = "INITIALIZE_REQUESTED"
            else:
                self.logger.warn("`initialize` only valid in INIT state")
            return

        if message["method"] == "initialized":
            if self.state == "INITIALIZE_REQUESTED":
                self.write_message(message)
                self.state = "READY"
                while True:
                    try:
                        m = self.messages_out_queue.popleft()
                    except IndexError:
                        break
                    else:
                        self.send(m)
            else:
                self.logger.warn("`initialized` only valid in INITIALIZE_REQUESTED state")
            return

        if message["method"] == "shutdown":
            if self.state == "READY":
                self.write_message(message)
                self.state = "SHUTDOWN_REQUESTED"
            else:
                self.logger.warn("`shutdown` only valid in READY state")
            return

        if message["method"] == "exit":
            self.write_message(message)
            self.state = "EXIT_REQUESTED"
            return

        if self.state == "SHUTDOWN_REQUESTED":
            self.logger.warn("Server `shutdown` has already been requested")
            return

        if self.state == "READY":
            self.write_message(message)
            return

        if self.state in ("INIT", "INITIALIZE_REQUESTED"):
            self.messages_out_queue.append(message)
            return


running_servers: dict[tuple[str, Optional[str]], Server] = {}


def ensure_server(config: ServerConfig) -> Server:
    if server := running_servers.get(config.identity()):
        if server.config == config and not server.in_shutdown_phase():
            return server

        handlers = server.handlers
        shutdown_server(server)

    else:
        handlers = {}

    new_server = start_server(config, handlers)
    running_servers[config.identity()] = new_server
    return new_server


def start_server(config: ServerConfig, handlers: dict[str, Callback] = {}) -> Server:
    proc = subprocess.Popen(
        config.cmd,
        cwd=config.root_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        startupinfo=util.create_startupinfo(),
    )
    assert proc.stdin
    assert proc.stdout
    assert proc.stderr
    run_on_new_thread(std_err_printer, logger, proc.stderr)
    logger.info(f"`{config.name}`> has started.")

    reader = proc.stdout
    writer = proc.stdin
    killer = partial(try_kill_proc, proc)
    server = Server(config, reader, writer, killer, handlers)
    server.add_listener(partial(on_log_message, server))

    folder_config = {
        "rootUri": Path(config.root_dir).as_uri() if config.root_dir else None,
        "rootPath": config.root_dir,
        "workspaceFolders": (
            [{"name": Path(config.root_dir).stem, "uri": Path(config.root_dir).as_uri()}]
            if config.root_dir else None
        ),
    }
    req = server.request("initialize", {
        **CLIENT_INFO,
        **folder_config,
        "capabilities": config.capabilities,
        "initializationOptions": config.initialization_options,
    })

    @req.on_response
    def on_initialize_response(msg):
        # print("msg capabilities", msg)
        try:
            server.capabilities = msg["result"]["capabilities"]
        except KeyError:
            pass
        server.notify("initialized")

    return server


class AnyLSP(Linter):
    __abstract__ = True

    def run(self, cmd: list[str] | None, code: str):
        if self.view.file_name() != __file__:
            raise PermanentError("only for this file.")
        if cmd is None:
            raise PermanentError("`cmd` must be defined.")

        initialization_options = self.settings.get("initialization_options", {})
        if not isinstance(initialization_options, dict):
            self.logger.error(
                f"`initialization_options` must be a dictionary/mapping."
                f" Got {initialization_options!r}"
            )
            raise PermanentError("wrong type for `initialization_options`")
        settings = self.settings.get("settings", {})
        if not isinstance(settings, dict):
            self.logger.error(
                f"`settings` must be a dictionary/mapping."
                f" Got {settings!r}"
            )
            raise PermanentError("wrong type for `initialization_options`")

        cwd = self.get_working_dir()
        config = ServerConfig(
            self.name,
            cmd,
            cwd,
            initialization_options=initialization_options,
            settings=settings
        )
        server = ensure_server(config)
        reason = (
            "on_modified"
            if get_server_for_view(self.view, self.name) == server
            else "on_load"
        )

        if reason == "on_load":
            server.add_listener(
                partial(diagnostics_handler, server, default_error_type=self.default_type)
            )
            remember_server_for_view(self.view, server)
            server.notify("textDocument/didOpen", unflatten({
                "textDocument.uri": canoncial_uri_for_view(self.view),
                "textDocument.languageId": language_id_for_view(self.view),
                "textDocument.version": self.view.change_count(),
                "textDocument.text": code,
            }))

        elif reason == "on_modified":
            server.notify("textDocument/didChange", unflatten({
                "textDocument.uri": canoncial_uri_for_view(self.view),
                "textDocument.version": self.view.change_count(),
                "contentChanges": [{ "text": code }]
            }))

        raise TransientError("lsp's answer on their own will.")


def handles(**kwargs) -> Callable[[Callable[P, None]], Callable[P, None]]:
    try:
        msg_arg_name = next(iter(kwargs.keys()))
    except StopIteration:
        raise TypeError("can't figure out the argument name of the message")

    wanted_method = kwargs[msg_arg_name]

    def decorator(fn: Callable[P, None]) -> Callable[P, None]:
        sig = inspect.signature(fn)

        @wraps(fn)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> None:
            bound = sig.bind(*args, **kwargs)
            msg = bound.arguments[msg_arg_name]
            if msg.get("method") == wanted_method:
                fn(*args, **kwargs)

        return wrapped
    return decorator


@handles(msg="window/logMessage")
def on_log_message(server: Server, msg: Notification) -> None:
    server.logger.log(
        translate_log_severity(msg["params"]["type"]),
        msg["params"]["message"]
    )


def translate_log_severity(type: int) -> int:
    return {
        1: logging.ERROR,
        2: logging.WARNING,
        3: logging.INFO,
        4: logging.DEBUG
    }.get(type, logging.WARNING)


@handles(msg="textDocument/publishDiagnostics")
def diagnostics_handler(server: Server, msg: Message, default_error_type: str = "error") -> None:
    # print("diagnostics_handler--")
    # print("msg", msg)
    linter_name = server.name
    uri = msg["params"]["uri"]
    file_name = canonical_name_from_uri(uri)
    view = view_for_file_name(file_name)
    if not view:
        server.logger.info(f"skip: no view for {file_name}")
        return
    # print("canoncial_uri_for_view", file_name, view.file_name())

    version = msg["params"]["version"]
    if view.change_count() != version:
        server.logger.info(f"skip: view has changed. {view.change_count()} -> {version}")
        return

    errors: list[persist.LintError] = []
    for diagnostic in msg["params"]["diagnostics"]:
        region = sublime.Region(
            view.text_point_utf16(
                diagnostic["range"]["start"]["line"],
                diagnostic["range"]["start"]["character"],
                clamp_column=True
            ),
            view.text_point_utf16(
                diagnostic["range"]["end"]["line"],
                diagnostic["range"]["end"]["character"],
                clamp_column=True
            )
        )
        error: persist.LintError = {
            "linter": linter_name,
            "filename": file_name,
            "msg": diagnostic["message"],
            "code": diagnostic.get("code", ""),
            "error_type": severity_to_type(diagnostic.get("severity"), default=default_error_type),
            "line": diagnostic["range"]["start"]["line"],
            "start": diagnostic["range"]["start"]["character"],
            "region": region,
            "offending_text": view.substr(region)
        }
        if "data" in diagnostic:
            error["data"] = diagnostic["data"]  # type: ignore[typeddict-unknown-key]
        error.update({
            "uid": make_error_uid(error),
            "priority": style.get_value("priority", error, 0)
        })
        errors.append(error)

    # print("out:", errors)
    # fan-in on Sublime's worker thread
    sublime.set_timeout_async(partial(
        sublime_linter.update_file_errors, file_name, linter_name, errors, reason=None
    ))


def severity_to_type(severity: int | None, default=DEFAULT_ERROR_TYPE) -> str:
    return {
        1: "error", 2: "warning", 3: "info", 4: "hint", None: default
    }.get(severity, default)



RUFF_NAME = "ruff-lsp"


class Ruff(AnyLSP):
    name = RUFF_NAME
    cmd = ('ruff', 'server', '--preview')
    defaults = {
        "selector": "source.python"
    }


@ignore_rules_inline(RUFF_NAME, except_for={
    # some indentation rules are not stylistic in python
    # the following violations cannot be ignored
    "E112",  # expected an indented block
    "E113",  # unexpected indentation
    "E116",  # unexpected indentation (comment)
    "E901",  # SyntaxError or IndentationError
    "E902",  # IOError
    "E999",  # SyntaxError
    "F722",  # syntax error in forward annotation
})
def ignore_ruff_code(error, view):
    # type: (persist.LintError, sublime.View) -> Iterator[TextRange]
    line = line_error_is_on(view, error)
    code = error["code"]
    yield (
        extend_existing_comment(
            r"(?i)# noqa:[\s]?(?P<codes>[A-Z]+[0-9]+((?:,\s?)[A-Z]+[0-9]+)*)",
            ", ",
            {code},
            line
        )
        or add_at_eol(
            "  # noqa: {}".format(code),
            line
        )
    )


@quick_actions_for(RUFF_NAME)
def ruff_fixes_provider(errors, _view):
    # type: (list[persist.LintError], Optional[sublime.View]) -> Iterator[QuickAction]
    def make_action(error):
        # type: (persist.LintError) -> QuickAction
        return QuickAction(
            f"{RUFF_NAME}: {{data[kind][suggestion]}}".format(**error),
            partial(ruff_fix_error, error),
            "{msg}".format(**error),
            solves=[error]
        )

    def except_(error):
        return "data" not in error
    yield from merge_actions_by_code_and_line(make_action, except_, errors, _view)


def ruff_fix_error(error, view) -> Iterator[TextRange]:
    """
    "data":{
       "code":"I001",
       "fix":{
          "applicability":"safe",
          "edits":[
             {
                "content":"...",
                "range":[
                   300,
                   729
                ]
             }
          ],
          "isolation_level":"NonOverlapping"
       },
       "kind":{
          "body":"Import block is un-sorted or un-formatted",
          "name":"UnsortedImports",
          "suggestion":"Organize imports"
       }
    },

    """
    fix_description = error["data"]["fix"]
    for edit in fix_description["edits"]:
        region = sublime.Region(
            *edit["range"]
        )
        yield TextRange(edit["content"] or "", region)


servers_attached_per_buffer: dict[int, dict[str, Server]] = defaultdict(dict)


def remember_server_for_view(view: sublime.View, server: Server) -> None:
    servers_attached_per_buffer[view.buffer_id()][server.name] = server


def get_server_for_view(view: sublime.View, name: str) -> Server | None:
    return servers_attached_per_buffer.get(view.buffer_id(), {}).get(name)


class DocumentListener(sublime_plugin.EventListener):
    def on_close(self, view: sublime.View) -> None:
        if view.clones():
            return

        did_close(view)

    def on_pre_close_window(self, window: sublime.Window) -> None:
        for buffer in {view.buffer() for view in window.views()}:
            view = buffer.primary_view()
            if view := buffer.primary_view():
                did_close(view)

    def on_exit(self) -> None:
        for (cmd, root_dir), server in running_servers.items():
            server.kill()


    def on_load(self, view):
        ...


def did_close(view: sublime.View) -> None:
    uri = canoncial_uri_for_view(view)
    for server in servers_attached_per_buffer.pop(view.buffer_id(), {}).values():
        server.notify("textDocument/didClose", {"uri": uri})


def join_popen(proc: subprocess.Popen, timeout: float) -> bool:
    try:
        proc.wait(timeout)
    except TimeoutError:
        return False
    else:
        return True


def join_thread(thread: threading.Thread, timeout: float) -> bool:
    thread.join(timeout)
    return not thread.is_alive()


WAIT_TIME = 0.5


def shutdown_server(server: Server) -> bool:
    if server.in_shutdown_phase():
        if not server.wait(WAIT_TIME):
            server.kill()
            return server.wait(WAIT_TIME)
        return True

    return shutdown_server_(server)


def shutdown_server_(server: Server) -> bool:
    cond = threading.Condition()

    def on_shutdown_response(_):
        server.notify("exit")
        if not server.wait(WAIT_TIME):
            server.kill()
            if not server.wait(WAIT_TIME):
                return

        with cond:
            cond.notify_all()

    req = server.request("shutdown")
    req.on_response(on_shutdown_response)
    try:
        req.wait(WAIT_TIME)
    except TimeoutError:
        req.cancel()
        server.kill()
        return server.wait(WAIT_TIME)

    with cond:
        ok = cond.wait(WAIT_TIME * 2)
    server.logger.info("shtdown in time" if ok else "shtdown too slow")
    return ok



class OkFuture(Future, Generic[T]):
    def on_response(self, fn: Callable[[T], None]) -> None:
        def wrapper(fut):
            try:
                result = fut.result()
            except Exception:
                pass
            else:
                fn(result)
        self.add_done_callback(wrapper)

    def wait(self, timeout: float) -> None:
        """Wait for the result; only ever raise TimeoutError"""
        try:
            self.result(timeout)
        except TimeoutError:
            raise
        except Exception:
            pass


def cleanup_servers() -> None:
    used_servers = {
        server.config.identity()
        for d in servers_attached_per_buffer.values()
        for server in d.values()
    }
    for identity, server in running_servers.items():
        if id not in used_servers:
            shutdown_server(server)


def language_id_for_view(view: sublime.View) -> str:
    main_scope = view.scope_name(0).split()[0]
    map = sublime.load_settings("language-ids.sublime-settings")
    return (
        map.get(main_scope)
        or main_scope.split(".")[1] if "." in main_scope else main_scope
    )

def canoncial_uri_for_view(view: sublime.View) -> str:
    if file_name := view.file_name():
        return to_uri(file_name)
    return f"buffer://{view.buffer_id()}"


def canonical_name_from_uri(uri) -> str:
    if uri.startswith("file:"):
        return from_uri(uri)
    if uri.startswith("buffer:"):
        return f"<untitled {uri[7:]}>"
    return uri


def view_for_file_name(file_name: str) -> sublime.View | None:
    for window in sublime.windows():
        if view := window.find_open_file(file_name):
            return view
    return None


def next_id() -> int:
    _counter.inc()
    return _counter.count()


def to_uri(path: str) -> str:
    return Path(path).as_uri()


def from_uri(uri: str) -> str:  # backport from Python 3.13
    """Return a new path from the given 'file' URI."""
    if not uri.startswith('file:'):
        raise ValueError(f"URI does not start with 'file:': {uri!r}")
    path = uri[5:]
    if path[:3] == '///':
        # Remove empty authority
        path = path[2:]
    elif path[:12] == '//localhost/':
        # Remove 'localhost' authority
        path = path[11:]
    if path[:3] == '///' or (path[:1] == '/' and path[2:3] in ':|'):
        # Remove slash before DOS device/UNC path
        path = path[1:]
    if path[1:2] == '|':
        # Replace bar with colon in DOS drive
        path = path[:1] + ':' + path[2:]
    from urllib.parse import unquote_to_bytes
    path_ = Path(os.fsdecode(unquote_to_bytes(path)))
    if not path_.is_absolute():
        raise ValueError(f"URI is not absolute: {uri!r}")
    return str(path_)


def merge_dicts(a: dict, b: dict) -> dict:
    """Merge given dicts `a` and `b` recursively and return a new dict."""
    c = {}
    common_keys = a.keys() & b.keys()

    for key in common_keys:
        if isinstance(a[key], dict) and isinstance(b[key], dict):
            c[key] = merge_dicts(a[key], b[key])
        else:
            # Otherwise, use the value from b
            c[key] = copy.deepcopy(b[key])

    for key in a.keys() - common_keys:
        c[key] = copy.deepcopy(a[key])

    for key in b.keys() - common_keys:
        c[key] = copy.deepcopy(b[key])

    return c


def std_err_printer(logger: logging.Logger, stream: IO[bytes]) -> None:
    for line in stream:
        logger.info(line.decode('utf-8', 'replace').rstrip())


def plugin_loaded():
    ...


def plugin_unloaded():
    for server in running_servers.values():
        server.kill()
