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
from functools import partial
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import (
    IO,
    Callable,
    Generic,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
)

from typing_extensions import NotRequired

T = TypeVar('T')


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

from .core.utils import Counter, run_on_new_thread, try_kill_proc, unflatten

logger = logging.getLogger('SublimeLinter.plugin.lsp')
JSON_RPC_MESSAGE = "Content-Length: {}\r\n\r\n{}"
CLIENT_INFO = unflatten({
    "processId": os.getpid(),
    "clientInfo.name": "SublimeLinter",
    "clientInfo.version": "4",
})
MINIMAL_CAPABILITIES = unflatten({
    "textDocument.synchronization.didSave": True,
    "textDocument.publishDiagnostics.relatedInformation": True,
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
    state: ServerStates = field(init=False, default="INIT")
    messages_out_queue: deque[Message] = field(default_factory=deque)
    pending_request_ids: dict[int, Future[Message]] = field(init=False, default_factory=dict)
    handlers: set[Callable[[Server, Message], None]] = field(default_factory=set)

    def __post_init__(self):
        reader_thread = run_on_new_thread(self.reader_loop)
        self.wait: Callable[[float], bool] = partial(join_thread, reader_thread)
        self._writer_lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.config.name

    def on_message(self, handler: Callable[[Server, Message], None]) -> None:
        self.handlers.add(handler)

    def kill(self):
        print("kill", self.name)
        self.killer()

    def reader_loop(self):
        while msg := parse_for_message(self.reader):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"<- {msg}")

            try:
                id = msg["id"]  # type: ignore[typeddict-item]
                fut = self.pending_request_ids.pop(id)
            except KeyError:
                pass
            else:
                fut.set_result(msg)

            for handler in self.handlers.copy():
                try:
                    handler(self, msg)
                except Exception as e:
                    print(f"{handler} raised {e!r}")

        self.state = "DEAD"
        print(f"`{self.name}`> is now dead.")


    def request(self, message: Request) -> OkFuture[Message]:
        fut: Future[Message]
        msg: Request = {"id": next_id(), **message}
        self.pending_request_ids[msg["id"]] = fut = OkFuture()
        self.send(msg)
        return fut

    def notify(self, method: str, params: dict = {}) -> None:
        self.send({"method": method, "params": params or {}})

    def write_message(self, message: Message) -> None:
        msg: Message = {"jsonrpc": "2.0", **message}

        if logger.isEnabledFor(logging.DEBUG):
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
            logger.debug(f"-> {sanitized}")

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
        if self.state == "DEAD":
            logger.warn("Server is already dead.")
            return

        if self.state == "EXIT_REQUESTED":
            logger.warn("Server `exit` has already been requested")
            return

        if message["method"] == "initialize":
            if self.state == "INIT":
                self.write_message(message)
                self.state = "INITIALIZE_REQUESTED"
            else:
                logger.warn("`initialize` only valid in INIT state")
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
                logger.warn("`initialized` only valid in INITIALIZE_REQUESTED state")
            return

        if message["method"] == "shutdown":
            if self.state == "READY":
                self.write_message(message)
                self.state = "SHUTDOWN_REQUESTED"
            else:
                logger.warn("`shutdown` only valid in READY state")
            return

        if message["method"] == "exit":
            self.write_message(message)
            self.state = "EXIT_REQUESTED"
            return

        if self.state == "SHUTDOWN_REQUESTED":
            logger.warn("Server `shutdown` has already been requested")
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
        else:
            shutdown_server(server)

    new_server = start_server(config)
    running_servers[config.identity()] = new_server
    return new_server


def start_server(config: ServerConfig) -> Server:
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
    run_on_new_thread(std_err_printer, config.name, proc.stderr)
    print(f"`{config.name}`> has started.")

    reader = proc.stdout
    writer = proc.stdin
    killer = partial(try_kill_proc, proc)
    server = Server(config, reader, writer, killer)

    folder_config = {
        "rootUri": Path(config.root_dir).as_uri() if config.root_dir else None,
        "rootPath": config.root_dir,
        "workspaceFolders": (
            [{"name": Path(config.root_dir).stem, "uri": Path(config.root_dir).as_uri()}]
            if config.root_dir else None
        ),
    }
    server.request({
        "method": "initialize",
        "params": {
            **CLIENT_INFO,
            **folder_config,
            "capabilities": config.capabilities,
            "initializationOptions": config.initialization_options,
        }
    }).on_response(lambda _: server.notify("initialized"))
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
            server.on_message(diagnostics_handler)
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


def diagnostics_handler(server: Server, msg: Message) -> None:
    try:
        method = msg["method"]
    except KeyError:
        return
    if method != "textDocument/publishDiagnostics":
        return

    linter_name = server.name
    uri = msg["params"]["uri"]
    file_name = canonical_name_from_uri(uri)
    view = view_for_file_name(file_name)
    if not view:
        print(f"skip: no view for {file_name}")
        return

    version = msg["params"]["version"]
    if view.change_count() != version:
        print(f"skip: view has changed. {view.change_count()} -> {version}")
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
            "error_type": severity_to_type(diagnostic.get("severity")),
            "line": diagnostic["range"]["start"]["line"],
            "start": diagnostic["range"]["start"]["character"],
            "region": region,
            "offending_text": view.substr(region)
        }
        error.update({
            "uid": make_error_uid(error),
            "priority": style.get_value("priority", error, 0)
        })
        errors.append(error)

    # fan-in on Sublime's worker thread
    sublime.set_timeout_async(partial(
        sublime_linter.update_file_errors, file_name, linter_name, errors, reason=None
    ))


DEFAULT_TYPE = "error"


def severity_to_type(severity: str | None, default=DEFAULT_TYPE) -> str:
    return {  # type: ignore[call-overload]
        1: "error", 2: "warning", 3: "info", 4: "hint"
    }.get(severity, default)


class Ruff(AnyLSP):
    name = "ruff-lsp"
    cmd = ('ruff', 'server', '--preview')
    defaults = {
        "selector": "source.python"
    }


servers_attached_per_buffer: dict[int, dict[str, Server]] = defaultdict(dict)


def remember_server_for_view(view: sublime.View, server: Server) -> None:
    servers_attached_per_buffer[view.buffer_id()][server.name] = server


def get_server_for_view(view: sublime.View, name: str) -> Server | None:
    return servers_attached_per_buffer.get(view.buffer_id(), {}).get(name)


class DocumentListener(sublime_plugin.EventListener):
    def on_close(self, view: sublime.View) -> None:
        if view.clones():
            print(f"{view} has still clones")
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
            print(f"killed: {server.name}")


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

    req = server.request({"method": "shutdown"})
    req.on_response(on_shutdown_response)
    try:
        req.result(WAIT_TIME)
    except TimeoutError:
        req.cancel()
        server.kill()
        return server.wait(WAIT_TIME)

    with cond:
        ok = cond.wait(WAIT_TIME * 2)
    print("shtdown in time" if ok else "shtdown too slow")
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


def std_err_printer(name: str, stream: IO[bytes]) -> None:
    for line in stream:
        print(f"<{name}>", line.decode('utf-8', 'replace').rstrip())


def plugin_loaded():
    ...


def plugin_unloaded():
    for (cmd, root_dir), server in running_servers.items():
        server.kill()
        print("SIGKILL", server.name)
