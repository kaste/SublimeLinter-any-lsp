from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import Future
import copy
from dataclasses import dataclass, field
from functools import partial
import inspect
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import threading
import time
import traceback

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

from ._lsp import AnyRequest, Message, Notification, Request, Response, handles
from .utils import Counter, inflate, read_path, run_on_new_thread, try_kill_proc

from typing import (
    IO,
    Callable,
    Dict,
    Generic,
    Iterator,
    Literal,
    Mapping,
    Optional,
    TypeVar,
)
from typing_extensions import (
    TypeAlias,
    overload,
)


__all__ = [
    "plugin_loaded",
    "plugin_unloaded",
    "DocumentListener",
    "ActivityMonitor",
]


def plugin_loaded():
    ...


def plugin_unloaded():
    for server in running_servers.values():
        server.kill()


T = TypeVar('T')

logger = logging.getLogger('SublimeLinter.plugin.lsp')
LOG_SEVERITY_MAP = {
    1: logging.ERROR,
    2: logging.WARNING,
    3: logging.INFO,
    4: logging.DEBUG
}


JSON_RPC_MESSAGE = "Content-Length: {}\r\n\r\n{}"
DEFAULT_ERROR_TYPE = "error"
SEVERITY_TO_ERROR_TYPE = {
    1: "error", 2: "warning", 3: "info", 4: "hint"
}

CLIENT_INFO = inflate({
    "processId": os.getpid(),
    "clientInfo.name": "SublimeLinter",
    "clientInfo.version": "4",
})
MINIMAL_CAPABILITIES = inflate({
    "textDocument.diagnostic.dynamicRegistration": True,
    "textDocument.diagnostic.relatedDocumentSupport": True,
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

ServerName: TypeAlias = str
RootDir: TypeAlias = Optional[str]
ServerIdentity: TypeAlias = "tuple[ServerName, RootDir]"
ServerStates: TypeAlias = Literal[
    "INIT", "INITIALIZE_REQUESTED", "READY", "SHUTDOWN_REQUESTED", "EXIT_REQUESTED", "DEAD"]
Callback = Callable[["Server", Message], None]
_counter = Counter()
next_id = _counter.next


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

    def identity(self) -> ServerIdentity:
        return (self.name, self.root_dir)


class Server:
    def __init__(
        self,
        config: ServerConfig,
        reader: IO[bytes],
        writer: IO[bytes],
        killer: Callable[[], None] = lambda: None,
        handlers: dict[str, Callback] = {}
    ):
        self.config = config
        self.reader = reader
        self.writer = writer
        self.killer = killer
        self.handlers = handlers.copy()
        self.logger = logging.getLogger(f"SublimeLinter.plugin.{self.name}")
        self.capabilities: dict[str, object] = {}
        self.last_interaction: float = time.monotonic()

        # Initialize state fields
        self.state: ServerStates = "INIT"
        self.messages_out_queue: deque[Message] = deque()
        self.pending_request_ids: dict[int, Future[Message]] = {}

        # Set up reader thread and lock
        self._writer_lock = threading.Lock()
        self._reader_thread = run_on_new_thread(self.reader_loop)

    @property
    def name(self) -> str:
        return self.config.name

    def has_capability(self, path: str) -> bool:
        return bool(read_path(self.capabilities, path))

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

    def wait(self, timeout: float) -> bool:
        return join_thread(self._reader_thread, timeout)

    @overload
    def kill(self, timeout: float) -> bool: ...
    @overload
    def kill(self) -> None: ...
    def kill(self, timeout: float | None = None) -> bool | None:
        self.logger.info(f"SIGKILL {self.name}")
        self.killer()
        if timeout is None:
            return None
        else:
            return self.wait(timeout)

    def request(self, method: str, params: dict = {}) -> OkFuture[Message]:
        fut: Future[Message]
        id = next_id()
        msg: AnyRequest = {"id": id, "method": method, "params": params.copy()}
        self.pending_request_ids[id] = fut = OkFuture()
        self.send(msg)
        return fut

    def respond(self, id: int, result: object = {}):
        msg: Response = {"id": id, "result": result}
        self.send(msg)

    def notify(self, method: str, params: dict = {}) -> None:
        self.send({"method": method, "params": params.copy()})

    def in_shutdown_phase(self) -> bool:
        return self.state in ("SHUTDOWN_REQUESTED", "EXIT_REQUESTED", "DEAD")

    def is_dead(self) -> bool:
        return self.state == "DEAD"

    def send(self, message: Message) -> None:
        # print("send message:", message)
        if self.state == "DEAD":
            self.logger.warn("Server is already dead.")
            return

        if self.state == "EXIT_REQUESTED":
            self.logger.warn("Server `exit` has already been requested")
            return

        method = message.get("method")
        if method == "initialize":
            if self.state == "INIT":
                self.write_message(message)
                self.state = "INITIALIZE_REQUESTED"
            else:
                self.logger.warn("`initialize` only valid in INIT state")
            return

        if method == "initialized":
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

        if method == "shutdown":
            if self.state == "READY":
                self.write_message(message)
                self.state = "SHUTDOWN_REQUESTED"
            else:
                self.logger.warn("`shutdown` only valid in READY state")
            return

        if method == "exit":
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

    def write_message(self, message: Message) -> None:
        msg: Message = {"jsonrpc": "2.0", **message}  # type: ignore[assignment]

        if self.logger.isEnabledFor(logging.DEBUG):
            sanitized = sanitize_message(msg)
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
            self.last_interaction = time.monotonic()

    def reader_loop(self):
        while msg := parse_for_message(self.reader):
            self.last_interaction = time.monotonic()
            # print(f"{self.name} <- {msg}")
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
                    handler(self, msg)
                except Exception:
                    traceback.print_exc()

        self.state = "DEAD"
        self.logger.info(f"`{self.name}`> is now dead.")
        print(f"`{self.name}`> is now dead.")


def sanitize_message(msg: Mapping) -> Mapping:
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
    return sanitized


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


### server lifetime


class RunningServers(Dict[ServerIdentity, Server]):
    def __call__(self) -> Iterator[Server]:
        self.gc()
        return iter(self.values())

    def gc(self) -> None:
        for identity, server in list(self.items()):
            if server.is_dead():
                del self[identity]


running_servers: RunningServers = RunningServers()
locks: dict[ServerIdentity, threading.Lock] = defaultdict(lambda: threading.Lock())
servers_attached_per_buffer: dict[int, dict[str, Server]] = defaultdict(dict)


def remember_server_for_view(view: sublime.View, server: Server) -> None:
    servers_attached_per_buffer[view.buffer_id()][server.name] = server


def get_server_for_view(view: sublime.View, name: str) -> Server | None:
    return servers_attached_per_buffer.get(view.buffer_id(), {}).get(name)


def attach_view_to_server(server: Server, view: sublime.View) -> bool:
    bid = view.buffer_id()
    name = server.name
    if server != servers_attached_per_buffer.get(bid, {}).get(name):
        servers_attached_per_buffer[bid][name] = server
        return True
    return False


def ensure_server_for_view(config: ServerConfig, view: sublime.View) -> tuple[Server, bool]:
    with locks[config.identity()]:
        server = ensure_server(config)
        attached = attach_view_to_server(server, view)
        return server, attached


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
    server.add_listener(on_workspace_configuration)
    server.add_listener(on_log_message)

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
        if config.settings:
            server.notify("workspace/didChangeConfiguration", {"settings": config.settings})
        server.notify("initialized")

    return server


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
        server.notify("textDocument/didClose", inflate({"textDocument.uri": uri}))


def attached_servers() -> set[Server]:
    return {
        server
        for d in servers_attached_per_buffer.values()
        for server in d.values()
    }


WAIT_TIME = 0.5
ONE_MINUTE = 60
KEEP_ALIVE_USED_INTERVAL  = 10 * ONE_MINUTE
KEEP_ALIVE_UNUSED_INTERVAL = 5 * ONE_MINUTE
USER_IS_IDLE_INTERVAL        = 5 * ONE_MINUTE
ACTIVITY_INTERVAL_AFTER_IDLE = 1 * ONE_MINUTE
last_activity: float = time.monotonic()

class ActivityMonitor(sublime_plugin.EventListener):
    def on_modified_async(self, view: sublime.View) -> None:
        global last_activity
        if random.random() > 0.01:
            return

        now, then = time.monotonic(), last_activity
        delta = now - then
        if delta > USER_IS_IDLE_INTERVAL:
            # User was idle and starts again; don't shutdown
            # all servers but wait for ACTIVITY_INTERVAL_AFTER_IDLE
            last_activity = now + ACTIVITY_INTERVAL_AFTER_IDLE
            # Because we shift into the future, delta is < 0
            # for that interval.
        elif delta > 0:
            last_activity = now
            cleanup_servers()


def cleanup_servers(
    *,
    keep_alive: tuple[float, float] = (KEEP_ALIVE_USED_INTERVAL, KEEP_ALIVE_UNUSED_INTERVAL)
) -> None:
    used_servers = attached_servers()
    current = time.monotonic()
    keep_alive_used, keep_alive_unused = keep_alive

    servers_to_shutdown = []
    for server in running_servers():
        idle_time = current - server.last_interaction
        max_idle_time = (
            keep_alive_used
            if server in used_servers
            else keep_alive_unused
        )

        if idle_time > max_idle_time:
            servers_to_shutdown.append((server, idle_time, max_idle_time))

    def fx():
        for server, idle_time, max_idle_time in servers_to_shutdown:
            server.logger.info(
                f"{server.name} idle for {idle_time:.1f}s (> {max_idle_time}s). Shutting down."
            )
            print(
                f"{server.name} idle for {idle_time:.1f}s (> {max_idle_time}s). Shutting down."
            )
            shutdown_server(server)

    if servers_to_shutdown:
        run_on_new_thread(fx)


def shutdown_server(server: Server, timeout: float = WAIT_TIME) -> bool:
    if server.in_shutdown_phase():
        return server.wait(timeout) or server.kill(timeout)

    ok = shutdown_server_(server, timeout)
    server.logger.info("shutdown in time" if ok else "shutdown too slow")
    return ok


def shutdown_server_(server: Server, timeout: float = WAIT_TIME) -> bool:
    req = server.request("shutdown")
    try:
        req.wait(timeout)
    except TimeoutError:
        req.cancel()
        return server.kill(timeout)
    else:
        server.notify("exit")
        return server.wait(timeout) or server.kill(timeout)


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


### Linter


class AnyLSP(Linter):
    __abstract__ = True

    def run(self, cmd: list[str] | None, code: str):
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
        server, attached = ensure_server_for_view(config, self.view)
        if attached:
            server.add_listener(
                partial(diagnostics_handler, default_error_type=self.default_type)
            )
            server.notify("textDocument/didOpen", inflate({
                "textDocument.uri": canoncial_uri_for_view(self.view),
                "textDocument.languageId": language_id_for_view(self.view),
                "textDocument.version": self.view.change_count(),
                "textDocument.text": code,
            }))

        else:
            server.notify("textDocument/didChange", inflate({
                "textDocument.uri": canoncial_uri_for_view(self.view),
                "textDocument.version": self.view.change_count(),
                "contentChanges": [{ "text": code }]
            }))

        if server.has_capability("diagnosticProvider"):
            cc = self.view.change_count()
            reason = self.context.get("reason")
            req = server.request("textDocument/diagnostic", inflate({
                "textDocument.uri": canoncial_uri_for_view(self.view),
            }))

            @req.on_response
            def on_diagnostics(msg):
                # print("on_diagnostics", self.name)
                try:
                    result = msg["result"]
                except KeyError:
                    self.logger.info(
                        f"no result for `textDocument/diagnostic`. "
                        f"error: {msg['error']['message']}"
                    )
                    return

                items = result.get("items")
                if items is not None:
                    parse_diagnostics(server, self.view, cc, items, self.default_type, reason)

                related_documents = result.get("relatedDocuments")
                if related_documents is not None:
                    for uri, report in related_documents.items():
                        items = report.get("items")
                        if items is not None:
                            parse_diagnostics(server, uri, None, items, self.default_type, reason)

        raise TransientError("lsp's answer on their own will.")


@handles("window/logMessage")
def on_log_message(server: Server, msg: Notification) -> None:
    server.logger.log(
        LOG_SEVERITY_MAP.get(msg["params"]["type"], logging.WARNING),
        msg["params"]["message"]
    )


@handles("workspace/configuration")
def on_workspace_configuration(server: Server, msg: Request) -> None:
    # print("--> msg", msg)
    result = [
        inflate({
            k:v
            for k, v in server.config.settings.items()
            if not (section := item.get("section")) or k.startswith(section)
        })
        for item in msg["params"]["items"]
    ]
    server.respond(msg["id"], result)


@handles("textDocument/publishDiagnostics")
def diagnostics_handler(
    server: Server, msg: Notification,
    *,
    default_error_type: str = "error"
) -> None:
    # print("diagnostics_handler", server.name)
    # print("diagnostics_handler--")
    # print("msg", msg)
    uri = msg["params"]["uri"]
    items = msg["params"]["diagnostics"]
    version = msg["params"]["version"]
    parse_diagnostics(server, uri, version, items, default_error_type, "on_modified")


def parse_diagnostics(
    server: Server,
    target: sublime.View | str,
    version: int | None,
    items: dict,
    default_error_type: str,
    reason: str | None
) -> None:
    linter_name = server.name
    if isinstance(target, str):
        file_name = canonical_name_from_uri(target)
        view = view_for_file_name(file_name)
        if not view:
            server.logger.info(f"skip: no view for {file_name}")
            return
    else:
        view = target

    if version and view.change_count() != version:
        server.logger.info(f"skip: view has changed. {view.change_count()} -> {version}")
        return

    read_out_and_broadcast_errors(linter_name, view, items, default_error_type, reason)


def read_out_and_broadcast_errors(
    linter_name: str,
    view: sublime.View,
    items: dict,
    default_error_type: str,
    reason: str | None
):
    file_name = util.canonical_filename(view)
    errors: list[persist.LintError] = []
    for diagnostic in items:
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
            "error_type": SEVERITY_TO_ERROR_TYPE.get(
                diagnostic.get("severity"), default_error_type
            ),
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
        sublime_linter.update_file_errors, file_name, linter_name, errors, reason
    ))



### Helper


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


def to_uri(path: str) -> str:
    return Path(path).as_uri()


def from_uri(uri: str) -> str:  # roughly taken from Python 3.13
    """Return a new path from the given 'file' URI."""
    from urllib.parse import unquote_to_bytes
    if not uri.startswith('file:'):
        raise ValueError(f"URI does not start with 'file:': {uri!r}")
    path = os.fsdecode(unquote_to_bytes(uri))
    path = path[5:]
    if path[:3] == '///':
        # Remove empty authority
        path = path[2:]
    elif path[:12] == '//localhost/':
        # Remove 'localhost' authority
        path = path[11:]
    if path[:3] == '///' or (path[:1] == '/' and path[2:3] in ':|'):
        # Remove slash before DOS device/UNC path
        path = path[1:]
        path = path[0].upper() + path[1:]
    if path[1:2] == '|':
        # Replace bar with colon in DOS drive
        path = path[:1] + ':' + path[2:]
    path_ = Path(path)
    if not path_.is_absolute():
        raise ValueError(f"URI is not absolute: {uri!r}.  Parsed so far: {path_!r}")
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
