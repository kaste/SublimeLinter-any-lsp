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
import copy
from dataclasses import dataclass, field
from functools import partial
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import (
    IO,
    Callable,
    Deque,
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
    "INIT", "INITIALIZE_REQUESTED", "READY", "SHUTDOWN_REQUESTED", "DEAD"]




@dataclass
class Server:
    name: str
    reader: IO[bytes]
    writer: IO[bytes]
    killer: Callable[[], None] = lambda: None
    state: ServerStates = "INIT"
    messages_out_queue: Deque[Message] = deque()
    pending_request_ids: dict[int, Callable[[Message], None]] = field(default_factory=dict)
    handlers: set[Callable[[Server, Message], None]] = field(default_factory=set)

    def on_message(self, handler: Callable[[Server, Message], None]) -> None:
        self.handlers.add(handler)

    def kill(self):
        self.killer()

    def reader_loop(self):
        while msg := parse_for_message(self.reader):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("<-", msg)
            else:
                try:
                    msg["id"]
                except KeyError:
                    print("<--", msg)
                else:
                    print("<-({})--".format(msg["id"]), msg)

            try:
                id = msg["id"]  # type: ignore[typeddict-item]
            except KeyError:
                ...
            else:
                try:
                    callback = self.pending_request_ids.pop(id)
                except KeyError:
                    ...
                else:
                    callback(msg)

            for handler in self.handlers.copy():
                try:
                    handler(self, msg)
                except Exception as e:
                    print(f"{handler} raised {e!r}")

        self.state = "DEAD"
        print(f"`{self.name}`> is now dead.")


    def request(self, message: Request, callback: Callable) -> None:
        msg: Request = {**{"id": next_id()}, **message}
        self.pending_request_ids[msg["id"]] = callback
        self.send(msg)

    def notify(self, method: str, params: dict = {}) -> None:
        self.send({"method": method, "params": params or {}})

    def write_message(self, message: Message) -> None:
        msg: Message = {**{"jsonrpc": "2.0"}, **message}

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
            logger.debug("->", sanitized)

        self.writer.write(encode_message(msg))
        self.writer.flush()


    def send(self, message: Message) -> None:
        if self.state == "DEAD":
            logger.warn("Server is already dead.")
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
            self.state = "DEAD"
            return

        if self.state == "SHUTDOWN_REQUESTED":
            logger.warn("Server shutdown has already been requested")
            return

        if self.state == "READY":
            self.write_message(message)
            return

        if self.state in ("INIT", "INITIALIZE_REQUESTED"):
            self.messages_out_queue.append(message)
            return


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



def ensure_server(name: str, cmd: list[str], root_dir: Optional[str]) -> Server:
    return get_server(name, root_dir) or start_server(name, cmd, root_dir)


running_servers: dict[tuple[str, Optional[str]], Server] = {}


def get_server(name: str, root_dir: Optional[str]) -> Server | None:
    if (server := running_servers.get((name, root_dir))) and server.state != "DEAD":
        return server
    return None


def start_server(name: str, cmd: list[str], root_dir: Optional[str]) -> Server:
    proc = subprocess.Popen(
        cmd,
        cwd=root_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        startupinfo=util.create_startupinfo(),
    )
    assert proc.stdin
    assert proc.stdout
    assert proc.stderr
    reader = proc.stdout
    writer = proc.stdin
    killer = partial(try_kill_proc, proc)
    server = Server(name, reader, writer, killer)
    running_servers[(name, root_dir)] = server
    run_on_new_thread(server.reader_loop)
    run_on_new_thread(std_err_printer, name, proc.stderr)
    folder_config = {
        "rootUri": Path(root_dir).as_uri() if root_dir else None,
        "rootPath": root_dir,
        "workspaceFolders": (
            [{"name": Path(root_dir).stem, "uri": Path(root_dir).as_uri()}]
            if root_dir else None
        ),
    }
    server.request({
        "method": "initialize",
        "params": {
            **CLIENT_INFO,
            **folder_config,
            "capabilities": MINIMAL_CAPABILITIES,
            "initializationOptions": {},
        }
    }, lambda _: server.send({"method": "initialized", "params": {}}))
    return server


class AnyLSP(Linter):
    __abstract__ = True

    def run(self, cmd: list[str] | None, code: str):
        if self.view.file_name() != __file__:
            raise PermanentError("only for this file.")
        if cmd is None:
            raise PermanentError("`cmd` must be defined.")

        # reasons: ("on_user_request", "on_save", "on_load", "on_modified")
        reason = (
            "on_modified"
            if get_server_for_view(self.view, self.name)
            else "on_load"
        )

        cwd = self.get_working_dir()
        server = ensure_server(self.name, cmd, cwd)
        server.on_message(diagnostics_handler)
        remember_server_for_view(self.view, self.name, server)
        if reason == "on_load":
            server.send({
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        # The text document's URI.
                        "uri": canoncial_uri_for_view(self.view),
                        # The text document's language identifier.
                        "languageId": language_id_for_view(self.view),
                        # The version number of this document (it will increase after each
                        # change, including undo/redo).
                        "version": self.view.change_count(),
                        # The content of the opened text document.
                        "text": code,
                    }
                }
            })

        elif reason in ("on_modified", "on_save"):
            server.send({
                "method": "textDocument/didChange",
                "params": {
                    "textDocument": {
                        "uri": canoncial_uri_for_view(self.view),
                        "version": self.view.change_count(),
                    },
                    "contentChanges": [{
                        "text": code
                    }]
                }
            })

        raise TransientError("lsp's answer on their own will.")


def diagnostics_handler(server: Server, msg: Message) -> None:
    try:
        method = msg["method"]
    except KeyError:
        return
    else:
        if method != "textDocument/publishDiagnostics":
            # print(f"skip: wrong method: {method}")
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

    errors: list[persist.LintError] = [
        {
            **error,  # type: ignore[typeddict-item]
            **{
                "uid": make_error_uid(error),
                "priority": style.get_value("priority", error, 0)
            }
        }
        for diagnostic in msg["params"]["diagnostics"]
        if (region := sublime.Region(
            view.text_point_utf16(
                diagnostic["range"]["start"]["line"],
                diagnostic["range"]["start"]["character"],
                clamp_column=True
            ),
            view.text_point_utf16(
                diagnostic["range"]["end"]["line"],
                diagnostic["range"]["end"]["character"],
                clamp_column=True
            ),
        )) or True
        if (error := {
            "linter": linter_name,
            "filename": file_name,
            "msg": diagnostic["message"],
            "code": diagnostic.get("code", ""),
            "error_type": severity_to_type(diagnostic.get("severity")),
            "line": diagnostic["range"]["start"]["line"],
            "start": diagnostic["range"]["start"]["character"],
            "region": region,
            "offending_text": view.substr(region)
        })
    ]
    # print("sublime_linter.update_file_errors", sublime_linter.update_file_errors)
    # print(file_name, linter_name, errors)
    sublime_linter.update_file_errors(file_name, linter_name, errors, reason=None)


DEFAULT_TYPE = "error"


def severity_to_type(severity: str | None, default=DEFAULT_TYPE) -> str:
    return {  # type: ignore[call-overload]
        1: "error", 2: "warning", 3: "info", 4: "hint"
    }.get(severity, default)


servers_attached_per_buffer: dict[int, dict[str, Server]] = defaultdict(dict)


def remember_server_for_view(view: sublime.View, name: str, server: Server) -> None:
    servers_attached_per_buffer[view.buffer_id()][name] = server


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


def shutdown_server(server: Server) -> None:
    def on_shutdown_response(_):
        server.send({"method": "exit"})
        try:
            server.wait(0.5)
        except TimeoutError:
            server.kill()
            print("kill", server.name)

    server.request({"method": "shutdown"}, on_shutdown_response)


def cleanup_servers() -> None:
    used_servers = {
        (server.name, server.root_dir)
        for d in servers_attached_per_buffer.values()
        for server in d.values()
    }
    unused_servers = running_servers.keys() - used_servers
    for name, root_dir in unused_servers:
        server = running_servers.get((name, root_dir))
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


class Ruff(AnyLSP):
    name = "ruff-lsp"
    cmd = ('ruff', 'server', '--preview')
    defaults = {
        "selector": "source.python"
    }


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


def std_err_printer(name: str, stream: IO[bytes]) -> None:
    for line in stream:
        print(f"<{name}>", line.decode('utf-8', 'replace').rstrip())


def plugin_loaded():
    ...


def plugin_unloaded():
    for (cmd, root_dir), server in running_servers.items():
        server.kill()
        print("SIGKILL", server.name)
