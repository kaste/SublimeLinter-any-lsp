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
import time
import traceback
from typing import (
    IO,
    Callable,
    Generic,
    Iterator,
    Literal,
    Mapping,
    Optional,
    TypedDict,
    TypeVar,
    Union,
    no_type_check,
)

from typing_extensions import Concatenate, NotRequired, ParamSpec, TypeAlias, get_args, overload

P = ParamSpec('P')
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

from . import _lsp
from .core.utils import Counter, inflate, read_path, run_on_new_thread, try_kill_proc

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
_counter = Counter()


class Message_(TypedDict, total=False):
    jsonrpc: str

class NotificationS(Message_):
    method: str

class Notification(Message_):
    method: str
    params: dict

AnyNotification = Union[NotificationS, Notification]

class RequestS(Message_):
    id: int
    method: str

class Request(Message_):
    id: int
    method: str
    params: dict

AnyRequest = Union[RequestS, Request]

class Response(Message_):
    id: int
    result: NotRequired[object]
    error: NotRequired[object]

Message = Union[RequestS, Request, Response, NotificationS, Notification]
Callback = Callable[["Server", Message], None]
MESSAGES_TO_TYPES = {
    _lsp.Notifications: NotificationS,
    _lsp.NotificationsWithParams: Notification,
    _lsp.Requests: RequestS,
    _lsp.RequestsWithParams: Request,
}


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


ServerName: TypeAlias = str
RootDir: TypeAlias = Optional[str]
ServerIdentity: TypeAlias = "tuple[ServerName, RootDir]"
ServerStates: TypeAlias = Literal[
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

    def kill(self):
        self.logger.info(f"SIGKILL {self.name}")
        self.killer()

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


running_servers: dict[ServerIdentity, Server] = {}
locks: dict[ServerIdentity, threading.Lock] = defaultdict(lambda: threading.Lock())



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
        with locks[config.identity()]:
            server = ensure_server(config)
            reason = (
                "on_modified"
                if get_server_for_view(self.view, self.name) == server
                else "on_load"
            )
            if reason == "on_load":
                remember_server_for_view(self.view, server)
                server.add_listener(
                    partial(diagnostics_handler, default_error_type=self.default_type)
                )

        if reason == "on_load":
            server.notify("textDocument/didOpen", inflate({
                "textDocument.uri": canoncial_uri_for_view(self.view),
                "textDocument.languageId": language_id_for_view(self.view),
                "textDocument.version": self.view.change_count(),
                "textDocument.text": code,
            }))

        elif reason == "on_modified":
            server.notify("textDocument/didChange", inflate({
                "textDocument.uri": canoncial_uri_for_view(self.view),
                "textDocument.version": self.view.change_count(),
                "contentChanges": [{ "text": code }]
            }))

        if server.has_capability("diagnosticProvider"):
            req = server.request("textDocument/diagnostic", inflate({
                "textDocument.uri": canoncial_uri_for_view(self.view),
            }))

            @req.on_response
            def on_diagnostics(msg):
                # print("on_diagnostics", self.name)
                try:
                    items = msg["result"]["items"]
                except KeyError:
                    pass
                else:
                    read_out_and_broadcast_errors(
                        self.name, self.view, items, self.default_type)

        raise TransientError("lsp's answer on their own will.")


# We want that `handler` turns specific message handlers into general message
# handlers.  Hence `R -> Message`.
R = TypeVar('R', bound=Message)

class MessageHandler(Generic[R]):
    """
    Internal helper class for the `handles` decorator factory.

    This *is* the actual decorator, just without the method to type inference.

    Attributes:
        wanted_method (str): The method name that this handler is interested in.

    Examples:
        @MessageHandler("shutdown")
        def handler_function(server, message: Message):
            pass

        notification = MessageHandler[Notification]
        request = MessageHandler[Request]
    """
    def __init__(self, wanted_method: str, /) -> None:
        self.wanted_method = wanted_method

    def __call__(self,
        fn: Callable[Concatenate[Server, R,       P], None]
    ) ->    Callable[Concatenate[Server, Message, P], None]:
        @wraps(fn)
        def wrapped(s, m, *args: P.args, **kwargs: P.kwargs) -> None:
            if m.get("method") == self.wanted_method:
                fn(s, m, *args, **kwargs)

        return wrapped


@overload
def handles(name: _lsp.Notifications) -> MessageHandler[NotificationS]: ...
@overload
def handles(name: _lsp.NotificationsWithParams) -> MessageHandler[Notification]: ...
@overload
def handles(name: _lsp.Requests) -> MessageHandler[RequestS]: ...
@overload
def handles(name: _lsp.RequestsWithParams) -> MessageHandler[Request]: ...
@overload
def handles(name: str, type_: type[R] | None = None) -> MessageHandler[R]: ...
@no_type_check
def handles(name, type_=None):
    """
    A decorator for message handlers that filters lsp messages based on their method name.

    Usage:
        # For known methods, the message type is checked according to the spec...
        @handles("shutdown")
        def handler_function(server, message: RequestS): ...

        # ...but can be overwritten.
        @handles("shutdown", Message)
        def handler_function(server, message: Message): ...

        # For unknown methods, the message type can be set arbitrarily.
        @handles("customMessage")            # <= no need to set `Notification` here
        def handler_function(server, message: Notification): ...  # <= but here

        # However, you can also create custom decorators for type safety.
        custom_message = handles("customMessage", Notification)
        custom_message: MessageHandler[Notification] = handles("customMessage")

        @custom_message
        def handler_function(server, message: Notification): ...  # <= type checked

    Available message types:
        Message
        AnyNotification, Notification, NotificationS
        AnyRequest, Request, RequestS

    The S-suffixed types are for messages without parameters.
    """
    if type_ is None:
        for methods, t in MESSAGES_TO_TYPES.items():
            if name in get_args(methods):
                type_ = t
                break
        else:
            type_ = Message  # <= this is a lie; R is unbound ("Never")
                             #    in accordance with the overload above
    return MessageHandler[type_](name)


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

    read_out_and_broadcast_errors(
        linter_name, view, msg["params"]["diagnostics"], default_error_type)


def read_out_and_broadcast_errors(
    linter_name: str,
    view: sublime.View,
    items: dict,
    default_error_type: str
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
                diagnostic.get("severity"),
                default_error_type),
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



RUFF_NAME = "ruff-lsp"


class Ruff(AnyLSP):
    name = RUFF_NAME
    cmd = ('ruff', 'server', '--preview')
    defaults = {
        # "disable": True,
        "selector": "source.python",
    }


class Pyright(AnyLSP):
    name = "pyright-lsp"
    cmd = ("pyright-langserver", "--stdio")
    defaults = {
        "disable": True,
        "selector": "source.python",
        "settings": {
            "python.analysis.autoSearchPaths": True,
            "python.analysis.diagnosticMode": "openFilesOnly",
            "python.analysis.useLibraryCodeForTypes": True,
        }
    }


class Eslint(AnyLSP):
    name = "eslint-lsp"
    cmd = ('vscode-eslint-language-server', '--stdio')
    defaults = {
        "env": {"DEBUG": "eslint:*,-eslint:code-path"},
        # "disable": True,
        "selector": "source.js",
        "settings": inflate({
            'validate': 'on',
            'packageManager': None,
            'useESLintClass': False,
            'experimental.useFlatConfig': False,
            # 'codeActionOnSave.enable': False,
            # 'codeActionOnSave.mode': 'all',
            'format': False,
            'quiet': False,
            "ignoreUntitled": False,
            "onIgnoredFiles": "off",
            'rulesCustomizations': [],
            'run': 'onType',
            'problems.shortenToSingleLine': False,
            'nodePath': None,
            'workingDirectory.mode': 'location',
            'codeAction.disableRuleComment.enable': True,
            'codeAction.disableRuleComment.location': 'separateLine',
            "codeAction.disableRuleComment.commentStyle": "line",
            # 'codeAction.showDocumentation.enable': True,
        })
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
        return read_path(error, "data.fix") is None
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


def attached_servers() -> set[ServerIdentity]:
    return {
        server.config.identity()
        for d in servers_attached_per_buffer.values()
        for server in d.values()
    }


WAIT_TIME = 0.5
ONE_MINUTE = 60
KEEP_ALIVE_USED_INTERVAL  = 10 * ONE_MINUTE
KEEP_ALIVE_UNUSED_INTERVAL = 5 * ONE_MINUTE

def cleanup_servers(*, keep_alive=(KEEP_ALIVE_USED_INTERVAL, KEEP_ALIVE_UNUSED_INTERVAL)) -> None:
    used_servers = attached_servers()
    current = time.monotonic()
    keep_alive_used, keep_alive_unused = keep_alive

    for identity, server in running_servers.items():
        if server.is_dead():
            continue

        idle_time = current - server.last_interaction
        max_idle_time = (
            keep_alive_used
            if identity in used_servers
            else keep_alive_unused
        )

        if idle_time > max_idle_time:
            server.logger.info(
                f"{server.name} idle for {idle_time:.1f}s (> {max_idle_time}s). Shutting down."
            )
            shutdown_server(server)


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
    server.logger.info("shutdown in time" if ok else "shutdown too slow")
    return ok



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


def next_id() -> int:
    _counter.inc()
    return _counter.count()


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


def plugin_loaded():
    ...


def plugin_unloaded():
    for server in running_servers.values():
        server.kill()
