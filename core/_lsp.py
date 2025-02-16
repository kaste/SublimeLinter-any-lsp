from __future__ import annotations

from functools import wraps

from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    Literal,
    TypedDict,
    TypeVar,
    Union,
    no_type_check,
)
from typing_extensions import (
    Concatenate,
    NotRequired,
    ParamSpec,
    TypeAlias,
    get_args,
    overload,
)


if TYPE_CHECKING:
    from .server import Server


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


# The following enumeration had been extracted from the 3.16 spec
Notifications: TypeAlias = Literal[
    "exit",
]

NotificationsWithParams: TypeAlias = Literal[
    "$/cancelRequest",
    "$/logTrace",
    "$/progress",
    "$/setTrace",
    "initialized",
    "telemetry/event",
    "textDocument/didChange",
    "textDocument/didClose",
    "textDocument/didOpen",
    "textDocument/didSave",
    "textDocument/publishDiagnostics",
    "textDocument/willSave",
    "window/logMessage",
    "window/showMessage",
    "window/workDoneProgress/cancel",
    "workspace/didChangeConfiguration",
    "workspace/didChangeWatchedFiles",
    "workspace/didChangeWorkspaceFolders",
    "workspace/didCreateFiles",
    "workspace/didDeleteFiles",
    "workspace/didRenameFiles",
]

Requests: TypeAlias = Literal[
    "shutdown",
    "workspace/codeLens/refresh",
    "workspace/semanticTokens/refresh",
    "workspace/workspaceFolders",
]

RequestsWithParams: TypeAlias = Literal[
    "callHierarchy/incomingCalls",
    "callHierarchy/outgoingCalls",
    "client/registerCapability",
    "client/unregisterCapability",
    "codeAction/resolve",
    "codeLens/resolve",
    "completionItem/resolve",
    "documentLink/resolve",
    "initialize",
    "textDocument/codeAction",
    "textDocument/codeLens",
    "textDocument/colorPresentation",
    "textDocument/completion",
    "textDocument/declaration",
    "textDocument/definition",
    "textDocument/documentColor",
    "textDocument/documentHighlight",
    "textDocument/documentLink",
    "textDocument/documentSymbol",
    "textDocument/foldingRange",
    "textDocument/formatting",
    "textDocument/hover",
    "textDocument/implementation",
    "textDocument/linkedEditingRange",
    "textDocument/moniker",
    "textDocument/onTypeFormatting",
    "textDocument/prepareCallHierarchy",
    "textDocument/prepareRename",
    "textDocument/rangeFormatting",
    "textDocument/references",
    "textDocument/rename",
    "textDocument/selectionRange",
    "textDocument/semanticTokens/full",
    "textDocument/semanticTokens/full/delta",
    "textDocument/semanticTokens/range",
    "textDocument/signatureHelp",
    "textDocument/typeDefinition",
    "textDocument/willSaveWaitUntil",
    "window/showDocument",
    "window/showMessageRequest",
    "window/workDoneProgress/create",
    "workspace/applyEdit",
    "workspace/configuration",
    "workspace/executeCommand",
    "workspace/symbol",
    "workspace/willCreateFiles",
    "workspace/willDeleteFiles",
    "workspace/willRenameFiles",
]
# End

AllKnownMethods: TypeAlias = Union[
    Notifications, NotificationsWithParams, Requests, RequestsWithParams
]
MESSAGES_TO_TYPES = {
    Notifications: NotificationS,
    NotificationsWithParams: Notification,
    Requests: RequestS,
    RequestsWithParams: Request,
}


# We want that `handler` turns specific message handlers into general message
# handlers.  Hence `R -> Message`.
P = ParamSpec('P')
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
def handles(name: Notifications) -> MessageHandler[NotificationS]: ...
@overload
def handles(name: NotificationsWithParams) -> MessageHandler[Notification]: ...
@overload
def handles(name: Requests) -> MessageHandler[RequestS]: ...
@overload
def handles(name: RequestsWithParams) -> MessageHandler[Request]: ...
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

