from .server import Server

from typing import Callable, Generic, Literal, TypedDict, TypeVar, Union, overload
from typing_extensions import Concatenate, NotRequired, ParamSpec, TypeAlias

class Message_(TypedDict, total=False):
    jsonrpc: str

class NotificationS(Message_):
    method: str

class Notification(Message_):
    method: str
    params: dict

class RequestS(Message_):
    id: int
    method: str

class Request(Message_):
    id: int
    method: str
    params: dict

class Response(Message_):
    id: int
    result: NotRequired[object]
    error: NotRequired[object]

AnyNotification: TypeAlias = Union[NotificationS, Notification]
AnyRequest: TypeAlias = Union[RequestS, Request]
Message: TypeAlias = Union[RequestS, Request, Response, NotificationS, Notification]


Notifications: TypeAlias = Literal[
    "exit"
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
    "workspace/didRenameFiles"
]

Requests: TypeAlias = Literal[
    "shutdown",
    "workspace/codeLens/refresh",
    "workspace/semanticTokens/refresh",
    "workspace/workspaceFolders"
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
    "workspace/willRenameFiles"
]


P = ParamSpec('P')
R = TypeVar('R', bound=Message)

class MessageHandler(Generic[R]):
    wanted_method: str
    def __init__(self, wanted_method: str, /) -> None: ...
    def __call__(self,
        fn: Callable[Concatenate[Server, R,       P], None]
    ) ->    Callable[Concatenate[Server, Message, P], None]: ...

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
