from __future__ import annotations

from functools import wraps

from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    TypedDict,
    TypeVar,
    Union,
)
from typing_extensions import (
    Concatenate,
    NotRequired,
    ParamSpec,
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

AnyNotification = Union[NotificationS, Notification]
AnyRequest = Union[RequestS, Request]
Message = Union[RequestS, Request, Response, NotificationS, Notification]

# ===
# Attention! The following types are heavily overwritten in the corresponding
# `.pyi` file.  The types herein are good-enough to demonstrate the intent.
# ===

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


def handles(name: str, type_: type[R] | None = None) -> MessageHandler[R]:
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
    return MessageHandler(name)
