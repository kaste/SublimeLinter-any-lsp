from __future__ import annotations

from typing import Literal, Union

from typing_extensions import TypeAlias

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

AllKnownMethods: TypeAlias = Union[
    Notifications, NotificationsWithParams, Requests, RequestsWithParams
]
