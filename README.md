# SublimeLinter-any-lsp 🎴

**SublimeLinter-any-lsp** is a plugin for [SublimeLinter](https://github.com/SublimeLinter/SublimeLinter) that allows you to use Language Server Protocol (LSP) servers as linters within Sublime Text.

## Status

I use it since May 2024.  Needs spread and contributions.  Unsure about which parts (if any) should be ported to SublimeLinter core.
Lack of time to make it shine, also maybe, because it just works for me.

## How It Works

This package implements a generic linter backend (`AnyLSP`) for SublimeLinter. It communicates with LSP servers, sends file contents, and parses diagnostics messages from the server to display them as linting errors/warnings in Sublime Text.

- Each LSP server defined by a linter class (e.g., `Ruff`, `Pyright`, `Eslint`) as usual, inheriting `AnyLSP` instead of `Linter`.
- Diagnostic messages from the LSP server are mapped to SublimeLinter errors and warnings.

## Design

Be open, user-friendly, playful.

```python
# Make a config
config = ServerConfig(
    "ruff-lsp",
    ("ruff", "server", "--preview"),
    "/user/knowles/and-her-working-dir",
)
# and ask for a server attached to a view
ensure_server_for_view(config, view)
# or a plain server
server = ensure_server(config)
# and communicate with it
req = server.request("textDocument/diagnostic", inflate({
    "textDocument.uri": canoncial_uri_for_view(self.view),
}))

@req.on_response
def on_diagnostics(msg):
    ....

# done.  Idle servers shutdown automatically. 🌆
```

### Example Linter Classes

```python
class Ruff(AnyLSP):
    name = "ruff-lsp"
    cmd = ("ruff", "server", "--preview")
    defaults = {"selector": "source.python"}

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
```

## Installation

Manual only.  Not listed on PackageControl

## Usage

- Ships ruff-lsp, pyright and eslint out-of-the-box
- Any other LSP (-configuration) must be added here, as a separate (private or public) package, or just in your User folder as a plugin.

## Configuration

Each linter class may be configured through SublimeLinter's settings system. For example, to enable or disable specific servers, or to pass settings to the underlying LSP.

Example:

```json
"SublimeLinter.linters.ruff-lsp.selector": "source.python",
```

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or improvements.

## License

[MIT](LICENSE)
