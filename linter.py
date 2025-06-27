from __future__ import annotations

from functools import partial

import sublime

from SublimeLinter.lint import persist
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

from .core import AnyLSP, inflate, read_path

from typing import Iterator


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
def ignore_ruff_code(error: persist.LintError, view: sublime.View) -> Iterator[TextRange]:
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
            f"  # noqa: {code}",
            line
        )
    )


@quick_actions_for(RUFF_NAME)
def ruff_fixes_provider(
    errors: list[persist.LintError],
    _view: sublime.View | None
) -> Iterator[QuickAction]:
    def make_action(error: persist.LintError) -> QuickAction:
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

