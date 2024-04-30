import io
import sys

from unittesting import DeferrableTestCase

from SublimeLinter.tests.parameterized import parameterized as p


linter_module = sys.modules['SublimeLinter-any-lsp.linter']


class TestMessageFormat(DeferrableTestCase):

    @p.expand([
        ({"method": "hello"}, b'Content-Length: 19\r\n\r\n{"method": "hello"}'),
    ])
    def testFormatMessage(self, input, expected):
        self.assertEqual(linter_module.format_message(input), expected)

    @p.expand([
        (b'Content-Length: 19\r\n\r\n{"method": "hello"}', {"method": "hello"}),
        (
            b'Content-Length: 19\r\n'
            b'Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n{"method": "hello"}',
            {"method": "hello"}
        ),
        (b'Ignored-Header: boo\r\nContent-Length: 19\r\n\r\n{"method": "hello"}', {"method": "hello"}),
        (b'\r\n\r\n{"method": "hello"}', None),
        (b'Content-Length: 19\r\n', None),
    ])
    def testParseMessage(self, input, expected):
        input_ = io.BytesIO(input)
        self.assertEqual(linter_module.parse_for_message(input_), expected)


class TestDictUnflattening(DeferrableTestCase):
    @p.expand([
        (
            {}, {}
        ),
        (
            {"foo": "bar"}, {"foo": "bar"}
        ),
        (
            {
                "textDocument.synchronization.didSave": True,
                "textDocument.publishDiagnostics.relatedInformation": True,
                "workspace.workspaceFolders": True,
                "window.workDoneProgress": True,
            },
            {
                "textDocument": {
                    "synchronization": {
                        "didSave": True,
                    },
                    "publishDiagnostics": {
                        "relatedInformation": True,
                    },
                },
                "workspace": {
                    "workspaceFolders": True,
                },
                "window": {
                    "workDoneProgress": True,
                },
            }
        ),
    ])
    def testUnflattenDict(self, input, expected):
        self.assertEqual(linter_module.unflatten(input), expected)
