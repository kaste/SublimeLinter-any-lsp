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
        self.assertEqual(linter_module.encode_message(input), expected)

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
        self.assertEqual(linter_module.inflate(input), expected)


class TestReadPath(DeferrableTestCase):
    @p.expand([
        (
            {"foo": {"bar": "baz"}}, "foo.bar", "baz"
        ),
        (
            {"foo": {"bar": "baz"}}, "foo.baz", None
        ),
        (
            {"foo": {"bar": "baz"}}, "foo", {"bar": "baz"}
        ),
        (
            {"foo": {"bar": "baz"}}, "foo.bar.baz", None
        ),
        (
            {}, "non.existent.path", None
        ),
    ])
    def testReadPath(self, input, path, expected):
        self.assertEqual(linter_module.read_path(input, path), expected)

    def testReadPathDefault(self):
        actual = linter_module.read_path(
            {"foo": {"bar": "baz"}},
            "foo.baz",
            "default"
        )
        self.assertEqual(actual, "default")


