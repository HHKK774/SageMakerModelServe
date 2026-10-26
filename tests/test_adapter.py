"""Local HTTP contract tests; no GPU or weights required."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import adapter

SSE = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'


class Backend(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data":[]}')

    def do_POST(self):
        self.server.payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.path = self.path
        body = SSE if self.server.payload.get('stream') else b'{"choices":[]}'
        self.send_response(400 if self.server.payload.get('fail') else 200)
        self.send_header('Content-Type', 'text/event-stream' if self.server.payload.get('stream') else 'application/json')
        self.end_headers()
        self.wfile.write(body)


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
        cls.front = ThreadingHTTPServer(('127.0.0.1', 0), adapter.Handler)
        cls.original = adapter.BACKEND
        adapter.BACKEND = f'http://127.0.0.1:{cls.backend.server_port}'
        for server in (cls.backend, cls.front):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        cls.url = f'http://127.0.0.1:{cls.front.server_port}'

    @classmethod
    def tearDownClass(cls):
        for server in (cls.front, cls.backend):
            server.shutdown()
            server.server_close()
        adapter.BACKEND = cls.original
        adapter.READY.clear()

    def setUp(self):
        adapter.READY.set()

    def request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        try:
            return urlopen(Request(self.url + path, data=body), timeout=3)
        except HTTPError as exc:
            return exc

    def test_readiness(self):
        adapter.READY.clear()
        with self.request('/ping') as r:
            self.assertEqual(r.status, 503)
        adapter.READY.set()
        for payload in (None, {}):
            with self.request('/ping', payload) as r:
                self.assertEqual(r.status, 200)

    def test_health_readiness(self):
        for ready, status, message in (
            (False, 503, 'llama-server not ready'),
            (True, 200, 'ready'),
        ):
            adapter.READY.set() if ready else adapter.READY.clear()
            for path in ('/health', '/health?check=1', '/ping'):
                with self.subTest(ready=ready, path=path):
                    with self.request(path) as r:
                        self.assertEqual(r.status, status)
                        self.assertEqual(json.load(r), {'message': message})

    def test_prompt_and_default_model(self):
        with self.request('/invocations', {'prompt': 'hello'}) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(r.read(), b'{"choices":[]}')
        self.assertEqual(self.backend.payload['messages'], [{'role': 'user', 'content': 'hello'}])
        self.assertEqual(self.backend.payload['model'], adapter.model_name())
        self.assertEqual(self.backend.path, '/v1/chat/completions')

    def test_sse_unchanged(self):
        with self.request('/invocations', {'messages': [], 'model': 'custom', 'stream': True}) as r:
            self.assertEqual(r.headers['Transfer-Encoding'], 'chunked')
            self.assertEqual(r.headers['Content-Type'], 'text/event-stream')
            self.assertEqual(r.read(), SSE)
        self.assertEqual(self.backend.payload['model'], 'custom')

    def test_passthrough_and_errors(self):
        for path in ('/v1/models', '/openai/v1/models'):
            with self.request(path) as r:
                self.assertEqual(r.read(), b'{"data":[]}')
        with self.request('/v1/chat/completions', {'fail': True}) as r:
            self.assertEqual(r.status, 400)
            self.assertEqual(r.read(), b'{"choices":[]}')
        with self.request('/invocations', {'prompt': 123}) as r:
            self.assertEqual(r.status, 400)
        with self.request('/unknown') as r:
            self.assertEqual(r.status, 404)

    def test_args(self):
        args = adapter.server_args({})
        self.assertIn(adapter.DEFAULT_MODEL, args)
        self.assertNotIn('--mmproj', args)
        self.assertNotIn('--jinja', args)
        args = adapter.server_args({'SM_LLAMACPP_JINJA': '', 'SM_LLAMACPP_MMPROJ': '/model/projector.gguf',
                                    'SM_LLAMACPP_CTX_SIZE': '1024', 'SM_EXTRA_ARGS': '--alias "my model"'})
        self.assertIn('--jinja', args)
        self.assertIn('/model/projector.gguf', args)
        self.assertIn('1024', args)
        self.assertEqual(args[-2:], ['--alias', 'my model'])

    def test_child_exit_clears_readiness(self):
        from unittest.mock import Mock
        adapter.STOP.clear()
        child = Mock(returncode=1)
        child.poll.return_value = 1
        adapter.monitor(child)
        self.assertFalse(adapter.READY.is_set())
        self.assertTrue(adapter.STOP.is_set())
        adapter.STOP.clear()


if __name__ == '__main__':
    unittest.main()
