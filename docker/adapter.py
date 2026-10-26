#!/usr/bin/env python3
"""Dependency-free SageMaker frontend for a supervised llama-server."""
import json
import os
import shlex
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler

BACKEND = "http://127.0.0.1:58585"
DEFAULT_MODEL = "/opt/ml/model/Qwen3.8-27B-Uncensored-Q6_K.gguf"
OPENER = build_opener(ProxyHandler({}))
READY = threading.Event()
STOP = threading.Event()


def model_name():
    return os.path.basename(os.environ.get("SM_LLAMACPP_MODEL", DEFAULT_MODEL))


def server_args(env=None):
    env = os.environ if env is None else env
    args = ["/app/llama-server", "--host", "127.0.0.1", "--port", "58585",
            "--model", env.get("SM_LLAMACPP_MODEL", DEFAULT_MODEL),
            "--n-gpu-layers", env.get("SM_LLAMACPP_N_GPU_LAYERS", "999"),
            "--ctx-size", env.get("SM_LLAMACPP_CTX_SIZE", "32768")]
    if env.get("SM_LLAMACPP_MMPROJ"):
        args += ["--mmproj", env["SM_LLAMACPP_MMPROJ"]]
    if "SM_LLAMACPP_JINJA" in env:
        args += ["--jinja"]
    return args + shlex.split(env.get("SM_EXTRA_ARGS", ""))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("adapter: " + fmt % args, flush=True)

    def reply(self, status, message):
        body = json.dumps({"message": message}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)

    def do_GET(self):
        self.handle_request()

    def do_POST(self):
        self.handle_request()

    def handle_request(self):
        path = self.path
        if path.split("?", 1)[0] in ("/ping", "/health"):
            self.reply(200 if READY.is_set() else 503,
                       "ready" if READY.is_set() else "llama-server not ready")
            return
        if path.startswith("/openai/v1/"):
            path = path[len("/openai"):]
        invocation = path == "/invocations" and self.command == "POST"
        if not invocation and not path.startswith("/v1/"):
            self.reply(404, "not found")
            return
        if not READY.is_set():
            self.reply(503, "llama-server not ready")
            return
        if self.headers.get("Transfer-Encoding"):
            self.reply(400, "request must use Content-Length")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 32 * 1024 * 1024:
                self.reply(413, "request too large")
                return
            self.connection.settimeout(300)
            body = self.rfile.read(length) if self.command == "POST" else None
            if invocation:
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("expected a JSON object")
                if "messages" not in payload and isinstance(payload.get("prompt"), str):
                    payload["messages"] = [{"role": "user", "content": payload.pop("prompt")}]
                if not isinstance(payload.get("messages"), list):
                    raise ValueError("messages array or prompt string required")
                payload.setdefault("model", model_name())
                body = json.dumps(payload).encode()
                path = "/v1/chat/completions"
        except (ValueError, UnicodeError) as exc:
            self.reply(400, str(exc))
            return
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json"),
                   "Accept": self.headers.get("Accept", "*/*")}
        # Do not forward SageMaker bearer tokens to the inference backend.
        request = Request(BACKEND + path, data=body, headers=headers, method=self.command)
        try:
            response = OPENER.open(request, timeout=300)
        except HTTPError as exc:
            response = exc  # Preserve backend error status and body.
        except (URLError, OSError):
            self.reply(502, "llama-server unavailable")
            return
        with response:
            self.send_response(response.code)
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                while True:
                    # read1 returns available bytes, rather than buffering entire SSE events.
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (OSError, ValueError) as exc:
                self.log_message("stream closed: %s", exc)


def monitor(child):
    while not STOP.is_set():
        if child.poll() is not None:
            READY.clear()
            print(f"llama-server exited with status {child.returncode}", flush=True)
            STOP.set()
            return
        try:
            with OPENER.open(BACKEND + "/health", timeout=2) as response:
                healthy = response.status == 200
        except (URLError, OSError):
            healthy = False
        if healthy and not READY.is_set():
            print("llama-server ready; /ping now returns 200", flush=True)
        if healthy:
            READY.set()
        else:
            READY.clear()
        STOP.wait(1)


def main():
    # SageMaker's trailing `serve` argument is deliberately ignored.
    def stop(signum, frame):
        READY.clear()
        STOP.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print("Starting llama-server on 127.0.0.1:58585; adapter on 0.0.0.0:8080", flush=True)
    child = subprocess.Popen(server_args())
    try:
        with ThreadingHTTPServer(("0.0.0.0", 8080), Handler) as server:
            server.timeout = 0.5
            threading.Thread(target=monitor, args=(child,), daemon=True).start()
            while not STOP.is_set():
                server.handle_request()
    finally:
        READY.clear()
        STOP.set()
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    return 1 if child.returncode not in (0, -signal.SIGTERM) else 0


if __name__ == "__main__":
    raise SystemExit(main())
