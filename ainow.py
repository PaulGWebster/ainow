#!/usr/bin/env python3
"""ainow — a small multi-provider agentic coding harness.

Usage:  ainow <provider>/<model>  [initial prompt ...]

Keys:   Ctrl-C  interrupt current generation / clear line  (does NOT exit)
        Ctrl-D  exit
        Ctrl-Q  exit
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

HOME = pathlib.Path.home()
CFG_DIR = HOME / ".config" / "ainow"
PROVIDERS_FILE = CFG_DIR / "providers.json"

# Built-in registry of free/public OpenAI-compatible models
# Each entry maps to a provider-compatible config with base_url + env_var
PUBLIC_MODELS: dict[str, dict] = {
    "public/gemini-2.0-flash": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
        "env_var": "GEMINI_API_KEY",
    },
    "public/gemini-2.5-pro": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-pro",
        "env_var": "GEMINI_API_KEY",
    },
    "public/mistral-large": {
        "base_url": "https://api.mistral.ai/v1/",
        "model": "mistral-large-latest",
        "env_var": "MISTRAL_API_KEY",
    },
    "public/mistral-small": {
        "base_url": "https://api.mistral.ai/v1/",
        "model": "mistral-small-latest",
        "env_var": "MISTRAL_API_KEY",
    },
    "public/groq-llama-3.3": {
        "base_url": "https://api.groq.com/openai/v1/",
        "model": "llama-3.3-70b-versatile",
        "env_var": "GROQ_API_KEY",
    },
    "public/groq-llama-4-scout": {
        "base_url": "https://api.groq.com/openai/v1/",
        "model": "llama-4-scout-17b-16e-instruct",
        "env_var": "GROQ_API_KEY",
    },
    "public/cerebras-llama-3.3": {
        "base_url": "https://api.cerebras.ai/v1/",
        "model": "llama3.3-70b",
        "env_var": "CEREBRAS_API_KEY",
    },
    "public/cerebras-qwen3-235b": {
        "base_url": "https://api.cerebras.ai/v1/",
        "model": "qwen3-235b",
        "env_var": "CEREBRAS_API_KEY",
    },
    "public/cerebras-gpt-oss-120b": {
        "base_url": "https://api.cerebras.ai/v1/",
        "model": "gpt-oss-120b",
        "env_var": "CEREBRAS_API_KEY",
    },
    "public/nvidia-llama-3.3": {
        "base_url": "https://integrate.api.nvidia.com/v1/",
        "model": "meta/llama-3.3-70b-instruct",
        "env_var": "NVIDIA_API_KEY",
    },
    "public/nvidia-mistral-large": {
        "base_url": "https://integrate.api.nvidia.com/v1/",
        "model": "mistralai/mistral-large-2-instruct",
        "env_var": "NVIDIA_API_KEY",
    },
    "public/github-gpt-4o": {
        "base_url": "https://models.inference.ai.azure.com/",
        "model": "gpt-4o",
        "env_var": "GITHUB_TOKEN",
    },
    "public/github-deepseek-r1": {
        "base_url": "https://models.inference.ai.azure.com/",
        "model": "DeepSeek-R1",
        "env_var": "GITHUB_TOKEN",
    },
    "public/github-llama-3.3": {
        "base_url": "https://models.inference.ai.azure.com/",
        "model": "Llama-3.3-70B-Instruct",
        "env_var": "GITHUB_TOKEN",
    },
    "public/cohere-command-a": {
        "base_url": "https://api.cohere.com/v1/",
        "model": "command-a",
        "env_var": "COHERE_API_KEY",
    },
    "public/openrouter-deepseek-r1": {
        "base_url": "https://openrouter.ai/api/v1/",
        "model": "deepseek/deepseek-r1",
        "env_var": "OPENROUTER_API_KEY",
    },
    "public/openrouter-llama-3.3": {
        "base_url": "https://openrouter.ai/api/v1/",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "env_var": "OPENROUTER_API_KEY",
    },
}
MODELS_CACHE = CFG_DIR / "models.json"
HISTORY_FILE = CFG_DIR / "history"
PROMPT_FMT_FILE = CFG_DIR / "prompt.format"
PREPROMPT_FILE = CFG_DIR / "preprompt.format"
POSTPROMPT_FILE = CFG_DIR / "postprompt.format"
CACHE_TTL = 24 * 3600

MAX_TOOL_OUTPUT = 30_000

LOG_DIR = CFG_DIR / "logs"
HTTPD_LOG = LOG_DIR / "httpd.log"
AINOW_LOG = LOG_DIR / "ainow.log"
HTTPD_PIDFILE = CFG_DIR / "httpd.pid"
DEFAULT_HTTPD_ROOT = CFG_DIR / "httpd"


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def load_providers() -> dict:
    if not PROVIDERS_FILE.exists():
        sys.exit(f"ainow: no config at {PROVIDERS_FILE}")
    return json.loads(PROVIDERS_FILE.read_text())["providers"]


def load_model_cache(*, force: bool = False) -> dict:
    if MODELS_CACHE.exists():
        try:
            cache = json.loads(MODELS_CACHE.read_text())
        except json.JSONDecodeError:
            return {}
        if not force and time.time() - cache.get("_ts", 0) > CACHE_TTL:
            return {}  # stale — trigger a refresh
        return cache
    return {}


def fetch_models(name: str, prov: dict) -> list[str]:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        prov["base_url"] + "/models",
        headers={"Authorization": "Bearer " + prov["api_key"]},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
    except Exception:
        return []
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    return sorted({str(m.get("id")) for m in rows if isinstance(m, dict) and m.get("id")})


def refresh_models(verbose: bool = True) -> dict:
    provs = load_providers()
    cache = {"_ts": time.time()}
    for name, prov in sorted(provs.items()):
        ids = fetch_models(name, prov)
        cache[name] = ids
        if verbose:
            status = f"{len(ids):5d} models" if ids else "  unavailable"
            print(f"  {name:12s} {status}")
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_CACHE.write_text(json.dumps(cache, indent=1))
    return cache


# --------------------------------------------------------------------------
# model helpers
# --------------------------------------------------------------------------
def _validate_model(provider: str, model: str) -> str | None:
    """Return an error message if the model isn't in cache, else None."""
    cache = load_model_cache()
    ids = cache.get(provider)
    if not ids:
        return None  # cache empty — let the API decide
    if model in ids:
        return None
    from difflib import get_close_matches
    suggestions = get_close_matches(model, ids, n=4, cutoff=0.3)
    msg = f"'{model}' not found in {provider} cache"
    if suggestions:
        msg += "; closest: " + ", ".join(suggestions)
    return msg


def _resolve_public_cfg(pub_cfg: dict) -> dict:
    """Resolve env_var to api_key for public model configs."""
    cfg = dict(pub_cfg)
    if "api_key" not in cfg and "env_var" in cfg:
        cfg["api_key"] = os.environ.get(cfg["env_var"], "")
    return cfg


def _validate_api_key(provider: str, pub_cfg: dict) -> None:
    """Warn if the required public-model API key is not set."""
    env_var = pub_cfg.get("env_var", "")
    if env_var and not os.environ.get(env_var):
        print(f"{C.ye}  warning: {env_var} env var is not set "
              f"— {provider} will fail at request time{C.r}")


def _fmt_prompt(provider: str, model: str) -> str:
    """Build the prompt string from config or default."""
    try:
        template = PROMPT_FMT_FILE.read_text().strip()
    except (OSError, FileNotFoundError):
        template = None
    if not template:
        return f"{C.gr}› {C.r}"
    # shorthand for model name: strip the provider prefix if present
    short = model.removeprefix(provider + "/") if model.startswith(provider + "/") else model
    now = time.strftime("%H:%M:%S")
    rendered = template.format(time=now, provider=provider, model=short)
    return f"{C.gr}{rendered}{C.r}"


# --------------------------------------------------------------------------
# context window helpers
# --------------------------------------------------------------------------
_CTX_WINDOWS: dict[str, int] = {
    "claude": 200_000, "gpt-4": 128_000, "gpt-4o": 128_000,
    "deepseek": 128_000, "kimi": 128_000, "gemini": 1_000_000,
    "llama": 128_000, "mistral": 128_000, "qwen": 128_000,
    "longcat": 128_000,
}
_DEFAULT_CTX_WINDOW = 128_000


def _ctx_window(model: str) -> int:
    m = model.lower()
    for key, size in _CTX_WINDOWS.items():
        if key in m:
            return size
    return _DEFAULT_CTX_WINDOW


def _count_tokens(text: str) -> int:
    """Estimate tokens — tries tiktoken, falls back to char/4."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _ctx_stats(agent) -> dict:
    msgs = agent.messages
    tokens = _count_tokens(json.dumps(msgs))
    window = getattr(agent, "ctx_window", _ctx_window(agent.model))
    pct = round(tokens / window * 100, 1) if window else 0
    return {"messages": len(msgs), "tokens": tokens, "window": window, "pct": pct}


def _ctx_format(template: str, agent, elapsed: float = 0) -> str:
    s = _ctx_stats(agent)
    short = agent.model.removeprefix(agent.provider + "/") if agent.model.startswith(agent.provider + "/") else agent.model
    now = time.strftime("%H:%M:%S")
    return template.format(
        time=now, provider=agent.provider, model=short,
        messages=s["messages"], tokens=s["tokens"],
        window=s["window"], pct=s["pct"],
        elapsed=f"{elapsed:.1f}",
    )


def _render_pre(agent) -> str | None:
    try:
        tmpl = PREPROMPT_FILE.read_text().strip()
    except (OSError, FileNotFoundError):
        return None
    if not tmpl:
        return None
    try:
        return _ctx_format(tmpl, agent)
    except Exception:
        return None


def _render_post(agent, elapsed: float) -> str | None:
    try:
        tmpl = POSTPROMPT_FILE.read_text().strip()
    except (OSError, FileNotFoundError):
        tmpl = "{elapsed}s · {tokens}/{window} ({pct}%)"
    if not tmpl:
        return None
    try:
        return _ctx_format(tmpl, agent, elapsed)
    except Exception:
        return None


# --------------------------------------------------------------------------
# httpd  (lazy imports — only loaded when /httpd or ainow httpd is used)
# --------------------------------------------------------------------------
_httpd_instance: "HTTPServer | None" = None
_httpd_thread: "threading.Thread | None" = None
_httpd_pending: int = 0  # upload notifications waiting to be processed


_DEFAULT_TEMPLATES = {
    PROMPT_FMT_FILE: "{provider}/{model}",
    PREPROMPT_FILE: "",
    POSTPROMPT_FILE: "{elapsed}s · {tokens}/{window} ({pct}%)",
}


def _ensure_dirs() -> None:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    HTTPD_LOG.touch(exist_ok=True)
    AINOW_LOG.touch(exist_ok=True)
    for path, content in _DEFAULT_TEMPLATES.items():
        if not path.exists():
            path.write_text(content + "\n")


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(AINOW_LOG, "a") as f:
        f.write(f"[{ts}] {msg}\n")


def _httpd_log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(HTTPD_LOG, "a") as f:
        f.write(f"[{ts}] {msg}\n")


# HTML upload form — served at GET /
_HTTPD_FORM = """\
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ainow httpd</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 600px; margin: 2rem auto; padding: 0 1rem; color: #222; }
  h1 { font-size: 1.4rem; }
  .drop { border: 2px dashed #aaa; border-radius: 8px; padding: 3rem 1rem; text-align: center; color: #666; }
  .drop.dragover { border-color: #222; color: #222; background: #f5f5f5; }
  input[type=file] { display: none; }
  .browse { color: #06c; cursor: pointer; text-decoration: underline; }
  ul { list-style: none; padding: 0; }
  li { padding: .4rem 0; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; }
  li a { color: #06c; text-decoration: none; }
  .size { color: #999; font-size: .85rem; }
  .status { margin-top: .5rem; color: #999; font-size: .85rem; }
</style></head><body>
<h1>ainow httpd</h1>
<div class="drop" id="drop"><span class="browse" id="browse">Browse</span> or drag files here</div>
<div class="status" id="status"></div>
<ul id="files"></ul>
<script>
const drop=document.getElementById("drop"),browse=document.getElementById("browse"),
      inp=document.createElement("input"),status=document.getElementById("status"),
      list=document.getElementById("files");
inp.type="file";inp.multiple=true;browse.onclick=()=>inp.click();
inp.onchange=()=>upload(inp.files);
drop.ondragover=e=>{e.preventDefault();drop.classList.add("dragover");};
drop.ondragleave=()=>drop.classList.remove("dragover");
drop.ondrop=e=>{e.preventDefault();drop.classList.remove("dragover");upload(e.dataTransfer.files);};
async function upload(files){for(let f of files){status.textContent="uploading "+f.name+"…";
 await fetch("/"+encodeURIComponent(f.name),{method:"PUT",body:f});status.textContent="done.";}
 loadFiles();}
async function loadFiles(){let r=await fetch("/.files");let fs=await r.json();
 list.innerHTML=fs.map(f=>`<li><a href="/${encodeURIComponent(f.name)}">${f.name}</a><span class="size">${f.size}</span></li>`).join("");}
loadFiles();
</script></body></html>"""


class _HttpdHandler:
    """Minimal HTTP handler: PUT uploads, GET serves files, GET / returns form,
    GET /.files returns JSON listing.  Auth via HTTP Basic."""
    def __init__(self, root: pathlib.Path, user: str, password: str):
        self.root = root.resolve()
        self.user = user
        self.password = password
        self.agent = None  # set by _httpd_start_bg for upload notifications

    def _check_auth(self) -> bool:
        import base64
        auth = getattr(self, "_auth_header", "")
        if not auth.startswith("Basic "):
            return False
        try:
            creds = base64.b64decode(auth[6:]).decode()
            u, _, p = creds.partition(":")
            return u == self.user and p == self.password
        except Exception:
            return False

    def _send_json(self, data, code: int = 200) -> bytes:
        body = json.dumps(data).encode()
        return self._response(code, body, "application/json")

    def _response(self, code: int, body: bytes, content_type: str = "application/octet-stream") -> bytes:
        import email.utils
        header = (
            f"HTTP/1.1 {code} {self._status(code)}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Date: {email.utils.formatdate(usegmt=True)}\r\n"
            f"Server: ainow-httpd\r\n"
            f"Connection: close\r\n\r\n"
        )
        return header.encode() + body

    @staticmethod
    def _status(code: int) -> str:
        return {200: "OK", 201: "Created", 400: "Bad Request", 401: "Unauthorized",
                403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed",
                500: "Internal Server Error"}.get(code, "Unknown")

    def _handle_get(self, path: str) -> bytes:
        if path == "/":
            return self._response(200, _HTTPD_FORM.encode(), "text/html; charset=utf-8")
        if path == "/.files":
            items = []
            if self.root.is_dir():
                for f in sorted(self.root.iterdir()):
                    if f.is_file():
                        items.append({"name": f.name, "size": f.stat().st_size})
            return self._send_json(items)

        # serve a file
        rel = path.lstrip("/")
        if ".." in rel or rel.startswith("/"):
            return self._response(403, b"Forbidden")
        fp = (self.root / rel).resolve()
        if not str(fp).startswith(str(self.root)):
            return self._response(403, b"Forbidden")
        if not fp.is_file():
            return self._response(404, b"Not Found")
        try:
            data = fp.read_bytes()
            ct = "text/plain" if fp.suffix in (".txt", ".py", ".md", ".log", ".json", ".xml", ".yml", ".yaml", ".cfg", ".ini", ".sh", ".bash", ".c", ".h", ".cpp", ".hpp", ".rs", ".go", ".js", ".ts", ".html", ".css") else "application/octet-stream"
            return self._response(200, data, ct)
        except Exception as e:
            return self._response(500, str(e).encode())

    def _handle_put(self, path: str, body: bytes) -> bytes:
        rel = path.lstrip("/")
        if not rel or ".." in rel or rel.startswith("/"):
            return self._response(400, b"Bad filename")
        self.root.mkdir(parents=True, exist_ok=True)
        fp = (self.root / rel).resolve()
        if not str(fp).startswith(str(self.root)):
            return self._response(403, b"Forbidden")
        try:
            fp.write_bytes(body)
            _httpd_log(f"PUT {rel} ({len(body)} bytes)")
            if self.agent is not None:
                global _httpd_pending
                self.agent.messages.append(
                    {"role": "user", "content": f"[httpd upload] {rel} ({len(body)} bytes)\n"
                     f"Path: {fp}"})
                _httpd_pending += 1
                print(f"\n{C.d}  ← httpd upload: {rel} ({len(body)} bytes)"
                      f"  (press Enter to process){C.r}\n")
            return self._response(201, b"Created")
        except Exception as e:
            return self._response(500, str(e).encode())

    def _extract_wsgi_headers(self, raw_resp: bytes):
        """Parse raw HTTP response into (status, headers) suitable for WSGI."""
        body_start = raw_resp.index(b"\r\n\r\n") + 4
        head = raw_resp[:body_start - 4].decode()
        status_line = head.split("\r\n")[0]
        status = status_line[9:]  # after "HTTP/1.1 "
        headers = []
        for line in head.split("\r\n")[1:]:
            k, _, v = line.partition(": ")
            if k.lower() not in ("connection", "date", "server"):
                headers.append((k, v))
        return status, headers, raw_resp[body_start:]

    def __call__(self, environ: dict, start_response) -> list[bytes]:
        # stash auth header for _check_auth
        setattr(self, "_auth_header", environ.get("HTTP_AUTHORIZATION", ""))
        if not self._check_auth():
            body = b"Unauthorized"
            header = [("Content-Type", "text/plain"), ("Content-Length", str(len(body))),
                      ("WWW-Authenticate", 'Basic realm="ainow"')]
            start_response("401 Unauthorized", header)
            return [body]

        method = environ["REQUEST_METHOD"]
        path = environ["PATH_INFO"]

        if method in ("GET", "HEAD"):
            resp = self._handle_get(path)
        elif method == "PUT":
            length = int(environ.get("CONTENT_LENGTH", 0))
            body = environ["wsgi.input"].read(length) if length > 0 else b""
            resp = self._handle_put(path, body)
        else:
            body = b"Method Not Allowed"
            start_response("405 Method Not Allowed", [("Content-Type", "text/plain"),
                            ("Content-Length", str(len(body)))])
            return [body]

        status, headers, body = self._extract_wsgi_headers(resp)
        start_response(status, headers)
        return [body]


def _make_httpd(root: pathlib.Path, user: str, password: str, port: int = 0,
                 agent=None):
    """Build and return an HTTPServer; port 0 = OS picks."""
    from wsgiref.simple_server import make_server, WSGIRequestHandler
    handler = _HttpdHandler(root, user, password)
    handler.agent = agent
    # WSGIRequestHandler is chatty to stderr; suppress it
    class _Quiet(WSGIRequestHandler):
        def log_message(self, format, *args):
            _httpd_log(f"{self.client_address[0]} {format % args}")
    srv = make_server("0.0.0.0", port, handler, handler_class=_Quiet)
    return srv


def _dangerous_root(root: pathlib.Path) -> bool:
    """Warn if the root is a sensitive directory."""
    r = root.resolve()
    dangerous = {pathlib.Path("/"), pathlib.Path("/home"), HOME,
                 HOME / ".ssh", HOME / ".gnupg", HOME / ".config"}
    return r in dangerous


# -- CLI entry points (foreground, blocking) -------------------------------
def httpd_start(root: pathlib.Path, user: str, password: str,
                port: int = 0, allow_dangerous: bool = False) -> None:
    """Foreground blocking server for `ainow httpd start`."""
    if _dangerous_root(root) and not allow_dangerous:
        sys.exit(f"ainow: refusing to serve from {root}. Use --allow-foot-bullet-root-mode "
                 "if you are absolutely sure.")

    import atexit
    _ensure_dirs()
    srv = _make_httpd(root, user, password, port)
    host, bound = srv.server_address
    url = f"http://{host}:{bound}/"
    HTTPD_PIDFILE.write_text(str(os.getpid()))
    _httpd_log(f"started on {url} (root={root}, user={user})")

    print(f"{C.b}ainow httpd{C.r}")
    print(f"  {C.gr}url{C.r}      {url}")
    print(f"  {C.gr}auth{C.r}     {user} : {password}")
    print(f"  {C.gr}root{C.r}     {root}")
    print(f"\n{C.d}  SIGINT (Ctrl-C) to stop{C.r}")

    def _cleanup():
        srv.shutdown()
        try:
            HTTPD_PIDFILE.unlink()
        except OSError:
            pass
        _httpd_log("stopped")

    atexit.register(_cleanup)

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()
        print(f"\n{C.d}httpd stopped{C.r}")


def httpd_stop() -> None:
    """Kill a running httpd via pidfile.  Returns whether something was killed."""
    if not HTTPD_PIDFILE.exists():
        print(f"{C.ye}httpd is not running (no pidfile){C.r}")
        return
    try:
        pid = int(HTTPD_PIDFILE.read_text().strip())
    except (ValueError, OSError):
        print(f"{C.re}corrupt pidfile at {HTTPD_PIDFILE}{C.r}")
        HTTPD_PIDFILE.unlink()
        return
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"{C.d}stopped httpd (pid {pid}){C.r}")
        _httpd_log(f"stopped by signal (pid {pid})")
    except ProcessLookupError:
        print(f"{C.ye}no process with pid {pid} — removing stale pidfile{C.r}")
    except PermissionError:
        print(f"{C.re}permission denied killing pid {pid}{C.r}")
    try:
        HTTPD_PIDFILE.unlink()
    except OSError:
        pass


# -- REPL integration (daemon thread, non-blocking) ------------------------
def _httpd_start_bg(agent, root, user, password, port=0, allow_dangerous=False):
    """Start httpd in a daemon thread.  Called from /httpd start in the REPL."""
    import atexit, threading
    global _httpd_instance, _httpd_thread
    if _httpd_instance is not None:
        print(f"{C.ye}  httpd is already running{C.r}")
        return
    if _dangerous_root(root) and not allow_dangerous:
        print(f"  {C.re}refusing to serve from {root}. "
              f"Use --allow-foot-bullet-root-mode if you are absolutely sure.{C.r}")
        return
    _ensure_dirs()
    srv = _make_httpd(root, user, password, port, agent=agent)
    _httpd_instance = srv
    _httpd_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    _httpd_thread.start()
    _httpd_log(f"started on http://0.0.0.0:{srv.server_port}/ (root={root}, user={user})")

    def _cleanup_bg():
        global _httpd_instance
        if _httpd_instance:
            _httpd_instance.shutdown()
            _httpd_instance = None
            _httpd_log("stopped (session end)")
    atexit.register(_cleanup_bg)

    host, bound = srv.server_address
    print(f"  {C.gr}httpd started{C.r}  {host}:{bound}  {C.d}{user}:{password}{C.r}")


def _httpd_stop_bg():
    """Stop the background httpd.  Called from /httpd stop in the REPL."""
    global _httpd_instance
    if _httpd_instance is None:
        print(f"{C.ye}  httpd is not running{C.r}")
        return
    _httpd_instance.shutdown()
    _httpd_instance = None
    _httpd_log("stopped (/httpd stop)")
    print(f"{C.d}  httpd stopped{C.r}")


def _httpd_repl_status() -> bool:
    """Print status; return True if running."""
    global _httpd_instance
    if _httpd_instance is not None:
        host, port = _httpd_instance.server_address
        print(f"{C.d}  httpd running on {host}:{port}{C.r}")
        return True
    if HTTPD_PIDFILE.exists():
        try:
            pid = int(HTTPD_PIDFILE.read_text().strip())
            print(f"{C.d}  httpd running (pid {pid}, standalone){C.r}")
            return True
        except Exception:
            pass
    print(f"{C.d}  httpd not running{C.r}")
    return False


# --------------------------------------------------------------------------
# completion  (must stay import-light: called on every TAB)
# --------------------------------------------------------------------------
def complete(word: str) -> None:
    provs = sorted(load_providers().keys())
    cache = load_model_cache()

    if "/" not in word:
        for p in provs:
            if p.startswith(word):
                print(p + "/")
        if "public".startswith(word):
            print("public/")
        return

    if word.startswith("public/"):
        fragment = word.removeprefix("public/")
        for key in sorted(PUBLIC_MODELS):
            if key.startswith(word):
                print(key)
        if not fragment:
            for key in sorted(PUBLIC_MODELS):
                print(key)
        return

    prov, _, frag = word.partition("/")
    if prov not in provs:
        return
    models = cache.get(prov) or []
    f = frag.lower()
    if not f:
        for m in models:
            print(f"{prov}/{m}")
        return

    # OpenRouter ids are "vendor/model", so match the model name too --
    # `openrouter/k` should surface moonshotai/kimi-k2, not everything
    # that happens to contain a "k".
    def rank(m: str) -> int | None:
        ml = m.lower()
        base = ml.rsplit("/", 1)[-1]
        if base.startswith(f):
            return 0          # kimi-k2, kat-coder-pro
        if ml.startswith(f):
            return 1          # kwaipilot/…
        if f in base:
            return 2          # …-kimi-…
        if f in ml:
            return 3          # vendor name only
        return None

    scored = [(r, m) for m in models if (r := rank(m)) is not None]
    for _, m in sorted(scored, key=lambda t: (t[0], t[1])):
        print(f"{prov}/{m}")


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------
def _clip(s: str) -> str:
    if len(s) > MAX_TOOL_OUTPUT:
        return s[:MAX_TOOL_OUTPUT] + f"\n… [truncated, {len(s) - MAX_TOOL_OUTPUT} more chars]"
    return s


def t_read_file(path: str, offset: int = 1, limit: int = 2000) -> str:
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        return f"error: no such file: {p}"
    if p.is_dir():
        return f"error: {p} is a directory (use list_dir)"
    try:
        lines = p.read_text(errors="replace").splitlines()
    except Exception as e:
        return f"error: {e}"
    sel = lines[max(0, offset - 1): max(0, offset - 1) + limit]
    width = len(str(offset + len(sel)))
    body = "\n".join(f"{i:>{width}}\t{l}" for i, l in enumerate(sel, start=offset))
    return _clip(body) if body else "(empty file)"


def t_write_file(path: str, content: str) -> str:
    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {p} ({len(content)} chars, {content.count(chr(10)) + 1} lines)"


def t_edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        return f"error: no such file: {p}"
    src = p.read_text(errors="replace")
    n = src.count(old_string)
    if n == 0:
        return "error: old_string not found (must match exactly, including indentation)"
    if n > 1 and not replace_all:
        return f"error: old_string occurs {n} times — add more context or set replace_all=true"
    p.write_text(src.replace(old_string, new_string) if replace_all
                 else src.replace(old_string, new_string, 1))
    return f"edited {p} ({n if replace_all else 1} replacement(s))"


def t_list_dir(path: str = ".") -> str:
    p = pathlib.Path(path).expanduser()
    if not p.is_dir():
        return f"error: not a directory: {p}"
    rows = []
    for e in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        rows.append(f"{'d' if e.is_dir() else '-'} {e.name}" +
                    ("" if e.is_dir() else f"  ({e.stat().st_size}b)"))
    return _clip("\n".join(rows)) or "(empty directory)"


def t_bash(command: str, timeout: int = 120) -> str:
    try:
        r = subprocess.run(command, shell=True, capture_output=True,
                           text=True, timeout=timeout, errors="replace")
    except subprocess.TimeoutExpired:
        return f"error: timed out after {timeout}s"
    out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
    if r.returncode != 0:
        out += f"\n[exit {r.returncode}]"
    return _clip(out.strip()) or f"(no output) [exit {r.returncode}]"


TOOLS = {
    "read_file":  (t_read_file,  {"path": "str"}, False),
    "list_dir":   (t_list_dir,   {"path": "str"}, False),
    "write_file": (t_write_file, {"path": "str"}, True),
    "edit_file":  (t_edit_file,  {"path": "str"}, True),
    "bash":       (t_bash,       {"command": "str"}, True),
}

TOOL_SCHEMA = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from disk. Returns line-numbered content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "offset": {"type": "integer", "description": "1-indexed start line"},
            "limit": {"type": "integer", "description": "Max lines to read"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List the contents of a directory. Defaults to current working directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path (default: .)"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write a file, creating or overwriting it entirely.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace an exact string in a file. old_string must match "
                       "byte-for-byte including indentation, and must be unique "
                       "unless replace_all is true.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean"}},
            "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command and return combined stdout/stderr.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "description": "Seconds, default 120"}},
            "required": ["command"]}}},
]

SYSTEM = """You are ainow, a command-line coding assistant running on the user's \
Linux machine with real filesystem and shell access.

You have tools: read_file, list_dir, write_file, edit_file, bash. Use them to \
inspect and change files directly rather than printing code for the user to \
copy. Prefer edit_file over rewriting whole files. Read a file before editing it.

Be concise. The user is in a terminal — use plain text only. No markdown of any \
kind (no **bold**, `backticks`, bullet lists) unless explicitly asked. Report \
what you actually did, and if a command failed, say so with the output rather \
than assuming it worked."""


# --------------------------------------------------------------------------
# interrupt plumbing
# --------------------------------------------------------------------------
class Interrupted(Exception):
    pass


class sigint_guard:
    """Ctrl-C during generation raises Interrupted instead of killing us."""

    def __enter__(self):
        self.prev = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._raise)
        return self

    def _raise(self, *_):
        raise Interrupted()

    def __exit__(self, *_):
        signal.signal(signal.SIGINT, self.prev)
        return False


# --------------------------------------------------------------------------
# ansi
# --------------------------------------------------------------------------
class C:
    d = "\033[2m"; b = "\033[1m"; r = "\033[0m"
    cy = "\033[36m"; gr = "\033[32m"; ye = "\033[33m"; re = "\033[31m"; ma = "\033[35m"


# --------------------------------------------------------------------------
# agent
# --------------------------------------------------------------------------
class Agent:
    def __init__(self, provider: str, model: str, prov_cfg: dict, auto: bool):
        from openai import OpenAI
        self.provider = provider
        self.model = model
        self.auto = auto
        self.client = OpenAI(base_url=prov_cfg["base_url"],
                             api_key=prov_cfg["api_key"], timeout=600.0)
        self.messages = [{"role": "system", "content": SYSTEM}]
        self.ctx_window = _ctx_window(model)
        self._last_elapsed = 0.0

    # -- approval -----------------------------------------------------
    def _approve(self, name: str, args: dict) -> bool:
        if self.auto or not TOOLS[name][2]:
            return True
        detail = args.get("command") or args.get("path") or ""
        print(f"\n{C.ye}  {name}{C.r} {C.d}{str(detail)[:160]}{C.r}")
        try:
            ans = input(f"  {C.b}run it?{C.r} [y/N/a=always] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if ans == "a":
            self.auto = True
            return True
        return ans in ("y", "yes")

    # -- one streamed assistant turn ----------------------------------
    def _stream_turn(self) -> dict:
        text_parts: list[str] = []
        calls: dict[int, dict] = {}
        printed_any = False

        stream = self.client.chat.completions.create(
            model=self.model, messages=self.messages,
            tools=TOOL_SCHEMA, tool_choice="auto", stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            if getattr(d, "content", None):
                sys.stdout.write(d.content)
                sys.stdout.flush()
                text_parts.append(d.content)
                printed_any = True
            for tc in (getattr(d, "tool_calls", None) or []):
                slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments
        if printed_any:
            print()

        msg = {"role": "assistant", "content": "".join(text_parts) or None}
        if calls:
            msg["tool_calls"] = [
                {"id": c["id"] or f"call_{i}", "type": "function",
                 "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                for i, c in sorted(calls.items())
            ]
        return msg

    # -- full turn incl. tool loop ------------------------------------
    def run(self, user_text: str) -> None:
        # context warning before sending
        s = _ctx_stats(self)
        if s["pct"] >= 90:
            print(f"{C.re}  ⚠ context {s['tokens']}/{s['window']} ({s['pct']}%) — consider /ctx compress{C.r}")
        elif s["pct"] >= 70:
            print(f"{C.ye}  ⚠ context {s['tokens']}/{s['window']} ({s['pct']}%){C.r}")

        t0 = time.time()
        self.messages.append({"role": "user", "content": user_text})
        try:
            with sigint_guard():
                while True:
                    msg = self._stream_turn()
                    self.messages.append(msg)
                    tcs = msg.get("tool_calls")
                    if not tcs:
                        self._last_elapsed = time.time() - t0
                        return

                    for tc in tcs:
                        name = tc["function"]["name"]
                        try:
                            args = json.loads(tc["function"]["arguments"] or "{}")
                        except json.JSONDecodeError as e:
                            result = f"error: bad tool arguments: {e}"
                        else:
                            if name not in TOOLS:
                                result = f"error: unknown tool {name}"
                            elif not self._approve(name, args):
                                result = "error: user declined this action"
                            else:
                                label = args.get("command") or args.get("path") or ""
                                _log(f"tool {name} {str(label)[:200]}")
                                print(f"{C.cy}  · {name}{C.r} {C.d}{str(label)[:120]}{C.r}")
                                try:
                                    result = TOOLS[name][0](**args)
                                except TypeError as e:
                                    result = f"error: bad arguments: {e}"
                                except Exception as e:
                                    result = f"error: {type(e).__name__}: {e}"
                        self.messages.append({"role": "tool", "tool_call_id": tc["id"],
                                              "content": str(result)})
        except Interrupted:
            print(f"\n{C.ye}  ^C interrupted{C.r}")
            # keep history valid: answer any dangling tool calls
            last = self.messages[-1]
            if last.get("role") == "assistant" and last.get("tool_calls"):
                done = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
                for tc in last["tool_calls"]:
                    if tc["id"] not in done:
                        self.messages.append({"role": "tool", "tool_call_id": tc["id"],
                                              "content": "error: interrupted by user"})
        except Exception as e:
            print(f"\n{C.re}  {type(e).__name__}: {e}{C.r}")
        self._last_elapsed = time.time() - t0


# --------------------------------------------------------------------------
# repl
# --------------------------------------------------------------------------
HELP = f"""{C.b}commands{C.r}
  /help          this
  /model <spec>  switch model, e.g. /model public/gemini-2.0-flash
  /models [pat]  list cached models for the current provider
  /httpd [start|stop]  file-transfer server (status if no args)
    start [-port N] [-root PATH] [-user U] [-pass P]
  /ctx [compress|window N]  context stats, compress, or set window
  /auto [on|off] toggle running tools without asking
  /clear         reset conversation
  /exit          quit
{C.b}keys{C.r}
  Ctrl-C  interrupt generation / clear the line  (does not exit)
  Ctrl-D  exit          Ctrl-Q  exit"""


def _handle_httpd_cmd(agent, args: str) -> None:
    """Parse and dispatch /httpd commands from the REPL."""
    import secrets
    parts = args.split()
    cmd = parts[0] if parts else ""

    if cmd == "start":
        # parse optional flags
        root = DEFAULT_HTTPD_ROOT
        user = "ainow"
        password = secrets.token_urlsafe(8)[:8]
        port = 0
        allow_dangerous = False
        i = 1
        while i < len(parts):
            if parts[i] == "-port" and i + 1 < len(parts):
                port = int(parts[i + 1]); i += 2
            elif parts[i] == "-user" and i + 1 < len(parts):
                user = parts[i + 1]; i += 2
            elif parts[i] == "-pass" and i + 1 < len(parts):
                password = parts[i + 1]; i += 2
            elif parts[i] == "-root" and i + 1 < len(parts):
                root = pathlib.Path(parts[i + 1]).expanduser(); i += 2
            elif parts[i] == "--allow-foot-bullet-root-mode":
                allow_dangerous = True; i += 1
            else:
                print(f"{C.ye}  unknown flag: {parts[i]}{C.r}")
                i += 1
        _httpd_start_bg(agent, root, user, password, port, allow_dangerous=allow_dangerous)
    elif cmd == "stop":
        _httpd_stop_bg()
    else:
        _httpd_repl_status()


def _ctx_compress(agent) -> str:
    """Summarize conversation: keep system + recent, summarise the middle."""
    msgs = agent.messages
    if len(msgs) <= 5:
        return "not enough context to compress"

    user_idxs = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    if len(user_idxs) <= 2:
        return "not enough user messages to compress"

    keep_from = user_idxs[-2]
    to_summarize = msgs[1:keep_from]

    print(f"{C.d}  summarising {len(to_summarize)} messages…{C.r}")
    summary_msgs = [
        {"role": "system", "content": "Summarise this conversation concisely. Keep key facts, decisions, file changes, and errors. Write in plain paragraphs."},
        {"role": "user", "content": json.dumps(to_summarize)},
    ]
    try:
        stream = agent.client.chat.completions.create(
            model=agent.model, messages=summary_msgs, stream=True,
        )
        parts = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
        summary = "".join(parts)
    except Exception as e:
        return f"compression failed: {e}"

    kept = msgs[keep_from:]
    agent.messages = [
        msgs[0],
        {"role": "user", "content": f"[Earlier conversation summary]\n{summary}"},
        {"role": "assistant", "content": "Understood. I'll continue from where we left off."},
    ] + kept
    new_tokens = _count_tokens(json.dumps(agent.messages))
    return f"compressed {len(to_summarize)} messages → {len(agent.messages)} ({new_tokens} tokens)"


def _oneshot(agent, prompt: str) -> None:
    """Run a single prompt, print the reply to stdout, and exit."""
    _ensure_dirs()
    if not MODELS_CACHE.exists():
        refresh_models()
    try:
        agent.chat(prompt)
    except requests.RequestException as e:
        print(f"ainow: API error: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print(file=sys.stderr)
        sys.exit(130)

    last = agent.messages[-1]
    if last["role"] == "assistant":
        text = last.get("content", "")
        # strip tool-call placeholders if present
        if isinstance(text, str):
            if "\n#" in text:
                text = text.split("\n#")[0]
            print(text)
    elif last.get("tool_calls"):
        print("(tool calls not supported in one-shot mode)", file=sys.stderr)
        sys.exit(3)

    _log(f"oneshot {agent.provider}/{agent.model} → {len(agent.messages)-1} msgs")


def _handle_ctx_cmd(agent, rest: str) -> None:
    """Parse and dispatch /ctx commands."""
    parts = rest.split()
    cmd = parts[0] if parts else ""

    if cmd == "compress":
        print(_ctx_compress(agent))
    elif cmd == "window" and len(parts) > 1:
        try:
            agent.ctx_window = int(parts[1])
            s = _ctx_stats(agent)
            print(f"{C.d}context window set to {agent.ctx_window} ({s['pct']}% used){C.r}")
        except ValueError:
            print(f"{C.re}invalid window size: {parts[1]}{C.r}")
    else:
        s = _ctx_stats(agent)
        # breakdown by role
        roles: dict[str, int] = {}
        for m in agent.messages:
            r = m.get("role", "?")
            roles[r] = roles.get(r, 0) + 1
        role_str = " ".join(f"{r}:{n}" for r, n in sorted(roles.items()))
        bar_w = 20
        filled = min(bar_w, int(s["pct"] / 100 * bar_w))
        bar = f"{'█' * filled}{'░' * (bar_w - filled)}"
        print(f"  {role_str}  {s['tokens']}/{s['window']} tokens  {bar} {s['pct']}%")


def repl(agent: Agent, provs: dict, first: str | None) -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings

    CFG_DIR.mkdir(parents=True, exist_ok=True)
    kb = KeyBindings()

    @kb.add("c-q")
    def _(event):
        event.app.exit(exception=EOFError, style="class:exiting")

    session = PromptSession(history=FileHistory(str(HISTORY_FILE)), key_bindings=kb)

    _log(f"session start {agent.provider}/{agent.model}  cwd={os.getcwd()}")
    print(f"{C.ma}ainow{C.r} {C.b}{agent.provider}/{agent.model}{C.r}  "
          f"{C.d}cwd {os.getcwd()}{C.r}")
    print(f"{C.d}/help for commands · Ctrl-C interrupts · Ctrl-D or Ctrl-Q exits{C.r}\n")

    pending = first
    while True:
        if pending is not None:
            line, pending = pending, None
            pre = _render_pre(agent)
            if pre:
                print(f"{C.d}{pre}{C.r}")
            print(f"{_fmt_prompt(agent.provider, agent.model)}{line}")
        else:
            try:
                pre = _render_pre(agent)
                if pre:
                    print(f"{C.d}{pre}{C.r}")
                line = session.prompt(ANSI(_fmt_prompt(agent.provider, agent.model)))
            except KeyboardInterrupt:      # Ctrl-C: clear line, stay alive
                continue
            except EOFError:               # Ctrl-D / Ctrl-Q: leave
                print("bye")
                return

        line = line.strip()
        if not line:
            global _httpd_pending
            if _httpd_pending:
                _httpd_pending = 0
                agent.run("")
                post = _render_post(agent, agent._last_elapsed)
                if post:
                    print(f"{C.d}{post}{C.r}")
                print()
            continue

        if line.startswith("/"):
            cmd, _, rest = line[1:].partition(" ")
            rest = rest.strip()
            if cmd in ("exit", "quit", "q"):
                print("bye")
                return
            if cmd == "help":
                print(HELP)
            elif cmd == "clear":
                agent.messages = agent.messages[:1]
                _log(f"context cleared ({agent.provider}/{agent.model})")
                print(f"{C.d}context cleared{C.r}")
            elif cmd == "ctx":
                _handle_ctx_cmd(agent, rest)
            elif cmd == "auto":
                if rest in ("on", "off"):
                    agent.auto = rest == "on"
                else:
                    agent.auto = not agent.auto
                print(f"{C.d}auto-approve {'on' if agent.auto else 'off'}{C.r}")
            elif cmd == "model":
                try:
                    p, m, pub_cfg = parse_spec(rest, provs)
                except SystemExit as e:
                    print(f"{C.re}{e}{C.r}")
                    continue
                if pub_cfg:
                    _validate_api_key(p, pub_cfg)
                else:
                    err = _validate_model(p, m)
                    if err:
                        print(f"{C.ye}  {err}{C.r}")
                cfg = _resolve_public_cfg(pub_cfg) if pub_cfg else provs[p]
                _log(f"model switch {agent.provider}/{agent.model} → {p}/{m}")
                agent.__init__(p, m, cfg, agent.auto)
                print(f"{C.d}now {p}/{m}{C.r}")
            elif cmd == "models":
                all_ids = load_model_cache().get(agent.provider, [])
                if not all_ids:
                    print("  (cache empty — run: ainow --refresh-models)")
                    continue
                ids = [i for i in all_ids if rest.lower() in i.lower()] if rest else all_ids
                if not ids:
                    print(f"  no match for {rest!r} in {len(all_ids)} models")
                    continue
                print("  " + "\n  ".join(ids))
                shown = f"{len(ids)} of {len(all_ids)}" if rest else str(len(ids))
                print(f"{C.d}  — {shown} models on {agent.provider}{C.r}")
            elif cmd == "httpd":
                _handle_httpd_cmd(agent, rest)
            else:
                print(f"{C.re}unknown command /{cmd}{C.r}")
            continue

        agent.run(line)
        post = _render_post(agent, agent._last_elapsed)
        if post:
            print(f"{C.d}{post}{C.r}")
        print()


# --------------------------------------------------------------------------
# entry
# --------------------------------------------------------------------------
def parse_spec(spec: str, provs: dict) -> tuple[str, str, dict | None]:
    """Parse model spec. Returns (provider, model, public_cfg_or_None)."""
    if spec.startswith("public/"):
        if spec not in PUBLIC_MODELS:
            sys.exit(f"ainow: unknown public model '{spec}'; try --help")
        pub = PUBLIC_MODELS[spec]
        return (spec, pub["model"], pub)
    if "/" not in spec:
        sys.exit(f"ainow: model must be <provider>/<model>; providers: {', '.join(sorted(provs))}")
    prov, _, model = spec.partition("/")
    if prov not in provs:
        sys.exit(f"ainow: unknown provider '{prov}'; have: {', '.join(sorted(provs))}")
    if not model:
        sys.exit(f"ainow: no model given for {prov}")
    return prov, model, None


def main() -> None:
    argv = sys.argv[1:]
    _ensure_dirs()

    # Listing output is routinely piped into head/less. Restore default SIGPIPE
    # so we die silently like any other unix tool instead of tracebacking on
    # shutdown. Deliberately NOT done for the REPL, where a broken HTTPS socket
    # would then kill the session.
    if argv and argv[0] in ("--complete", "--models", "-m", "--providers"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    # fast paths — no heavy imports
    if argv and argv[0] == "--complete":
        complete(argv[1] if len(argv) > 1 else "")
        return
    if argv and argv[0] in ("--refresh-models", "--refresh"):
        print("refreshing model cache…")
        refresh_models()
        return
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        provs = load_providers()
        print("providers: " + ", ".join(sorted(provs)))
        print(f"public: {len(PUBLIC_MODELS)} built-in free models (no provider config needed)")
        return
    if argv[0] == "--public":
        print(f"built-in free/public models ({len(PUBLIC_MODELS)}):")
        for key in sorted(PUBLIC_MODELS):
            pub = PUBLIC_MODELS[key]
            print(f"  {key:30s} → {pub['base_url']}  [{pub.get('env_var', '')}]")
        return
    if argv[0] == "--providers":
        for n, p in sorted(load_providers().items()):
            print(f"  {n:12s} {p['base_url']}")
        print(f"\n  -- public ({len(PUBLIC_MODELS)} built-in, use --public to list) --")
        return
    if argv[0] in ("--models", "-m"):
        cache = load_model_cache()
        prov = argv[1] if len(argv) > 1 else None
        pat = (argv[2] if len(argv) > 2 else "").lower()
        if not prov:
            for n in sorted(k for k in cache if not k.startswith("_")):
                print(f"  {n:12s} {len(cache[n]):5d} models")
            print(f"\n  ainow --models <provider> [pattern]")
            return
        ids = cache.get(prov)
        if ids is None:
            sys.exit(f"ainow: no cached models for '{prov}'; "
                     f"have: {', '.join(sorted(k for k in cache if not k.startswith('_')))}")
        hits = [i for i in ids if pat in i.lower()] if pat else ids
        for i in hits:
            print(i)
        sys.stdout.flush()   # so the stderr summary lands after the list
        print(f"— {len(hits)} of {len(ids)}" if pat else f"— {len(ids)} models",
              file=sys.stderr)
        return

    # httpd subcommand — standalone file-transfer server
    if argv and argv[0] == "httpd":
        _ensure_dirs()
        import secrets
        subcmd = argv[1] if len(argv) > 1 else ""
        if subcmd == "start":
            root = DEFAULT_HTTPD_ROOT
            user = "ainow"
            password = secrets.token_urlsafe(8)[:8]
            port = 0
            allow_dangerous = False
            args = argv[2:]
            i = 0
            while i < len(args):
                if args[i] == "-port" and i + 1 < len(args):
                    port = int(args[i + 1]); i += 2
                elif args[i] == "-user" and i + 1 < len(args):
                    user = args[i + 1]; i += 2
                elif args[i] == "-pass" and i + 1 < len(args):
                    password = args[i + 1]; i += 2
                elif args[i] == "-root" and i + 1 < len(args):
                    root = pathlib.Path(args[i + 1]).expanduser(); i += 2
                elif args[i] == "--allow-foot-bullet-root-mode":
                    allow_dangerous = True; i += 1
                else:
                    print(f"{C.ye}unknown flag: {args[i]}{C.r}")
                    i += 1
            httpd_start(root, user, password, port, allow_dangerous=allow_dangerous)
        elif subcmd == "stop":
            httpd_stop()
        else:
            # status
            if _httpd_repl_status():
                pass  # already printed
            else:
                print(f"{C.d}usage: ainow httpd [start|stop]{C.r}")
        return

    auto = False
    oneshot_prompt = None

    # parse flags before positional model spec
    if "--yolo" in argv:
        auto = True
        argv.remove("--yolo")
    if "-c" in argv:
        idx = argv.index("-c")
        if idx + 1 < len(argv):
            oneshot_prompt = argv[idx + 1]
        argv = argv[:idx]

    provs = load_providers()
    prov, model, pub_cfg = parse_spec(argv[0], provs)
    first = " ".join(argv[1:]) or None

    if pub_cfg:
        _validate_api_key(prov, pub_cfg)

    if not MODELS_CACHE.exists():
        print("building model cache (first run)…")
        refresh_models()

    if not pub_cfg:
        err = _validate_model(prov, model)
        if err:
            print(f"{C.ye}{err}{C.r}")

    agent = Agent(prov, model, _resolve_public_cfg(pub_cfg) if pub_cfg else provs[prov], auto)

    if oneshot_prompt:
        _oneshot(agent, oneshot_prompt)
        return

    try:
        repl(agent, provs, first)
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # piping into head/less: die quietly, not with a traceback
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
