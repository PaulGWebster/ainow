#!/usr/bin/env python3
"""ainow — a small multi-provider agentic coding harness.

Usage:  ainow <provider>/<model>  [initial prompt ...]

Flags:  --yolo      auto-approve all tool calls
        -c PROMPT   one-shot: run prompt and exit (no REPL)
        --allow-foot-bullet-root-mode   allow running as root

Keys:   Ctrl-C  interrupt current generation / clear line  (does NOT exit)
        Ctrl-D  exit
        Ctrl-Q  exit
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time

HOME = pathlib.Path.home()
CFG_DIR = HOME / ".config" / "ainow"
PROVIDERS_FILE = CFG_DIR / "providers.json"

# Built-in registry — OpenAI-compatible models, no provider config needed.
# Three classifiers: free/, paid/, local/.  public/ is a backward-compat alias.
FREE_MODELS: dict[str, dict] = {
    "free/gemini-2.0-flash": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
        "env_var": "GEMINI_API_KEY",
        "ctx_window": 1_000_000,
    },
    "free/gemini-2.5-pro": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-pro",
        "env_var": "GEMINI_API_KEY",
        "ctx_window": 1_000_000,
    },
    "free/gemini-2.5-flash": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
        "env_var": "GEMINI_API_KEY",
        "ctx_window": 1_000_000,
    },
    "free/mistral-large": {
        "base_url": "https://api.mistral.ai/v1/",
        "model": "mistral-large-latest",
        "env_var": "MISTRAL_API_KEY",
    },
    "free/mistral-small": {
        "base_url": "https://api.mistral.ai/v1/",
        "model": "mistral-small-latest",
        "env_var": "MISTRAL_API_KEY",
    },
    "free/groq-llama-3.3": {
        "base_url": "https://api.groq.com/openai/v1/",
        "model": "llama-3.3-70b-versatile",
        "env_var": "GROQ_API_KEY",
    },
    "free/groq-llama-4-scout": {
        "base_url": "https://api.groq.com/openai/v1/",
        "model": "llama-4-scout-17b-16e-instruct",
        "env_var": "GROQ_API_KEY",
    },
    "free/groq-llama-4-maverick": {
        "base_url": "https://api.groq.com/openai/v1/",
        "model": "llama-4-maverick-17b-128e-instruct",
        "env_var": "GROQ_API_KEY",
    },
    "free/cerebras-llama-3.3": {
        "base_url": "https://api.cerebras.ai/v1/",
        "model": "llama3.3-70b",
        "env_var": "CEREBRAS_API_KEY",
    },
    "free/cerebras-qwen3-235b": {
        "base_url": "https://api.cerebras.ai/v1/",
        "model": "qwen3-235b",
        "env_var": "CEREBRAS_API_KEY",
    },
    "free/cerebras-gpt-oss-120b": {
        "base_url": "https://api.cerebras.ai/v1/",
        "model": "gpt-oss-120b",
        "env_var": "CEREBRAS_API_KEY",
    },
    "free/nvidia-llama-3.3": {
        "base_url": "https://integrate.api.nvidia.com/v1/",
        "model": "meta/llama-3.3-70b-instruct",
        "env_var": "NVIDIA_API_KEY",
    },
    "free/nvidia-mistral-large": {
        "base_url": "https://integrate.api.nvidia.com/v1/",
        "model": "mistralai/mistral-large-2-instruct",
        "env_var": "NVIDIA_API_KEY",
    },
    "free/github-gpt-4o": {
        "base_url": "https://models.inference.ai.azure.com/",
        "model": "gpt-4o",
        "env_var": "GITHUB_TOKEN",
    },
    "free/github-deepseek-r1": {
        "base_url": "https://models.inference.ai.azure.com/",
        "model": "DeepSeek-R1",
        "env_var": "GITHUB_TOKEN",
    },
    "free/github-llama-3.3": {
        "base_url": "https://models.inference.ai.azure.com/",
        "model": "Llama-3.3-70B-Instruct",
        "env_var": "GITHUB_TOKEN",
    },
    "free/cohere-command-a": {
        "base_url": "https://api.cohere.com/v1/",
        "model": "command-a",
        "env_var": "COHERE_API_KEY",
    },
}

PAID_MODELS: dict[str, dict] = {
    "paid/openrouter-deepseek-r1": {
        "base_url": "https://openrouter.ai/api/v1/",
        "model": "deepseek/deepseek-r1",
        "env_var": "OPENROUTER_API_KEY",
    },
    "paid/openrouter-llama-3.3": {
        "base_url": "https://openrouter.ai/api/v1/",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "env_var": "OPENROUTER_API_KEY",
    },
}

LOCAL_MODELS: dict[str, dict] = {}

# Combined lookup — public/ prefix remains as backward-compat alias
PUBLIC_MODELS: dict[str, dict] = {}
for _prefix, _reg in ("free", FREE_MODELS), ("paid", PAID_MODELS), ("local", LOCAL_MODELS):
    for _key, _cfg in _reg.items():
        PUBLIC_MODELS["public/" + _key.removeprefix(_prefix + "/")] = _cfg
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


def fetch_models(name: str, prov: dict) -> tuple[list[str], dict]:
    """Return (sorted ids, {id: capability meta}) from /v1/models.

    Moonshot and others return extra fields per model (context_length,
    supports_reasoning, think_efforts, supports_image_in …). We keep the ones
    we know how to use so the harness can auto-configure itself per model."""
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
        return [], {}
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return [], {}
    ids, meta = [], {}
    for m in rows:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        mid = str(m["id"])
        ids.append(mid)
        keep = {}
        for k in ("context_length", "supports_reasoning", "supports_image_in",
                  "supports_video_in", "supports_dynamic_tools",
                  "think_efforts", "reasoning_efforts", "supports_thinking_type"):
            if k in m:
                keep[k] = m[k]
        if keep:
            meta[mid] = keep
    return sorted(set(ids)), meta


def refresh_models(verbose: bool = True) -> dict:
    provs = load_providers()
    cache = {"_ts": time.time()}
    for name, prov in sorted(provs.items()):
        ids, meta = fetch_models(name, prov)
        cache[name] = ids
        if meta:
            cache["_meta_" + name] = meta
        if verbose:
            status = f"{len(ids):5d} models" if ids else "  unavailable"
            print(f"  {name:12s} {status}")
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_CACHE.write_text(json.dumps(cache, indent=1))
    _log(f"model cache refreshed ({len(provs)} providers)")
    return cache


def _model_meta(provider: str, model: str) -> dict:
    """Capability metadata for a model from the cache, if any was stored."""
    return (load_model_cache().get("_meta_" + provider) or {}).get(model, {})


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
        _log(f"warning: {env_var} not set for {provider}")
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
# Longest prefix wins: put specific model ids before family substrings.
# kimi-k3 is advertised at 1048576 by /v1/models; kimi family otherwise 1M.
_CTX_WINDOWS: dict[str, int] = {
    "kimi-k3": 1_048_576, "kimi": 1_000_000,
    "claude": 200_000, "gpt-4": 128_000, "gpt-4o": 128_000,
    "deepseek": 128_000, "gemini": 1_000_000,
    "llama": 128_000, "mistral": 128_000, "qwen": 128_000,
    "longcat": 128_000,
}
_DEFAULT_CTX_WINDOW = 128_000


def _ctx_window(model: str) -> int:
    m = model.lower()
    # most specific (longest) key first so "kimi-k3" beats "kimi"
    for key in sorted(_CTX_WINDOWS, key=len, reverse=True):
        if key in m:
            return _CTX_WINDOWS[key]
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
    usage = getattr(agent, "last_usage", {}) or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    return {"messages": len(msgs), "tokens": tokens, "window": window, "pct": pct,
            "cached": cached, "reasoning": reasoning}


def _ctx_format(template: str, agent, elapsed: float = 0) -> str:
    s = _ctx_stats(agent)
    short = agent.model.removeprefix(agent.provider + "/") if agent.model.startswith(agent.provider + "/") else agent.model
    now = time.strftime("%H:%M:%S")
    return template.format(
        time=now, provider=agent.provider, model=short,
        messages=s["messages"], tokens=s["tokens"],
        window=s["window"], pct=s["pct"],
        elapsed=f"{elapsed:.1f}",
        cached=s["cached"], reasoning=s["reasoning"],
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


# -- session transcript ------------------------------------------------------
# A per-session, append-as-it-happens plain-text record of the conversation
# (user turns, assistant text, tool calls + results). Survives a crash, unlike the
# in-memory message list. One file per session: logs/transcript-<ts>-<pid>.md
_TRANSCRIPT = None


def _transcript_start(provider: str, model: str) -> None:
    global _TRANSCRIPT
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    _TRANSCRIPT = LOG_DIR / f"transcript-{ts}-{os.getpid()}.md"
    _tx_write(f"# ainow transcript — {provider}/{model}\n"
              f"# started {time.strftime('%Y-%m-%d %H:%M:%S')}  pid {os.getpid()}\n\n")


def _tx_write(text: str) -> None:
    if _TRANSCRIPT is None:
        return
    try:
        with open(_TRANSCRIPT, "a") as f:
            f.write(text)
    except OSError:
        pass


def _tx(role: str, text: str) -> None:
    ts = time.strftime("%H:%M:%S")
    _tx_write(f"\n### [{ts}] {role}\n{text}\n")


# -- mid-task nudges ---------------------------------------------------------
# When the model is blocked inside a long tool call, the REPL would normally
# freeze and the user cannot steer it.  We run the tool in a worker thread and
# let the main thread poll stdin; any line the user types is queued and then
# injected as a clearly-marked user message right after the current batch of
# tool results.  Disabled for non-tty / one-shot stdin so unattended pipelines
# behave exactly as before.
_NUDGE_LOCK = threading.Lock()
_NUDGE_QUEUE: list[str] = []


def _queue_nudge(line: str) -> None:
    text = line.rstrip("\n").rstrip("\r").strip()
    if not text:
        return
    with _NUDGE_LOCK:
        _NUDGE_QUEUE.append(text)
    print(f"{C.d}  [nudge queued]{C.r}", flush=True)


def _drain_nudges(agent) -> None:
    """Append queued mid-task user lines as user messages."""
    with _NUDGE_LOCK:
        queued = _NUDGE_QUEUE[:]
        _NUDGE_QUEUE[:] = []
    for text in queued:
        marked = f"[user nudge mid-task] {text}"
        agent.messages.append({"role": "user", "content": marked})
        _tx("paul", marked)


def _run_with_nudges(fn, interactive: bool, *args, **kwargs):
    """Run a callable, collecting stdin lines as nudges if interactive+tty."""
    # Non-interactive (one-shot, piped stdin): keep the old blocking behaviour.
    if not interactive or not sys.stdin.isatty():
        return fn(*args, **kwargs)

    result = [None]
    error = [None]

    def worker():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:  # capture so main thread can re-raise
            error[0] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    stdin_fd = sys.stdin.fileno()
    try:
        while t.is_alive():
            try:
                ready, _, _ = select.select([stdin_fd], [], [], 0.2)
            except InterruptedError:
                # Signal (likely Ctrl-C) interrupted the syscall; loop so the
                # custom SIGINT handler's Interrupted exception can propagate.
                continue
            if ready:
                try:
                    line = sys.stdin.readline()
                except (OSError, EOFError):
                    break
                if not line:
                    break
                _queue_nudge(line)
    finally:
        # Give the worker a moment to finish cleanly; if Ctrl-C fired the caller
        # will raise Interrupted after this function returns.
        t.join(timeout=2.0)

    if error[0] is not None:
        raise error[0]
    return result[0]


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
_BUILTIN_REGISTRIES = {"free": FREE_MODELS, "paid": PAID_MODELS,
                      "local": LOCAL_MODELS, "public": PUBLIC_MODELS}


def complete(word: str) -> None:
    provs = sorted(load_providers().keys())
    cache = load_model_cache()

    if "/" not in word:
        for p in provs:
            if p.startswith(word):
                print(p + "/")
        for cls in ("free", "paid", "local", "public"):
            if cls.startswith(word):
                print(cls + "/")
        return

    cls = word.partition("/")[0]
    if cls in _BUILTIN_REGISTRIES:
        fragment = word.removeprefix(cls + "/")
        for key in sorted(_BUILTIN_REGISTRIES[cls]):
            if not fragment or key.startswith(word):
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


# -- multimodal attachments --------------------------------------------------
# kimi-k3 supports image and video input (supports_image_in / supports_video_in
# in /v1/models). A `file.<ext>` key on a tool-call record marks it as a binary
# attachment; run() converts it to an image_url / video_url content part.
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif",
             ".svg", ".heic", ".heif", ".avif", ".ico"}
_VID_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v", ".3gp"}
_FILE_URI_RE = re.compile(r"^(image|video)://(\S+)$", re.I)


def _file_part(path: str, kind: str) -> dict | None:
    """Build an OpenAI content part for a local image/video file (data URI)."""
    import base64
    p = pathlib.Path(path).expanduser()
    if not p.is_file():
        return None
    mime = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
        ".svg": "image/svg+xml", ".tiff": "image/tiff", ".tif": "image/tiff",
        ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
        ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
    }.get(p.suffix.lower())
    if mime is None:
        mime = ("image/" if kind == "image" else "video/") + p.suffix.lstrip(".")
    data = base64.b64encode(p.read_bytes()).decode()
    url = f"data:{mime};base64,{data}"
    if kind == "video":
        return {"type": "video_url", "video_url": {"url": url}}
    return {"type": "image_url", "image_url": {"url": url}}


def _content_with_attachments(text: str) -> str | list:
    """Extract image:// and video:// URIs from a prompt into content parts."""
    parts: list[dict] = []
    rest: list[str] = []
    for tok in text.split():
        m = _FILE_URI_RE.match(tok)
        if m:
            part = _file_part(m.group(2), m.group(1).lower())
            if part:
                parts.append(part)
                continue
        rest.append(tok)
    body = " ".join(rest)
    if not parts:
        return text
    if body:
        parts.insert(0, {"type": "text", "text": body})
    return parts


def t_read_file(path: str, offset: int = 1, limit: int = 2000) -> str:
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        return f"error: no such file: {p}"
    if p.is_dir():
        return f"error: {p} is a directory (use list_dir)"
    # kimi-k3 can see images/videos — don't dump base64 into the transcript;
    # hand back a URI the model can reference as image://… in a follow-up turn.
    ext = p.suffix.lower()
    if ext in _IMG_EXTS | _VID_EXTS:
        kind = "image" if ext in _IMG_EXTS else "video"
        return (f"{kind} file: {p} ({p.stat().st_size} bytes)\n"
                f"To view it, ask the user to attach it, or if you have it, "
                f"reference it in your reply as {kind}://{p}")
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


def t_bash(command: str, timeout: int = 120, background: bool = False) -> str:
    # Two hard rules for every child we spawn:
    #  1. stdin is ALWAYS /dev/null. An interactive-ish child (notably ssh) must never
    #     inherit the REPL's terminal stdin — that is what blocked the user from typing.
    #  2. The child runs in its own session/process group, and a timeout kills the WHOLE
    #     group. subprocess.run's timeout only kills the direct child (the shell), which
    #     is how orphaned ssh processes survived and held the terminal/channel open.
    if background:
        return _bg_run(command)
    p = subprocess.Popen(command, shell=True,
                         stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, errors="replace", start_new_session=True)
    try:
        out_s, err_s = p.communicate(timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return f"error: timed out after {timeout}s (process group killed)"
    out = (out_s or "") + (("\n[stderr]\n" + err_s) if err_s else "")
    if rc != 0:
        out += f"\n[exit {rc}]"
    return _clip(out.strip()) or f"(no output) [exit {rc}]"


# -- background jobs ---------------------------------------------------------
# Long-running reduce/backtest/build commands can outlive a context compaction.
# We keep a small registry under logs/bg/ so the harness remembers them across
# summaries and can list/read/kill them later.
_BG_DIR = LOG_DIR / "bg"
_BG_REGISTRY_FILE = _BG_DIR / "jobs.json"
_BG_REGISTRY: dict[str, dict] = {}
_BG_REGISTRY_LOADED = False


def _bg_init() -> None:
    global _BG_REGISTRY_LOADED
    if _BG_REGISTRY_LOADED:
        return
    _BG_DIR.mkdir(parents=True, exist_ok=True)
    if _BG_REGISTRY_FILE.exists():
        try:
            data = json.loads(_BG_REGISTRY_FILE.read_text())
            _BG_REGISTRY.update(data)
        except Exception as e:
            _log(f"background jobs registry load failed: {e}")
    _BG_REGISTRY_LOADED = True


def _bg_save() -> None:
    _BG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _BG_REGISTRY_FILE.write_text(json.dumps(_BG_REGISTRY, indent=2))
    except Exception as e:
        _log(f"background jobs registry save failed: {e}")


def _bg_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _bg_reap() -> None:
    """Mark dead jobs, reap zombies, and persist."""
    changed = False
    for job in _BG_REGISTRY.values():
        if not job.get("alive"):
            continue
        pid = job["pid"]
        # Try to reap it if it is our child; if waitpid returns the pid, it died.
        try:
            pid2, _ = os.waitpid(pid, os.WNOHANG)
            if pid2 != 0:
                job["alive"] = False
                job["finished"] = time.time()
                changed = True
                continue
        except (ChildProcessError, OSError):
            pass
        # Not (yet) reapable; check whether the pid still exists at all.
        if not _bg_is_alive(pid):
            job["alive"] = False
            job["finished"] = time.time()
            changed = True
    if changed:
        _bg_save()


def _bg_next_id() -> str:
    ids = [int(k) for k in _BG_REGISTRY if k.isdigit()]
    return str(max(ids, default=0) + 1)


def _bg_run(command: str) -> str:
    _bg_init()
    _BG_DIR.mkdir(parents=True, exist_ok=True)
    jid = _bg_next_id()
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = _BG_DIR / f"job-{jid}-{ts}.log"
    log_fp = log_path.open("w")
    try:
        p = subprocess.Popen(command, shell=True,
                             stdin=subprocess.DEVNULL,
                             stdout=log_fp, stderr=subprocess.STDOUT,
                             text=True, errors="replace",
                             start_new_session=True, close_fds=True)
    except Exception as e:
        log_fp.close()
        return f"error: failed to spawn background job: {e}"
    now = time.time()
    _BG_REGISTRY[jid] = {
        "command": command,
        "pid": p.pid,
        "log": str(log_path),
        "started": now,
        "alive": True,
    }
    _bg_save()
    return f"background job {jid} started (pid {p.pid}) → {log_path}"


def _bg_list() -> str:
    _bg_init()
    _bg_reap()
    if not _BG_REGISTRY:
        return "no background jobs"
    rows = []
    for jid in sorted(_BG_REGISTRY, key=lambda k: int(k)):
        j = _BG_REGISTRY[jid]
        elapsed = time.time() - j["started"]
        status = "running" if j.get("alive") else "finished"
        rows.append(f"{jid}: [{status}] {elapsed:.1f}s  {j['command']}")
    return "\n".join(rows)


def _bg_log_tail(job_id: str, lines: int = 50) -> str:
    _bg_init()
    job = _BG_REGISTRY.get(job_id)
    if not job:
        return f"error: no job {job_id}"
    p = pathlib.Path(job["log"])
    if not p.exists():
        return f"error: log not found: {p}"
    try:
        with p.open("r", errors="replace") as f:
            buf = f.readlines()
        out = "".join(buf[-lines:])
        return _clip(out.strip()) or "(log empty)"
    except Exception as e:
        return f"error: reading log: {e}"


def _bg_kill(job_id: str, sig: int = signal.SIGKILL) -> str:
    _bg_init()
    job = _BG_REGISTRY.get(job_id)
    if not job:
        return f"error: no job {job_id}"
    pid = job["pid"]
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError):
        pass
    _bg_reap()
    return f"job {job_id} killed (pid {pid})"


def t_bash_jobs(action: str = "list", job_id: str = "", lines: int = 50,
                signal_name: str = "SIGKILL") -> str:
    action = action.lower()
    if action == "list":
        return _bg_list()
    if action == "log":
        if not job_id:
            return "error: job_id required for log"
        return _bg_log_tail(job_id, lines)
    if action == "kill":
        if not job_id:
            return "error: job_id required for kill"
        sig = getattr(signal, signal_name, signal.SIGKILL)
        return _bg_kill(job_id, sig)
    return f"error: unknown action {action!r} (try list/log/kill)"


# Persistent "working memory" journal target. Configured via env so nothing private
# (host, path) is baked into the public repo:
#   AINOW_JOURNAL_SSH   ssh target, e.g. "user@host"   (required to enable)
#   AINOW_JOURNAL_FILE  absolute path to the journal on that host (required)
# If either is unset the journal tool reports itself disabled.
# ~/.config/ainow/journal.env (gitignored, KEY=VALUE or `export KEY=VALUE` lines) is
# loaded automatically so the user need not export these by hand.
def _load_journal_env() -> None:
    try:
        for line in (CFG_DIR / "journal.env").read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.removeprefix("export ").strip()
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except (OSError, FileNotFoundError):
        pass


_load_journal_env()
_JOURNAL_SSH = os.environ.get("AINOW_JOURNAL_SSH", "")
_JOURNAL_FILE = os.environ.get("AINOW_JOURNAL_FILE", "")


def t_journal(text: str, section: str = "LOG") -> str:
    """Append a durable entry to the persistent working-memory journal.

    section=LOG     -> append under the dated heading in the LOG (default; use for
                       decisions, findings, state changes worth surviving a reboot).
    section=THREADS -> append a bullet to OPEN THREADS (a new watch-item / to-do).
    """
    if not (_JOURNAL_SSH and _JOURNAL_FILE):
        return ("journal disabled: set AINOW_JOURNAL_SSH and AINOW_JOURNAL_FILE "
                "to your persistent journal target")
    section = section.upper()
    if section not in ("LOG", "THREADS"):
        return f"error: section must be LOG or THREADS, got {section!r}"
    # base64-encode the text so it survives the remote shell untouched, decode it in
    # a small Python editor script piped over ssh stdin. Immune to any characters.
    import base64 as _b64
    date = time.strftime("%Y-%m-%d")
    enc = _b64.b64encode(text.encode()).decode()
    pre = (
        "import base64\n"
        f"p={_JOURNAL_FILE!r}\n"
        f"text=base64.b64decode('{enc}').decode()\n"
        "s=open(p).read()\n"
    )
    if section == "LOG":
        body = pre + (
            f"hdr='### {date}'\n"
            "if hdr not in s:\n"
            "    s=s.rstrip('\\n')+'\\n\\n'+hdr+'\\n'\n"
            "s=s.rstrip('\\n')+'\\n- '+text+'\\n'\n"
            "open(p,'w').write(s)\n"
            "print('logged')\n"
        )
    else:  # THREADS — insert after the full OPEN THREADS heading line
        body = pre + (
            "import re\n"
            "m=re.search(r'^## OPEN THREADS.*$', s, re.M)\n"
            "assert m, 'no OPEN THREADS heading'\n"
            "line='- [ ] '+text+'\\n'\n"
            "s=s[:m.end()]+'\\n'+line+s[m.end():]\n"
            "open(p,'w').write(s)\n"
            "print('thread added')\n"
        )
    # Use the same orphan-ssh-safe pattern as t_bash: own session, kill the whole
    # process group on timeout.  Otherwise a stuck ssh could outlive the harness.
    p = subprocess.Popen(["ssh", _JOURNAL_SSH, "python3", "-"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, errors="replace",
                         start_new_session=True)
    try:
        out_s, err_s = p.communicate(input=body, timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return "error: journal ssh timed out after 30s (process group killed)"
    except Exception as e:
        return f"error: journal write failed: {e}"
    out = (out_s or "") + (err_s or "")
    return out.strip() or f"(ssh exit {p.returncode})"


TOOLS = {
    "read_file":  (t_read_file,  {"path": "str"}, False),
    "list_dir":   (t_list_dir,   {"path": "str"}, False),
    "write_file": (t_write_file, {"path": "str"}, True),
    "edit_file":  (t_edit_file,  {"path": "str"}, True),
    "bash":       (t_bash,       {"command": "str", "background": "bool"}, True),
    "bash_jobs":  (t_bash_jobs,  {"action": "str", "job_id": "str", "lines": "int", "signal_name": "str"}, False),
    "journal":    (t_journal,    {"text": "str", "section": "str"}, False),
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
            "timeout": {"type": "integer", "description": "Seconds, default 120"},
            "background": {"type": "boolean", "description": "Run detached and return a job id"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "bash_jobs",
        "description": "Manage background bash jobs: list, read a log tail, or kill by id.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "description": "list, log, or kill"},
            "job_id": {"type": "string", "description": "Job id (required for log/kill)"},
            "lines": {"type": "integer", "description": "Tail lines for log action, default 50"},
            "signal_name": {"type": "string", "description": "Signal for kill, default SIGKILL"}},
            "required": ["action"]}}},
    {"type": "function", "function": {
        "name": "journal",
        "description": "Persist a durable note to the persistent working-memory "
                       "journal (configured via AINOW_JOURNAL_SSH/AINOW_JOURNAL_FILE). "
                       "Call this the moment a decision, finding, or state change "
                       "worth surviving a reboot happens — do not wait for end of "
                       "session. section=LOG (default) logs a dated entry; "
                       "section=THREADS adds an OPEN THREADS item.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "The note to persist"},
            "section": {"type": "string", "description": "LOG (default) or THREADS"}},
            "required": ["text"]}}},
]

# -- plugin tools ------------------------------------------------------------
# Drop-in python files under ~/.config/ainow/tools.d/ (or AINOW_TOOLS_DIR)
# register extra tools WITHOUT editing ainow.py.  This is how private tools
# (e.g. trading reads) live in the gitignored overlay instead of forking the
# public repo — the harness stays generic.
#
# A plugin may either export:
#   TOOL_SPEC = {"name": ..., "schema": {...params...}, "call": callable,
#                "description": "...", "requires_approval": False}
# or define:
#   def register(registry): registry.add(...)
#
# The schema is the JSON-Schema parameters object ("type":"object", properties,
# required).  Broken plugins are warned and skipped.
class _PluginReg:
    def __init__(self):
        self.tools: dict[str, tuple] = {}
        self.schemas: list[dict] = []

    def add(self, name: str, parameters_schema: dict, call, description: str = "",
            requires_approval: bool = False) -> None:
        self.tools[name] = (call, parameters_schema, requires_approval, True)
        self.schemas.append({"type": "function", "function": {
            "name": name,
            "description": "[plugin] " + (description or f"Plugin tool {name}"),
            "parameters": parameters_schema}})


def _load_plugins() -> None:
    tools_dir = pathlib.Path(os.environ.get("AINOW_TOOLS_DIR",
                                           str(CFG_DIR / "tools.d")))
    if not tools_dir.is_dir():
        return
    for fp in sorted(tools_dir.glob("*.py")):
        try:
            spec = importlib.util.spec_from_file_location(fp.stem, fp)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            reg = _PluginReg()
            if hasattr(mod, "register") and callable(mod.register):
                mod.register(reg)
            elif hasattr(mod, "TOOL_SPEC"):
                s = mod.TOOL_SPEC
                reg.add(s["name"], s["schema"], s["call"],
                        s.get("description", ""), s.get("requires_approval", False))
            else:
                _log(f"plugin {fp}: no TOOL_SPEC or register(); skipped")
                continue
            for name, (call, params, appr, _) in reg.tools.items():
                if name in TOOLS:
                    _log(f"plugin {fp}: tool {name} conflicts with built-in; skipped")
                    print(f"{C.ye}  warning: plugin tool {name!r} conflicts with "
                          f"built-in and was skipped{C.r}")
                    continue
                TOOLS[name] = (call, params, appr)
            TOOL_SCHEMA.extend(reg.schemas)
            if reg.tools:
                _log(f"plugin loaded: {fp.name} ({len(reg.tools)} tools)")
        except Exception as e:
            _log(f"plugin {fp} failed: {e}")
            print(f"{C.ye}  warning: plugin {fp.name} failed to load: {e}{C.r}")


SYSTEM = """You are ainow, a command-line coding assistant running on the user's \
Linux machine with real filesystem and shell access.

You have tools: read_file, list_dir, write_file, edit_file, bash, bash_jobs, \
journal, plus any plugin tools loaded from ~/.config/ainow/tools.d/. Use them \
to inspect and change files directly rather than printing code for the user to \
copy. Prefer edit_file over rewriting whole files. Read a file before editing it. \
Use the journal tool to persist any decision, finding, or state change worth \
surviving a reboot the moment it happens, not just at end of session.

Be concise. The user is in a terminal — use plain text only. No markdown of any \
kind (no **bold**, `backticks`, bullet lists) unless explicitly asked. Report \
what you actually did, and if a command failed, say so with the output rather \
than assuming it worked."""


# Optional private system-prompt overlay. If ~/.config/ainow/system.local exists,
# its contents are appended to SYSTEM. This is the place for private context (hosts,
# paths, ongoing projects) that must NOT be committed to the public repo. The file
# is gitignored (see .gitignore).
def _system() -> str:
    try:
        local = (CFG_DIR / "system.local").read_text().strip()
    except (OSError, FileNotFoundError):
        local = ""
    return SYSTEM + (("\n\n" + local) if local else "")


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
# ansi — honour NO_COLOR and non-tty
# --------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


class C:
    d  = "\033[2m" if _USE_COLOR else ""
    b  = "\033[1m" if _USE_COLOR else ""
    r  = "\033[0m" if _USE_COLOR else ""
    cy = "\033[36m" if _USE_COLOR else ""
    gr = "\033[32m" if _USE_COLOR else ""
    ye = "\033[33m" if _USE_COLOR else ""
    re = "\033[31m" if _USE_COLOR else ""
    ma = "\033[35m" if _USE_COLOR else ""


# Load drop-in plugin tools now that TOOLS/TOOL_SCHEMA/C are all defined.
_load_plugins()


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
        self.messages = [{"role": "system", "content": _system()}]
        self._last_elapsed = 0.0
        # Capability metadata cached from /v1/models (context_length,
        # think_efforts, supports_image_in …). Providers.json wins if explicit.
        meta = _model_meta(provider, model)
        self.meta = meta
        self.ctx_window = (prov_cfg.get("ctx_window")
                           or meta.get("context_length")
                           or _ctx_window(model))
        # kimi-k3 thinking. None = provider default (k3 defaults to "max").
        # Valid efforts are advertised by /v1/models: low | high | max.
        # Note kimi-k3 is thinking-only — it cannot be switched off entirely.
        self.think_effort = prov_cfg.get("think_effort")
        efforts = (meta.get("think_efforts") or {})
        self.valid_efforts = efforts.get("valid_efforts") or ["low", "high", "max"]
        self.show_reasoning = True
        self.web_search = False          # register moonshot builtin $web_search
        self.last_usage: dict = {}       # usage from the final chunk of the last turn
        self.interactive_repl = False    # set True by the interactive REPL for nudge watching

    def _extra_body(self) -> dict:
        """Provider extension params for the request."""
        body: dict = {}
        if self.think_effort:
            body["think_effort"] = self.think_effort
        return body

    def _tools(self) -> list:
        tools = list(TOOL_SCHEMA)
        if self.web_search:
            tools.append({"type": "builtin_function",
                          "function": {"name": "$web_search"}})
        return tools

    # -- approval -----------------------------------------------------
    def _approve(self, name: str, args: dict) -> bool:
        if self.auto or not TOOLS[name][2]:
            return True
        detail = args.get("command") or args.get("path") or ""
        # Non-interactive (one-shot / piped stdin): there is no human to answer the
        # prompt, and blocking on input() would hang forever. One-shot is meant for
        # unattended runs, so auto-approve. (To force a prompt, run interactively.)
        if not sys.stdin.isatty():
            _log(f"tool auto-approved (non-interactive) {name} {str(detail)[:120]}")
            print(f"{C.ye}  {name}{C.r} {C.d}{str(detail)[:160]}{C.r} {C.d}[auto: non-interactive]{C.r}")
            return True
        print(f"\n{C.ye}  {name}{C.r} {C.d}{str(detail)[:160]}{C.r}")
        try:
            ans = input(f"  {C.b}run it?{C.r} [y/N/a=always] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            _log(f"tool rejected {name} {str(detail)[:120]}")
            return False
        if ans == "a":
            self.auto = True
        if ans not in ("y", "yes", "a"):
            _log(f"tool rejected {name} {str(detail)[:120]}")
            return False
        return True

    # -- one streamed assistant turn ----------------------------------
    def _stream_turn(self) -> dict:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: dict[int, dict] = {}
        printed_any = False
        thinking_shown = False

        # Strip out-of-band keys (prefixed _) before the wire.
        wire = [{k: v for k, v in m.items() if not k.startswith("_")}
                for m in self.messages]
        stream = self.client.chat.completions.create(
            model=self.model, messages=wire,
            tools=self._tools(), tool_choice="auto", stream=True,
            stream_options={"include_usage": True},
            extra_body=self._extra_body(),
        )
        for chunk in stream:
            if getattr(chunk, "usage", None):
                self.last_usage = chunk.usage.model_dump()
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            # kimi-k3 always thinks; reasoning streams as reasoning_content deltas
            rc = getattr(d, "reasoning_content", None)
            if rc:
                reasoning_parts.append(rc)
                if self.show_reasoning:
                    if not thinking_shown:
                        print(f"{C.d}  ── thinking ──")
                        thinking_shown = True
                    sys.stdout.write(rc)
                    sys.stdout.flush()
            if getattr(d, "content", None):
                if thinking_shown:
                    print(f"\n  ────────────{C.r}")
                    thinking_shown = False
                    printed_any = False  # let content set it; header ends its own line
                sys.stdout.write(d.content)
                sys.stdout.flush()
                text_parts.append(d.content)
                printed_any = True
            for tc in (getattr(d, "tool_calls", None) or []):
                slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": "",
                                                   "type": "function"})
                if tc.id:
                    slot["id"] = tc.id
                if getattr(tc, "type", None):
                    slot["type"] = tc.type
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments
        if thinking_shown:
            print(f"\n  ────────────{C.r}")
        elif printed_any:
            print()

        msg = {"role": "assistant", "content": "".join(text_parts) or None}
        # Persist reasoning in history. kimi-k3 accepts reasoning_content echoed back,
        # so the model keeps sight of its own chain of thought across turns.
        reasoning = "".join(reasoning_parts)
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            builtin_ids = []
            tcs = []
            for i, c in sorted(calls.items()):
                if c["type"] == "builtin_function":
                    builtin_ids.append(c["id"] or f"call_{i}")
                tc = {"id": c["id"] or f"call_{i}", "type": "function",
                      "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                tcs.append(tc)
            # History echo: moonshot 400s ("tokenization failed") on re-posted
            # type=builtin_function — normalize to function, which it accepts.
            msg["tool_calls"] = tcs
            if builtin_ids:
                msg["_builtin_ids"] = builtin_ids
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
        self.messages.append({"role": "user",
                              "content": _content_with_attachments(user_text)})
        _tx("paul", user_text)
        try:
            with sigint_guard():
                while True:
                    msg = self._stream_turn()
                    self.messages.append(msg)
                    if msg.get("reasoning_content"):
                        _tx("ainow thinking", msg["reasoning_content"])
                    if msg.get("content"):
                        _tx("ainow", msg["content"])
                    tcs = msg.get("tool_calls")
                    if not tcs:
                        self._last_elapsed = time.time() - t0
                        return

                    builtin_ids = set(msg.get("_builtin_ids") or [])
                    for tc in tcs:
                        name = tc["function"]["name"]
                        tc_id = tc["id"]
                        # Builtin (moonshot $web_search etc.): search happens server-side;
                        # echo the arguments back as the tool result per the docs.
                        if tc_id in builtin_ids:
                            args = json.loads(tc["function"]["arguments"] or "{}")
                            label = args.get("query") or args.get("search_id") or ""
                            _log(f"builtin {name} {str(label)[:200]}")
                            print(f"{C.cy}  · {name}{C.r} {C.d}{str(label)[:120]}{C.r}")
                            # Per moonshot docs: echo the arguments back as the tool result.
                            result = tc["function"]["arguments"]
                        else:
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
                                        result = _run_with_nudges(
                                            TOOLS[name][0], self.interactive_repl, **args)
                                    except TypeError as e:
                                        result = f"error: bad arguments: {e}"
                                    except Exception as e:
                                        result = f"error: {type(e).__name__}: {e}"
                        _tx(f"tool:{name}", f"args: {tc['function']['arguments']}\n→ {result}")
                        self.messages.append({"role": "tool", "tool_call_id": tc_id,
                                              "content": str(result)})
                    # Inject any user lines typed while the tools were running.  We
                    # drain here *after* every tool_call in this assistant turn has a
                    # matching tool result, preserving OpenAI's tool-call ordering.
                    _drain_nudges(self)
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
  /model <spec>  switch model, e.g. /model free/gemini-2.5-flash, /model paid/openrouter-llama-3.3
  /models [pat]  list cached models for the current provider
  /httpd [start|stop]  file-transfer server (status if no args)
    start [-port N] [-root PATH] [-user U] [-pass P]
  /tokens        token/context stats (alias: /ctx)
  /ctx [compress|window N]  context stats, compress, or set window
  /think [low|high|max|off]  show/set kimi-k3 think effort (off = provider default)
  /reasoning [on|off]        show the model's reasoning as it streams
  /websearch [on|off]        register moonshot builtin $web_search tool
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
    # Sanitise for the summary pass: drop out-of-band keys, reasoning chains,
    # and any base64 media payloads — the summariser only needs the gist.
    to_summarize = []
    for m in msgs[1:keep_from]:
        m2 = {k: v for k, v in m.items()
              if not k.startswith("_") and k != "reasoning_content"}
        if isinstance(m2.get("content"), list):
            m2["content"] = " ".join(
                p.get("text", "[media]") if p.get("type") == "text" else "[media]"
                for p in m2["content"])
        to_summarize.append(m2)

    print(f"{C.d}  summarising {len(to_summarize)} messages…{C.r}")
    summary_msgs = [
        {"role": "system", "content": "Summarise this conversation concisely. Keep key facts, decisions, file changes, and errors. Write in plain paragraphs."},
        {"role": "user", "content": json.dumps(to_summarize)},
    ]
    try:
        stream = agent.client.chat.completions.create(
            model=agent.model, messages=summary_msgs, stream=True,
            extra_body={"think_effort": "low"},
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
    _transcript_start(agent.provider, agent.model)
    try:
        agent.run(prompt)
    except Interrupted:
        print(file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"ainow: API error: {e}", file=sys.stderr)
        sys.exit(2)

    # The reply already streamed to stdout; only print it again if the final
    # message didn't (e.g. content is structured, not text).
    last = agent.messages[-1]
    if last["role"] == "assistant":
        text = last.get("content", "")
        if isinstance(text, list):
            print(" ".join(p.get("text", "") for p in text if p.get("type") == "text"))
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
    agent.interactive_repl = True

    cwd = os.path.realpath(os.getcwd())
    _transcript_start(agent.provider, agent.model)
    _log(f"session start {agent.provider}/{agent.model}  cwd={cwd}  auto={agent.auto}")
    print(f"{C.ma}ainow{C.r} {C.b}{agent.provider}/{agent.model}{C.r}  "
          f"{C.d}cwd {cwd}{C.r}")
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
                _log(f"session end {agent.provider}/{agent.model}  msgs={len(agent.messages)}")
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
                _log(f"session end {agent.provider}/{agent.model}  msgs={len(agent.messages)}")
                print("bye")
                return
            if cmd == "help":
                print(HELP)
            elif cmd == "clear":
                agent.messages = agent.messages[:1]
                _log(f"context cleared ({agent.provider}/{agent.model})")
                print(f"{C.d}context cleared{C.r}")
            elif cmd in ("ctx", "tokens"):
                _handle_ctx_cmd(agent, rest)
            elif cmd == "auto":
                if rest in ("on", "off"):
                    agent.auto = rest == "on"
                else:
                    agent.auto = not agent.auto
                _log(f"auto {agent.auto} ({agent.provider}/{agent.model})")
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
            elif cmd == "think":
                valid = getattr(agent, "valid_efforts", ["low", "high", "max"])
                if rest in valid:
                    agent.think_effort = rest
                    print(f"{C.d}think effort = {rest}{C.r}")
                elif rest in ("off", "default", ""):
                    if rest:
                        agent.think_effort = None
                        print(f"{C.d}think effort = provider default{C.r}")
                    else:
                        cur = agent.think_effort or "provider default (k3: max)"
                        print(f"{C.d}think effort = {cur}  (valid: {', '.join(valid)}){C.r}")
                else:
                    print(f"{C.re}usage: /think [{'|'.join(valid)}|off]{C.r}")
            elif cmd == "reasoning":
                if rest in ("on", "off"):
                    agent.show_reasoning = rest == "on"
                else:
                    agent.show_reasoning = not agent.show_reasoning
                print(f"{C.d}reasoning display {'on' if agent.show_reasoning else 'off'}{C.r}")
            elif cmd == "websearch":
                if rest in ("on", "off"):
                    agent.web_search = rest == "on"
                else:
                    agent.web_search = not agent.web_search
                print(f"{C.d}web_search builtin {'on' if agent.web_search else 'off'}{C.r}")
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
def _resolve_builtin(spec: str) -> dict | None:
    """Look up a built-in model by its classifier prefix: free/, paid/, local/, or public/."""
    cls, _, name = spec.partition("/")
    registry = {"free": FREE_MODELS, "paid": PAID_MODELS,
                "local": LOCAL_MODELS, "public": PUBLIC_MODELS}.get(cls)
    if registry is not None:
        return registry.get(spec)
    return None


def parse_spec(spec: str, provs: dict) -> tuple[str, str, dict | None]:
    """Parse model spec. Returns (provider, model, public_cfg_or_None)."""
    if "/" in spec:
        cls = spec.partition("/")[0]
        if cls in ("free", "paid", "local", "public"):
            cfg = _resolve_builtin(spec)
            if cfg is not None:
                return (spec, cfg["model"], cfg)
            sys.exit(f"ainow: unknown {cls} model '{spec}'; try --{cls}")
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
    if argv and argv[0] in ("--complete", "--models", "-m", "--providers",
                             "--public", "--free", "--paid", "--local"):
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
        print(f"built-in: {len(FREE_MODELS)} free, {len(PAID_MODELS)} paid"
              + (f", {len(LOCAL_MODELS)} local" if LOCAL_MODELS else "")
              + " (--free, --paid, --local to list)")
        return
    if argv[0] in ("--public", "--free", "--paid", "--local"):
        label = argv[0][2:]  # strip --
        registry = {"public": PUBLIC_MODELS, "free": FREE_MODELS,
                    "paid": PAID_MODELS, "local": LOCAL_MODELS}.get(label, PUBLIC_MODELS)
        tag = f"{label}{' (backward compat)' if label == 'public' else ''}"
        print(f"built-in {tag} models ({len(registry)}):")
        for key in sorted(registry):
            pub = registry[key]
            print(f"  {key:35s} → {pub['base_url']}  [{pub.get('env_var', '')}]")
        return
    if argv[0] == "--providers":
        for n, p in sorted(load_providers().items()):
            print(f"  {n:12s} {p['base_url']}")
        print(f"\n  built-in: {len(FREE_MODELS)} free, {len(PAID_MODELS)} paid"
              + (f", {len(LOCAL_MODELS)} local" if LOCAL_MODELS else "")
              + " (--free, --paid, --local to list) --")
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
    allow_root = False

    # parse flags before positional model spec
    if "--yolo" in argv:
        auto = True
        argv.remove("--yolo")
    if "-c" in argv:
        idx = argv.index("-c")
        if idx + 1 < len(argv):
            oneshot_prompt = argv[idx + 1]
        argv = argv[:idx]
    if "--allow-foot-bullet-root-mode" in argv:
        allow_root = True
        argv.remove("--allow-foot-bullet-root-mode")

    if os.geteuid() == 0 and not allow_root:
        sys.exit(
            "ainow: refusing to run as root. Use --allow-foot-bullet-root-mode to override."
        )

    provs = load_providers()
    prov, model, pub_cfg = parse_spec(argv[0], provs)
    first = " ".join(argv[1:]) or None

    if pub_cfg:
        _validate_api_key(prov, pub_cfg)

    stale = False
    if MODELS_CACHE.exists():
        try:
            entry = json.loads(MODELS_CACHE.read_text())
            stale = time.time() - entry.get("_ts", 0) > CACHE_TTL
        except json.JSONDecodeError:
            stale = True
    if not MODELS_CACHE.exists() or stale:
        print(f"{'rebuilding stale' if stale else 'building'} model cache…")
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
