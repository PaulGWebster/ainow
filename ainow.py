#!/usr/bin/env python3
"""ainow — a small multi-provider agentic coding harness.

Usage:  ainow <provider>/<model>  [initial prompt ...]

Flags:  --yolo      auto-approve all tool calls
        -c PROMPT   one-shot: run prompt and exit (no REPL)
        --comm none|home|local  peer comm socket mode (env AINOW_COMM overrides default)
        --label LABEL           instance label for comm peers (default: pid or workspace)
        -w|--workspace NAME     load workspace (env AINOW_WORKSPACE as default)
        --allow-foot-bullet-root-mode   allow running as root

Keys:   Ctrl-C  interrupt current generation / clear line  (does NOT exit)
        Ctrl-D  exit
        Ctrl-Q  exit
"""
from __future__ import annotations

import atexit
import importlib.util
import json
import os
import pathlib
import re
import select
import signal
import socket
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
import types

HOME = pathlib.Path.home()
CFG_DIR = HOME / ".config" / "ainow"
PROVIDERS_FILE = CFG_DIR / "providers.json"

# Built-in registry — OpenAI-compatible models, no provider config needed.
# Three classifiers: free/, paid/, local/.  public/ is a backward-compat alias.
FREE_MODELS: dict[str, dict] = {
    "free/gemini-3.5-flash-lite": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.5-flash-lite",
        "env_var": "GEMINI_API_KEY",
        "ctx_window": 1_000_000,
    },
    "free/gemini-3.6-flash": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.6-flash",
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
    "free/groq-gpt-oss-120b": {
        "base_url": "https://api.groq.com/openai/v1/",
        "model": "openai/gpt-oss-120b",
        "env_var": "GROQ_API_KEY",
    },    "free/groq-qwen3.8-27b": {
        "base_url": "https://api.groq.com/openai/v1/",
        "model": "qwen/qwen3.8-27b",
        "env_var": "GROQ_API_KEY",
    },                    "free/nvidia-llama-3.3": {
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

# -- per-instance AF_UNIX comm socket ---------------------------------------
# Home scope is user-private: only this uid can list/create/remove instances.
COMM_HOME_DIR = HOME / "ainow"              # --comm home  (mode 0700)
# Local scope is box-wide and cross-user: sticky-bit directory so anyone can
# create their own instance dir but nobody can remove another user's.
# Env override lets tests isolate instances without touching the real /tmp/ainow.
COMM_LOCAL_DIR = pathlib.Path(os.environ.get("AINOW_COMM_LOCAL", "/tmp/ainow"))
_COMM_MODE: str | None = None
_COMM_DIR: pathlib.Path | None = None
_COMM_SOCK_PATH: pathlib.Path | None = None
_COMM_REGISTRY_PATH: pathlib.Path | None = None
_COMM_LABEL: str | None = None
_COMM_LISTENER: socket.socket | None = None
_COMM_THREAD: threading.Thread | None = None
_COMM_SHUTDOWN = threading.Event()
_COMM_CLEANED = False

# -- reactive comm: wake an idle REPL prompt when a peer message arrives -----
# Without this, a comm message only gets processed at the START of the next
# human-driven turn (_drain_nudges in Agent.run) -- an idle prompt just shows
# the "[msg from X]" banner and sits there until a human notices and types
# something. _REPL_SESSION lets the comm listener thread (which is NOT the
# thread running the prompt_toolkit event loop) find the live prompt
# Application and ask it to exit early via the documented thread-safe path
# (loop.call_soon_threadsafe), causing session.prompt() to return _COMM_WAKE
# instead of a typed line. The REPL loop then runs a turn with no keystroke
# needed. Two safeguards: never discards a half-typed line (skips the
# auto-exit if the buffer is non-empty), and caps consecutive
# auto-triggered turns so two auto-reactive instances messaging each other
# cannot ping-pong forever unsupervised.
_REPL_SESSION = None  # type: ignore[var-annotated]
_COMM_WAKE = object()
_COMM_AUTORUN_STREAK = 0
_COMM_AUTORUN_MAX = 3


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

    # NOTE: some providers (Cerebras via Cloudflare, OKX historically) 403 the
    # urllib default User-Agent; send a browser UA so /models works everywhere.
    req = urllib.request.Request(
        prov["base_url"] + "/models",
        headers={"Authorization": "Bearer " + prov["api_key"],
                 "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ainow"},
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
        # Google's endpoint namespaces ids as "models/X"; the canonical (and
        # copy-pasteable) form for chat calls is the bare "X". Strip the prefix
        # at ingestion so listings, validation, and requests all use bare ids.
        if mid.startswith("models/"):
            mid = mid[len("models/"):]
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
def _validate_model(provider: str, model: str) -> tuple[str | None, str]:
    """Return (error_or_None, resolved_model_id).

    Providers namespace ids ('qwen/qwen3.8-27b', 'openai/gpt-oss-120b'); typing the
    bare tail is natural and unambiguous when exactly one cached id ends with it —
    resolve to the FULL cached id in that case (the bare tail 404s at the API).
    """
    cache = load_model_cache()
    ids = cache.get(provider)
    if not ids:
        return None, model  # cache empty — let the API decide
    if model in ids:
        return None, model
    suffix = [i for i in ids if i.endswith("/" + model)]
    if len(suffix) == 1:
        return None, suffix[0]
    from difflib import get_close_matches
    suggestions = get_close_matches(model, ids, n=4, cutoff=0.3)
    msg = f"'{model}' not found in {provider} cache"
    if suggestions:
        msg += "; closest: " + ", ".join(suggestions)
    return msg, model


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
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(AINOW_LOG, "a") as f:
            f.write(f"[{ts}] {msg}\n")
    except OSError:
        pass


def _httpd_log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(HTTPD_LOG, "a") as f:
        f.write(f"[{ts}] {msg}\n")


# -- per-instance AF_UNIX comm socket ---------------------------------------
# Each ainow instance may expose a UNIX-domain socket for lightweight peer
# messaging.  Mode is selected by --comm (env AINOW_COMM overrides the default).
#   none  = disabled
#   home  = ~/ainow/              (user-private, mode 0700)
#   local = /tmp/ainow/           (box-wide sticky-bit directory, mode 01777)
#
# Layout: per-instance DIRECTORY <root>/ainow.<pid>/ containing:
#   instance      -- the AF_UNIX socket
#   registry.json -- {"pid","label","model","cwd","started","workspace"}
# The directory is the future extension point (extra state files, lockfiles,
# mounts, etc.).
#
# Permission split:
#   * The model-facing tools comm_list/comm_send are registered with
#     needs_approval=True: every model-initiated peer message/task must be
#     explicitly approved by the user.
#   * Non-model local processes (watchers, crons, daemons, user scripts) may
#     write newline-delimited JSON directly to any ainow.<pid>/instance socket
#     with no approval gate.  This is by design: user infra runs under the
#     user's own authority and does not need an additional human-in-the-loop
#     check.

def _comm_instance_dir(dir_: pathlib.Path, pid: int) -> pathlib.Path:
    """Per-instance directory under the comm root."""
    return dir_ / f"ainow.{pid}"


def _comm_sock_path(dir_: pathlib.Path, pid: int) -> pathlib.Path:
    """AF_UNIX socket lives inside the instance directory."""
    return _comm_instance_dir(dir_, pid) / "instance"


def _comm_reg_path(dir_: pathlib.Path, pid: int) -> pathlib.Path:
    """Registry lives next to the socket inside the instance directory."""
    return _comm_instance_dir(dir_, pid) / "registry.json"


def _comm_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Some process we cannot signal owns it; do not treat as stale.
        return True


def _comm_probe(sock: pathlib.Path) -> str:
    """Try to connect to a peer socket.  Returns 'alive', 'refused', or 'error'."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect(str(sock))
        s.close()
        return "alive"
    except ConnectionRefusedError:
        return "refused"
    except OSError:
        return "error"
    finally:
        try:
            s.close()
        except OSError:
            pass


def _comm_read_registry(dir_: pathlib.Path, pid: int) -> dict:
    p = _comm_reg_path(dir_, pid)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {"pid": pid, "label": str(pid)}


def _comm_unlink_pair(dir_: pathlib.Path, pid: int) -> None:
    """Remove an instance's socket + registry, then the instance directory.

    Also cleans legacy flat layout files (ainow.<pid> and ainow.<pid>.json)
    if they exist, so upgrades from the old format do not crash.
    """
    inst_dir = _comm_instance_dir(dir_, pid)
    if inst_dir.exists():
        for name in ("instance", "registry.json"):
            try:
                (inst_dir / name).unlink()
            except OSError:
                pass
        try:
            inst_dir.rmdir()
        except OSError:
            pass
    # legacy flat layout
    for p in (dir_ / f"ainow.{pid}", dir_ / f"ainow.{pid}.json"):
        try:
            p.unlink()
        except OSError:
            pass


def _comm_reap_stale(dir_: pathlib.Path) -> None:
    """Drop dead sockets from the directory before listing or sending.

    Handles both the current per-instance-directory layout
    (<root>/ainow.<pid>/instance) and the legacy flat socket layout.
    """
    for entry in list(dir_.glob("ainow.*")):
        if entry.is_dir():
            try:
                pid = int(entry.name.split(".", 1)[1])
            except (ValueError, IndexError):
                continue
            if pid == os.getpid():
                continue
            sock = entry / "instance"
            status = _comm_probe(sock)
        elif entry.is_file() and entry.suffix != ".json":
            # legacy flat socket
            try:
                pid = int(entry.name.split(".", 1)[1])
            except (ValueError, IndexError):
                continue
            if pid == os.getpid():
                continue
            status = _comm_probe(entry)
        else:
            continue
        if status == "alive":
            continue
        if status == "refused" and not _comm_pid_alive(pid):
            _comm_unlink_pair(dir_, pid)


def _comm_scan(dir_: pathlib.Path, own_pid: int, singleton: bool) -> tuple[str, dict | None]:
    """Scan directory before binding.

    In local mode (singleton=True) only one instance may hold the comm
    directory; a live peer means exit(3) and a refused socket with a live
    owner means exit(4).

    In home mode (singleton=False) multiple instances coexist; we only
    remove stale dead entries and our own reused-pid socket.

    Handles both the current per-instance-directory layout and the legacy
    flat socket layout.

    Returns (action, info):
      'proceed'  = caller may bind
      'alive'    = a live peer exists -> exit(3)
      'mismatch' = socket refused but pid alive -> exit(4)
    """
    for entry in list(dir_.glob("ainow.*")):
        if entry.is_dir():
            try:
                pid = int(entry.name.split(".", 1)[1])
            except (ValueError, IndexError):
                continue
            sock = entry / "instance"
            status = _comm_probe(sock)
        elif entry.is_file():
            if entry.suffix == ".json":
                continue  # legacy registry; handled with legacy socket
            try:
                pid = int(entry.name.split(".", 1)[1])
            except (ValueError, IndexError):
                continue
            sock = entry
            status = _comm_probe(sock)
        else:
            continue
        if status == "alive":
            if singleton:
                info = _comm_read_registry(dir_, pid)
                return "alive", info
            continue
        if status != "refused":
            continue
        if pid == own_pid:
            # A socket from a previous process that reused our pid; we own it now.
            _comm_unlink_pair(dir_, pid)
            continue
        if _comm_pid_alive(pid):
            if singleton:
                info = _comm_read_registry(dir_, pid)
                return "mismatch", info
            continue
        _comm_unlink_pair(dir_, pid)
    return "proceed", None


def _comm_wake_idle_prompt() -> None:
    """If the REPL is idle at the prompt, wake it to process the message now.

    Safe to call from the comm listener thread: uses prompt_toolkit's own
    thread-safe exit path (loop.call_soon_threadsafe), the same pattern the
    library uses internally for cross-thread interaction with a running
    Application. No-ops if there's no live REPL, the prompt isn't currently
    active, the user has a half-typed line pending (never discard input),
    or we've auto-woken too many times in a row without a human turn (loop
    guard against two auto-reactive instances ping-ponging each other).
    """
    session = _REPL_SESSION
    if session is None:
        return
    app = session.app
    if not app.is_running or app.loop is None:
        return
    if session.default_buffer.text:
        return  # don't clobber a half-typed line
    if _COMM_AUTORUN_STREAK >= _COMM_AUTORUN_MAX:
        # By-design skip, but silent otherwise: the "[msg from X]" line above
        # already printed, so without this the human sees a message announced
        # and then nothing happen, with no indication it's queued rather than
        # lost or broken. It IS queued -- any keystroke (even bare Enter)
        # drains it on the next turn and resets the streak.
        print(f"{C.d}  [auto-wake paused after {_COMM_AUTORUN_STREAK} consecutive "
              f"replies -- press Enter to process the queued message]{C.r}", flush=True)
        return
    try:
        app.loop.call_soon_threadsafe(app.exit, _COMM_WAKE)
    except RuntimeError:
        pass


def _comm_listener() -> None:
    """Daemon thread accepting newline-delimited JSON peer messages."""
    global _COMM_LISTENER
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(_COMM_SOCK_PATH))
    except OSError:
        # Bind path taken despite the scan: treat as a true collision and exit
        # the way the design specifies, with a clear message, not a traceback.
        print(f"{C.re}another ainow holds this comm socket (pid {os.getpid()}){C.r}",
              file=sys.stderr, flush=True)
        os._exit(3)
    sock.listen(4)
    _COMM_LISTENER = sock
    while not _COMM_SHUTDOWN.is_set():
        try:
            sock.settimeout(0.2)
            conn, _ = sock.accept()
        except socket.timeout:
            continue
        except OSError:
            # Socket closed out from under us (e.g. /comm relisten retiring
            # this thread for a fresh one) -- exit quietly, not a crash.
            break
        try:
            with conn.makefile("r") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    from_ = str(payload.get("from", "unknown"))
                    kind = payload.get("kind", "message")
                    text = payload.get("text", "")
                    if kind not in ("message", "task"):
                        continue
                    if not isinstance(text, str):
                        continue
                    print(f"{C.d}  [msg from {from_}]{C.r}", flush=True)
                    _log(f"comm received from {from_} kind={kind}")
                    _queue_nudge(text, tag=f"comm from {from_} ({kind})", notify=False)
                    _comm_wake_idle_prompt()
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass


def _comm_cleanup() -> None:
    """Close listener and remove this instance's socket, registry, and dir."""
    global _COMM_CLEANED
    if _COMM_CLEANED:
        return
    _COMM_CLEANED = True
    _COMM_SHUTDOWN.set()
    if _COMM_LISTENER is not None:
        try:
            _COMM_LISTENER.close()
        except OSError:
            pass
    if _COMM_SOCK_PATH is not None:
        try:
            _COMM_SOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass
    if _COMM_REGISTRY_PATH is not None:
        try:
            _COMM_REGISTRY_PATH.unlink(missing_ok=True)
        except OSError:
            pass
    if _COMM_SOCK_PATH is not None:
        try:
            _COMM_SOCK_PATH.parent.rmdir()
        except OSError:
            pass


def _comm_signal_handler(signum: int, _frame) -> None:
    _comm_cleanup()
    sys.exit(128 + signum)


def _comm_startup(mode: str, label: str | None, model: str,
                  workspace: str | None = None) -> None:
    """Bind the comm socket for this instance, or exit if a peer is alive."""
    global _COMM_MODE, _COMM_DIR, _COMM_SOCK_PATH, _COMM_REGISTRY_PATH, _COMM_LABEL
    if mode == "none":
        return
    if mode == "home":
        dir_ = COMM_HOME_DIR
    elif mode == "local":
        dir_ = COMM_LOCAL_DIR
    else:
        # Unknown mode: disable rather than crash.
        print(f"{C.ye}  warning: unknown --comm mode {mode!r}; disabling comm{C.r}")
        return

    _COMM_MODE = mode
    _COMM_DIR = dir_
    _COMM_LABEL = label or None

    dir_.mkdir(parents=True, exist_ok=True)
    if mode == "home":
        os.chmod(dir_, 0o700)
    else:
        # Sticky-bit world-writable directory: any user may create an instance
        # dir, but no user may remove another user's instance dir.
        os.chmod(dir_, 0o1777)

    pid = os.getpid()
    inst_dir = _comm_instance_dir(dir_, pid)
    inst_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(inst_dir, 0o700)
    _COMM_SOCK_PATH = _comm_sock_path(dir_, pid)
    _COMM_REGISTRY_PATH = _comm_reg_path(dir_, pid)

    # Peers COEXIST in both scopes — that is the whole point of the comm wire.
    # The startup guard is only for (a) stale entries (cleaned as a courtesy) and
    # (b) a leftover socket bearing OUR OWN pid (reclaimed); a live peer is never
    # a reason to refuse to start. A true same-pid collision can't happen on one
    # box, but if the bind path is somehow taken we exit(3) at bind time below.
    singleton = False
    action, info = _comm_scan(dir_, pid, singleton)
    if action == "alive":
        lbl = info.get("label", pid) if info else pid
        print(f"{C.re}another ainow is alive (pid {info.get('pid', pid)}, label {lbl}){C.r}")
        sys.exit(3)
    if action == "mismatch":
        lbl = info.get("label", pid) if info else pid
        peer = info.get("pid", "?") if info else "?"
        print(f"{C.re}stale socket for live process pid {peer}, label {lbl} — refusing to start{C.r}")
        sys.exit(4)

    registry = {
        "pid": pid,
        "label": label or str(pid),
        "model": model,
        "cwd": str(pathlib.Path.cwd().resolve()),
        "started": time.time(),
        "workspace": workspace,
    }
    _COMM_REGISTRY_PATH.write_text(json.dumps(registry))

    t = threading.Thread(target=_comm_listener, daemon=True)
    t.start()
    _COMM_THREAD = t

    atexit.register(_comm_cleanup)
    signal.signal(signal.SIGTERM, _comm_signal_handler)
    # Keep the default SIGINT behaviour in place; the handler just guarantees
    # socket cleanup if a signal arrives outside the REPL's own handlers.
    signal.signal(signal.SIGINT, _comm_signal_handler)


def _comm_relisten(mode: str | None, model: str, workspace: str | None) -> str:
    """Tear down and re-establish this instance's comm registration in place.

    Recovers a live session from its comm directory/socket/registry having
    been removed out from under it by something outside ainow (observed:
    /tmp/ainow vanished mid-session, cause unidentified after investigation
    -- this is the mitigation regardless of root cause), and doubles as a
    way to switch --comm mode without restarting the process. mode=None
    keeps the current mode/dir; pass "local" or "home" to switch.
    """
    global _COMM_SHUTDOWN, _COMM_LISTENER, _COMM_THREAD, _COMM_CLEANED
    old_label = _COMM_LABEL
    old_mode = _COMM_MODE or "local"

    # Stop the old listener thread cleanly before starting a new one -- it
    # may already be running against a socket path that no longer exists.
    _COMM_SHUTDOWN.set()
    if _COMM_LISTENER is not None:
        try:
            _COMM_LISTENER.close()
        except OSError:
            pass
    if _COMM_THREAD is not None and _COMM_THREAD.is_alive():
        _COMM_THREAD.join(timeout=2.0)

    # Best-effort teardown of old state. Every step tolerates the path
    # already being gone -- that's the whole scenario this exists for.
    if _COMM_SOCK_PATH is not None:
        try:
            _COMM_SOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass
    if _COMM_REGISTRY_PATH is not None:
        try:
            _COMM_REGISTRY_PATH.unlink(missing_ok=True)
        except OSError:
            pass
    if _COMM_SOCK_PATH is not None:
        try:
            _COMM_SOCK_PATH.parent.rmdir()
        except OSError:
            pass

    _COMM_SHUTDOWN = threading.Event()
    _COMM_CLEANED = False  # let atexit's cleanup run again for the new registration

    _comm_startup(mode or old_mode, old_label, model, workspace)
    return f"relistening: mode={_COMM_MODE}  dir={_COMM_DIR}  pid={os.getpid()}  label={_COMM_LABEL or os.getpid()}"

    _log(f"comm {mode} socket {_COMM_SOCK_PATH} label={registry['label']}")


# -- workspaces ---------------------------------------------------------------
# A workspace is a named project context: a root directory, an optional startup
# bootstrap command, optional journal defaults, and metadata.  Workspaces are
# stored as pure data in ~/ainow/workspaces/<name>.json; they are NEVER executed
# directly.  Loading a workspace only affects CONTEXT and DEFAULTS — we never
# chdir and never touch repos, because magic relocation breaks tool paths that
# the model emits relative to ainow's cwd.
WORKSPACE_DIR = HOME / "ainow" / "workspaces"
_WORKSPACE_NAME: str | None = None
_WORKSPACE_BOOTSTRAP_OUTPUT: str = ""
_WORKSPACE_NUDGE_SHOWN = False
_WORKSPACE_SESSION_START = 0.0
_WORKSPACE_TOOL_CALLS = 0
_FIRST_USER_LINE: str | None = None


# Env defaults for the lazy "this session is getting long" workspace nudge.
# Time and tool-call thresholds are intentionally soft/ignorable.
_WORKSPACE_NUDGE_MINS = float(os.environ.get("AINOW_WORKSPACE_NUDGE_MINS", "30"))
_WORKSPACE_NUDGE_CALLS = int(os.environ.get("AINOW_WORKSPACE_NUDGE_CALLS", "25"))


def _workspace_path(name: str) -> pathlib.Path:
    return WORKSPACE_DIR / f"{name}.json"


def _workspace_list() -> list[str]:
    """Return sorted workspace names from the index.

    Malformed entries are skipped with a warning so a broken file does not
    break /workspace list or the unknown-name suggestion UX.
    """
    if not WORKSPACE_DIR.is_dir():
        return []
    names = []
    for p in sorted(WORKSPACE_DIR.glob("*.json")):
        if not p.stem:
            continue
        data = _workspace_load_json(p.stem)
        if data is None:
            continue
        names.append(p.stem)
    return names


def _workspace_load_json(name: str) -> dict | None:
    """Load a workspace entry, validating only the shape enough to skip junk."""
    p = _workspace_path(name)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"{C.ye}  warning: workspace {name!r} is malformed ({e}); skipping{C.r}")
        return None
    if not isinstance(data, dict):
        print(f"{C.ye}  warning: workspace {name!r} is not an object; skipping{C.r}")
        return None
    return data


def _workspace_apply_defaults(name: str, data: dict) -> None:
    """Apply workspace journal defaults when explicit env is not set."""
    global _JOURNAL_SSH, _JOURNAL_FILE
    if not os.environ.get("AINOW_JOURNAL_SSH") and data.get("journal_ssh"):
        _JOURNAL_SSH = str(data["journal_ssh"])
    if not os.environ.get("AINOW_JOURNAL_FILE") and data.get("journal_file"):
        _JOURNAL_FILE = str(data["journal_file"])


def _workspace_run_bootstrap(name: str, data: dict) -> str:
    """Run the workspace bootstrap command (with cwd=root) and return stdout."""
    bootstrap = data.get("bootstrap")
    if not bootstrap:
        return ""
    root = data.get("root")
    cwd = pathlib.Path(root).expanduser().resolve() if root else pathlib.Path.cwd()
    try:
        proc = subprocess.run(
            bootstrap, shell=True, cwd=str(cwd),
            capture_output=True, text=True, timeout=60, errors="replace")
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            out = f"[bootstrap exited {proc.returncode}]\n{out}"
        return out.strip()
    except subprocess.TimeoutExpired:
        return "[bootstrap timed out after 60s]"
    except Exception as e:
        return f"[bootstrap failed: {type(e).__name__}: {e}]"


def _workspace_load(name: str) -> dict:
    """Load a workspace by name or exit 2 listing available names."""
    global _WORKSPACE_NAME, _WORKSPACE_BOOTSTRAP_OUTPUT
    data = _workspace_load_json(name)
    if data is None:
        available = _workspace_list()
        if available:
            print(f"ainow: unknown workspace {name!r}; available: {', '.join(available)}")
        else:
            print(f"ainow: unknown workspace {name!r}; no workspaces saved")
        sys.exit(2)
    _WORKSPACE_NAME = name
    _workspace_apply_defaults(name, data)
    _WORKSPACE_BOOTSTRAP_OUTPUT = _workspace_run_bootstrap(name, data)
    return data


def _workspace_save(name: str, model: str, transcript: pathlib.Path | None = None,
                     notes: str | None = None, bootstrap: str | None = None) -> str:
    """Create or overwrite a workspace index entry from the current session.

    notes/bootstrap are only overwritten when explicitly passed (not None) so a
    plain re-save (e.g. from the periodic nudge) doesn't clobber a previously
    configured bootstrap command.
    """
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    p = _workspace_path(name)
    existing = _workspace_load_json(name) or {}
    data = {
        "name": name,
        "root": str(pathlib.Path.cwd().resolve()),
        "bootstrap": existing.get("bootstrap", "") if bootstrap is None else bootstrap,
        "journal_ssh": existing.get("journal_ssh", ""),
        "journal_file": existing.get("journal_file", ""),
        "label": existing.get("label", name),
        "notes": existing.get("notes", "") if notes is None else notes,
        "created": existing.get("created") or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
    }
    if existing.get("transcript"):
        data["transcript"] = existing["transcript"]
    if transcript:
        data["transcript"] = str(transcript)
        # Maintain a stable symlink in the workspace index dir for discoverability.
        link = WORKSPACE_DIR / f"{name}.transcript.md"
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(transcript)
        except OSError:
            pass
    p.write_text(json.dumps(data, indent=2) + "\n")
    return f"saved workspace {name!r}"


def _workspace_show(name: str) -> str:
    data = _workspace_load_json(name)
    if data is None:
        return f"error: no workspace {name!r}"
    return json.dumps(data, indent=2)


def _workspace_forget(name: str) -> str:
    """Remove the workspace registry entry (the JSON file), never project files."""
    p = _workspace_path(name)
    if not p.exists():
        return f"error: no workspace {name!r}"
    try:
        p.unlink()
        link = WORKSPACE_DIR / f"{name}.transcript.md"
        if link.is_symlink():
            link.unlink()
    except OSError as e:
        return f"error: could not remove workspace entry: {e}"
    return f"forgot workspace {name!r}"


def _workspace_slug(first_line: str | None) -> str:
    """Generate a friendly workspace slug from the first user line or cwd."""
    if first_line:
        text = first_line.strip().lower()
        text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
        if text:
            return text[:40]
    return pathlib.Path.cwd().name or "workspace"


def _workspace_maybe_nudge(first_line: str | None) -> None:
    """Once per session, nudge the user to save a long workspace-less session."""
    global _WORKSPACE_NUDGE_SHOWN
    if _WORKSPACE_NUDGE_SHOWN or _WORKSPACE_NAME is not None:
        return
    mins = (time.time() - _WORKSPACE_SESSION_START) / 60.0
    calls = _WORKSPACE_TOOL_CALLS
    if mins >= _WORKSPACE_NUDGE_MINS or calls >= _WORKSPACE_NUDGE_CALLS:
        _WORKSPACE_NUDGE_SHOWN = True
        slug = _workspace_slug(first_line)
        print(f"{C.d}  this session is getting long — /workspace save <name>? maybe '{slug}'{C.r}", flush=True)


def t_workspace(action: str, name: str = "", text: str = "",
                model: str = "", transcript: str = "", bootstrap: str = "") -> str:
    """Model-facing workspace tool (save/show/list/forget).

    Registered with needs_approval=True because model-initiated workspace
    mutations affect project context.
    """
    action = action.lower()
    if action == "list":
        names = _workspace_list()
        return "workspaces: " + ", ".join(names) if names else "no workspaces"
    if action == "show":
        if not name:
            return "error: name required"
        return _workspace_show(name)
    if action == "forget":
        if not name:
            return "error: name required"
        return _workspace_forget(name)
    if action == "save":
        if not name:
            return "error: name required"
        tx = pathlib.Path(transcript) if transcript else None
        return _workspace_save(name, model or "unknown", tx,
                                notes=text or None, bootstrap=bootstrap or None)
    return f"error: unknown action {action!r} (try list/show/save/forget)"


# -- session transcript ------------------------------------------------------
# A per-session, append-as-it-happens plain-text record of the conversation
# (user turns, assistant text, tool calls + results). Survives a crash, unlike the
# in-memory message list. One file per session: logs/transcript-<ts>-<pid>.md
_TRANSCRIPT = None


def _transcript_start(provider: str, model: str) -> None:
    global _TRANSCRIPT, _WORKSPACE_SESSION_START
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    _TRANSCRIPT = LOG_DIR / f"transcript-{ts}-{os.getpid()}.md"
    _WORKSPACE_SESSION_START = time.time()
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
_NUDGE_QUEUE: list[tuple[str, str]] = []


def _queue_nudge(line: str, tag: str = "user nudge mid-task",
                 notify: bool = True) -> None:
    text = line.rstrip("\n").rstrip("\r").strip()
    if not text:
        return
    with _NUDGE_LOCK:
        _NUDGE_QUEUE.append((tag, text))
    if notify:
        # Typing is echo-suppressed while a tool runs (see _stdin_no_echo) to
        # stop it interleaving with concurrent tool output, so echo the
        # captured text back here — otherwise there'd be no confirmation at
        # all of what was actually captured.
        print(f"{C.ye}  [nudge queued] {text}{C.r}", flush=True)


def _drain_nudges(agent) -> None:
    """Append queued nudges/comm messages as clearly-marked user messages."""
    with _NUDGE_LOCK:
        queued = _NUDGE_QUEUE[:]
        _NUDGE_QUEUE[:] = []
    for tag, text in queued:
        marked = f"[{tag}] {text}"
        agent.messages.append({"role": "user", "content": marked})
        _tx("paul", marked)


def _kill_orphaned_jobs(pre_running: set) -> None:
    """Kill any bash job started during the just-interrupted call.

    Ctrl-C only unblocks the REPL's main thread; the worker thread (and the
    subprocess it's waiting on inside t_bash) keeps running unless we
    explicitly kill it here. Only touches jobs that started during THIS call
    (not in pre_running), never pre-existing background jobs the user
    intentionally left running.
    """
    with _JOBS_LOCK:
        new_running = [jid for jid, j in _BG_REGISTRY.items()
                       if j.get("alive") and jid not in pre_running]
    for jid in new_running:
        job = _BG_REGISTRY.get(jid, {})
        pid = job.get("pid")
        _job_kill(jid, "killed-interrupt")
        print(f"{C.ye}  killed job {jid} (pid {pid}) after Ctrl-C{C.r}", flush=True)


class _stdin_no_echo:
    """Disable local tty echo (keeping canonical line-editing) for a block.

    While a tool runs, the model's own output is being printed concurrently
    from the main thread; the kernel's live per-keystroke echo of anything
    the user types races against those prints on the same terminal, so
    typed text and tool output visibly interleave/garble mid-word. ICANON
    stays on, so the line discipline still buffers input and processes
    backspace correctly — the user just doesn't see it happen live. The
    captured text is echoed back deliberately once queued (see
    _queue_nudge) so there's still confirmation of what was sent.
    """

    def __init__(self, fd: int):
        self.fd = fd
        self.old = None

    def __enter__(self):
        try:
            self.old = termios.tcgetattr(self.fd)
            new = termios.tcgetattr(self.fd)
            new[3] &= ~termios.ECHO
            termios.tcsetattr(self.fd, termios.TCSANOW, new)
        except (termios.error, OSError):
            self.old = None
        return self

    def __exit__(self, *_):
        if self.old is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSANOW, self.old)
            except (termios.error, OSError):
                pass
        return False


class _stdin_echo_restore:
    """Temporarily force tty echo back on inside an active _stdin_no_echo block.

    The turn-wide echo suppression exists to stop typed input racing
    concurrent print() output (see _stdin_no_echo) — but the tool-approval
    prompt ("run it? [y/N/a=always]") is a synchronous, blocking input()
    call with nothing else printing at that moment, so there's no race to
    guard against there, only lost visibility into what's being typed.
    Restores exactly the surrounding (suppressed) state on exit, so the
    outer suppression resumes correctly afterward.
    """

    def __init__(self, fd: int):
        self.fd = fd
        self.old = None

    def __enter__(self):
        try:
            self.old = termios.tcgetattr(self.fd)
            new = termios.tcgetattr(self.fd)
            new[3] |= termios.ECHO
            termios.tcsetattr(self.fd, termios.TCSANOW, new)
        except (termios.error, OSError):
            self.old = None
        return self

    def __exit__(self, *_):
        if self.old is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSANOW, self.old)
            except (termios.error, OSError):
                pass
        return False


def _run_with_nudges(fn, interactive: bool, *args, **kwargs):
    """Run a callable, collecting stdin lines as nudges if interactive+tty."""
    # Non-interactive (one-shot, piped stdin): keep the old blocking behaviour.
    if not interactive or not sys.stdin.isatty():
        return fn(*args, **kwargs)

    with _JOBS_LOCK:
        pre_running = {jid for jid, j in _BG_REGISTRY.items() if j.get("alive")}

    result = [None]
    error = [None]

    def worker():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:  # capture so main thread can re-raise
            error[0] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # Echo is suppressed for the whole turn (see run()), not just this one
    # tool call, so typing between tool calls doesn't race concurrent prints.
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
        # will raise Interrupted after this function returns. If it's still
        # alive after the grace period, it's blocked on a subprocess we
        # started — kill it rather than leaving it orphaned.
        t.join(timeout=2.0)
        if t.is_alive():
            _kill_orphaned_jobs(pre_running)

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


def t_bash(command: str, background: bool = False, max_s: int | None = None,
           idle_s: int | None = None, timeout: int | None = None) -> str:
    # Every bash execution becomes a managed job in logs/bg/<id>/.  Foreground
    # calls wait on the job; if they outlive AINOW_FG_MAX_S the tool returns a
    # job id so the model can poll with bash_jobs.  Background calls return
    # immediately.  max_s and idle_s apply to all jobs.
    if max_s is None and timeout is not None:
        max_s = timeout
    if max_s is None:
        max_s = _DEFAULT_MAX_S
    if idle_s is None:
        idle_s = _DEFAULT_IDLE_S
    jid, job = _job_run(command, max_s=max_s, idle_s=idle_s)
    if background:
        return f"background job {jid} started (pid {job['pid']}) → {job['dir']}"
    fg_max = _default_fg_max_s()
    return _job_wait_foreground(jid, fg_max)


# -- job runner ---------------------------------------------------------------
# All shell execution (foreground and background) goes through the same runner.
# Each job gets its own directory under logs/bg/<id>/ with cmd, stdout.log,
# stderr.log, rc, and state.  Two timers protect against runaways:
#   max_s   -> process-group SIGKILL after this many seconds (0 = disabled)
#   idle_s  -> SIGKILL if no new stdout/stderr bytes arrive for this long
#              (the hung-ssh killer; 0 = disabled).
_DEFAULT_MAX_S = 600
_DEFAULT_IDLE_S = 180
_DEFAULT_FG_MAX_S = 120
_JOBS_LOCK = threading.RLock()


def _default_fg_max_s() -> float:
    try:
        return float(os.environ.get("AINOW_FG_MAX_S", _DEFAULT_FG_MAX_S))
    except (ValueError, TypeError):
        return _DEFAULT_FG_MAX_S


def _read_job_log(jid: str, lines: int | None = None) -> str:
    job = _BG_REGISTRY.get(jid)
    if not job:
        return ""
    # New jobs store a directory; old registry entries stored a single log file.
    jdir = pathlib.Path(job.get("dir") or os.path.dirname(job.get("log") or ""))
    parts = []
    for name in ("stdout.log", "stderr.log"):
        try:
            data = (jdir / name).read_text(errors="replace")
            if data:
                parts.append(data)
        except (OSError, FileNotFoundError):
            pass
    if not parts and job.get("log"):
        try:
            data = pathlib.Path(job["log"]).read_text(errors="replace")
            if data:
                parts.append(data)
        except (OSError, FileNotFoundError):
            pass
    text = "\n".join(parts)
    if lines is not None:
        text = "\n".join(text.splitlines()[-lines:])
    return text


def _job_mark(jid: str, state: str, rc: int | None = None) -> None:
    """Write state/rc files and update the in-memory registry."""
    job = _BG_REGISTRY.get(jid)
    if not job:
        return
    jdir = pathlib.Path(job["dir"])
    try:
        (jdir / "state").write_text(state)
    except (OSError, FileNotFoundError):
        pass
    if rc is not None:
        try:
            (jdir / "rc").write_text(str(rc))
        except (OSError, FileNotFoundError):
            pass
    with _JOBS_LOCK:
        job["state"] = state
        if rc is not None:
            job["rc"] = rc
        _bg_save()


def _job_finalize(jid: str, rc: int | None) -> None:
    """Move a job to its final state once the process has exited."""
    job = _BG_REGISTRY.get(jid)
    if not job:
        return
    if not job.get("alive"):
        # Already finalized via a kill; just record rc if we have it.
        if rc is not None:
            _job_mark(jid, job.get("state", "done"), rc)
        return
    reason = job.get("killed_reason")
    state = reason if reason else "done"
    job["alive"] = False
    job["finished"] = time.time()
    _job_mark(jid, state, rc)


def _job_kill(jid: str, reason: str) -> None:
    job = _BG_REGISTRY.get(jid)
    if not job or not job.get("alive"):
        return
    pid = job["pid"]
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    job["alive"] = False
    job["finished"] = time.time()
    job["killed_reason"] = reason
    _job_mark(jid, reason)


def _job_reader(pipe, fp, job: dict) -> None:
    try:
        for line in iter(pipe.readline, ""):
            fp.write(line)
            fp.flush()
            job["_last_byte"] = time.time()
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass
        try:
            fp.close()
        except Exception:
            pass


def _job_watch(jid: str, proc: subprocess.Popen, readers: list[threading.Thread],
               max_s: int, idle_s: int) -> None:
    job = _BG_REGISTRY[jid]
    started = job["started"]
    deadline = started + max_s if max_s else None
    while proc.poll() is None:
        now = time.time()
        if deadline and now > deadline:
            _job_kill(jid, "killed-max")
            break
        if idle_s:
            last_byte = job.get("_last_byte", started)
            if now - last_byte > idle_s:
                _job_kill(jid, "killed-idle")
                break
        time.sleep(0.2)
    for r in readers:
        r.join(timeout=2)
    try:
        rc = proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        rc = None
    _job_finalize(jid, rc)


def _job_run(command: str, max_s: int = _DEFAULT_MAX_S,
             idle_s: int = _DEFAULT_IDLE_S) -> tuple[str, dict]:
    _bg_init()
    jid = _bg_next_id()
    jdir = _BG_DIR / jid
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "cmd").write_text(command)
    (jdir / "state").write_text("running")
    (jdir / "rc").write_text("")
    out_fp = open(jdir / "stdout.log", "w")
    err_fp = open(jdir / "stderr.log", "w")
    try:
        p = subprocess.Popen(
            command, shell=True, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace", start_new_session=True, close_fds=True,
        )
    except Exception as e:
        out_fp.close()
        err_fp.close()
        state = "error"
        (jdir / "state").write_text(state)
        (jdir / "rc").write_text("-1")
        now = time.time()
        job = {
            "command": command, "pid": 0, "started": now,
            "alive": False, "finished": now, "dir": str(jdir),
            "state": state, "rc": -1,
            "max_s": max_s, "idle_s": idle_s,
        }
        with _JOBS_LOCK:
            _BG_REGISTRY[jid] = job
            _bg_save()
        return jid, job

    now = time.time()
    job = {
        "command": command, "pid": p.pid, "started": now, "alive": True,
        "dir": str(jdir), "state": "running",
        "max_s": max_s, "idle_s": idle_s,
    }
    with _JOBS_LOCK:
        _BG_REGISTRY[jid] = job
        _bg_save()

    t_out = threading.Thread(target=_job_reader, args=(p.stdout, out_fp, job), daemon=True)
    t_err = threading.Thread(target=_job_reader, args=(p.stderr, err_fp, job), daemon=True)
    t_out.start()
    t_err.start()

    watcher = threading.Thread(
        target=_job_watch,
        args=(jid, p, [t_out, t_err], max_s, idle_s),
        daemon=True,
    )
    watcher.start()
    return jid, job


def _job_wait_foreground(jid: str, fg_max: float) -> str:
    job = _BG_REGISTRY[jid]
    start = time.time()
    while True:
        state = job.get("state", "running")
        if not job.get("alive", False) and state != "running":
            return _job_collect(jid)
        elapsed = time.time() - start
        if fg_max > 0 and elapsed > fg_max:
            tail = _read_job_log(jid, lines=20).strip()
            return (
                f"job {jid} is still running after {elapsed:.1f}s and has been "
                f"moved to the background — it is NOT paused, it keeps running.\n"
                f"Don't poll it immediately; there is usually nothing new to see "
                f"yet. Continue with other work or tell the user it's running, "
                f"then check back with bash_jobs action=log job_id={jid} after "
                f"a real delay (tens of seconds), not a follow-up call right away.\n"
                f"--- recent log ---\n"
                f"{tail if tail else '(no output yet)'}"
            )
        time.sleep(0.2)


def _job_collect(jid: str) -> str:
    job = _BG_REGISTRY.get(jid)
    if not job:
        return f"error: no job {jid}"
    jdir = pathlib.Path(job.get("dir") or os.path.dirname(job.get("log") or ""))
    rc_text = ""
    try:
        rc_text = (jdir / "rc").read_text().strip()
    except (OSError, FileNotFoundError):
        pass
    try:
        rc = int(rc_text)
    except (ValueError, TypeError):
        rc = None
    out = _read_job_log(jid)
    if rc is not None and rc != 0:
        out += f"\n[exit {rc}]"
    state = job.get("state", "done")
    if state != "done":
        out += f"\n[state: {state}]"
    out = out.strip()
    return _clip(out) or (f"(no output) [exit {rc}]" if rc is not None else "(no output)")


# -- background job registry --------------------------------------------------
# Backward-compatible persistent registry under logs/bg/jobs.json.  New jobs
# store a directory instead of a single log file, but the same keys (command,
# pid, started, alive) remain so existing consumers keep working.
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
    """Mark dead jobs and finalize any that the watcher hasn't picked up yet."""
    for jid, job in list(_BG_REGISTRY.items()):
        if not job.get("alive"):
            continue
        pid = job["pid"]
        try:
            pid2, rc = os.waitpid(pid, os.WNOHANG)
            if pid2 != 0:
                _job_finalize(jid, rc if rc != 0 else 0)
                continue
        except (ChildProcessError, OSError):
            pass
        if not _bg_is_alive(pid):
            _job_finalize(jid, None)


def _bg_next_id() -> str:
    ids = [int(k) for k in _BG_REGISTRY if k.isdigit()]
    return str(max(ids, default=0) + 1)


def _bg_list() -> str:
    _bg_init()
    _bg_reap()
    if not _BG_REGISTRY:
        return "no background jobs"
    rows = []
    for jid in sorted(_BG_REGISTRY, key=lambda k: int(k)):
        j = _BG_REGISTRY[jid]
        if j.get("alive"):
            elapsed = time.time() - j["started"]
            status = j.get("state", "running")
        else:
            elapsed = (j.get("finished") or time.time()) - j["started"]
            status = j.get("state", "finished")
        rows.append(f"{jid}: [{status}] {elapsed:.1f}s  {j['command']}")
    return "\n".join(rows)


def _bg_log_tail(job_id: str, lines: int = 50) -> str:
    _bg_init()
    _bg_reap()
    job = _BG_REGISTRY.get(job_id)
    if not job:
        return f"error: no job {job_id}"
    return _clip(_read_job_log(job_id, lines=lines).strip()) or "(log empty)"


def _bg_kill(job_id: str, sig: int = signal.SIGKILL) -> str:
    _bg_init()
    job = _BG_REGISTRY.get(job_id)
    if not job:
        return f"error: no job {job_id}"
    if not job.get("alive"):
        return f"job {job_id} is not running"
    pid = job["pid"]
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError):
        pass
    _job_kill(job_id, "killed-user")
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


# -- todo list ----------------------------------------------------------------
# Live in-flight scratch list.  Durable cross-session items that matter after a
# reboot belong in the user's trading journal OPEN THREADS; /todo is for tasks
# active in the current session.
TODO_FILE = CFG_DIR / "todo.md"
_TODO_HUD = True
_TODO_RE = re.compile(r"^- \[( |x)\] (.*)$")


def _todo_load() -> list[tuple[bool, str]]:
    if not TODO_FILE.exists():
        return []
    items: list[tuple[bool, str]] = []
    for line in TODO_FILE.read_text(errors="replace").splitlines():
        m = _TODO_RE.match(line.strip())
        if m:
            items.append((m.group(1) == "x", m.group(2)))
    return items


def _todo_save(items: list[tuple[bool, str]]) -> None:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"- {'[x]' if done else '[ ]'} {text}" for done, text in items)
    TODO_FILE.write_text(body + "\n")


def _todo_list() -> str:
    items = _todo_load()
    if not items:
        return "no todos"
    lines = []
    for i, (done, text) in enumerate(items, 1):
        mark = "x" if done else " "
        lines.append(f"{i}: [{mark}] {text}")
    return "\n".join(lines)


def _todo_add(text: str) -> str:
    items = _todo_load()
    items.append((False, text.strip()))
    _todo_save(items)
    return f"added todo {len(items)}"


def _todo_done(n: int) -> str:
    items = _todo_load()
    if not 1 <= n <= len(items):
        return f"error: no todo {n}"
    items[n - 1] = (True, items[n - 1][1])
    _todo_save(items)
    return f"marked todo {n} done"


def _todo_rm(n: int) -> str:
    items = _todo_load()
    if not 1 <= n <= len(items):
        return f"error: no todo {n}"
    removed = items.pop(n - 1)
    _todo_save(items)
    return f"removed todo {n}: {removed[1]}"


def _todo_edit(n: int, text: str) -> str:
    items = _todo_load()
    if not 1 <= n <= len(items):
        return f"error: no todo {n}"
    items[n - 1] = (items[n - 1][0], text.strip())
    _todo_save(items)
    return f"edited todo {n}"


def _todo_hud_line() -> str | None:
    if not _TODO_HUD:
        return None
    items = _todo_load()
    open_items = [(i, text) for i, (done, text) in enumerate(items, 1) if not done]
    if not open_items:
        return None
    n, text = open_items[0]
    count = len(open_items)
    return f"[todo] {n}: {text} ({count} open)"


def _todo_hud_toggle(state: str) -> None:
    global _TODO_HUD
    _TODO_HUD = state.lower() in ("on", "1", "true", "yes")


def t_todo(action: str = "list", n: int = 0, text: str = "") -> str:
    action = action.lower()
    if action == "list":
        return _todo_list()
    if action == "add":
        if not text:
            return "error: text required"
        return _todo_add(text)
    if action == "done":
        if n <= 0:
            return "error: n required"
        return _todo_done(n)
    if action == "rm":
        if n <= 0:
            return "error: n required"
        return _todo_rm(n)
    if action == "edit":
        if n <= 0 or not text:
            return "error: n and text required"
        return _todo_edit(n, text)
    return f"error: unknown action {action!r} (try list/add/done/rm)"


def t_comm_list() -> str:
    """List live ainow instances in the current comm directory."""
    if _COMM_MODE == "none" or _COMM_DIR is None:
        return "comm disabled"
    _comm_reap_stale(_COMM_DIR)
    rows = []
    for reg in sorted(_COMM_DIR.glob("ainow.*/registry.json")):
        try:
            info = json.loads(reg.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows.append({
            "pid": info.get("pid", "?"),
            "label": info.get("label", info.get("pid", "?")),
            "model": info.get("model", "?"),
            "cwd": info.get("cwd", "?"),
            "workspace": info.get("workspace") or "",
        })
    if not rows:
        return "no live ainow instances"
    lines = ["pid  label  model  workspace  cwd"]
    lines += [f"{r['pid']}\t{r['label']}\t{r['model']}\t{r['workspace']}\t{r['cwd']}" for r in rows]
    return "\n".join(lines)


def t_comm_send(target: str, text: str, kind: str = "message") -> str:
    """Send a newline-delimited JSON message/task to another ainow instance.

    Model-initiated sends are gated by Agent._approve (needs_approval=True).
    User infrastructure may write to the socket directly without approval.
    """
    if _COMM_MODE == "none" or _COMM_DIR is None:
        return "comm disabled"
    if kind not in ("message", "task"):
        return "error: kind must be 'message' or 'task'"
    _comm_reap_stale(_COMM_DIR)
    target = str(target)
    matches = []
    for reg in _COMM_DIR.glob("ainow.*/registry.json"):
        try:
            info = json.loads(reg.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        pid = str(info.get("pid", ""))
        label = str(info.get("label", ""))
        if target == pid or target == label:
            matches.append(info)
    if not matches:
        return f"error: no live instance matching {target!r}"
    if len(matches) > 1:
        return f"error: ambiguous target {target!r}"
    info = matches[0]
    sock = _comm_sock_path(_COMM_DIR, info["pid"])
    payload = json.dumps({
        "from": _COMM_LABEL or str(os.getpid()),
        "kind": kind,
        "text": text,
    })
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(str(sock))
        s.sendall((payload + "\n").encode())
        s.close()
    except OSError as e:
        return f"error: could not send to pid {info['pid']}: {e}"
    _log(f"comm sent {kind} to pid {info['pid']} label={info.get('label', '')}")
    return f"sent {kind} to {info.get('label', info['pid'])} (pid {info['pid']})"


TOOLS = {
    # name: (callable, arg-hint-dict, requires-approval, is-plugin)
    "read_file":  (t_read_file,  {"path": "str"}, False, False),
    "list_dir":   (t_list_dir,   {"path": "str"}, False, False),
    "write_file": (t_write_file, {"path": "str"}, True, False),
    "edit_file":  (t_edit_file,  {"path": "str"}, True, False),
    "bash":       (t_bash,       {"command": "str", "background": "bool", "max_s": "int", "idle_s": "int"}, True, False),
    "bash_jobs":  (t_bash_jobs,  {"action": "str", "job_id": "str", "lines": "int", "signal_name": "str"}, False, False),
    "todo":       (t_todo,       {"action": "str", "n": "int", "text": "str"}, False, False),
    "journal":    (t_journal,    {"text": "str", "section": "str"}, False, False),
    "comm_list":  (t_comm_list,  {}, True, False),
    "comm_send":  (t_comm_send,  {"target": "str", "text": "str", "kind": "str"}, True, False),
    "workspace":  (t_workspace,  {"action": "str", "name": "str", "text": "str", "model": "str", "transcript": "str"}, True, False),
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
            "background": {"type": "boolean", "description": "Run detached and return a job id"},
            "max_s": {"type": "integer", "description": "Maximum runtime in seconds (0 disables, default 600)"},
            "idle_s": {"type": "integer", "description": "Kill if idle this many seconds (0 disables, default 180)"}},
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
        "name": "todo",
        "description": "Manage the live in-flight todo list (add/done/rm/list).",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "description": "list, add, done, rm, or edit"},
            "n": {"type": "integer", "description": "Item number for done/rm/edit"},
            "text": {"type": "string", "description": "Text for add/edit"}},
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
    {"type": "function", "function": {
        "name": "comm_list",
        "description": "List live ainow instances in the current comm directory. "
                       "Requires user approval because it exposes peer metadata.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "comm_send",
        "description": "Send a message or task to another live ainow instance. "
                       "Call comm_list first to discover a valid target pid/label — "
                       "sending to a guessed or previously-known target may fail if "
                       "that instance is no longer live. Requires explicit user "
                       "approval for every model-initiated send.",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "pid or label of the peer, from comm_list"},
            "text": {"type": "string", "description": "Message text"},
            "kind": {"type": "string", "description": "message (default) or task"}},
            "required": ["target", "text"]}}},
    {"type": "function", "function": {
        "name": "workspace",
        "description": "Manage workspaces (save/show/list/forget). "
                       "Model-initiated mutations require user approval.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "description": "list, show, save, or forget"},
            "name": {"type": "string", "description": "Workspace name (required for show/save/forget)"},
            "text": {"type": "string", "description": "Notes for save"},
            "bootstrap": {"type": "string", "description": "Shell command for save. Run "
                          "(cwd=root) on every future load of this workspace and its "
                          "stdout/stderr injected into context — this is how a workspace "
                          "actually resumes context, e.g. 'tail -c 4000 <transcript path>' "
                          "or a project-specific status script."},
            "model": {"type": "string", "description": "Model string to record for save"},
            "transcript": {"type": "string", "description": "Transcript path to record for save"}},
            "required": ["action"]}}},
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
            # Use a synthetic module name so filenames with dashes/digits still load.
            mod_name = f"ainow_plugin_{fp.stem}"
            spec = importlib.util.spec_from_file_location(mod_name, fp)
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
                TOOLS[name] = (call, params, appr, True)
            TOOL_SCHEMA.extend(reg.schemas)
            if reg.tools:
                _log(f"plugin loaded: {fp.name} ({len(reg.tools)} tools)")
        except Exception as e:
            _log(f"plugin {fp} failed: {e}")
            print(f"{C.ye}  warning: plugin {fp.name} failed to load: {e}{C.r}")


SYSTEM = """You are ainow, a command-line coding assistant running on the user's \
Linux machine with real filesystem and shell access.

You have tools: read_file, list_dir, write_file, edit_file, bash, bash_jobs, \
todo, journal, workspace, plus any plugin tools loaded from ~/.config/ainow/tools.d/. \
Use them to inspect and change files directly rather than printing code for the \
user to copy. Prefer edit_file over rewriting whole files. Read a file before \
editing it. Use the journal tool to persist any decision, finding, or state change \
worth surviving a reboot the moment it happens, not just at end of session.

Be concise. The user is in a terminal — use plain text only. No markdown of any \
kind (no **bold**, `backticks`, bullet lists) unless explicitly asked. Report \
what you actually did, and if a command failed, say so with the output rather \
than assuming it worked."""


# Optional private system-prompt overlay. If ~/.config/ainow/system.local exists,
# its contents are appended to SYSTEM. This is the place for private context (hosts,
# paths, ongoing projects) that must NOT be committed to the public repo. The file
# is gitignored (see .gitignore).
# If a workspace was loaded and its bootstrap produced output, that output is
# injected here too (same pattern as the trading bootstrap).
def _system() -> str:
    try:
        local = (CFG_DIR / "system.local").read_text().strip()
    except (OSError, FileNotFoundError):
        local = ""
    out = SYSTEM
    if _WORKSPACE_BOOTSTRAP_OUTPUT:
        out += "\n\n[workspace bootstrap]\n" + _WORKSPACE_BOOTSTRAP_OUTPUT
    if local:
        out += "\n\n" + local
    return out


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
# ansi / themes — honour NO_COLOR and non-tty
# --------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _theme_code(code: str) -> str:
    return code if _USE_COLOR else ""


def _active_theme_name() -> str:
    """AINOW_THEME env wins, otherwise ~/.config/ainow/theme, otherwise default."""
    name = os.environ.get("AINOW_THEME", "")
    if not name:
        try:
            name = (CFG_DIR / "theme").read_text().strip()
        except (OSError, FileNotFoundError):
            name = ""
    return name if name else "default"


# Centralised palettes.  Keys must be present for every theme; code uses C.d/C.b/etc.
THEME: dict[str, dict[str, str]] = {
    "default": {
        "d":  _theme_code("\033[2m"),
        "b":  _theme_code("\033[1m"),
        "r":  _theme_code("\033[0m"),
        "cy": _theme_code("\033[36m"),
        "gr": _theme_code("\033[32m"),
        "ye": _theme_code("\033[33m"),
        "re": _theme_code("\033[31m"),
        "ma": _theme_code("\033[35m"),
    },
    "amber": {
        # amber keeps reasoning dim but makes the working palette warm/visible
        "d":  _theme_code("\033[2m"),
        "b":  _theme_code("\033[1;38;5;208m"),
        "r":  _theme_code("\033[0m"),
        "cy": _theme_code("\033[38;5;208m"),
        "gr": _theme_code("\033[38;5;214m"),
        "ye": _theme_code("\033[38;5;220m"),
        "re": _theme_code("\033[38;5;202m"),
        "ma": _theme_code("\033[38;5;166m"),
    },
    "mono": {
        # monochrome: only dim/bold distinguish emphasis; errors get bold
        "d":  _theme_code("\033[2m"),
        "b":  _theme_code("\033[1m"),
        "r":  _theme_code("\033[0m"),
        "cy": _theme_code("\033[37m"),
        "gr": _theme_code("\033[37m"),
        "ye": _theme_code("\033[37m"),
        "re": _theme_code("\033[1m"),
        "ma": _theme_code("\033[37m"),
    },
    "high-contrast": {
        # bright bold colours; reasoning remains intentionally dim
        "d":  _theme_code("\033[2m"),
        "b":  _theme_code("\033[1m"),
        "r":  _theme_code("\033[0m"),
        "cy": _theme_code("\033[1;96m"),
        "gr": _theme_code("\033[1;92m"),
        "ye": _theme_code("\033[1;93m"),
        "re": _theme_code("\033[1;91m"),
        "ma": _theme_code("\033[1;95m"),
    },
}

_ACTIVE_THEME = _active_theme_name()
if _ACTIVE_THEME not in THEME:
    _ACTIVE_THEME = "default"

# Namespace so the rest of the file can keep using C.gr, C.d, etc.
C = types.SimpleNamespace(**THEME[_ACTIVE_THEME])


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
            with _stdin_echo_restore(sys.stdin.fileno()):
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
                # Gemini's OpenAI-compat layer sends index=null on tool-call
                # deltas (unlike OpenAI/Kimi's integer index); fall back to
                # slot 0 rather than merging distinct parallel calls under a
                # shared None key.
                idx = tc.index if tc.index is not None else 0
                slot = calls.setdefault(idx, {"id": "", "name": "", "args": "",
                                              "type": "function", "extra_content": None})
                if tc.id:
                    slot["id"] = tc.id
                if getattr(tc, "type", None):
                    slot["type"] = tc.type
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments
                # Gemini (thinking models): the function call's thought
                # signature must be replayed verbatim on the next request's
                # tool_calls or the API 400s with "Function call is missing
                # a thought_signature" -- capture it here so it round-trips
                # through history unchanged (see wire-stripping below, which
                # only strips top-level message keys, not nested tool_call
                # fields like this one).
                extra = getattr(tc, "extra_content", None)
                if extra:
                    slot["extra_content"] = extra
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
                if c.get("extra_content"):
                    tc["extra_content"] = c["extra_content"]
                tcs.append(tc)
            # History echo: moonshot 400s ("tokenization failed") on re-posted
            # type=builtin_function — normalize to function, which it accepts.
            msg["tool_calls"] = tcs
            if builtin_ids:
                msg["_builtin_ids"] = builtin_ids
        return msg

    # -- full turn incl. tool loop ------------------------------------
    def run(self, user_text: str) -> None:
        global _FIRST_USER_LINE, _WORKSPACE_TOOL_CALLS
        if _FIRST_USER_LINE is None and user_text:
            _FIRST_USER_LINE = user_text

        # Drain any comm messages (or mid-task nudges from the previous turn)
        # before starting the next turn, so peers can inject context at the
        # boundary without waiting for a tool call.
        _drain_nudges(self)

        # context warning before sending
        s = _ctx_stats(self)
        if s["pct"] >= 90:
            print(f"{C.re}  ⚠ context {s['tokens']}/{s['window']} ({s['pct']}%) — consider /ctx compress{C.r}")
        elif s["pct"] >= 70:
            print(f"{C.ye}  ⚠ context {s['tokens']}/{s['window']} ({s['pct']}%){C.r}")

        t0 = time.time()
        if user_text:
            # Only append a user message when there's actual new text. run("")
            # is used to process something already-drained/pre-queued (a comm
            # wake, an httpd-upload notification) -- appending an empty-string
            # message on top of that 400s at the API ("message must not be
            # empty"), since it lands right after another user-role message
            # from _drain_nudges above with nothing assistant/tool in between.
            self.messages.append({"role": "user",
                                  "content": _content_with_attachments(user_text)})
            _tx("paul", user_text)
        try:
            # Suppress tty echo for the whole turn, not just individual tool
            # calls: the model prints continuously throughout (streamed
            # text, "· toolname" lines between tool calls) and any of that
            # racing the kernel's live echo of concurrent typing is what
            # produces visibly garbled/interleaved output. No-ops safely if
            # stdin isn't a tty (one-shot / piped runs).
            with _stdin_no_echo(sys.stdin.fileno()), sigint_guard():
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
                        _workspace_maybe_nudge(_FIRST_USER_LINE)
                        return

                    builtin_ids = set(msg.get("_builtin_ids") or [])
                    for tc in tcs:
                        name = tc["function"]["name"]
                        tc_id = tc["id"]
                        _WORKSPACE_TOOL_CALLS += 1
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
                                    except Interrupted:
                                        # Ctrl-C during a tool call: stop the turn
                                        # cleanly instead of treating this as a
                                        # tool error and looping into another API
                                        # call (which is what produced the
                                        # confusing APIConnectionError before).
                                        raise
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
        _workspace_maybe_nudge(_FIRST_USER_LINE)


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
  /workers [id]    list live runners, or tail-follow a runner's log until keypress
  /todo [add|done|rm|edit|hud]  in-flight todo list (hud on|off)
  /comm          list live ainow instances in the current comm directory
  /comm relisten [local|home]  re-register on the comm mesh (recovers from
                 the comm dir/socket being removed externally; also switches mode)
  /workspace list|show <name>|save <name> [notes]|bootstrap <name> <cmd>|forget <name>
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


def _workers_list() -> str:
    _bg_init()
    _bg_reap()
    running = [
        (jid, job) for jid, job in _BG_REGISTRY.items()
        if job.get("alive") and job.get("state") == "running"
    ]
    if not running:
        return "no live workers"
    rows = []
    for jid, job in sorted(running, key=lambda t: int(t[0])):
        elapsed = time.time() - job["started"]
        rows.append(f"{jid}: {elapsed:.1f}s  {job['command']}")
    return "\n".join(rows)


def _workers_peek(job_id: str) -> None:
    _bg_init()
    _bg_reap()
    job = _BG_REGISTRY.get(job_id)
    if not job:
        print(f"{C.re}error: no job {job_id}{C.r}")
        return
    jdir = pathlib.Path(job.get("dir") or os.path.dirname(job.get("log") or ""))
    print(f"{C.d}peeking job {job_id} (any key to return){C.r}")
    if not sys.stdin.isatty():
        print(_read_job_log(job_id, lines=40) or "(log empty)")
        return
    files = {}
    for name in ("stdout.log", "stderr.log"):
        p = jdir / name
        try:
            f = open(p, "r", errors="replace")
            f.seek(0, 2)
            files[name] = f
        except OSError:
            pass
    if not files:
        print("(no log files)")
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            for f in files.values():
                chunk = f.read()
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
            ready, _, _ = select.select([fd], [], [], 0.2)
            if ready:
                ch = os.read(fd, 1)
                if ch:
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        for f in files.values():
            try:
                f.close()
            except OSError:
                pass
    print()


def _handle_todo_cmd(rest: str) -> None:
    parts = rest.split(None, 1)
    sub = parts[0].lower() if parts else ""
    arg = parts[1] if len(parts) > 1 else ""
    if sub == "add":
        if not arg:
            print("usage: /todo add <text>")
        else:
            print(_todo_add(arg))
    elif sub == "done":
        try:
            print(_todo_done(int(arg)))
        except (ValueError, IndexError):
            print("usage: /todo done <n>")
    elif sub == "rm":
        try:
            print(_todo_rm(int(arg)))
        except (ValueError, IndexError):
            print("usage: /todo rm <n>")
    elif sub == "edit":
        n_str, _, text = arg.partition(" ")
        try:
            print(_todo_edit(int(n_str), text))
        except (ValueError, IndexError):
            print("usage: /todo edit <n> <text>")
    elif sub == "hud":
        if arg.lower() in ("on", "off"):
            _todo_hud_toggle(arg)
            print(f"todo hud {'on' if _TODO_HUD else 'off'}")
        else:
            print("usage: /todo hud on|off")
    else:
        print(_todo_list())


def _handle_workspace_cmd(agent, rest: str) -> None:
    parts = rest.split(None, 1)
    sub = parts[0].lower() if parts else ""
    arg = parts[1] if len(parts) > 1 else ""
    global _WORKSPACE_NAME, _COMM_LABEL
    if sub == "list":
        print(t_workspace("list"))
    elif sub == "show":
        if not arg:
            print("usage: /workspace show <name>")
        else:
            print(t_workspace("show", name=arg))
    elif sub == "forget":
        if not arg:
            print("usage: /workspace forget <name>")
        else:
            print(t_workspace("forget", name=arg))
    elif sub == "save":
        if not arg:
            print("usage: /workspace save <name> [notes...]")
        else:
            name, _, notes = arg.partition(" ")
            print(_workspace_save(name, agent.model, _TRANSCRIPT, notes=notes or None))
            _WORKSPACE_NAME = name
            # Upgrade the comm label from pid to workspace name so peers see it.
            _COMM_LABEL = name
            if _COMM_REGISTRY_PATH and _COMM_REGISTRY_PATH.exists():
                try:
                    reg = json.loads(_COMM_REGISTRY_PATH.read_text())
                    reg["label"] = name
                    reg["workspace"] = name
                    _COMM_REGISTRY_PATH.write_text(json.dumps(reg))
                except (OSError, json.JSONDecodeError):
                    pass
    elif sub == "bootstrap":
        # Set the shell command that runs (cwd=root) on every future -w load of
        # this workspace, with its stdout/stderr injected into context. This is
        # the only thing that makes -w actually resume context, not just labels.
        name, _, cmd = arg.partition(" ")
        if not name or not cmd:
            print("usage: /workspace bootstrap <name> <shell command>")
        elif name not in _workspace_list():
            print(f"ainow: unknown workspace {name!r}; save it first with "
                  f"/workspace save {name}")
        else:
            print(_workspace_save(name, agent.model, bootstrap=cmd))
    else:
        print("usage: /workspace list|show <name>|save <name> [notes]|"
              "bootstrap <name> <cmd>|forget <name>")


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
    global _REPL_SESSION
    _REPL_SESSION = session
    agent.interactive_repl = True

    cwd = os.path.realpath(os.getcwd())
    _transcript_start(agent.provider, agent.model)
    _log(f"session start {agent.provider}/{agent.model}  cwd={cwd}  auto={agent.auto}")
    print(f"{C.ma}ainow{C.r} {C.b}{agent.provider}/{agent.model}{C.r}  "
          f"{C.d}cwd {cwd}{C.r}")
    if _ACTIVE_THEME != "default":
        print(f"{C.d}theme: {_ACTIVE_THEME}{C.r}")
    print(f"{C.d}/help for commands · Ctrl-C interrupts · Ctrl-D or Ctrl-Q exits{C.r}\n")

    pending = first
    while True:
        hud = _todo_hud_line()
        if pending is not None:
            line, pending = pending, None
            pre = _render_pre(agent)
            if pre:
                print(f"{C.d}{pre}{C.r}")
            if hud:
                print(f"{C.d}{hud}{C.r}")
            print(f"{_fmt_prompt(agent.provider, agent.model)}{line}")
        else:
            try:
                pre = _render_pre(agent)
                if pre:
                    print(f"{C.d}{pre}{C.r}")
                if hud:
                    print(f"{C.d}{hud}{C.r}")
                line = session.prompt(ANSI(_fmt_prompt(agent.provider, agent.model)))
            except KeyboardInterrupt:      # Ctrl-C: clear line, stay alive
                continue
            except EOFError:               # Ctrl-D / Ctrl-Q: leave
                _log(f"session end {agent.provider}/{agent.model}  msgs={len(agent.messages)}")
                print("bye")
                return

        global _COMM_AUTORUN_STREAK
        if line is _COMM_WAKE:
            # A comm message arrived while idle; run a turn with no keystroke
            # to process it now instead of waiting for the next human input.
            _COMM_AUTORUN_STREAK += 1
            agent.run("")
            post = _render_post(agent, agent._last_elapsed)
            if post:
                print(f"{C.d}{post}{C.r}")
            print()
            continue

        # Any real human input (even a bare Enter) demonstrates a human is
        # actually present -- reset the auto-wake loop guard.
        _COMM_AUTORUN_STREAK = 0

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
                    err, m = _validate_model(p, m)
                    if err:
                        print(f"{C.ye}  {err}{C.r}")
                        print(f"{C.d}  staying on {agent.provider}/{agent.model}{C.r}")
                        continue
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
            elif cmd == "workers":
                if not rest:
                    print(_workers_list())
                else:
                    _workers_peek(rest.split()[0])
            elif cmd == "todo":
                _handle_todo_cmd(rest)
            elif cmd == "comm":
                sub, _, arg = rest.partition(" ")
                if sub == "relisten":
                    new_mode = arg.strip() or None
                    if new_mode and new_mode not in ("local", "home"):
                        print("usage: /comm relisten [local|home]")
                    else:
                        print(_comm_relisten(new_mode, agent.model, _WORKSPACE_NAME))
                else:
                    print(t_comm_list())
            elif cmd == "workspace":
                _handle_workspace_cmd(agent, rest)
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
    comm_mode = os.environ.get("AINOW_COMM", "local")
    instance_label: str | None = None
    workspace_name: str | None = os.environ.get("AINOW_WORKSPACE")

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
    if "--comm" in argv:
        idx = argv.index("--comm")
        if idx + 1 < len(argv):
            comm_mode = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2:]
    if "--label" in argv:
        idx = argv.index("--label")
        if idx + 1 < len(argv):
            instance_label = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2:]
    if "--workspace" in argv:
        idx = argv.index("--workspace")
        if idx + 1 < len(argv):
            workspace_name = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2:]
    if "-w" in argv:
        idx = argv.index("-w")
        if idx + 1 < len(argv):
            workspace_name = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2:]

    if os.geteuid() == 0 and not allow_root:
        sys.exit(
            "ainow: refusing to run as root. Use --allow-foot-bullet-root-mode to override."
        )

    provs = load_providers()
    prov, model, pub_cfg = parse_spec(argv[0], provs)
    first = " ".join(argv[1:]) or None

    # Load workspace (if requested) before binding comm so the comm label and
    # journal defaults come from the workspace. Workspace loading only affects
    # context/defaults; we never chdir or touch repos.
    if workspace_name:
        _workspace_load(workspace_name)
        if not instance_label:
            instance_label = workspace_name

    # Bind the per-instance comm socket before any network work so a second
    # instance fails fast with a clean message instead of after model refresh.
    _comm_startup(comm_mode, instance_label, model, workspace=workspace_name)

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
        err, model = _validate_model(prov, model)
        if err:
            print(f"{C.ye}{err}{C.r}")
            sys.exit(2)

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
