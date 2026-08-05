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
MODELS_CACHE = CFG_DIR / "models.json"
HISTORY_FILE = CFG_DIR / "history"
CACHE_TTL = 24 * 3600

MAX_TOOL_OUTPUT = 30_000


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def load_providers() -> dict:
    if not PROVIDERS_FILE.exists():
        sys.exit(f"ainow: no config at {PROVIDERS_FILE}")
    return json.loads(PROVIDERS_FILE.read_text())["providers"]


def load_model_cache() -> dict:
    if MODELS_CACHE.exists():
        try:
            return json.loads(MODELS_CACHE.read_text())
        except json.JSONDecodeError:
            pass
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
# completion  (must stay import-light: called on every TAB)
# --------------------------------------------------------------------------
def complete(word: str) -> None:
    provs = sorted(load_providers().keys())
    cache = load_model_cache()

    if "/" not in word:
        for p in provs:
            if p.startswith(word):
                print(p + "/")
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
        "description": "List the contents of a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path"}},
            "required": ["path"]}}},
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

Be concise. The user is in a terminal — no markdown headers or bullet-heavy \
formatting unless asked. Report what you actually did, and if a command failed, \
say so with the output rather than assuming it worked."""


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
        self.messages.append({"role": "user", "content": user_text})
        try:
            with sigint_guard():
                while True:
                    msg = self._stream_turn()
                    self.messages.append(msg)
                    tcs = msg.get("tool_calls")
                    if not tcs:
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


# --------------------------------------------------------------------------
# repl
# --------------------------------------------------------------------------
HELP = f"""{C.b}commands{C.r}
  /help          this
  /model <spec>  switch model, e.g. /model deepseek/deepseek-v4-pro
  /models [pat]  list cached models for the current provider
  /auto [on|off] toggle running tools without asking
  /clear         reset conversation
  /exit          quit
{C.b}keys{C.r}
  Ctrl-C  interrupt generation / clear the line  (does not exit)
  Ctrl-D  exit          Ctrl-Q  exit"""


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

    print(f"{C.ma}ainow{C.r} {C.b}{agent.provider}/{agent.model}{C.r}  "
          f"{C.d}cwd {os.getcwd()}{C.r}")
    print(f"{C.d}/help for commands · Ctrl-C interrupts · Ctrl-D or Ctrl-Q exits{C.r}\n")

    pending = first
    while True:
        if pending is not None:
            line, pending = pending, None
            print(f"{C.gr}› {C.r}{line}")
        else:
            try:
                line = session.prompt(ANSI(f"{C.gr}› {C.r}"))
            except KeyboardInterrupt:      # Ctrl-C: clear line, stay alive
                continue
            except EOFError:               # Ctrl-D / Ctrl-Q: leave
                print("bye")
                return

        line = line.strip()
        if not line:
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
                print(f"{C.d}context cleared{C.r}")
            elif cmd == "auto":
                if rest in ("on", "off"):
                    agent.auto = rest == "on"
                else:
                    agent.auto = not agent.auto
                print(f"{C.d}auto-approve {'on' if agent.auto else 'off'}{C.r}")
            elif cmd == "model":
                try:
                    p, m = parse_spec(rest, provs)
                except SystemExit as e:
                    print(f"{C.re}{e}{C.r}")
                    continue
                agent.__init__(p, m, provs[p], agent.auto)
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
            else:
                print(f"{C.re}unknown command /{cmd}{C.r}")
            continue

        agent.run(line)
        print()


# --------------------------------------------------------------------------
# entry
# --------------------------------------------------------------------------
def parse_spec(spec: str, provs: dict) -> tuple[str, str]:
    if "/" not in spec:
        sys.exit(f"ainow: model must be <provider>/<model>; providers: {', '.join(sorted(provs))}")
    prov, _, model = spec.partition("/")
    if prov not in provs:
        sys.exit(f"ainow: unknown provider '{prov}'; have: {', '.join(sorted(provs))}")
    if not model:
        sys.exit(f"ainow: no model given for {prov}")
    return prov, model


def main() -> None:
    argv = sys.argv[1:]

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
        return
    if argv[0] == "--providers":
        for n, p in sorted(load_providers().items()):
            print(f"  {n:12s} {p['base_url']}")
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

    auto = False
    if "--yolo" in argv:
        auto = True
        argv.remove("--yolo")

    provs = load_providers()
    prov, model = parse_spec(argv[0], provs)
    first = " ".join(argv[1:]) or None

    if not MODELS_CACHE.exists():
        print("building model cache (first run)…")
        refresh_models()

    agent = Agent(prov, model, provs[prov], auto)
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
