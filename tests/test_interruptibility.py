#!/usr/bin/env python3
"""Regression/feature tests for the feature/interruptibility batch.

Run from the repo root:
    python3 tests/test_interruptibility.py

Each test runs in its own subprocess with a temporary HOME so nothing in
~/.config/ainow is touched and env/state does not leak between tests.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(script: str, env_extra: dict | None = None, input_data: str = "", timeout: float = 60) -> str:
    home = tempfile.mkdtemp()
    env = os.environ.copy()
    env["HOME"] = home
    env["NO_COLOR"] = "1"
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        input=input_data,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    combined = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise AssertionError(f"subprocess failed (rc={proc.returncode}):\n{combined}")
    return combined


def test_py_compile():
    print("py_compile...")
    subprocess.run([sys.executable, "-m", "py_compile", "ainow.py"], cwd=REPO_ROOT, check=True)
    print("  ok")


def test_a_auto_background():
    print("(a) foreground auto-background...")
    script = r'''
import os, sys, time, re
sys.path.insert(0, '.')
import ainow
os.environ["AINOW_FG_MAX_S"] = "3"
res = ainow.t_bash("sleep 6; echo done", max_s=20, idle_s=0)
assert "job " in res and "is still running" in res, f"expected still-running note: {res!r}"
m = re.search(r"job (\d+) is still running", res)
jid = m.group(1)
for _ in range(40):
    time.sleep(0.25)
    if ainow._BG_REGISTRY.get(jid, {}).get("state") == "done":
        break
log = ainow.t_bash_jobs("log", job_id=jid, lines=10)
assert "done" in log, f"expected completion in log: {log!r}"
print("PASS")
'''
    assert "PASS" in _run(script, env_extra={"AINOW_FG_MAX_S": "3"}, timeout=30)
    print("  ok")


def test_b_idle_kill():
    print("(b) idle kill...")
    script = r'''
import os, sys, time
sys.path.insert(0, '.')
import ainow
t0 = time.time()
res = ainow.t_bash("sleep 60", max_s=600, idle_s=3)
elapsed = time.time() - t0
assert 2 < elapsed < 5, f"idle kill took {elapsed:.1f}s"
assert "killed-idle" in res, f"expected killed-idle: {res!r}"
print("PASS")
'''
    assert "PASS" in _run(script, timeout=20)
    print("  ok")


def test_c_max_kill():
    print("(c) max-time kill...")
    script = r'''
import os, sys, time
sys.path.insert(0, '.')
import ainow
t0 = time.time()
res = ainow.t_bash("sleep 1000", max_s=2, idle_s=0)
elapsed = time.time() - t0
assert 1.5 < elapsed < 4, f"max kill took {elapsed:.1f}s"
assert "killed-max" in res, f"expected killed-max: {res!r}"
print("PASS")
'''
    assert "PASS" in _run(script, timeout=20)
    print("  ok")


def test_d_todo_roundtrip():
    print("(d) todo round-trip...")
    script = r'''
import os, sys, io
sys.path.insert(0, '.')
import ainow

assert ainow.t_todo("add", text="alpha") == "added todo 1"
assert ainow.t_todo("add", text="beta") == "added todo 2"
assert "1: [ ] alpha" in ainow.t_todo("list")
assert ainow.t_todo("done", n=1) == "marked todo 1 done"
assert ainow.t_todo("edit", n=2, text="beta edited") == "edited todo 2"
assert ainow.t_todo("rm", n=1) == "removed todo 1: alpha"

todo_path = os.path.join(os.environ["HOME"], ".config", "ainow", "todo.md")
contents = open(todo_path).read().strip()
assert contents == "- [ ] beta edited", f"bad todo.md: {contents!r}"

hud = ainow._todo_hud_line()
assert hud is not None and "beta edited" in hud and "1 open" in hud, f"bad hud: {hud!r}"

buf = io.StringIO()
old = sys.stdout
sys.stdout = buf
ainow._handle_todo_cmd("add gamma")
ainow._handle_todo_cmd("done 1")
ainow._handle_todo_cmd("rm 1")
sys.stdout = old
out = buf.getvalue()
assert "added" in out and "marked" in out and "removed" in out
assert open(todo_path).read().strip() == "- [ ] gamma"
print("PASS")
'''
    assert "PASS" in _run(script, timeout=20)
    print("  ok")


def test_e_themes():
    print("(e) themes load without KeyError...")
    script = r'''
import os, sys, types
sys.path.insert(0, '.')
import ainow
for name in ainow.THEME:
    c = types.SimpleNamespace(**ainow.THEME[name])
    _ = (c.d, c.b, c.r, c.cy, c.gr, c.ye, c.re, c.ma)
    print(name)
print("PASS")
'''
    out = _run(script, timeout=20)
    assert "PASS" in out
    for name in ["default", "amber", "mono", "high-contrast"]:
        assert name in out
    print("  ok")


def test_f_nudge_oneshot_regression():
    print("(f) nudge/oneshot regression...")
    home = tempfile.mkdtemp()
    os.makedirs(os.path.join(home, ".config", "ainow"), exist_ok=True)
    cfg = '{"providers": {"dummy": {"base_url": "http://127.0.0.1:1", "api_key": "x"}}}'
    with open(os.path.join(home, ".config", "ainow", "providers.json"), "w") as f:
        f.write(cfg)

    # Non-interactive _run_with_nudges path must not crash.
    script1 = r'''
import os, sys
sys.path.insert(0, '.')
import ainow
assert ainow._run_with_nudges(lambda: "ok", False) == "ok"
print("PASS")
'''
    env = {"HOME": home, "NO_COLOR": "1"}
    assert "PASS" in _run(script1, env_extra=env, timeout=20)

    # oneshot -c with piped input should fail cleanly (API error), not traceback.
    proc = subprocess.run(
        [sys.executable, "ainow.py", "dummy/model", "-c", "hello"],
        cwd=REPO_ROOT,
        env=env,
        input="nudge line\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = proc.stdout + proc.stderr
    assert "Traceback" not in combined, f"unexpected traceback:\n{combined}"
    # It returns cleanly (exit 0) even though the dummy provider cannot connect.
    print("  ok")


def test_g_plugins():
    print("(g) plugin tools load...")
    home = tempfile.mkdtemp()
    tools_dir = os.path.join(home, "tools.d")
    os.makedirs(tools_dir)
    plugin = '''
TOOL_SPEC = {
    "name": "demo_plugin_hello",
    "schema": {"type": "object", "properties": {"who": {"type": "string"}}, "required": []},
    "call": lambda who="world": f"hello {who}",
    "description": "demo plugin",
    "requires_approval": False,
}
'''
    with open(os.path.join(tools_dir, "demo.py"), "w") as f:
        f.write(plugin)
    script = r'''
import os, sys
sys.path.insert(0, '.')
import ainow
print(list(ainow.TOOLS.keys()))
'''
    env = {"HOME": home, "NO_COLOR": "1", "AINOW_TOOLS_DIR": tools_dir}
    out = _run(script, env_extra=env, timeout=20)
    assert "demo_plugin_hello" in out
    print("  ok")


def main():
    test_py_compile()
    test_a_auto_background()
    test_b_idle_kill()
    test_c_max_kill()
    test_d_todo_roundtrip()
    test_e_themes()
    test_f_nudge_oneshot_regression()
    test_g_plugins()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
