#!/usr/bin/env python3
"""Tests for the workspace loader and slash commands.

Run from the repo root:
    venv/bin/python tests/test_workspace.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_home() -> str:
    home = tempfile.mkdtemp()
    cfg = os.path.join(home, ".config", "ainow")
    os.makedirs(cfg, exist_ok=True)
    providers = '{"providers": {"dummy": {"base_url": "http://127.0.0.1:1", "api_key": "x"}}}'
    with open(os.path.join(cfg, "providers.json"), "w") as f:
        f.write(providers)
    now = time.time()
    cache = '{"_ts": %s, "dummy": ["m"]}' % now
    with open(os.path.join(cfg, "models.json"), "w") as f:
        f.write(cache)
    return home


def _env(home: str) -> dict:
    env = os.environ.copy()
    env["HOME"] = home
    env["NO_COLOR"] = "1"
    env["AINOW_COMM_LOCAL"] = os.path.join(home, "comm-local")
    return env


def test_a_unknown_workspace():
    print("(a) -w with unknown name exits 2 listing names...")
    home = _make_home()
    ws_dir = os.path.join(home, "ainow", "workspaces")
    os.makedirs(ws_dir, exist_ok=True)
    for name in ("alpha", "beta"):
        open(os.path.join(ws_dir, f"{name}.json"), "w").write(json.dumps({"name": name}))
    env = _env(home)
    proc = subprocess.run(
        [sys.executable, "ainow.py", "dummy/m", "-w", "gamma", "-c", "hi", "--comm", "none"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 2, f"expected exit 2, got {proc.returncode}: {proc.stdout} {proc.stderr}"
    combined = proc.stdout + proc.stderr
    assert "unknown workspace" in combined, combined
    assert "alpha" in combined and "beta" in combined, combined
    print("  ok")


def test_b_workspace_bootstrap_and_label():
    print("(b) -w with valid workspace injects bootstrap and labels instance...")
    home = _make_home()
    ws_dir = os.path.join(home, "ainow", "workspaces")
    os.makedirs(ws_dir, exist_ok=True)
    ws = {
        "name": "sandbox",
        "root": os.path.join(home, "project"),
        "bootstrap": "echo 'PROJECT=sandbox-bootstrapped'",
        "label": "sandbox",
        "created": "2024-01-01T00:00:00",
        "model": "dummy/m",
    }
    os.makedirs(ws["root"], exist_ok=True)
    open(os.path.join(ws_dir, "sandbox.json"), "w").write(json.dumps(ws))

    # Directly load the workspace and check bootstrap injection + label default.
    script = f'''
import os, sys
sys.path.insert(0, {REPO_ROOT!r})
import ainow
os.makedirs(os.path.join(os.environ["HOME"], ".config", "ainow"), exist_ok=True)
ainow._workspace_load("sandbox")
assert ainow._WORKSPACE_NAME == "sandbox", ainow._WORKSPACE_NAME
assert "PROJECT=sandbox-bootstrapped" in ainow._WORKSPACE_BOOTSTRAP_OUTPUT, ainow._WORKSPACE_BOOTSTRAP_OUTPUT
assert "sandbox-bootstrapped" in ainow._system(), ainow._system()
print("BOOTSTRAP_OK")
'''
    env = _env(home)
    out = subprocess.run([sys.executable, "-u", "-c", script], cwd=REPO_ROOT, env=env,
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0 and "BOOTSTRAP_OK" in out.stdout, f"{out.stdout} {out.stderr}"

    # Spawn a persistent helper and inspect the comm socket while alive.
    helper = f'''
import os, sys, time
sys.path.insert(0, {REPO_ROOT!r})
import ainow
os.makedirs(os.path.join(os.environ["HOME"], ".config", "ainow"), exist_ok=True)
ainow._workspace_load("sandbox")
# Main() passes workspace name as the default label when --label is absent.
ainow._comm_startup("home", "sandbox", "dummy/m", workspace="sandbox")
print("READY", ainow._COMM_LABEL, flush=True)
input()
'''
    proc = subprocess.Popen([sys.executable, "-u", "-c", helper], cwd=REPO_ROOT,
                            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        ready = proc.stdout.readline()
        assert "READY" in ready, f"helper failed: {ready!r}"
        assert "sandbox" in ready, f"label not defaulted: {ready!r}"
        comm_root = os.path.join(home, "ainow")
        inst_dirs = [d for d in os.listdir(comm_root) if d.startswith("ainow.")]
        assert inst_dirs, f"no instance dir created: {comm_root}"
        reg_path = os.path.join(comm_root, inst_dirs[0], "registry.json")
        reg = json.loads(open(reg_path).read())
        assert reg.get("label") == "sandbox", reg
        assert reg.get("workspace") == "sandbox", reg
    finally:
        try:
            proc.stdin.write("\n")
            proc.stdin.flush()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait()
    print("  ok")


def test_c_workspace_roundtrip():
    print("(c) /workspace save/list/show/forget round-trip, files never deleted...")
    home = _make_home()
    ws_dir = os.path.join(home, "ainow", "workspaces")
    os.makedirs(ws_dir, exist_ok=True)
    # Create a small existing workspace entry with bootstrap/notes.
    existing = {
        "name": "demo",
        "root": os.path.join(home, "project"),
        "bootstrap": "echo demo-bootstrap",
        "journal_ssh": "user@host",
        "journal_file": "/path/to/journal.md",
        "label": "demo-label",
        "notes": "keep me",
        "created": "2024-01-01T00:00:00",
        "model": "dummy/m",
    }
    os.makedirs(existing["root"], exist_ok=True)
    marker = os.path.join(existing["root"], "project_file.txt")
    open(marker, "w").write("project data")
    open(os.path.join(ws_dir, "demo.json"), "w").write(json.dumps(existing))

    script = f'''
import os, sys
sys.path.insert(0, {REPO_ROOT!r})
import ainow
from types import SimpleNamespace
os.makedirs(os.path.join(os.environ["HOME"], ".config", "ainow"), exist_ok=True)

# Simulate a REPL agent and transcript.
ainow._transcript_start("dummy", "m")
agent = SimpleNamespace(model="dummy/m")

# Save should update root/model/transcript but preserve bootstrap/journal/notes.
print(ainow._workspace_save("demo", agent.model, ainow._TRANSCRIPT))
print(ainow._workspace_show("demo"))
print(ainow.t_workspace("list"))
print(ainow.t_workspace("forget", name="demo"))
print(ainow.t_workspace("list"))
'''
    env = _env(home)
    out = subprocess.run([sys.executable, "-u", "-c", script], cwd=REPO_ROOT, env=env,
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, f"round-trip failed: {out.stdout} {out.stderr}"
    combined = out.stdout + out.stderr
    assert "saved workspace" in combined, combined
    assert "demo-bootstrap" in combined, combined  # bootstrap preserved
    assert "keep me" in combined, combined  # notes preserved
    assert "demo-label" in combined, combined  # label preserved
    assert "project_file.txt" not in combined, combined  # project files not touched
    assert "workspaces:" in combined and "demo" in combined, combined  # list
    assert "no workspaces" in combined, combined  # after forget

    # The project file must still exist.
    assert os.path.exists(marker), "workspace forget deleted project files"
    # The registry entry (JSON) must be gone.
    assert not os.path.exists(os.path.join(ws_dir, "demo.json")), "registry entry not removed"
    print("  ok")


def test_d_nudge_once():
    print("(d) threshold nudge fires once and only once...")
    home = _make_home()
    env = _env(home)
    script = f'''
import os, sys
sys.path.insert(0, {REPO_ROOT!r})
import ainow
from types import SimpleNamespace
os.makedirs(os.path.join(os.environ["HOME"], ".config", "ainow"), exist_ok=True)

# Tiny thresholds: 0 minutes and 1 tool call.
os.environ["AINOW_WORKSPACE_NUDGE_MINS"] = "0"
os.environ["AINOW_WORKSPACE_NUDGE_CALLS"] = "1"
ainow._WORKSPACE_NUDGE_MINS = 0
ainow._WORKSPACE_NUDGE_CALLS = 1
ainow._WORKSPACE_SESSION_START = 0
ainow._WORKSPACE_TOOL_CALLS = 0
ainow._WORKSPACE_NUDGE_SHOWN = False
ainow._WORKSPACE_NAME = None
ainow._FIRST_USER_LINE = "hello world"

# First call should trigger the nudge.
ainow._workspace_maybe_nudge("hello world")
# Second call should not print again.
ainow._workspace_maybe_nudge("hello world")
print("NUDGE_TEST_DONE")
'''
    out = subprocess.run([sys.executable, "-u", "-c", script], cwd=REPO_ROOT, env=env,
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, f"nudge test failed: {out.stdout} {out.stderr}"
    lines = [ln for ln in (out.stdout + out.stderr).splitlines() if "getting long" in ln]
    assert len(lines) == 1, f"expected exactly one nudge line, got {lines}"
    assert "hello-world" in lines[0], lines[0]  # slug from first user line
    print("  ok")


def test_e_malformed_skipped():
    print("(e) malformed workspace entries are skipped with a warning...")
    home = _make_home()
    ws_dir = os.path.join(home, "ainow", "workspaces")
    os.makedirs(ws_dir, exist_ok=True)
    open(os.path.join(ws_dir, "good.json"), "w").write(json.dumps({"name": "good"}))
    open(os.path.join(ws_dir, "bad.json"), "w").write("not json")
    open(os.path.join(ws_dir, "weird.json"), "w").write("[1, 2, 3]")
    script = f'''
import os, sys
sys.path.insert(0, {REPO_ROOT!r})
import ainow
os.makedirs(os.path.join(os.environ["HOME"], ".config", "ainow"), exist_ok=True)
names = ainow._workspace_list()
assert names == ["good"], names
print("SKIP_OK")
'''
    env = _env(home)
    out = subprocess.run([sys.executable, "-u", "-c", script], cwd=REPO_ROOT, env=env,
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0 and "SKIP_OK" in out.stdout, f"{out.stdout} {out.stderr}"
    assert "warning" in (out.stdout + out.stderr).lower(), "missing warning for malformed entry"
    print("  ok")


def test_f_env_workspace():
    print("(f) AINOW_WORKSPACE env is honoured...")
    home = _make_home()
    ws_dir = os.path.join(home, "ainow", "workspaces")
    os.makedirs(ws_dir, exist_ok=True)
    ws = {"name": "envtest", "root": home, "model": "dummy/m"}
    open(os.path.join(ws_dir, "envtest.json"), "w").write(json.dumps(ws))
    env = _env(home)
    env["AINOW_WORKSPACE"] = "envtest"
    proc = subprocess.run(
        [sys.executable, "ainow.py", "dummy/m", "-c", "hi", "--comm", "none"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    # Should not exit 2 (unknown workspace); it may fail at API time, but not
    # at workspace-load time.
    assert proc.returncode != 2, f"workspace env not honoured: {proc.stdout} {proc.stderr}"
    print("  ok")


def main():
    test_a_unknown_workspace()
    test_b_workspace_bootstrap_and_label()
    test_c_workspace_roundtrip()
    test_d_nudge_once()
    test_e_malformed_skipped()
    test_f_env_workspace()
    print("\nALL WORKSPACE TESTS PASSED")


if __name__ == "__main__":
    main()
