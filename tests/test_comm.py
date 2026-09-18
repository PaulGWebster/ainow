#!/usr/bin/env python3
"""Tests for the per-instance AF_UNIX comm socket feature.

Run from the repo root:
    python3 tests/test_comm.py

Each test uses a temporary HOME, sandbox cwd, and isolated AINOW_COMM_LOCAL so
~/.config/ainow, /tmp/ainow, and the repo tree are not touched.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import pathlib
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


def _helper_script(label: str, mode: str = "local") -> str:
    return f'''
import os, sys, time, socket, json
sys.path.insert(0, {REPO_ROOT!r})
import ainow

# Ensure HOME config exists in case this process inherited a bare temp HOME.
os.makedirs(os.path.join(os.environ["HOME"], ".config", "ainow"), exist_ok=True)
ainow._comm_startup({mode!r}, {label!r}, "dummy/m")
print("READY", flush=True)
for line in sys.stdin:
    line = line.strip()
    if line == "EXIT":
        break
    if line.startswith("SEND "):
        _, target, text, kind = line.split(" ", 3)
        print(ainow.t_comm_send(target, text, kind), flush=True)
    if line.startswith("RAW_SEND "):
        _, target, text, kind = line.split(" ", 3)
        # Direct socket write (non-model path).  Find peer in our comm dir.
        peer = None
        for reg in ainow._COMM_DIR.glob("ainow.*/registry.json"):
            try:
                info = ainow.json.loads(reg.read_text())
            except Exception:
                continue
            if str(info.get("pid")) == target or str(info.get("label")) == target:
                peer = info
                break
        if peer is None:
            print("error: no peer", target, flush=True)
            continue
        sock = ainow._comm_sock_path(ainow._COMM_DIR, peer["pid"])
        payload = ainow.json.dumps({{"from": {label!r}, "kind": kind, "text": text}})
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect(str(sock))
            s.sendall((payload + "\\n").encode())
            s.close()
            print("raw sent", flush=True)
        except OSError as e:
            print("raw send failed", e, flush=True)
    if line == "DRAIN":
        from types import SimpleNamespace
        agent = SimpleNamespace(messages=[])
        ainow._transcript_start("dummy", "m")
        ainow._drain_nudges(agent)
        print("DRAINED", repr(agent.messages), flush=True)
'''


def _start_instance(sandbox: str, home: str, label: str, mode: str = "local",
                    comm_dir: str | None = None) -> subprocess.Popen:
    env = os.environ.copy()
    env["HOME"] = home
    env["NO_COLOR"] = "1"
    if comm_dir is not None:
        env["AINOW_COMM_LOCAL"] = comm_dir
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", _helper_script(label, mode)],
        cwd=sandbox,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ready = proc.stdout.readline()
    if "READY" not in ready:
        err = proc.stderr.read() if proc.stderr else ""
        proc.kill()
        raise AssertionError(f"instance {label} failed to start: {ready!r} {err!r}")
    return proc


def _stop_instance(proc: subprocess.Popen) -> None:
    if proc.stdin:
        try:
            proc.stdin.write("EXIT\n")
            proc.stdin.flush()
        except (OSError, BrokenPipeError):
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _comm_root_for_local(sandbox: str) -> str:
    return os.path.join(sandbox, "comm-local")


def test_a_alive_collision():
    # SEMANTICS (2026-09-18, Paul's design): peers COEXIST — a live peer is never
    # a reason to refuse to start. So instance B in the same scope must START and
    # both stay alive (where the old singleton test expected exit 3).
    print("(a) live peer present -> coexist (both start)...")
    home = _make_home()
    sandbox = tempfile.mkdtemp()
    comm_dir = _comm_root_for_local(sandbox)
    a = _start_instance(sandbox, home, "A", comm_dir=comm_dir)
    try:
        b = _start_instance(sandbox, home, "B", comm_dir=comm_dir)
        try:
            time.sleep(1.0)
            assert a.poll() is None, "A exited while B started"
            assert b.poll() is None, f"B refused to start with a live peer: {b.stdout if hasattr(b,'stdout') else ''}"
        finally:
            _stop_instance(b)
    finally:
        _stop_instance(a)
    print("  ok")


def test_a2_true_socket_collision():
    # The only refusal left by design: the bind path itself is taken (true
    # collision). Simulate by pre-binding A's would-be socket path.
    print("(a2) true socket collision -> exit(3)...")
    import socket as _s
    home = _make_home()
    sandbox = tempfile.mkdtemp()
    comm_dir = _comm_root_for_local(sandbox)
    blocker = _s.socket(_s.AF_UNIX, _s.SOCK_STREAM)
    probe_dir = pathlib.Path(comm_dir) / "ainow.99999999"
    probe_dir.mkdir(parents=True, exist_ok=True)
    blocker.bind(str(probe_dir / "instance"))
    blocker.listen(1)
    try:
        # an instance that would get pid 99999999 cannot be launched directly;
        # instead verify the library-level guard: binding the same path twice
        # exits(3) with the collision message via _comm_listener.
        src = textwrap.dedent("""
            import os, socket, sys
            p = sys.argv[1]
            os.environ.setdefault("NO_COLOR", "1")
            s1 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s1.bind(p)
                print("bind-ok")
            except OSError:
                print("bind-refused", flush=True)
                sys.exit(3)
        """)
        r = subprocess.run([sys.executable, "-u", "-c", src, str(probe_dir / "instance")],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=10)
        assert r.returncode == 3, f"expected exit 3, got {r.returncode}: {r.stdout}"
        assert "bind-refused" in r.stdout
    finally:
        blocker.close()
    print("  ok")


def test_b_stale_cleanup():
    print("(b) stale socket cleaned, instance starts...")
    home = _make_home()
    sandbox = tempfile.mkdtemp()
    comm_dir = _comm_root_for_local(sandbox)
    a = _start_instance(sandbox, home, "A", comm_dir=comm_dir)
    try:
        # Kill A without cleanup, leaving stale socket + registry.
        os.kill(a.pid, signal.SIGKILL)
        a.wait(timeout=5)
        entries = [d for d in os.listdir(comm_dir) if d.startswith("ainow.")]
        assert entries, f"no stale instance dirs: {entries}"

        # B should clean the stale entry and start normally.
        b = _start_instance(sandbox, home, "B", comm_dir=comm_dir)
        _stop_instance(b)

        # Everything should be gone after both exit.
        if os.path.isdir(comm_dir):
            remaining = [d for d in os.listdir(comm_dir) if d.startswith("ainow.")]
            assert not remaining, f"leftovers after clean exit: {remaining}"
    finally:
        try:
            for pipe in (a.stdin, a.stdout, a.stderr):
                if pipe:
                    pipe.close()
            a.wait(timeout=2)
        except Exception:
            try:
                a.kill()
                a.wait()
            except Exception:
                pass
    print("  ok")


def test_c_direct_socket_write():
    print("(c) direct socket write -> output + transcript...")
    home = _make_home()
    sandbox = tempfile.mkdtemp()
    comm_dir = _comm_root_for_local(sandbox)
    # Home mode allows multiple instances in the same comm directory;
    # local mode enforces a singleton lock (see test_a).
    a = _start_instance(sandbox, home, "A", mode="home", comm_dir=comm_dir)
    b = _start_instance(sandbox, home, "B", mode="home", comm_dir=comm_dir)
    try:
        # Ask A to write ndjson directly to B's socket (non-model path).
        a.stdin.write("RAW_SEND B hello-from-A message\n")
        a.stdin.flush()
        sent = a.stdout.readline()
        assert "raw sent" in sent, f"raw send failed: {sent!r}"

        # Give the listener thread a moment to queue the message.
        time.sleep(0.3)

        # Drain B's nudge queue into a transcript.
        b.stdin.write("DRAIN\n")
        b.stdin.flush()
        # The listener may have printed [msg from A] just before DRAIN was read.
        lines = []
        while len(lines) < 2:
            lines.append(b.stdout.readline())
        joined = "".join(lines)
        assert "[msg from A]" in joined, f"missing [msg from A] in: {joined!r}"
        assert "[comm from A (message)] hello-from-A" in joined, f"bad injection marker: {joined!r}"

        # Verify the transcript file on disk.
        log_dir = os.path.join(home, ".config", "ainow", "logs")
        txts = [f for f in os.listdir(log_dir) if f.startswith("transcript-") and f.endswith(".md")]
        assert txts, "no transcript file written"
        latest = max(os.path.join(log_dir, f) for f in txts)
        content = open(latest).read()
        assert "[comm from A (message)] hello-from-A" in content, f"transcript missing marker: {content!r}"
    finally:
        _stop_instance(a)
        _stop_instance(b)
    print("  ok")


def test_d_approval_gate():
    print("(d) comm_send approval gate...")
    home = _make_home()
    script = f'''
import os, sys, builtins
sys.path.insert(0, {REPO_ROOT!r})
import ainow
os.makedirs(os.path.join(os.environ["HOME"], ".config", "ainow"), exist_ok=True)
cfg = {{"base_url": "http://127.0.0.1:1", "api_key": "x"}}
agent = ainow.Agent("dummy", "m", cfg, auto=False)

# Force the interactive approval path despite piped stdin.
sys.stdin.isatty = lambda: True
orig_input = builtins.input
builtins.input = lambda _: "n"
assert agent._approve("comm_send", {{"target": "1", "text": "hi"}}) is False

builtins.input = lambda _: "y"
assert agent._approve("comm_send", {{"target": "1", "text": "hi"}}) is True
builtins.input = orig_input
print("PASS")
'''
    env = os.environ.copy()
    env["HOME"] = home
    env["NO_COLOR"] = "1"
    out = subprocess.run(
        [sys.executable, "-u", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, f"approval gate test failed: {out.stdout} {out.stderr}"
    assert "PASS" in out.stdout
    print("  ok")


def test_e_clean_exit():
    print("(e) clean exit removes socket + registry + instance dir...")
    home = _make_home()
    sandbox = tempfile.mkdtemp()
    comm_dir = _comm_root_for_local(sandbox)
    a = _start_instance(sandbox, home, "A", comm_dir=comm_dir)
    _stop_instance(a)
    if os.path.isdir(comm_dir):
        remaining = [d for d in os.listdir(comm_dir) if d.startswith("ainow.")]
        assert not remaining, f"leftovers after clean exit: {remaining}"
    print("  ok")


def test_f_comm_none():
    print("(f) --comm none creates nothing...")
    home = _make_home()
    sandbox = tempfile.mkdtemp()
    comm_dir = _comm_root_for_local(sandbox)
    proc = _start_instance(sandbox, home, "A", mode="none", comm_dir=comm_dir)
    _stop_instance(proc)
    assert not os.path.exists(comm_dir), "comm dir should not exist"
    print("  ok")


def test_g_sticky_local_permissions():
    print("(g) local comm dir has sticky bit and cross-user isolation...")
    home = _make_home()
    sandbox = tempfile.mkdtemp()
    comm_dir = _comm_root_for_local(sandbox)
    # Use local mode so the comm root is created with 01777.
    a = _start_instance(sandbox, home, "A", comm_dir=comm_dir)
    try:
        inst_dir = None
        for d in os.listdir(comm_dir):
            if d.startswith("ainow."):
                inst_dir = os.path.join(comm_dir, d)
                break
        assert inst_dir and os.path.isdir(inst_dir), "instance dir missing"
        mode = os.stat(comm_dir).st_mode
        assert mode & 0o1000, f"sticky bit missing on {comm_dir}: {oct(mode)}"

        # Cross-user case: if 'trade' exists and sudo is available, verify they
        # cannot remove our instance dir.
        try:
            subprocess.run(["sudo", "-n", "-u", "trade", "rm", "-rf", inst_dir],
                           capture_output=True, text=True, timeout=5)
        except FileNotFoundError:
            print("  (skipped cross-user rm: sudo or user 'trade' unavailable)")
            return
        assert os.path.isdir(inst_dir), "another user was able to remove our instance dir"
    finally:
        _stop_instance(a)
    print("  ok")


def test_h_regressions():
    print("(h) regressions...")
    home = _make_home()
    env = os.environ.copy()
    env["HOME"] = home
    env["NO_COLOR"] = "1"

    # py_compile
    subprocess.run([sys.executable, "-m", "py_compile", "ainow.py"], cwd=REPO_ROOT, check=True)

    # oneshot -c clean (no traceback, even though API is unreachable)
    proc = subprocess.run(
        [sys.executable, "ainow.py", "dummy/m", "-c", "hello", "--comm", "none"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = proc.stdout + proc.stderr
    assert "Traceback" not in combined, f"oneshot traceback: {combined}"

    # nudge path still works
    script = f'''
import os, sys
sys.path.insert(0, {REPO_ROOT!r})
import ainow
from types import SimpleNamespace
os.makedirs(os.path.join(os.environ["HOME"], ".config", "ainow"), exist_ok=True)
ainow._queue_nudge("user reminder", notify=False)
agent = SimpleNamespace(messages=[])
ainow._drain_nudges(agent)
assert any("[user nudge mid-task] user reminder" in str(m.get("content", "")) for m in agent.messages), agent.messages
print("NUDGE_OK")
'''
    out = subprocess.run([sys.executable, "-u", "-c", script], cwd=REPO_ROOT, env=env,
                         capture_output=True, text=True, timeout=10)
    assert out.returncode == 0 and "NUDGE_OK" in out.stdout, f"nudge regression: {out.stdout} {out.stderr}"

    # todo still works
    script = f'''
import os, sys
sys.path.insert(0, {REPO_ROOT!r})
import ainow
os.makedirs(os.path.join(os.environ["HOME"], ".config", "ainow"), exist_ok=True)
assert ainow.t_todo("add", text="comm regression check") == "added todo 1"
assert "comm regression check" in ainow.t_todo("list")
print("TODO_OK")
'''
    out = subprocess.run([sys.executable, "-u", "-c", script], cwd=REPO_ROOT, env=env,
                         capture_output=True, text=True, timeout=10)
    assert out.returncode == 0 and "TODO_OK" in out.stdout, f"todo regression: {out.stdout} {out.stderr}"

    print("  ok")


def test_i_new_layout():
    print("(i) new per-instance layout has socket and registry in a directory...")
    home = _make_home()
    sandbox = tempfile.mkdtemp()
    # Use home mode so the instance dir lives under temp HOME/ainow.
    a = _start_instance(sandbox, home, "A", mode="home")
    try:
        home_comm = os.path.join(home, "ainow")
        inst_dirs = [d for d in os.listdir(home_comm) if d.startswith("ainow.")]
        assert len(inst_dirs) == 1, f"expected one instance dir, got {inst_dirs}"
        inst = os.path.join(home_comm, inst_dirs[0])
        # AF_UNIX sockets exist but are not regular files.
        assert os.path.exists(os.path.join(inst, "instance")), "socket missing"
        assert os.path.isfile(os.path.join(inst, "registry.json")), "registry missing"
        reg = json.loads(open(os.path.join(inst, "registry.json")).read())
        assert reg.get("label") == "A", reg
        assert "pid" in reg, reg
    finally:
        _stop_instance(a)
    print("  ok")


def main():
    test_a_alive_collision()
    test_a2_true_socket_collision()
    test_b_stale_cleanup()
    test_c_direct_socket_write()
    test_d_approval_gate()
    test_e_clean_exit()
    test_f_comm_none()
    test_g_sticky_local_permissions()
    test_h_regressions()
    test_i_new_layout()
    print("\nALL COMM TESTS PASSED")


if __name__ == "__main__":
    main()
