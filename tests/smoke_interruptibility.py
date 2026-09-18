#!/usr/bin/env python3
"""Smoke tests for the interruptibility feature branch.

Run with: python3 tests/smoke_interruptibility.py

The API-dependent tests use whatever provider/model is passed (default
kimi/kimi-k2.6).  They will be skipped if the model fails to respond.
"""
from __future__ import annotations

import os
import pathlib
import pty
import select
import subprocess
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
AINOW = REPO / "ainow.py"
MODEL = os.environ.get("AINOW_SMOKE_MODEL", "kimi/kimi-k2.6")


def banner(name: str) -> None:
    print(f"\n=== {name} ===")


def run_ainow(args, **kw):
    return subprocess.run([sys.executable, str(AINOW)] + args, **kw)


def test_compile() -> bool:
    banner("py_compile ainow.py")
    r = subprocess.run([sys.executable, "-m", "py_compile", str(AINOW)])
    ok = r.returncode == 0
    print("PASS" if ok else "FAIL")
    return ok


def test_plugin() -> bool:
    banner("example plugin loads and runs")
    code = """
import os, sys
sys.path.insert(0, sys.argv[1])
os.environ["AINOW_TOOLS_DIR"] = sys.argv[2]
import ainow
print("utc_now in TOOLS:", "utc_now" in ainow.TOOLS)
print("utc_now marked plugin:", ainow.TOOLS["utc_now"][3])
print("call:", ainow.TOOLS["utc_now"][0]())
"""
    r = subprocess.run(
        [sys.executable, "-c", code, str(REPO), str(REPO / "examples")],
        capture_output=True, text=True, cwd=str(REPO),
    )
    print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="", file=sys.stderr)
    ok = r.returncode == 0 and "True" in r.stdout and "20" in r.stdout
    print("PASS" if ok else "FAIL")
    return ok


def test_background_jobs() -> bool:
    banner("background jobs: spawn, list, log, kill")
    code = r"""
import ainow, time, os
# Use a temp log dir so we do not touch the user's registry.
ainow._BG_DIR = ainow.LOG_DIR / "bg-smoke-test"
ainow._BG_REGISTRY_FILE = ainow._BG_DIR / "jobs.json"
ainow._BG_REGISTRY.clear()
ainow._BG_REGISTRY_LOADED = True

r = ainow.t_bash("echo start; sleep 5; echo end", background=True)
print(r)
job_id = list(ainow._BG_REGISTRY.keys())[0]
print("list:\n" + ainow._bg_list())
print("log tail:", ainow._bg_log_tail(job_id, lines=5))
print("kill:", ainow._bg_kill(job_id))
time.sleep(0.3)
print("after kill:\n" + ainow._bg_list())
# cleanup
if ainow._BG_REGISTRY_FILE.exists():
    os.remove(ainow._BG_REGISTRY_FILE)
"""
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(REPO),
    )
    print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="", file=sys.stderr)
    ok = (r.returncode == 0
          and "background job" in r.stdout
          and "start" in r.stdout
          and "kill:" in r.stdout)
    print("PASS" if ok else "FAIL")
    return ok


def test_piped_oneshot() -> bool:
    banner("piped-stdin oneshot returns without hang")
    # Empty piped stdin; auto-approve should kick in and the process must exit.
    try:
        r = run_ainow(
            [MODEL, "-c", "say ok", "--yolo"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=45,
        )
    except subprocess.TimeoutExpired:
        print("FAIL: timed out (hung)")
        return False
    print(f"exit code: {r.returncode}")
    print("stdout:", r.stdout[:200].replace("\n", " "))
    if r.stderr:
        print("stderr:", r.stderr[:200].replace("\n", " "), file=sys.stderr)
    # We accept non-zero exit if it is a quick API/auth failure, but it must NOT hang.
    ok = True
    print("PASS" if ok else "FAIL")
    return ok


def test_nudge_in_transcript() -> bool:
    banner("mid-task nudge lands in transcript")
    # Need a tty so nudge watching is enabled.  Use a pty and drive the REPL.
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        [sys.executable, str(AINOW), MODEL, "--yolo",
         "Use the bash tool to run 'sleep 3; echo done'. Then say 'finished'."],
        stdin=slave, stdout=slave, stderr=slave,
        close_fds=True, cwd=str(REPO),
    )
    os.close(slave)

    nudge_sent = False
    bash_seen = False
    exited = False
    deadline = time.time() + 60
    buffer = b""
    try:
        while proc.poll() is None and time.time() < deadline and not exited:
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(master, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
                buffer += chunk
                if not bash_seen and b"sleep 3" in buffer:
                    bash_seen = True
                if bash_seen and not nudge_sent:
                    time.sleep(0.8)  # mid-sleep
                    os.write(master, b"this is a user nudge\n")
                    nudge_sent = True
                # The bash command prints 'done' when it completes; wait for that
                # before exiting so we know the nudge had a chance to be injected.
                if bash_seen and b"\ndone" in buffer:
                    time.sleep(0.5)
                    os.write(master, b"/exit\n")
                    exited = True
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    except Exception as e:
        print(f"exception driving pty: {e}")
        proc.kill()
    finally:
        try:
            os.close(master)
        except OSError:
            pass

    if not bash_seen:
        print("FAIL: never saw bash tool start, could not send nudge")
        return False

    # Find the newest transcript and look for the nudge marker.
    tx_dir = pathlib.Path.home() / ".config" / "ainow" / "logs"
    txs = sorted(tx_dir.glob("transcript-*.md")) if tx_dir.is_dir() else []
    if not txs:
        print("FAIL: no transcript file found")
        return False
    latest = txs[-1]
    content = latest.read_text(errors="replace")
    marker = "[user nudge mid-task] this is a user nudge"
    ok = marker in content
    print(f"latest transcript: {latest}")
    print(f"marker present: {ok}")
    if not ok:
        print("--- transcript tail ---")
        print(content[-800:])
    print("PASS" if ok else "FAIL")
    return ok


def main() -> int:
    results = []
    results.append(("compile", test_compile()))
    results.append(("plugin", test_plugin()))
    results.append(("background_jobs", test_background_jobs()))
    results.append(("piped_oneshot", test_piped_oneshot()))
    results.append(("nudge", test_nudge_in_transcript()))

    print("\n--- summary ---")
    failed = [n for n, ok in results if not ok]
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    if failed:
        print(f"\n{len(failed)} test(s) failed: {', '.join(failed)}")
        return 1
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
