#!/usr/bin/env python3
"""A minimal peer for ainow's AF_UNIX comm mesh.

Lets a non-ainow process register itself as a discoverable peer, so a live
ainow instance's comm_list/comm_send tools can see it and talk to it — no
ainow instance or API key required, since this speaks ainow's local
peer-discovery protocol directly rather than being one. Useful for driving
a conversation between a live ainow session and another LLM/agent/script
interactively, without wiring up a second full provider.

Protocol (read directly out of ainow.py's _comm_startup/_comm_listener/
t_comm_list/t_comm_send, not guessed):
  <comm_dir>/ainow.<pid>/instance       AF_UNIX SOCK_STREAM socket
  <comm_dir>/ainow.<pid>/registry.json  {"pid","label","model","cwd","started","workspace"}
  wire messages: newline-delimited JSON {"from","kind","text"}, kind in
    ("message","task")

Usage:
  tools/comm_peer.py listen --label claude [--comm local|home] [--comm-dir PATH]
      Registers as a peer and prints incoming messages to stdout as they
      arrive (one JSON line each). Runs until Ctrl-C; cleans up on exit.

  tools/comm_peer.py send --label claude <target> <text> [--comm local|home] [--comm-dir PATH]
      One-shot send to a live peer (ainow instance or another comm_peer.py),
      found by pid or label, matching ainow's own t_comm_send lookup.

--comm must match how the target ainow instance was started (default
'local', i.e. AINOW_COMM_LOCAL or /tmp/ainow; 'home' is ~/ainow).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import socket
import sys
import time

HOME = pathlib.Path.home()


def comm_dir(mode: str, override: str | None) -> pathlib.Path:
    if override:
        return pathlib.Path(override)
    if mode == "home":
        return HOME / "ainow"
    return pathlib.Path(os.environ.get("AINOW_COMM_LOCAL", "/tmp/ainow"))


def instance_dir(dir_: pathlib.Path, pid: int) -> pathlib.Path:
    return dir_ / f"ainow.{pid}"


def find_target(dir_: pathlib.Path, target: str) -> dict | None:
    matches = []
    for reg in dir_.glob("ainow.*/registry.json"):
        try:
            info = json.loads(reg.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if str(info.get("pid", "")) == target or str(info.get("label", "")) == target:
            matches.append(info)
    if len(matches) == 1:
        return matches[0]
    return None


def cmd_send(args: argparse.Namespace) -> int:
    dir_ = comm_dir(args.comm, args.comm_dir)
    info = find_target(dir_, args.target)
    if not info:
        print(f"error: no unique live instance matching {args.target!r}", file=sys.stderr)
        return 1
    sock_path = instance_dir(dir_, info["pid"]) / "instance"
    payload = json.dumps({"from": args.label, "kind": args.kind, "text": args.text})
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(2.0)
    try:
        s.connect(str(sock_path))
        s.sendall((payload + "\n").encode())
    except OSError as e:
        print(f"error: could not reach {args.target!r}: {e}", file=sys.stderr)
        return 1
    finally:
        s.close()
    print(f"sent to {info.get('label')} (pid {info['pid']})")
    return 0


def cmd_listen(args: argparse.Namespace) -> int:
    dir_ = comm_dir(args.comm, args.comm_dir)
    pid = os.getpid()
    inst = instance_dir(dir_, pid)
    dir_.mkdir(parents=True, exist_ok=True)
    if args.comm == "local":
        os.chmod(dir_, 0o1777)
    else:
        try:
            os.chmod(dir_, 0o700)
        except OSError:
            pass
    inst.mkdir(parents=True, exist_ok=True)
    os.chmod(inst, 0o700)
    sock_path = inst / "instance"
    reg_path = inst / "registry.json"

    if sock_path.exists():
        sock_path.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(4)

    reg_path.write_text(json.dumps({
        "pid": pid, "label": args.label, "model": "claude-code (interactive)",
        "cwd": str(pathlib.Path.cwd().resolve()), "started": time.time(),
        "workspace": None,
    }))
    print(f"# listening as {args.label!r} (pid {pid}) on {sock_path}", file=sys.stderr)
    print(f"# comm dir: {dir_}", file=sys.stderr)

    def cleanup(*_a):
        try:
            sock_path.unlink(missing_ok=True)
            reg_path.unlink(missing_ok=True)
            inst.rmdir()
        except OSError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        while True:
            conn, _ = srv.accept()
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
                        print(json.dumps(payload), flush=True)
            finally:
                conn.close()
    finally:
        cleanup()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="claude")
    ap.add_argument("--comm", choices=("local", "home"), default="local")
    ap.add_argument("--comm-dir", default=None)
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("listen")
    sp = sub.add_parser("send")
    sp.add_argument("target")
    sp.add_argument("text")
    sp.add_argument("--kind", choices=("message", "task"), default="message")
    args = ap.parse_args()
    if args.mode == "listen":
        return cmd_listen(args)
    return cmd_send(args)


if __name__ == "__main__":
    sys.exit(main())
