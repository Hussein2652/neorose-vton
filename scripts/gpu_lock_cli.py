#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def lock_path() -> str:
    locks_dir = os.environ.get("LOCKS_DIR", os.path.join("storage", "locks"))
    os.makedirs(locks_dir, exist_ok=True)
    return os.path.join(locks_dir, "gpu.lock")


def cmd_status(args) -> int:
    p = lock_path()
    if not os.path.exists(p):
        print("free")
        return 0
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        owner = data.get("owner")
        pid = data.get("pid")
        ts = float(data.get("ts", 0))
        age = time.time() - ts
        print(json.dumps({"status": "reserved", "owner": owner, "pid": pid, "age_sec": round(age, 3)}))
    except Exception:
        print("reserved")
    return 0


def cmd_acquire(args) -> int:
    p = lock_path()
    if os.path.exists(p):
        print("already reserved", file=sys.stderr)
        return 1
    data = {"owner": args.owner or "manual", "pid": os.getpid(), "ts": time.time()}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print("acquired")
    return 0


def cmd_release(args) -> int:
    p = lock_path()
    if not os.path.exists(p):
        print("already free")
        return 0
    if args.force:
        try:
            os.remove(p)
            print("released")
            return 0
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    # non-force: require owner match if file is valid json
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        owner = data.get("owner")
        if args.owner and owner != args.owner:
            print(f"lock held by {owner}; use --force to override", file=sys.stderr)
            return 3
    except Exception:
        pass
    try:
        os.remove(p)
        print("released")
        return 0
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU reservation lock CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("status", help="print lock status (free/reserved)")
    sp.set_defaults(func=cmd_status)
    sp = sub.add_parser("acquire", help="acquire the GPU lock")
    sp.add_argument("--owner", default=None)
    sp.set_defaults(func=cmd_acquire)
    sp = sub.add_parser("release", help="release the GPU lock")
    sp.add_argument("--owner", default=None)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_release)
    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

