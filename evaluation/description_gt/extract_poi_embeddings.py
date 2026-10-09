#!/usr/bin/env python3
"""
extract_poi_embeddings.py — pull just the museum/attraction POI vectors out of the
3.44 GB data/socal/poi_embeddings.json.

The full cache is a single JSON object {"<row_index>": [floats...], ...}. Loading it
with json.load costs roughly 8-10 GB of RAM as Python lists, which the project's
"run ONE graph-resident chain at a time" rule makes unwise. Only ~766 rows are ever
needed for set3's description filters (museum + attraction rows that carry a usable
description), so this streams the file once and writes a small subset.

This is semantically exact for the evaluation: filtration only ever scores POIs whose
node is already in the candidate set, and those candidates are always museum or
attraction rows for the description stops.

Values are FLAT float arrays, so a value ends at the first ']' after its '[' — no
nesting to track.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "socal" / "poi_embeddings.json"
OUT = ROOT / "outputs" / "evaluation" / "description_gt" / "poi_embeddings_subset.json"

KEY_RE = re.compile(rb'"(\d+)"\s*:\s*\[')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--want", required=True, help="JSON list of row indices to keep")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--chunk", type=int, default=8 << 20)
    args = ap.parse_args()

    want = {int(i) for i in json.loads(Path(args.want).read_text())}
    print(f"[extract] want {len(want)} row indices from {args.src}")

    found: dict[str, list] = {}
    buf = b""
    pos = 0                      # absolute offset of buf[0]
    capture_key: str | None = None
    capture_parts: list[bytes] = []
    total = Path(args.src).stat().st_size
    read = 0

    with open(args.src, "rb") as fh:
        while True:
            chunk = fh.read(args.chunk)
            if not chunk:
                break
            read += len(chunk)
            buf += chunk

            while True:
                if capture_key is not None:
                    end = buf.find(b"]")
                    if end == -1:
                        capture_parts.append(buf)
                        buf = b""
                        break
                    capture_parts.append(buf[:end])
                    raw = b"".join(capture_parts)
                    try:
                        found[capture_key] = json.loads(b"[" + raw + b"]")
                    except Exception as e:
                        sys.exit(f"parse failure on row {capture_key}: {e}")
                    capture_key, capture_parts = None, []
                    buf = buf[end + 1:]
                    continue

                m = KEY_RE.search(buf)
                if not m:
                    # keep a tail long enough to hold a split key token
                    if len(buf) > 64:
                        buf = buf[-64:]
                    break
                key = m.group(1).decode()
                buf = buf[m.end():]
                if int(key) in want:
                    capture_key, capture_parts = key, []
                # else: skip ahead to this value's ']' without buffering it
                else:
                    end = buf.find(b"]")
                    while end == -1:
                        nxt = fh.read(args.chunk)
                        if not nxt:
                            break
                        read += len(nxt)
                        buf = nxt
                        end = buf.find(b"]")
                    if end == -1:
                        break
                    buf = buf[end + 1:]

            if read % (512 << 20) < args.chunk:
                print(f"    {read/total:5.1%}  captured {len(found)}")

    print(f"[extract] captured {len(found)}/{len(want)} vectors")
    missing = sorted(want - {int(k) for k in found})
    if missing:
        print(f"[extract] {len(missing)} rows have NO cached embedding "
              f"(they can never match a description filter): e.g. {missing[:8]}")
    dims = {len(v) for v in found.values()}
    print(f"[extract] vector dims present: {dims}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(found))
    print(f"[extract] wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
