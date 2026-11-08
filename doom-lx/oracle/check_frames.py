#!/usr/bin/env python3
"""Compare frames from the weights against the reference manifest.

    python check_frames.py <frames-dir> [--first 6] [--last 121] [--manifest oracle-192x120.md5]

The manifest holds the md5 of each frame's PPM pixel payload (the bytes after the
`P6\\n<w> <h>\\n255\\n` header), one `md5  frame-NNNN.ppm` line per frame. Every frame from
--first to --last (the 116-frame clip is 6..121) has to be present in <frames-dir> and match;
exit status is 0 only if all do.
"""
import argparse
import hashlib
import sys
from pathlib import Path


def payload(path: Path) -> bytes:
    data = path.read_bytes()
    pos, fields = 0, []
    while len(fields) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        end = pos
        while not data[end:end + 1].isspace():
            end += 1
        fields.append(data[pos:end])
        pos = end
    if fields[0] != b"P6" or fields[3] != b"255":
        raise ValueError(f"{path.name}: not an 8-bit P6 PPM")
    return data[pos + 1:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("frames", type=Path)
    ap.add_argument("--first", type=int, default=6)
    ap.add_argument("--last", type=int, default=121)
    ap.add_argument("--manifest", type=Path, default=Path(__file__).with_name("oracle-192x120.md5"))
    a = ap.parse_args()

    want = {}
    for line in a.manifest.read_text().splitlines():
        if line.strip():
            digest, name = line.split()
            if a.first <= int(name[6:10]) <= a.last:
                want[name] = digest

    bad, missing = [], []
    for name, digest in want.items():
        f = a.frames / name
        if not f.exists():
            missing.append(name)
        elif hashlib.md5(payload(f)).hexdigest() != digest:
            bad.append(name)

    ok = len(want) - len(bad) - len(missing)
    print(f"{ok}/{len(want)} frames match {a.manifest.name}")
    for name in bad:
        print(f"  MISMATCH {name}")
    for name in missing:
        print(f"  MISSING  {name}")
    return 0 if ok == len(want) else 1


if __name__ == "__main__":
    sys.exit(main())
