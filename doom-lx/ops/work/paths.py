"""Resolve the paths these dev scripts need, without hardcoding a machine.

Earlier versions of every script in this directory opened with a hardcoded
`/home/<someone>/...` path, which only ever worked on the box the work happened
on. Import this instead:

    from paths import ELF_FILE, OPS, PORT   # absolute paths
    import paths; paths.add_lx()            # put the `lx` package on sys.path

`LX` points at the linuxformer checkout that contains the `lx` package. Set
`LX_PATH` if it is not a sibling of this repo.
"""
import os
import sys
from pathlib import Path

OPS = Path(__file__).resolve().parent          # .../doom-lx/ops/work
REPO = OPS.parents[2] if len(OPS.parents) > 2 else OPS   # a shallow dir such as /content/lxsrc has no third parent
DOOMLX = OPS.parent.parent                     # .../doom-lx
PORT = DOOMLX / "port"
ELF_FILE = PORT / "build" / "doom-lf.elf"
FRAMES_DIR = OPS / "weights-frames"

LX = Path(os.environ.get("LX_PATH", REPO.parent / "linuxformer" / "lx"))


def add_lx():
    """Put the linuxformer `lx` package and this directory on sys.path."""
    for p in (LX, OPS):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return LX
