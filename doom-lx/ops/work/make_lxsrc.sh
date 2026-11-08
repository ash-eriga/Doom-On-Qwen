#!/bin/sh
# Assemble the `lxsrc` directory that run_kaggle.py (--lxsrc) and colab_run.py (lxsrc.zip) expect,
# from the linuxformer commit pinned in doom-lx/LINUXFORMER_COMMIT:
#
#     lxsrc/lx/            linuxformer's lx/ directory (it holds the `lx` package, lx/lx/)
#     lxsrc/run_kaggle.py  lxsrc/colab_bench.py  lxsrc/paths.py
#
#     make_lxsrc.sh <out-dir> [path to an existing linuxformer checkout]
#
# Without a checkout it clones https://github.com/itsrealranky/linuxformer. Either way the commit
# is checked out detached and compared with the pin, so a different linuxformer cannot slip in.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
PIN=$(cat "$HERE/../../LINUXFORMER_COMMIT")
OUT=${1:?usage: make_lxsrc.sh <out-dir> [linuxformer checkout]}
SRC=${2:-}

if [ -z "$SRC" ]; then
    SRC=$(mktemp -d)/linuxformer
    git clone -q https://github.com/itsrealranky/linuxformer "$SRC"
fi
git -C "$SRC" checkout -q --detach "$PIN"
[ "$(git -C "$SRC" rev-parse HEAD)" = "$PIN" ] || { echo "linuxformer is not at $PIN" >&2; exit 1; }
[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || { echo "linuxformer checkout has local changes" >&2; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT"
cp -r "$SRC/lx" "$OUT/lx"
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +
cp "$HERE/kaggle/run_kaggle.py" "$HERE/colab_bench.py" "$HERE/paths.py" "$OUT/"
echo "lxsrc ready in $OUT (linuxformer $PIN)"
