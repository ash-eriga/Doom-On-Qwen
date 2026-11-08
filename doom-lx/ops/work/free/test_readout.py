#!/usr/bin/env python3
"""Unit test: the readout inverts the snapshot exactly, on CPU, no GPU and no weights needed.

Builds the circuit and a boot-state machine, creates the chip cache (zero states, the same
shapes Fast.init_cache makes; the packed blob is loaded with map_location=meta, so nothing
is materialised), injects the snapshot, then compares the readout's RAM byte for byte with
the machine's RAM. The forward pass is never run.

    python test_readout.py <packed.pt> <doom-lf.elf>
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ops/work: colab_bench

import paths; paths.add_lx()  # LX_PATH env picks the linuxformer checkout with the lx package
import colab_bench as cb

RAM_SIZE, TEXT_END = 0x600000, 0x40000
FC_OFF, FB_PTR_OFF = 0x562DD0, 0x4A2D94


def main():
    packed, elf = Path(sys.argv[1]), Path(sys.argv[2])
    cb.RAM_SIZE, cb.TEXT_END = RAM_SIZE, TEXT_END
    cb.ELF = elf

    import torch
    from transformers import DynamicCache, Qwen3_5TextConfig
    from lx import core, hf, spec as S
    from lx.ref import Machine
    from readout import Readout

    c = core.build(core.Config(ram_size=RAM_SIZE, text_end=TEXT_END))
    ram, entry = cb.load_elf(elf, RAM_SIZE)
    m = Machine(ram, TEXT_END)
    m.anchor, m.n = (entry - S.RAM_BASE) >> 2, 0

    blob = torch.load(packed, map_location="meta", weights_only=False)
    meta = blob["meta"]  # nk, nv, kd, vd, n_chip
    cfg = Qwen3_5TextConfig(**blob["tc"])

    cache = DynamicCache(config=cfg)
    conv = 2 * meta["nk"] * meta["kd"] + meta["nv"] * meta["vd"]
    for L in range(c.n_layers()):
        cache.update_conv_state(
            torch.zeros(1, conv, 1, dtype=torch.bfloat16), L,
            conv_kernel_size=cfg.linear_conv_kernel_dim)
        cache.layers[L].update_recurrent_state(
            torch.zeros(1, meta["nv"], meta["kd"], meta["vd"], dtype=torch.float32))
    print(f"cache: nk={meta['nk']} nv={meta['nv']} kd={meta['kd']} vd={meta['vd']} "
          f"({meta['n_chip']} chip layers)")

    core.snapshot(c, m)
    hf.inject(c, cache)

    ro = Readout(c, cache)

    words = torch.arange(TEXT_END // 4, RAM_SIZE // 4, dtype=torch.long)
    got = ro._words(words).tolist()
    want = list(struct.unpack_from(f"<{len(words)}I", m.ram, TEXT_END))
    bad = [(i, w, g) for i, (w, g) in enumerate(zip(want, got)) if w != g]
    print(f"RAM words: {len(want) - len(bad)}/{len(want)} match the snapshot")
    if bad:
        print("  first mismatches (addr, want, got):",
              [(hex(S.RAM_BASE + TEXT_END + 4 * i), hex(w), hex(g)) for i, w, g in bad[:8]])

    fc, = ro._words(torch.tensor([FC_OFF >> 2], dtype=torch.long))
    (fbptr,) = ro._words(torch.tensor([FB_PTR_OFF >> 2], dtype=torch.long))
    mfc = struct.unpack_from("<I", m.ram, FC_OFF)[0]
    mfb = struct.unpack_from("<I", m.ram, FB_PTR_OFF)[0]
    print(f"frame counter {int(fc)} vs {mfc}, fb ptr {hex(int(fbptr))} vs {hex(mfb)}")
    ok = not bad and int(fc) == mfc and int(fbptr) == mfb
    print("READOUT", "OK" if ok else "BROKEN")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
