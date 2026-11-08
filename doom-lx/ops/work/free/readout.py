"""Read the guest's RAM out of the model's Gated DeltaNet state.

The circuit keeps every RAM word as 32 bit-columns of one state row: guest word `w` (byte
`4*w` above RAM_BASE) lives in RAM block `w >> 11` at value head `w & 7`, row `(w >> 3) & 255`,
columns 2..33 of the head's 128-dim value space (see `core.py`'s RAM tables and `snapshot()`).
A set bit is A_BIT (128), an unset one is 0, and the row's ballast sits in columns 0..1, so a
threshold at A_BIT / 2 decodes the word. This is the read side of `snapshot()`, which only
writes. Words below TEXT_END live in the read-only ROM blocks and are not served here.

    from readout import Readout
    ro = Readout(c, cache)               # c: the built circuit, cache: the Session's cache
    ro.read_bytes(off, n)                # n bytes of guest RAM (offsets from RAM_BASE)
    ro.frame()                           # the framebuffer as PPM bytes (palette applied)

"""
import struct

import torch

A_BIT = 2**7
THRESH = A_BIT / 2
BIT_COLS = slice(2, 34)
RAM_BLOCK_WORDS = 2048  # one key head of RAM: 8 value heads x 256 rows
POW = (1 << torch.arange(32, dtype=torch.int64))


def _node_index(c, name: str) -> int:
    node = c.tables[name]
    for i, n in enumerate(c.nodes):
        if n is node:
            return i
    raise KeyError(name)


class Readout:
    def __init__(self, c, cache, palette: bytes | None = None,
                 fb_ptr_off: int = 0, fb_w: int = 0, fb_h: int = 0) -> None:
        from lx import hf

        self.c, self.cache = c, cache
        self.r = c.target.value_heads // c.target.key_heads
        self.blk = {}
        for b in c.ram_blocks:
            name = f"ram{b}"
            if name not in c.tables:
                continue
            ni = _node_index(c, name)
            L, j = hf.head_slots(c)[ni]
            self.blk[b] = (L, j, len(c.nodes[ni].vheads()))
        assert self.blk, "no RAM blocks found"
        self.palette = palette
        self.fb = (fb_ptr_off, fb_w, fb_h)

    def _words(self, w: torch.Tensor) -> torch.Tensor:
        """Guest word indices `w` (long tensor, from RAM_BASE) as uint32 values."""
        first = next(iter(self.blk))
        dev = self.cache.layers[self.blk[first][0]].recurrent_states[0].device
        w = w.to(dev)
        out = torch.zeros(len(w), dtype=torch.int64, device=dev)
        done = torch.zeros(len(w), dtype=torch.bool, device=dev)
        for b, (L, j, _) in sorted(self.blk.items()):
            sel = ((w >> 11) == b) & ~done
            if not sel.any():
                continue
            st = self.cache.layers[L].recurrent_states[0]  # (1, nv, kd, vd)
            wi = w[sel] & (RAM_BLOCK_WORDS - 1)
            bits = st[0, j * self.r + (wi & 7), wi >> 3, BIT_COLS].float()  # (n, 32)
            out[sel] = ((bits > THRESH).double() @ POW.double().to(dev)).long()
            done |= sel
        if not done.all():
            raise KeyError(f"words {w[~done].tolist()[:4]} are in no RAM block "
                           "(below TEXT_END?)")
        return out

    def read_bytes(self, base: int, n: int) -> bytes:
        """`n` bytes of guest RAM, `base` = byte offset from RAM_BASE."""
        w0, off = base >> 2, (base & 3)
        dev = "cpu"
        words = self._words(torch.arange(w0, w0 + ((off + n + 3) >> 2), dtype=torch.long))
        raw = b"".join(struct.pack("<I", int(v) & 0xFFFFFFFF) for v in words.tolist())
        return raw[off:off + n]

    def frame(self) -> bytes:
        """The guest's framebuffer as PPM bytes, like run_kaggle's grab()."""
        ptr_off, w, h = self.fb
        (fbptr,) = self._words(torch.tensor([ptr_off >> 2], dtype=torch.long)).tolist()
        fb = fbptr - 0x80000000  # spec.RAM_BASE
        pix = self.read_bytes(fb, w * h)
        out = bytearray(f"P6\n{w} {h}\n255\n".encode())
        for v in pix:
            out += self.palette[3 * v:3 * v + 3]
        return bytes(out)
