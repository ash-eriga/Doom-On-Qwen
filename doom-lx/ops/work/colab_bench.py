#!/usr/bin/env python3
"""The chip on a Colab GPU, built without ever holding the dense model.

The circuit's weights are 99.98% zeros, but `hf.build_model` emits them dense: 51 layers of the
LX geometry is 34.4 GiB, more than a T4 (16 GB) or a free Colab VM (12.7 GB RAM). So this emits
one layer at a time, packs it into `lx.fast`'s nonzero form, and frees the dense copy before the
next layer. The GPU then holds the embedding (3.3 GiB) plus the nonzeros (a few MB).

    python colab_bench.py check            # streamed == dense, on a config small enough for both
    python colab_bench.py bench [tokens]   # tok/s of the real Doom circuit

`check` is the test that matters: it compares the streamed weights against
`hf.build_model` + `Fast.__init__` for a tiny circuit, tensor by tensor.
"""
import argparse
import gc
import os
import struct
import sys
import time
from pathlib import Path
from paths import ELF_FILE, LX

# the local checkout, or the clone on the Colab VM
for _p in ("/content/lxsrc/lx", LX):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break

import torch

from lx import core, spec as S
from lx.emit import check_bf16, embedding_rows, row
from lx.fast import PROJ, Fast, _nz_cols
from lx.graph import ONE, Expr
from lx.ir import FFN, KD, VD, K, Head

TEXT_END = 0x40000
RAM_SIZE = 0x580000
ELF = Path(os.environ.get("DOOM_ELF", "/content/doom-lf.elf"))
if not ELF.exists():
    ELF = Path(ELF_FILE)


def _check_bf16(name: str, t: torch.Tensor, rows: int = 4096) -> None:
    """`emit.check_bf16` in row blocks: it holds a fp32 copy of the weight (1 GiB for in_proj_qkv)
    beside the tensor it checks, which is most of a free Colab VM's headroom."""
    if t.dim() == 0 or t.shape[0] <= rows:
        check_bf16(name, t)
        return
    for a in range(0, t.shape[0], rows):
        check_bf16(f"{name}[{a}:]", t[a:a + rows])


# ---------------------------------------------------------------------------- one layer's weights
def emit_layer(g, lay, tc, L, dtype=torch.bfloat16) -> dict:
    """`emit_layers`' loop body for layer L alone (the alloc/keep branches of the merged exit
    are not used by the plain chip model)."""
    h, nk, nv = tc["hidden_size"], tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    inter = tc["intermediate_size"]
    r = nv // nk
    assert r * nk == nv
    assert tc["linear_key_head_dim"] == KD and tc["linear_value_head_dim"] == VD
    f32 = torch.float32
    one = Expr({ONE: 1})
    qkv = torch.zeros(2 * nk * KD + nv * VD, h, dtype=f32)
    convw = torch.zeros(2 * nk * KD + nv * VD, 1, tc["linear_conv_kernel_dim"], dtype=f32)
    zw = torch.zeros(nv * VD, h, dtype=f32)
    bw = torch.zeros(nv, h, dtype=f32)
    aw = torch.zeros(nv, h, dtype=f32)
    ow = torch.zeros(h, nv * VD, dtype=f32)
    gate = torch.zeros(inter, h, dtype=f32)
    up = torch.zeros(inter, h, dtype=f32)
    down = torch.zeros(h, inter, dtype=f32)
    heads = [n for n in g.nodes if isinstance(n, Head) and n.level == 2 * L + 1]
    for j, n in enumerate(heads):
        vhs = n.vheads()
        assert len(vhs) <= r, (n.name, len(vhs), r)
        for base, chans in ((j * KD, n.q), (nk * KD + j * KD, n.k)):
            for i, ch in chans.items():
                row(ch.expr, lay, qkv[base + i])
                for lag, w in enumerate(ch.taps):
                    convw[base + i, 0, -1 - lag] = w
        for x, vhd in enumerate(vhs):
            vh = j * r + x
            base = 2 * nk * KD + vh * VD
            for i, ch in vhd.v.items():
                row(ch.expr, lay, qkv[base + i])
                for lag, w in enumerate(ch.taps):
                    convw[base + i, 0, -1 - lag] = w
            beta = Expr({ONE: 128}) if vhd.beta is None else 256 * vhd.beta - 128 * one
            row(beta, lay, bw[vh])
            row(vhd.a, lay, aw[vh])
            for col, e in vhd.z.items():
                row(e, lay, zw[vh * VD + col])
            for col, d, w in vhd.out:
                ow[lay.dim(d), vh * VD + col] += w
    units = []
    for n in g.nodes:
        if isinstance(n, FFN) and n.level == 2 * L + 2:
            units += [(coef, a, b, n.out) for coef, a, b in n.terms]
    assert len(units) <= inter, (L, len(units), inter)
    for u, (coef, a, b, d) in enumerate(units):
        row(b, lay, gate[u], K)
        row(a, lay, up[u], 1 / K)
        down[lay.dim(d), u] = float(coef)
    layer = {
        "input_layernorm.weight": torch.zeros(h, dtype=f32),
        "post_attention_layernorm.weight": torch.zeros(h, dtype=f32),
        "linear_attn.in_proj_qkv.weight": qkv,
        "linear_attn.in_proj_z.weight": zw,
        "linear_attn.in_proj_a.weight": aw,
        "linear_attn.in_proj_b.weight": bw,
        "linear_attn.conv1d.weight": convw,
        "linear_attn.dt_bias": torch.full((nv,), -110.0, dtype=f32),
        "linear_attn.A_log": torch.zeros(nv, dtype=f32),
        "linear_attn.norm.weight": torch.ones(VD, dtype=f32),
        "linear_attn.out_proj.weight": ow,
        "mlp.gate_proj.weight": gate,
        "mlp.up_proj.weight": up,
        "mlp.down_proj.weight": down,
    }
    out = {}
    for k, t in layer.items():
        _check_bf16(f"layers.{L}.{k}", t)
        out[k] = t.to(dtype)
        layer[k] = None  # free the fp32 copy as soon as its bf16 form exists
    return out


def embedding_chunked(g, lay, dtype=torch.bfloat16, rows=16384) -> torch.Tensor:
    """The embedding (N_VOCAB x hidden) in bf16, built a block of rows at a time: the whole
    matrix in fp32 first would be twice the size it ends up."""
    from lx.hf import N_VOCAB

    e = torch.empty(N_VOCAB, lay.hidden, dtype=dtype)
    for a in range(0, N_VOCAB, rows):
        e[a:a + rows] = embedding_rows(g, lay, list(range(a, min(a + rows, N_VOCAB))),
                                       lay.hidden).to(dtype)
    return e


# --------------------------------------------------------------------------- the streamed build
PROJ_PREFIXES = ("linear_attn.in_proj_qkv.", "linear_attn.in_proj_z.", "linear_attn.in_proj_a.",
                 "linear_attn.in_proj_b.", "linear_attn.out_proj.", "mlp.gate_proj.",
                 "mlp.up_proj.", "mlp.down_proj.")


class _W:
    """A module stand-in holding one weight."""

    def __init__(self, w):
        self.weight = w


class _LayerView:
    """The parts of an emitted layer that `Fast._gdn` and `Fast._mlp` read, without a module."""

    def __init__(self, sd):
        self.in_proj_qkv = _W(sd["linear_attn.in_proj_qkv.weight"])
        self.conv1d = _W(sd["linear_attn.conv1d.weight"])
        self.in_proj_z = _W(sd["linear_attn.in_proj_z.weight"])
        self.in_proj_a = _W(sd["linear_attn.in_proj_a.weight"])
        self.in_proj_b = _W(sd["linear_attn.in_proj_b.weight"])
        self.out_proj = _W(sd["linear_attn.out_proj.weight"])
        self.A_log = sd["linear_attn.A_log"]
        self.dt_bias = sd["linear_attn.dt_bias"]
        self.gate_proj = _W(sd["mlp.gate_proj.weight"])
        self.up_proj = _W(sd["mlp.up_proj.weight"])
        self.down_proj = _W(sd["mlp.down_proj.weight"])


def _rss() -> float:
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * 4096 / 2**30
    except OSError:
        return 0.0


def stream_fast(c, lay, tc, dtype=torch.bfloat16, log=print, with_head=True, upto=None,
                with_decode=False):
    """A Fast over the chip layers, built one layer at a time: the model keeps only what Fast
    reads back later (norms, conv, A_log, dt_bias), and the projections are freed as each layer
    is packed."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from lx.hf import CODE_GAIN

    n = c.n_layers()
    cfg = Qwen3_5TextConfig(**tc)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(cfg)
    f = Fast.__new__(Fast)
    f.model, f.n_chip = model, n
    f.nk, f.nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    f.kd, f.vd = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    f.r = f.nv // f.nk
    f.layers, f.dec, f.stats, f._conv_c = [], None, None, None
    f.n_chip = n if upto is None else upto
    t0 = time.time()
    for L in range(n if upto is None else upto):
        sd = emit_layer(c, lay, tc, L, dtype=dtype)
        tgt = model.model.layers[L]
        small = {k: v for k, v in sd.items() if not any(k.startswith(s) for s in PROJ_PREFIXES)}
        missing, unexpected = tgt.load_state_dict(small, strict=False, assign=True)
        assert not unexpected, (L, unexpected[:4])
        # the dense projections never enter the model: Fast reads them once, through a shim,
        # and they are freed with it. Loading them into the module and nulling them afterwards
        # kept 0.6 GiB per layer alive (assign=True on meta parameters).
        view = _LayerView(sd)
        f.layers.append((f._gdn(view), f._mlp(view)))
        if with_decode:
            # Fast._dec reads the dense projections through the module, so they are lent to it
            # for the length of this call and dropped after: it copies what it keeps.
            ga = tgt.linear_attn
            ga.in_proj_z.weight = torch.nn.Parameter(sd["linear_attn.in_proj_z.weight"],
                                                     requires_grad=False)
            ga.out_proj.weight = torch.nn.Parameter(sd["linear_attn.out_proj.weight"],
                                                    requires_grad=False)
            f.dec = (f.dec or []) + [f._dec(L)]
            ga.in_proj_z.weight = ga.out_proj.weight = None
        for m in tgt.modules():  # as place_lean: the projections Fast holds are dropped
            for name in PROJ:
                lin = getattr(m, name, None)
                if isinstance(lin, torch.nn.Linear):
                    lin.weight = None
        del sd, small
        gc.collect()
        if L % 10 == 9 or L == n - 1:
            log(f"  layer {L + 1}/{n}  {time.time() - t0:.0f}s  rss {_rss():.2f} GiB")
    f.head_cols = f.w_head = None
    if not with_head:
        return model, f

    sd = {"model.embed_tokens.weight": embedding_chunked(c, lay, dtype)}
    norm = torch.full((lay.hidden,), -1.0)
    for d in c.keep:
        norm[lay.dim(d)] = CODE_GAIN
    sd["model.norm.weight"] = norm.to(dtype)
    for k, p in model.state_dict().items():  # the zero attention layer
        if k in sd or k.startswith("lm_head."):
            continue
        if p is None or p.device.type != "meta":
            continue
        if k.startswith(f"model.layers.{n}."):  # the zero attention layer
            sd[k] = torch.zeros(p.shape, dtype=dtype)
            continue
        if any(k.split(".", 3)[-1].startswith(s) for s in PROJ_PREFIXES):
            continue  # a projection Fast already holds, packed
        raise AssertionError(f"unexpected meta weight {k}")
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected, unexpected[:4]
    # what may still be meta afterwards: the zero attention layer's own weights, and the
    # projections Fast holds packed (never loaded into the model at all)
    left = [k for k, p in model.state_dict().items()
            if p is not None and p.device.type == "meta"]
    ok = all(k.startswith(f"model.layers.{n}.") or k.startswith("lm_head.")
             or any(k.split(".", 3)[-1].startswith(s) for s in PROJ_PREFIXES) for k in left)
    assert ok, [k for k in left if not (k.startswith(f"model.layers.{n}.")
                                        or k.startswith("lm_head.")
                                        or any(k.split(".", 3)[-1].startswith(s)
                                               for s in PROJ_PREFIXES))][:6]
    del sd
    model.tie_weights()
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    model.model.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=cfg)
    model.layout = lay
    f.head_cols = _nz_cols(model.lm_head.weight)
    f.w_head = model.lm_head.weight[:, f.head_cols].contiguous()
    return model, f


def to_gpu(f, dev):
    """`Fast.to` without the decode plan: moves the packed tensors and the head. The decode plan
    (`Fast._dec`) rebuilds per-layer tensors from the dense projections, which the streamed build
    no longer holds; building it needs a second streaming pass."""
    from dataclasses import fields

    def mv(o):
        if isinstance(o, torch.Tensor):
            return o.to(dev)
        return type(o)(mv(v) for v in o) if isinstance(o, (list, tuple)) else o

    for obj in [*(x for pair in f.layers for x in pair)]:
        for fl in fields(obj):
            setattr(obj, fl.name, mv(getattr(obj, fl.name)))
    f.head_cols, f.w_head = f.head_cols.to(dev), f.w_head.to(dev)
    for L, (g, _) in enumerate(f.layers):
        vh = (g.kh[:, None] * f.r + torch.arange(f.r, device=dev)).reshape(-1)
        g.a_neg = -f.model.model.layers[L].linear_attn.A_log.to(dev).float()[vh].exp()
    return f




# ------------------------------------------------------------------------------------- the bench
def load_elf(path, size):
    data = path.read_bytes()
    assert data[:4] == b"\x7fELF" and data[4] == 1, "not an ELF32"
    phoff, = struct.unpack_from("<I", data, 28)
    phentsize, phnum = struct.unpack_from("<HH", data, 42)
    ram, entry = bytearray(size), None
    for i in range(phnum):
        p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz = struct.unpack_from(
            "<6I", data, phoff + i * phentsize)
        if p_type != 1:
            continue
        if entry is None:
            entry, = struct.unpack_from("<I", data, 24)
        base = p_paddr - S.RAM_BASE
        assert 0 <= base and base + p_memsz <= size, hex(p_paddr)
        ram[base:base + p_filesz] = data[p_offset:p_offset + p_filesz]
    return ram, entry


def report():
    import os
    import shutil
    import subprocess

    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  {p.total_memory / 2**30:.1f} GiB, sm_{p.major}{p.minor}, {p.multi_processor_count} SMs")
    print(f"host RAM {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30:.1f} GiB, "
          f"free {shutil.disk_usage('/content').free / 2**30:.1f} GiB disk")
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                         capture_output=True, text=True).stdout.strip())


class _Shim:
    """Just what emit_layers reads of a Graph: a graph holding one layer's nodes."""

    def __init__(self, nodes, n=1):
        self.nodes, self._n = nodes, n

    def n_layers(self):
        return self._n


def _packed_bytes(o) -> int:
    n = 0
    for name in o.__dataclass_fields__:
        x = getattr(o, name)
        if isinstance(x, torch.Tensor):
            n += x.numel() * x.element_size()
        elif isinstance(x, list):
            for p in x:
                if p is not None:
                    n += sum(t.numel() * t.element_size() for t in p)
    return n


def _tc(c, lay) -> dict:
    from lx import hf

    return dict(hf.BASE, hidden_size=lay.hidden, vocab_size=hf.N_VOCAB,
                num_hidden_layers=c.n_layers() + 1,
                layer_types=["linear_attention"] * c.n_layers() + ["full_attention"],
                linear_key_head_dim=KD, linear_value_head_dim=VD,
                linear_num_key_heads=c.target.key_heads,
                linear_num_value_heads=c.target.value_heads,
                intermediate_size=c.target.intermediate, eos_token_id=S.tok("HALT"),
                pad_token_id=hf.PAD, bos_token_id=hf.PAD)


def check_weights(sample: int = 7) -> list:
    """emit_layer's dense weights against emit_layers' for sampled layers."""
    import copy

    from lx import hf
    from lx.emit import emit_layers

    c = core.build(core.Config(ram_size=RAM_SIZE, text_end=TEXT_END))
    lay = hf.pick_layout(len(c.dims))
    tc = _tc(c, lay)
    n = c.n_layers()
    print(f"circuit {c.stats()} hidden {lay.hidden}")

    bad, t0 = [], time.time()
    for L in list(range(0, n, sample)) + [n - 1]:
        nodes = [copy.copy(x) for x in c.nodes if x.level in (2 * L + 1, 2 * L + 2)]
        for x in nodes:
            x.level -= 2 * L
        want = emit_layers(_Shim(nodes), lay, tc, 0, "model.", dtype=torch.bfloat16)
        got = emit_layer(c, lay, tc, L, dtype=torch.bfloat16)
        assert {k.removeprefix("model.layers.0.") for k in want} == set(got)
        for k in got:
            if not torch.equal(want[f"model.layers.0.{k}"], got[k]):
                bad.append((L, k, int((want[f"model.layers.0.{k}"] != got[k]).sum())))
        del want, got
        gc.collect()
        print(f"  layer {L:2d}: weights {'ok' if not bad else 'MISMATCH'}  "
              f"({time.time() - t0:.0f}s)")
        if bad:
            break
    return bad


def check_packed(upto: int = 6) -> list:
    """The Fast that stream_fast packs, against Fast on emit_layers' one-layer weights."""
    import copy

    from lx import hf
    from lx.emit import emit_layers
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    c = core.build(core.Config(ram_size=RAM_SIZE, text_end=TEXT_END))
    lay = hf.pick_layout(len(c.dims))
    tc = _tc(c, lay)
    model, f = stream_fast(c, lay, tc, with_head=False, upto=upto, log=lambda s: None)
    bad = []
    for L in range(len(f.layers)):
        nodes = [copy.copy(x) for x in c.nodes if x.level in (2 * L + 1, 2 * L + 2)]
        for x in nodes:
            x.level -= 2 * L
        want = emit_layers(_Shim(nodes), lay, tc, 0, "model.", dtype=torch.bfloat16)
        with torch.device("meta"):
            shell = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**tc))
        tgt = shell.model.layers[0]
        tgt.load_state_dict({k.removeprefix("model.layers.0."): v for k, v in want.items()},
                            strict=False, assign=True)
        for got, exp, tag in ((f.layers[L][0], Fast._gdn(f, tgt.linear_attn), "gdn"),
                              (f.layers[L][1], Fast._mlp(f, tgt.mlp), "mlp")):
            for name in got.__dataclass_fields__:
                x, y = getattr(got, name), getattr(exp, name)
                if isinstance(x, torch.Tensor):
                    if not torch.equal(x, y):
                        bad.append((L, f"{tag}.{name}"))
                elif isinstance(x, list):
                    for i, (p, q) in enumerate(zip(x, y)):
                        if p is not None and not all(torch.equal(a, b) for a, b in zip(p, q)):
                            bad.append((L, f"{tag}.{name}[{i}]"))
                elif x != y:
                    bad.append((L, f"{tag}.{name}"))
        del want, tgt, shell
        gc.collect()
        print(f"  layer {L:2d}: packed {'ok' if not bad else 'MISMATCH'}, "
              f"{_packed_bytes(f.layers[L][0]) / 2**20:.1f} MiB")
    return bad


def check(sample: int = 7, phase: str = "both"):
    print("== check: per-layer emit against lx.emit.emit_layers ==")
    bad = []
    if phase in ("both", "weights"):
        bad += check_weights(sample)
    if not bad and phase in ("both", "packed"):
        bad += check_packed()
    if bad:
        print(f"MISMATCH: {bad[:10]}")
        raise SystemExit(1)
    print("EXACT: the streamed build matches the repo's emitter, tensor for tensor")




def bench(ntok: int):
    print("== bench: the real Doom circuit ==")
    report()
    t0 = time.time()
    c = core.build(core.Config(ram_size=RAM_SIZE, text_end=TEXT_END))
    from lx import hf, ref as R

    lay = hf.pick_layout(len(c.dims))
    print(f"circuit {c.stats()} hidden {lay.hidden} ({time.time() - t0:.0f}s)")

    ram, entry = load_elf(ELF, RAM_SIZE)

    def reset():
        """A machine at reset, as the snapshot needs it (not the one that proposed the trace)."""
        r = R.Machine(ram, TEXT_END)
        r.anchor, r.n = (entry - S.RAM_BASE) >> 2, 0
        return r

    tracem = reset()
    toks = [S.tok("GO")]
    t0 = time.time()
    while len(toks) < ntok and not tracem.halted:
        toks += tracem.step()
    print(f"{len(toks)} real tokens from the reference ({time.time() - t0:.1f}s, "
          f"{tracem.instret:,} instructions)")

    tc = _tc(c, lay)
    t0 = time.time()
    model, f = stream_fast(c, lay, tc)
    packed = sum(_packed_bytes(g) + _packed_bytes(ml) for g, ml in f.layers)
    print(f"streamed build {time.time() - t0:.0f}s, packed nonzeros {packed / 2**30:.2f} GiB")
    dev = "cuda"
    f.head_block = 8192
    f.lean = True
    # the published 59,236 tok/s use this: without it the conv runs as a 24,576-group conv1d
    f.compile = os.environ.get("CB_COMPILE", "1") == "1"
    t0 = time.time()
    model.to(dev)
    to_gpu(f, dev)
    print(f"to GPU {time.time() - t0:.0f}s, "
          f"allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    from lx import session as SS

    chunk = int(os.environ.get("CB_CHUNK", "16384"))
    s = SS.Session(c, model, reset(), chunk, 0, f)
    torch.cuda.synchronize()

    # verified prefill: the circuit's prediction at every position against the trace. The first
    # chunk is warm-up (fla's kernels are compiled per shape), so it is timed separately.
    pre = toks[:-1]
    n = wrong = 0
    t0 = time.time()
    r = s.verify(pre[:256])
    torch.cuda.synchronize()
    warm = time.time() - t0
    wrong += r["wrong"]
    n += 256
    if wrong:
        print("  warm-up wrong:", r["first_wrong"][:3])
    t0 = time.time()
    for a in range(256, len(pre), chunk):
        r = s.verify(pre[a:a + chunk])
        wrong += r["wrong"]
        n += len(pre[a:a + chunk])
        if r["wrong"]:
            print("  first wrong:", r["first_wrong"][:3])
            break
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"warm-up 256 tokens in {warm:.1f}s = {256 / warm:.1f} tok/s")
    tpi = len(toks) / max(tracem.instret, 1)
    print(f"prefill {n} tokens in {dt:.1f}s = {n / dt:.0f} tok/s, {wrong} wrong")
    print(f"  = {n / dt / tpi:.0f} guest instr/s  (1 instr = {tpi:.3f} tok)")

    # decode: one token at a time, the model fed its own prediction, CUDA graph
    if ntok >= 512 and f.dec is not None:
        t0 = time.time()
        k = 256
        for t in pre[n:n + k]:
            s.step(t)
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"decode {k} tokens in {dt:.1f}s = {k / dt:.0f} tok/s "
              f"= {k / dt / tpi:.0f} guest instr/s")
    print(f"peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="report",
                    choices=["check", "weights", "packed", "bench", "report"])
    ap.add_argument("n", nargs="?", type=int, default=4096)
    a = ap.parse_args()
    if a.what == "check":
        check()
    elif a.what == "weights":
        check(sample=a.n, phase="weights")
    elif a.what == "packed":
        check(phase="packed")
    elif a.what == "bench":
        bench(a.n)
    else:
        report()
