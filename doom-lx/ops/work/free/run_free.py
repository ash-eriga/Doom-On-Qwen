#!/usr/bin/env python3
"""Free-run: the model's own predictions become its next input.

Verified mode feeds the reference machine's tokens (teacher forcing): a wrong prediction is
discarded, so the state stays on rails and any error rate is survivable. Here, after a short
burn-in, every input chunk is the model's own previous chunk of predictions — block-wise
autoregression at chunked speed. The proposer machine (lx.ref.Machine) still runs alongside,
but only to time the schedule's KEY tokens (the keyboard is host input) and to log
mismatches; the model never consumes its tokens. Wrong predictions now matter: one wrong
token enters the state, so a clean run needs the head's argmax to never flip.

Frames are read out of the Gated DeltaNet state (readout.py) and md5-checked against the
oracle manifest. With zero wrong predictions this is the same gameplay, produced entirely
inside the model: the guest's schedule is baked in, the host feeds GO and nothing else.

    python run_free.py --data <packed.pt, doom-lf.elf, doom1.wad, sched.txt, oracle md5> \
        --lxsrc <lxsrc> --work <out> [--fp32-head] [--resume]
"""
import argparse
import hashlib
import os
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths; paths.add_lx()
import colab_bench as cb


def payload_md5(ppm: bytes) -> str:
    """md5 of the PPM pixel payload (everything after the header), like check_frames.py."""
    pos, fields = 0, []
    while len(fields) < 4:
        while ppm[pos:pos + 1].isspace():
            pos += 1
        end = pos
        while not ppm[end:end + 1].isspace():
            end += 1
        fields.append(ppm[pos:end])
        pos = end
    return hashlib.md5(ppm[pos + 1:]).hexdigest()


def load_manifest(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split()
            out[name] = digest
    return out


def build_fast(blob):
    from lx.fast import Fast, _GDN, _MLP
    f = Fast.__new__(Fast)
    f.n_chip = blob["meta"]["n_chip"]
    f.nk, f.nv = blob["meta"]["nk"], blob["meta"]["nv"]
    f.kd, f.vd = blob["meta"]["kd"], blob["meta"]["vd"]
    f.r = f.nv // f.nk
    f.head_cols, f.w_head = blob["head_cols"], blob["w_head"]
    f.layers = [(_GDN(**g), _MLP(**m)) for g, m in zip(blob["gdn"], blob["mlp"])]
    f.dec, f.stats, f._conv_c, f._steps = None, None, None, None
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input/doom-lx")
    ap.add_argument("--lxsrc", default="/kaggle/working/lxsrc")
    ap.add_argument("--work", default="/kaggle/working")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--ntok", type=int, default=110_000_000)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--dump", default="6-121")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fp32-head", action="store_true",
                    help="compute the output head in fp32 (kills one-value-off argmax flips)")
    ap.add_argument("--manifest", default=None, help="oracle md5 manifest")
    ap.add_argument("--fc-off", default="0x562DD0")
    ap.add_argument("--key-off", default="0x562DCC")
    ap.add_argument("--demo-off", default="0x4684C0")
    ap.add_argument("--fb-off", default="0x4A2D94")
    ap.add_argument("--fb-w", type=int, default=192)
    ap.add_argument("--fb-h", type=int, default=120)
    ap.add_argument("--packed", default=None)
    a = ap.parse_args()

    FC_OFF, KEY_OFF, DEMO_OFF, FB_PTR_OFF = (int(a.fc_off, 0), int(a.key_off, 0),
                                             int(a.demo_off, 0), int(a.fb_off, 0))
    FB_W, FB_H = a.fb_w, a.fb_h
    RAM_SIZE, TEXT_END = 0x600000, 0x40000

    data, work = Path(a.data), Path(a.work)
    outdir = Path(a.outdir or work / "weights-frames")
    elf, wad = data / "doom-lf.elf", data / "doom1.wad"
    packed_pt = Path(a.packed) if a.packed else data / "packed.pt"
    ckpt = work / "free-session.pt"
    manifest = load_manifest(Path(a.manifest) if a.manifest else data / "oracle-192x120.md5")

    sys.path.insert(0, str(Path(a.lxsrc) / "lx"))
    for cand in (Path(a.lxsrc), Path(a.lxsrc).parent):
        if (cand / "colab_bench.py").exists():
            sys.path.insert(0, str(cand))
            break
    import colab_bench as _cb
    _cb.RAM_SIZE, _cb.TEXT_END = RAM_SIZE, TEXT_END
    _cb.ELF = elf

    import torch
    t_start = time.time()
    cap = a.hours * 3600.0
    maj, mi = torch.cuda.get_device_capability()
    os.environ["LX_ARCH"] = f"sm_{maj}{mi}a" if maj == 9 else f"sm_{maj}{mi}"

    from lx import core, hf, session as SS, spec as S
    from lx.fast import _Norm
    from lx.ref import Machine
    from readout import Readout
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    props = torch.cuda.get_device_properties(0)
    print(f"GPU {torch.cuda.get_device_name(0)} sm_{props.major}{props.minor} "
          f"mem={props.total_memory / 2**30:.1f} GiB", flush=True)

    c = core.build(core.Config(ram_size=RAM_SIZE, text_end=TEXT_END))
    ram, entry = _cb.load_elf(elf, RAM_SIZE)
    m0 = Machine(ram, TEXT_END)
    m0.anchor, m0.n = (entry - S.RAM_BASE) >> 2, 0

    blob = torch.load(packed_pt, map_location="cpu", weights_only=False)
    cfg = Qwen3_5TextConfig(**blob["tc"])
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(cfg)
    model.load_state_dict(blob["sd"], strict=False, assign=True)
    proj = tuple(_cb.PROJ_PREFIXES)
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        if p is None or p.device.type != "meta":
            continue
        parts = name.split(".")
        obj = model
        for attr in parts[:-1]:
            obj = getattr(obj, attr)
        if any(name.split(".", 3)[-1].startswith(x) for x in proj):
            setattr(obj, parts[-1], torch.nn.Parameter(torch.empty(0), requires_grad=False)
                    if isinstance(p, torch.nn.Parameter) else torch.empty(0))
            continue
        z = torch.zeros(p.shape, dtype=p.dtype)
        setattr(obj, parts[-1], torch.nn.Parameter(z, requires_grad=False)
                if isinstance(p, torch.nn.Parameter) else z)
    model.model.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=cfg)
    model.layout = hf.pick_layout(len(c.dims))
    model.tie_weights()

    f = build_fast(blob)
    f.model = model
    f.head_block, f.lean, f.compile = 8192, True, True
    f.head_vblock = 8192
    model.to("cuda")
    _cb.to_gpu(f, "cuda")
    if a.fp32_head:
        w32 = f.w_head.float()
        norm, cols = model.model.norm, f.head_cols

        def head_argmax32(h, block=None):
            out, block = [], block or f.head_block
            for i in range(0, h.shape[0], block):
                hn = _Norm(norm, h[i:i + block])(cols)
                out.append((hn.float() @ w32.T).argmax(-1))
            return torch.cat(out)
        f.head_argmax = head_argmax32
        print(f"fp32 head on ({w32.numel() * 4 / 2**30:.1f} GiB)", flush=True)

    if a.resume and ckpt.exists():
        d = torch.load(ckpt, weights_only=False)
        s = SS.Session(c, model, None, a.chunk, 0, f)
        s._set_state([(cs.to(s.dev), rs.to(s.dev)) for cs, rs in d["state"]])
        s.next, s.last, s.n, s.waiting = d["next"], d["last"], d["n"], d["waiting"]
        s.ref = d["ref"]
        pre, ppm, at, last_fc, saved = d["pre"], d["ppm"], d["at"], d["lastfc"], d["saved"]
        inp = d["inp"]
        print(f"resumed at token {s.n:,}", flush=True)
    else:
        if a.fresh and ckpt.exists():
            ckpt.unlink()
        s = SS.Session(c, model, m0, a.chunk, 0, f)
        pre, ppm, at, last_fc, saved = [], {}, {}, -1, set()
        inp = None
    tracem = s.ref
    ro = Readout(c, s.cache, RK_palette(str(wad)), FB_PTR_OFF, FB_W, FB_H)
    GO = S.tok("GO")

    # the baked schedule drives the guest; the host feeds nothing after GO
    wanted = set()
    for part in a.dump.replace(" ", "").split(","):
        if "-" in part[1:]:
            x, y = part.split("-", 1)
            wanted.update(range(int(x), int(y) + 1))
        else:
            wanted.add(int(part))

    BUF = 4 * a.chunk
    instret0 = tracem.instret
    n0 = s.n
    target = s.n + a.ntok

    def propose_until(abs_target):
        """Reference tokens up to abs_target — mismatch logging only (the guest drives
        itself through the baked schedule; the runner feeds the model nothing but GO)."""
        while s.n + len(pre) < abs_target and not tracem.halted:
            fc = struct.unpack_from("<I", tracem.ram, FC_OFF)[0]
            if fc != last_fc:
                if fc in wanted and fc not in at:
                    at[fc] = s.n + len(pre)
                last_fc = fc
            pre.extend(tracem.step())

    def step_chunk(inp):
        """One chunked pass on chip-token ids; returns the next predictions (list)."""
        e = model.model.embed_tokens(torch.tensor([inp], device="cuda") + s.offset)
        return f.head_argmax(f.forward(e, s.cache)[0]).tolist()

    def save_all():
        torch.save({"state": [(cs.cpu(), rs.cpu()) for cs, rs in s._state()],
                    "next": s.next, "last": s.last, "n": s.n, "waiting": s.waiting,
                    "ref": tracem, "pre": pre, "at": at, "ppm": ppm, "lastfc": last_fc,
                    "saved": saved, "inp": inp}, ckpt)

    def frames_from_state():
        """The model's current frame, read out of the GDN state, md5'd against the oracle."""
        (fc,) = ro._words(torch.tensor([FC_OFF >> 2], dtype=torch.long)).tolist()
        if fc not in wanted or fc in saved:
            return
        outdir.mkdir(parents=True, exist_ok=True)
        got = ro.frame()
        d = payload_md5(got)
        want = manifest.get(f"frame-{fc:04d}.ppm")
        (outdir / f"frame-{fc:04d}.ppm").write_bytes(got)
        saved.add(fc)
        print(f"  frame {fc} from state: "
              f"{'MATCH' if d == want else 'MISMATCH ' + d}", flush=True)

    inp = inp or [GO]
    wrong, firsts = 0, []
    t0 = time.time()
    free_since = None
    last_ckpt = s.n
    while s.n < target and not tracem.halted:
        if time.time() - t_start > cap:
            print("  stopping at the time cap", flush=True)
            break
        propose_until(s.n + BUF + len(inp))
        preds = step_chunk(inp)
        for i in range(len(inp)):  # diagnostics: the model ran on its own predictions
            j = s.n + i + 1
            if j < len(pre) and preds[i] != pre[j]:
                wrong += 1
                if len(firsts) < 8:
                    firsts.append((j, pre[j], preds[i]))
        s.n += len(inp)
        s.next = preds[-1]
        del pre[:len(inp)]
        if free_since is None and any(v <= s.n for v in at.values()):
            free_since = s.n
            print(f"free-run from token {s.n:,} ({time.time() - t0:.0f}s in)", flush=True)
        inp = preds
        frames_from_state()
        if wanted and saved >= wanted:
            print("all wanted frames are in; stopping", flush=True)
            break
        el = time.time() - t0
        if (s.n - n0) // a.chunk % 8 == 0:
            print(f"  {s.n:,}/{target:,} fed, {(s.n - n0) / el:.0f} tok/s, {wrong} wrong, "
                  f"{time.time() - t_start:.0f}s/{cap:.0f}s", flush=True)
        if s.n - last_ckpt >= 8_000_000:
            save_all()
            last_ckpt = s.n
        if wrong > 1000:
            print("FAILING: the trajectory diverged; stopping to save compute", flush=True)
            break

    frames_from_state()
    save_all()
    torch.cuda.synchronize()
    el = time.time() - t0
    print(f"DONE at token {s.n:,} ({s.n - n0:,} this run): {wrong} wrong, "
          f"{(s.n - n0) / el:.0f} tok/s, "
          f"{(s.n - n0) / max(tracem.instret - instret0, 1):.3f} tok/instr, "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak, {el:.0f}s",
          flush=True)
    print(f"free since token {free_since}; first wrong: {firsts[:8]}", flush=True)
    print(f"frames from state: {sorted(saved)}", flush=True)


def RK_palette(wad_path):
    """The Doom palette as run_kaggle's _palette builds it (import without running main)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_kaggle", Path(__file__).resolve().parents[1] / "kaggle" / "run_kaggle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._palette(wad_path)


if __name__ == "__main__":
    main()
