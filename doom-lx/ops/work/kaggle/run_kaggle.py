#!/usr/bin/env python3
"""The drive stage of modal_doom.py, without Modal: for a Kaggle GPU notebook (or any
single machine with the files).

Same logic, same prints, same resumable session.pt -- a run started on Modal continues here and
the other way round, because packed.pt and session.pt are the same formats.

Inputs (paths via flags, defaults match the Kaggle dataset layout):
    packed.pt   the streamed build (4.08 GiB, from the Modal Volume)
    session.pt  optional resume point (token 3,403,776 of the boot, 0 wrong)
    doom-lf.elf + doom1.wad  the game and its WAD (the proposer runs on CPU)
    lxsrc/      the linuxformer checkout (cb.py and the lx package)

    python run_kaggle.py --data /kaggle/input/doom-lx --lxsrc /kaggle/working/lxsrc \
        --ntok 41000000 --hours 2.5 --dump 1-50 --play "0 0;1 9;3 25;..."

Frames land in --outdir as frame-NNNN.ppm as they are verified; the session checkpoint is written
next to session.pt every ~2M tokens.
"""
import argparse
import os
import struct
import sys
import time
from pathlib import Path

RAM_SIZE = 0x600000        # must match port-lf/doom-lf.ld
TEXT_END = 0x40000

# Offsets from riscv64-unknown-elf-nm, minus RAM_BASE.
FC_OFF = 0x562DD0          # qwen_frame_count
KEY_OFF = 0x562DCC         # qwen_key_state (bit per key)
DEMO_OFF = 0x4684C0        # qwen_demo_enabled (0 hands control to the host)
FB_PTR_OFF = 0x4A2D94      # DG_ScreenBuffer (a pointer to the framebuffer)

PLAY_E1M5 = ("0 0;1 9;3 25;5 17;36 25;37 17;45 25;46 17;52 25;53 17;58 25;60 17;67 25;68 17;"
             "70 25;71 17;72 25;73 17;74 24;76 16")


def _palette(wad_path):
    b = Path(wad_path).read_bytes()
    n, d = struct.unpack_from("<II", b, 4)
    for i in range(n):
        off, size, name = struct.unpack_from("<II8s", b, d + 16 * i)
        if name.rstrip(b"\0") == b"PLAYPAL":
            return b[off:off + 768]
    raise SystemExit("PLAYPAL not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input/doom-lx", help="dir with packed.pt, "
                    "session.pt, doom-lf.elf, doom1.wad")
    ap.add_argument("--lxsrc", default="/kaggle/working/lxsrc", help="unzipped linuxformer repo")
    ap.add_argument("--work", default="/kaggle/working", help="writable dir for the checkpoint")
    ap.add_argument("--outdir", default=None, help="frames dir (default WORK/weights-frames)")
    ap.add_argument("--ntok", type=int, default=41_000_000)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--hours", type=float, default=2.5, help="wall clock cap for this run")
    ap.add_argument("--dump", default="1-50", help="frame numbers/ranges to capture")
    ap.add_argument("--play", default=PLAY_E1M5, help='"frame key" schedule; frame = rendered no.')
    ap.add_argument("--fresh", action="store_true", help="ignore/overwrite session.pt")
    ap.add_argument("--bench", action="store_true",
                    help="run only the warm-up chunk, print the honest tok/s, exit")
    ap.add_argument("--budget-dollars", type=float, default=0.0,
                    help="with --price-per-hour: trim the trace so the run fits this budget")
    ap.add_argument("--price-per-hour", type=float, default=0.0)
    ap.add_argument("--head-block", type=int, default=32768)
    ap.add_argument("--head-vblock", type=int, default=8192)
    a = ap.parse_args()

    data, work = Path(a.data), Path(a.work)
    outdir = Path(a.outdir or work / "weights-frames")
    elf = data / "doom-lf.elf"
    wad = data / "doom1.wad"
    packed_pt = data / "packed.pt"
    resume = work / "session.pt"
    for p in (elf, wad, packed_pt):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    if (data / "session.pt").exists() and not resume.exists():
        # the dataset is read-only on Kaggle: start from the uploaded checkpoint
        import shutil
        shutil.copy(data / "session.pt", resume)

    sys.path.insert(0, str(Path(a.lxsrc) / "lx"))
    sys.path.insert(0, str(Path(a.lxsrc) / "lxsrc" / "lx"))
    for cand in (Path(a.lxsrc), Path(a.lxsrc).parent):
        cbp = cand / "colab_bench.py"
        if cbp.exists():
            sys.path.insert(0, str(cand))
            break
    import cb
    cb.RAM_SIZE, cb.TEXT_END = RAM_SIZE, TEXT_END
    cb.ELF = elf

    import torch
    t_start = time.time()
    cap = a.hours * 3600.0

    maj, mi = torch.cuda.get_device_capability()
    os.environ["LX_ARCH"] = f"sm_{maj}{mi}a" if maj == 9 else f"sm_{maj}{mi}"

    from lx import core, hf, spec as S
    from lx import session as SS
    from lx.fast import Fast, _GDN, _MLP
    from lx.ref import Machine
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    props = torch.cuda.get_device_properties(0)
    print(f"GPU {torch.cuda.get_device_name(0)} sm_{props.major}{props.minor} "
          f"SMs={props.multi_processor_count} mem={props.total_memory / 2**30:.1f} GiB "
          f"arch={os.environ['LX_ARCH']}", flush=True)
    print(f"cap {a.hours:.2f}h = {cap:.0f}s of wall time", flush=True)

    c = core.build(core.Config(ram_size=RAM_SIZE, text_end=TEXT_END))
    lay = hf.pick_layout(len(c.dims))
    ram, entry = cb.load_elf(cb.ELF, RAM_SIZE)

    def reset():
        r = Machine(ram, TEXT_END)
        r.anchor, r.n = (entry - S.RAM_BASE) >> 2, 0
        return r

    sched = []
    for part in a.play.split(";"):
        part = part.strip()
        if part:
            f, m = part.split()
            sched.append((int(f), int(m, 0)))
    sched.sort()

    def tick(m, fc, last):
        """Keys at a frame boundary; this is a human at the keyboard."""
        if fc == last or not sched:
            return
        mask = 0
        for f, k in sched:
            if f <= fc:
                mask = k
        struct.pack_into("<I", m.ram, DEMO_OFF, 0)
        struct.pack_into("<I", m.ram, KEY_OFF, mask)

    t0 = time.time()
    blob = torch.load(packed_pt, map_location="cpu", weights_only=False)
    cfg = Qwen3_5TextConfig(**blob["tc"])
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(cfg)
    model.load_state_dict(blob["sd"], strict=False, assign=True)
    # Whatever is still on the meta device was never given numbers by the streamed build. Two
    # very different cases, and confusing them costs 37 GiB:
    #   * the projections Fast holds packed -- the module's copy is unused, so it is dropped
    #     rather than filled (filling would allocate the whole dense model);
    #   * the zero attention layer's own weights, which are zeros by construction.
    proj = tuple(cb.PROJ_PREFIXES)
    n_zero = n_drop = 0
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        if p is None or p.device.type != "meta":
            continue
        parts = name.split(".")
        obj = model
        for attr in parts[:-1]:
            obj = getattr(obj, attr)
        if any(name.split(".", 3)[-1].startswith(x) for x in proj):
            setattr(obj, parts[-1],
                    torch.nn.Parameter(torch.empty(0), requires_grad=False)
                    if isinstance(p, torch.nn.Parameter) else torch.empty(0))
            n_drop += 1
            continue
        z = torch.zeros(p.shape, dtype=p.dtype)
        if isinstance(p, torch.nn.Parameter):
            setattr(obj, parts[-1], torch.nn.Parameter(z, requires_grad=False))
        else:
            setattr(obj, parts[-1], z)
        n_zero += 1
    print(f"meta tensors: {n_zero} zeroed, {n_drop} dropped (Fast holds them packed)", flush=True)
    model.model.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=cfg)
    model.layout = lay
    model.tie_weights()

    f = Fast.__new__(Fast)
    f.model = model
    f.n_chip = blob["meta"]["n_chip"]
    f.nk, f.nv = blob["meta"]["nk"], blob["meta"]["nv"]
    f.kd, f.vd = blob["meta"]["kd"], blob["meta"]["vd"]
    f.r = f.nv // f.nk
    f.head_cols, f.w_head = blob["head_cols"], blob["w_head"]
    f.layers = [(_GDN(**g), _MLP(**m)) for g, m in zip(blob["gdn"], blob["mlp"])]
    f.dec, f.stats, f._conv_c = None, None, None
    print(f"loaded packed build {time.time() - t0:.0f}s", flush=True)

    f.head_block, f.lean, f.compile = a.head_block, True, True
    f.head_vblock = a.head_vblock
    t0 = time.time()
    model.to("cuda")
    cb.to_gpu(f, "cuda")
    print(f"to GPU {time.time() - t0:.0f}s, allocated "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
          f"{time.time() - t_start:.0f}s elapsed", flush=True)

    if a.fresh and resume.exists():
        resume.unlink()
    t0 = time.time()
    if resume.exists():
        s = SS.Session.load(c, model, str(resume), a.chunk, 0, f)
        done = s.n
        print(f"resumed at token {done:,} ({time.time() - t0:.0f}s)", flush=True)
    else:
        s, done = SS.Session(c, model, reset(), a.chunk, 0, f), 0
        print(f"fresh session ({time.time() - t0:.0f}s)", flush=True)
    torch.cuda.synchronize()
    tracem = s.ref

    pal = _palette(wad)
    wanted = set()
    for part in a.dump.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part[1:]:
            x, y = part.split("-", 1)
            wanted.update(range(int(x), int(y) + 1))
        else:
            wanted.add(int(part))

    def grab(m):
        """The framebuffer as a PPM, following the pointer the guest set up."""
        fb = struct.unpack_from("<I", m.ram, FB_PTR_OFF)[0] - S.RAM_BASE
        out = bytearray(b"P6\n320 200\n255\n")
        for v in bytes(m.ram[fb:fb + 320 * 200]):
            out += pal[3 * v:3 * v + 3]
        return bytes(out)

    # The proposer: tokens from where the model's state ends, and the pixels they encode.
    # ``at`` is the token index each frame first appeared at: a frame is the model's only when
    # every one of those tokens has been verified.
    t0 = time.time()
    toks, ppm, at, last = [], {}, {}, -1
    while len(toks) < a.ntok and not tracem.halted:
        fc = struct.unpack_from("<I", tracem.ram, FC_OFF)[0]
        tick(tracem, fc, last)
        last = fc
        if fc in wanted and fc not in at:
            at[fc] = done + len(toks)
            ppm[fc] = grab(tracem)
        toks.extend(tracem.step())
    tpi = (done + len(toks)) / max(tracem.instret, 1)
    print(f"{len(toks):,} new tokens proposed in {time.time() - t0:.1f}s "
          f"(machine at {tracem.instret:,} instructions, {tpi:.3f} tok/instr), "
          f"frames {min(at) if at else '-'}..{max(at) if at else '-'}", flush=True)

    torch.cuda.synchronize()
    pre = toks

    # torch.compile compiles one kernel set per shape, so time the compiler separately.
    t0 = time.time()
    r = s.verify(pre[:a.chunk])
    torch.cuda.synchronize()
    dt0 = time.time() - t0
    first = min(a.chunk, len(pre))
    print(f"warm-up {first} tokens in {dt0:.1f}s (includes compilation), "
          f"{r['wrong']} wrong, {first / dt0:.0f} tok/s", flush=True)
    if r["wrong"]:
        print("  first wrong:", r["first_wrong"][:3], flush=True)
    if r["wrong"] > first * 0.05:
        print(f"FAILING FAST: {r['wrong']} of {first} wrong in the first chunk at token "
              f"{done:,} -- the state did not take", flush=True)
        return
    if a.bench:
        # Measure-and-stop: the warm-up above already includes compilation, so this rate is
        # the honest planning number. No checkpoint, no frames, nothing written.
        print(f"BENCH: {first / dt0:.0f} tok/s over {first} tokens on "
              f"{torch.cuda.get_device_name(0)} (chunk {a.chunk}, head_block {f.head_block}, "
              f"head_vblock {f.head_vblock}); run cost of 43.6M tokens at this rate: "
              f"{43_600_000 / (first / dt0) / 3600:.1f} h. Nothing saved.", flush=True)
        return

    def flush(upto):
        """Frames whose every token is verified, written into outdir."""
        outdir.mkdir(parents=True, exist_ok=True)
        saved = []
        for fc, blob_ppm in ppm.items():
            if at[fc] <= upto and not (outdir / f"frame-{fc:04d}.ppm").exists():
                (outdir / f"frame-{fc:04d}.ppm").write_bytes(blob_ppm)
                saved.append(fc)
        return saved

    n, wrong, firsts = first, r["wrong"], list(r["first_wrong"])
    t0 = time.time()
    stop = "end"
    k = 0
    report = max(1, 1_000_000 // a.chunk)
    save_every = max(1, 2_000_000 // a.chunk)
    while a.chunk + (k + 1) * a.chunk <= len(pre):
        seg = pre[a.chunk + k * a.chunk:a.chunk + (k + 1) * a.chunk]
        r = s.verify(seg)
        wrong += r["wrong"]
        n += len(seg)
        if r["wrong"]:
            firsts += r["first_wrong"]
        k += 1
        el = time.time() - t0
        if k == 1 and a.budget_dollars and a.price_per_hour:
            # First steady chunk = the honest rate (the warm-up above spent its time compiling).
            # Whatever does not fit the budget is cut here, on chunk boundaries, and the frames
            # beyond the cut are dropped, so every frame promised is a frame fully verified.
            rate1 = len(seg) / max(el, 1e-9)
            remain = a.budget_dollars * 3600.0 / a.price_per_hour - (time.time() - t_start)
            afford = max(0, int(rate1 * remain * 0.9))
            cut = a.chunk + (afford // a.chunk) * a.chunk
            if cut < len(pre):
                for fc in [fc for fc in ppm if at[fc] > cut]:
                    del ppm[fc], at[fc]
                print(f"budget: {rate1:.0f} tok/s steady, {remain:.0f}s left -> cutting to "
                      f"{cut:,} tokens, {len(ppm)} frames kept", flush=True)
                pre = pre[:cut]
        if k % report == 0:
            print(f"  {done + n:,}/{done + len(pre):,} verified, {(n - first) / el:.0f} tok/s, "
                  f"{el:.0f}s, {wrong} wrong, {time.time() - t_start:.0f}s/{cap:.0f}s",
                  flush=True)
        if k % save_every == 0:
            flush(done + n)
            s.save(str(resume))
        if time.time() - t_start > cap:
            stop = "budget"
            print(f"  stopping at the time cap with {n:,} of {len(pre):,} tokens", flush=True)
            break
    if stop == "end" and n < len(pre):
        tail = pre[n:]
        r = s.verify(tail)
        wrong += r["wrong"]
        n += len(tail)
        if r["wrong"]:
            firsts += r["first_wrong"]
    torch.cuda.synchronize()
    dt = time.time() - t0
    ran = n - first
    rate = ran / dt if dt else 0
    print(f"VERIFIED to token {done + n:,} ({ran:,} this run), {wrong} wrong, stop={stop}",
          flush=True)
    print(f"  {rate:.0f} tok/s, {rate / tpi:.0f} guest instr/s, "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak", flush=True)
    print(f"  {time.time() - t_start:.0f}s wall", flush=True)
    if firsts:
        print(f"  first wrong: {firsts[:6]}", flush=True)

    s.save(str(resume))
    saved = flush(done + n)
    print(f"saved state at token {done + n:,}; {len(saved)} frames written this run -> {outdir}",
          flush=True)


if __name__ == "__main__":
    main()
