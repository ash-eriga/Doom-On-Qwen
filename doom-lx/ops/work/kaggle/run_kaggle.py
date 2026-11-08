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

# Offsets from riscv64-unknown-elf-nm, minus RAM_BASE. These belong to the 320x200 BLOCKS=3
# build; every other build moves them -- pass --fc-off/--key-off/--demo-off/--fb-off (values
# from nm on the build's ELF, minus 0x80000000). The framebuffer size follows the build too.
FC_OFF = 0x562DD0          # qwen_frame_count
KEY_OFF = 0x562DCC         # qwen_key_state (bit per key)
DEMO_OFF = 0x4684C0        # qwen_demo_enabled (0 hands control to the host)
FB_PTR_OFF = 0x4A2D94      # DG_ScreenBuffer (a pointer to the framebuffer)
FB_W, FB_H = 320, 200

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


def emit(a):
    """Stream the circuit for the ELF in --data into packed.pt (pure CPU, ~7 min). Run this on a
    cheap CPU job when the build changes; a GPU job then only loads. Same code as modal_doom's
    emit stage."""
    import torch
    for p in (Path(a.lxsrc) / "lx", Path(a.lxsrc)):
        sys.path.insert(0, str(p))
    import colab_bench as cb   # the file is colab_bench.py; "import cb" only worked on Modal,
                               # which mounted it renamed to cb.py
    cb.RAM_SIZE, cb.TEXT_END = RAM_SIZE, TEXT_END
    cb.ELF = Path(a.data) / "doom-lf.elf"
    from lx import core, hf
    t0 = time.time()
    c = core.build(core.Config(ram_size=RAM_SIZE, text_end=TEXT_END))
    lay = hf.pick_layout(len(c.dims))
    tc = cb._tc(c, lay)
    print(f"circuit {c.stats()} hidden {lay.hidden} layers {c.n_layers()} "
          f"({time.time() - t0:.0f}s)", flush=True)
    t0 = time.time()
    model, f = cb.stream_fast(c, lay, tc)
    tot = sum(cb._packed_bytes(g) + cb._packed_bytes(ml) for g, ml in f.layers)
    print(f"streamed {time.time() - t0:.0f}s, packed nonzeros {tot / 2**30:.2f} GiB", flush=True)
    blob = {
        "gdn": [{k: getattr(g, k) for k in g.__dataclass_fields__} for g, _ in f.layers],
        "mlp": [{k: getattr(m, k) for k in m.__dataclass_fields__} for _, m in f.layers],
        "head_cols": f.head_cols,
        "w_head": f.w_head,
        "sd": {k: v for k, v in model.state_dict().items()
               if v is not None and v.device.type != "meta"},
        "tc": tc,
        "meta": {"nk": f.nk, "nv": f.nv, "kd": f.kd, "vd": f.vd, "n_chip": f.n_chip},
    }
    out = Path(a.work) / "packed.pt"
    torch.save(blob, out)
    print(f"wrote {out} {out.stat().st_size / 2**30:.2f} GiB", flush=True)


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
    ap.add_argument("--emit", action="store_true",
                    help="CPU only: build the circuit from --data/doom-lf.elf into --work/packed.pt")
    ap.add_argument("--fc-off", default="0x562DD0", help="qwen_frame_count offset (nm - RAM_BASE)")
    ap.add_argument("--key-off", default="0x562DCC", help="qwen_key_state offset")
    ap.add_argument("--demo-off", default="0x4684C0", help="qwen_demo_enabled offset")
    ap.add_argument("--fb-off", default="0x4A2D94", help="DG_ScreenBuffer offset")
    ap.add_argument("--fb-w", type=int, default=320, help="framebuffer width")
    ap.add_argument("--fb-h", type=int, default=200, help="framebuffer height")
    ap.add_argument("--packed", default=None,
                    help="packed.pt to load (default DATA/packed.pt; the emit+verify flow points "
                    "this at the freshly emitted WORK/packed.pt)")
    ap.add_argument("--head-block", type=int, default=32768)
    ap.add_argument("--head-vblock", type=int, default=8192)
    a = ap.parse_args()

    if a.emit:
        emit(a)
        return

    global FC_OFF, KEY_OFF, DEMO_OFF, FB_PTR_OFF, FB_W, FB_H
    FC_OFF = int(a.fc_off, 0)
    KEY_OFF = int(a.key_off, 0)
    DEMO_OFF = int(a.demo_off, 0)
    FB_PTR_OFF = int(a.fb_off, 0)
    FB_W, FB_H = a.fb_w, a.fb_h

    data, work = Path(a.data), Path(a.work)
    outdir = Path(a.outdir or work / "weights-frames")
    elf = data / "doom-lf.elf"
    wad = data / "doom1.wad"
    packed_pt = Path(a.packed) if a.packed else data / "packed.pt"
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
    import colab_bench as cb
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
        out = bytearray(f"P6\n{FB_W} {FB_H}\n255\n".encode())
        for v in bytes(m.ram[fb:fb + FB_W * FB_H]):
            out += pal[3 * v:3 * v + 3]
        return bytes(out)

    # The proposer, interleaved with the verifier. The machine may only run a bounded window
    # (BUF tokens) ahead of the verified frontier: a session saved while the proposer is far
    # ahead captures the machine at the END of the trace with n at the frontier -- internally
    # inconsistent, and resuming from it lands in the wrong ground (v7 and v10's failure).
    # ``at`` is the token index each frame first appeared at: a frame is the model's only when
    # every one of those tokens has been verified.
    BUF = 4 * a.chunk
    pre = []
    ppm, at = {}, {}
    last = -1
    instret0 = tracem.instret
    target = s.n + a.ntok

    def propose_until(abs_target):
        """Step the machine until the buffer reaches abs_target absolute tokens (or it halts)."""
        nonlocal last
        while s.n + len(pre) < abs_target and not tracem.halted:
            fc = struct.unpack_from("<I", tracem.ram, FC_OFF)[0]
            tick(tracem, fc, last)
            last = fc
            if fc in wanted and fc not in at:
                at[fc] = s.n + len(pre)
                ppm[fc] = grab(tracem)
            pre.extend(tracem.step())

    # torch.compile compiles one kernel set per shape, so time the compiler separately.
    propose_until(s.n + BUF)
    t0 = time.time()
    seg = pre[:a.chunk]
    r = s.verify(seg)
    del pre[:a.chunk]
    torch.cuda.synchronize()
    dt0 = time.time() - t0
    first = len(seg)
    print(f"warm-up {first} tokens in {dt0:.1f}s (includes compilation), "
          f"{r['wrong']} wrong, {first / dt0:.0f} tok/s", flush=True)
    if r["wrong"]:
        print("  first wrong:", r["first_wrong"][:3], flush=True)
    if r["wrong"] > first * 0.05:
        print(f"FAILING FAST: {r['wrong']} of {first} wrong in the first chunk at token "
              f"{s.n:,} -- the state did not take", flush=True)
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

    wrong, firsts = r["wrong"], list(r["first_wrong"])
    n0 = s.n
    last_saved = s.n
    t0 = time.time()
    stop = "end"
    k = 0
    report = max(1, 1_000_000 // a.chunk)
    save_every = max(1, 2_000_000 // a.chunk)
    while s.n < target and not tracem.halted:
        if time.time() - t_start > cap:
            stop = "budget"
            print(f"  stopping at the time cap at token {s.n:,}", flush=True)
            break
        # fill the window, then verify it down to (almost) empty
        propose_until(s.n + BUF)
        while len(pre) >= a.chunk:
            seg = pre[:a.chunk]
            del pre[:a.chunk]
            r = s.verify(seg)
            wrong += r["wrong"]
            if r["wrong"]:
                firsts += r["first_wrong"]
            k += 1
            el = time.time() - t0
            if k % report == 0:
                print(f"  {s.n:,}/{target:,} verified, {(s.n - n0) / el:.0f} tok/s, "
                      f"{el:.0f}s, {wrong} wrong, {time.time() - t_start:.0f}s/{cap:.0f}s",
                      flush=True)
            if time.time() - t_start > cap:
                stop = "budget"
                print(f"  stopping at the time cap at token {s.n:,}", flush=True)
                break
        if stop != "end":
            break
        # safe checkpoint moment: the machine sits exactly at the verified frontier
        if s.n - last_saved >= 2_000_000:
            flush(s.n)
            s.save(str(resume))
            last_saved = s.n

    if stop == "end" and pre:
        r = s.verify(pre)
        del pre[:]
        wrong += r["wrong"]
        if r["wrong"]:
            firsts += r["first_wrong"]
    torch.cuda.synchronize()
    dt = time.time() - t0
    ran = s.n - n0
    rate = ran / dt if dt else 0
    tpi = (s.n - done) / max(tracem.instret - instret0, 1)
    print(f"VERIFIED to token {s.n:,} ({ran:,} this run), {wrong} wrong, stop={stop}",
          flush=True)
    print(f"  {rate:.0f} tok/s, {rate / tpi:.0f} guest instr/s, "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak", flush=True)
    print(f"  {time.time() - t_start:.0f}s wall", flush=True)
    if firsts:
        print(f"  first wrong: {firsts[:6]}", flush=True)

    flush(s.n)
    s.save(str(resume))
    print(f"saved state at token {s.n:,}; -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
