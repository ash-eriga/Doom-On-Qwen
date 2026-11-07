# Running the in-weights Doom trace

This is the operational half of doom-lx: it takes an emitted circuit plus a Doom ELF and produces
verified frames on a rented GPU. Everything here was used for the two clips in
`../STATE.md` sections 13-16.

## What is in this directory

| file | what it does |
|---|---|
| `colab_setup.py` | pins torch 2.11.0 / transformers 5.22 / fla 0.5.2 on a Colab VM, including the three fixes the stock image needs |
| `colab_run.py` | the run driver: fetch artifacts, verify their md5, resume or start fresh, stream the verifier, upload frames and checkpoint |
| `run-192x120.ipynb` | the same flow as a notebook, for driving it by hand |
| `../kaggle/run_kaggle.py` | the verifier itself (chunked prefill + strict token check); not duplicated here |
| `../colab_bench.py` | ELF loader, streaming circuit builder, GPU transfer helpers |

## Prerequisites

1. **An emitted circuit for the exact ELF.** Different `.text` bytes mean a different weight
   set, so `packed.pt` is per-build. Emit it on CPU:

       python ../kaggle/run_kaggle.py --emit --data <dir with doom-lf.elf> --lxsrc <lxsrc> --work <out>

2. **A dataset holding the artifacts.** `colab_run.py` reads them from one HF dataset repo:
   `packed-*.pt`, `doom-lf-*.elf`, `doom1.wad`, `lxsrc.zip`, `sched.txt`.
   `lxsrc.zip` is the linuxformer `lx` package plus `run_kaggle.py` and `colab_bench.py`.

3. **`HF_TOKEN` with write access** to that dataset, plus `HF_DATASET=<owner>/<dataset>`.

## Run

```bash
colab new -s doom192 -gpu A100
colab exec -s doom192 -f colab_setup.py --timeout 1700
colab exec -s doom192 -f colab_run.py --timeout 9000 \
    --env HF_TOKEN=... --env HF_DATASET=<owner>/<dataset> \
    --env SET=192 --env RESUME=0 --env HOURS=3.0 --env NTOK=95000000
```

The CLI drops the kernel after ~30 s without output, so live with `nohup` + a short poller rather
than one long `exec`.

## Budget by tokens, not by constants

These are the measured totals, not a prediction from a constant:

| build | instr/frame (`fastemu-lf`) | tokens verified, frames 6-121 | runs needed |
|---|---|---|---|
| 96x60 | 226,105 | 64.7M | 2 (6..82, then 83..121) |
| 192x120 | 459,923 | **106.1M** | 3 (95.0M, +6.0M, +5.0M) |

Instructions per frame grow 2.03x for 4x the pixels: the engine dominates and pixel work is the
minority.

The trap is tokens per instruction. The 192x120 build measures **1.606**, not the 1.5087 constant
earlier notes used. Budgeting 95M from the old constant stops at frame 119, which is exactly what
happened and why two resumes were needed. Use `--bench` on a fresh VM or read the first 1M-token
line of the run, then size `NTOK` from the real rate.

## Measured throughput

| GPU | arch | tok/s | note |
|---|---|---|---|
| T4 | sm_75 | 1,030 | bf16 has no tensor cores here; fla-bound |
| A10G | sm_86 | 10,533 | |
| L40S | sm_89 | 17,700 | 96x60 clip |
| A100 40GB | sm_80 | 19,109 | 192x120 clip, 18.75 GiB peak |

Peak memory is ~18.75 GiB, so a 24 GB card is enough; the 80 GB A100 buys nothing here.

## How the verification works

`run_kaggle.py` interleaves two things. The reference machine (`lx.ref.Machine`, the executable
spec) proposes the token stream and holds the ground RAM. The circuit verifies that stream as
chunked prefill. The machine is never allowed more than a bounded window ahead of the verified
frontier, so a checkpoint can only be saved with the machine sitting at that frontier.

A frame is written only when every token it is made of has been verified. The claim to check is
therefore byte equality with `fastemu-lf` on the same ELF and schedule, not "the model looks
plausible".

`fastemu-lf` is the gate:

```bash
cc -O3 -o fastemu-lf ../../../port/native/fastemu-lf.c -lm   # lives in the port submodule
./fastemu-lf doom-lf.elf --wad doom1.wad \
  --fbp-sym DG_ScreenBuffer --fc-sym qwen_frame_count \
  --keys-sym qwen_key_state --demo-sym qwen_demo_enabled \
  --frames 130 --script sched.txt \
  --dump-every 1 --dump-from 6 --dump-to 130 --dump-dir ref --guest-out none
```

The `--fb-w/--fb-h` passed to the runner select how many pixels are read from the framebuffer as a
linear prefix. `screenblocks=11` builds render at the full viewport width, so reading 96x60 or
192x120 from a 320x200 buffer is a prefix, not a strided crop.

## Known open item

About **4.2e-6 of predictions are wrong** (394 + 28 + 28 over 106.1 M tokens). They do not affect
the frames, which are byte-exact, but they retire any claim that the circuit predicts *every*
token. The pattern is narrow: one-value-off pairs (`65536`/`65537`, `65557`), which points at a
rounding boundary in the fast path rather than a state-space error.

## Attribution

The code here is the project's own. What it stands on is not: Doom and DoomGeneric (GPL-2.0,
© id Software / ozkl / Chocolate Doom contributors), the **LX machine from linuxformer** (MIT), and
**base Qwen3.5-4B** (Apache-2.0) as the weights the circuit is emitted into. The emit technique is
lineage from Qwenputer (MIT, Divyansh Agrawal); its checkpoint is not used here. Full chain in
`../../CREDITS.md`.

