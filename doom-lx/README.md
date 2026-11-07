# doom-lx — technical detail

The Doom engine on the LX machine, verified token by token. The project overview is in the
[top-level README](../README.md); this page is the part you need to rerun or audit it.

The machine is **linuxformer**'s LX (`lx.ref.Machine`, `lx.fast`), the weights are **base
Qwen3.5-4B** widened by `lx.core.build`, and the guest is **DoomGeneric** built for LX. See
[`CREDITS.md`](CREDITS.md).

| folder | what is in it |
|---|---|
| `port/` | the DoomGeneric port for LX (submodule, own history): `port-lf/build.sh`, the native emulator `native/fastemu-lf.c`, the speed patch |
| `ops/work/` | operation: `STATE.md` (the full ledger), `runs/` (GPU drivers + notebook), `kaggle/` (verifier and packaging), `preview/` (clips, contact sheets) |
| `ops/work/RUNLOG-*.md` | per-run reports, L40S and A100 |

## Verification

`run_kaggle.py` interleaves a proposer and a checker. `lx.ref.Machine` proposes the token stream
and holds the ground RAM; the circuit verifies that stream as chunked prefill. The machine is never
allowed more than a bounded window ahead of the verified frontier, so a checkpoint can only be
saved with both at the same instruction.

A frame is written only when every token composing it has verified. The oracle is `fastemu-lf`, a
native C emulator rendering from the same ELF and the same schedule:

```bash
cc -O3 -o fastemu-lf port/native/fastemu-lf.c -lm
./fastemu-lf doom-lf.elf --wad doom1.wad \
  --fbp-sym DG_ScreenBuffer --fc-sym qwen_frame_count \
  --keys-sym qwen_key_state --demo-sym qwen_demo_enabled \
  --frames 130 --script sched.txt \
  --dump-every 1 --dump-from 6 --dump-to 121 --dump-dir ref --guest-out none
```

Frames compare byte for byte on the PPM payload. All 116 in both clips are exact.

## Framebuffer geometry

`screenblocks=11` builds render the full viewport width, so a 96x60 or 192x120 frame is the **first
5,760 or 23,040 pixels of the 320x200 buffer**, read as a linear prefix. It is not a strided crop.
Reading it with a 96-wide stride produces a flattened band, which is a mistake already made once
here.

## Budget by tokens, not by constants

| build | instr/frame (`fastemu-lf`) | tokens verified, frames 6-121 | runs needed |
|---|---|---|---|
| 96x60 | 226,105 | 64.7M | 2 (6..82, then 83..121) |
| 192x120 | 459,923 | **106.1M** | 3 (95.0M, +6.0M, +5.0M) |

Instructions per frame grow 2.03x for 4x the pixels: the engine dominates, and pixel work is the
minority.

The trap is tokens per instruction. 192x120 measures **1.606**, not the 1.5087 constant earlier
notes used. Budgeting 95M from the old constant stops at frame 119 — which is what happened, and
why two resumes were needed.

## Measured throughput

| GPU | arch | tok/s | note |
|---|---|---|---|
| T4 | sm_75 | 1,030 | no bf16 tensor cores; fla-bound |
| A10G | sm_86 | 10,533 | |
| L40S | sm_89 | 17,700 | 96x60 clip |
| A100 40GB | sm_80 | 19,109 | 192x120 clip, 18.75 GiB peak |

Peak memory is ~18.75 GiB, so a 24 GB card is enough. An 80 GB A100 buys nothing here.

## Open defect

About 4.2e-6 of predictions are wrong at 192x120, and about 1.6e-5 at 96x60. They do not affect
the frames, which are oracle-exact, but they retire any claim that the circuit predicts *every*
token. The pattern is narrow: one-value-off pairs (`65536`/`65537`, `65557`), pointing at a
rounding boundary in the fast path rather than a state-space error.

Separately, a `lightlevel` bug in the guest (§12 of the ledger) keeps the player in the line of
fire, so the demo dies early. Nothing below that is blocked on the circuit.

## Reproducing

[`README.md`](../README.md#reproducing-it) has the commands,
[`ops/work/runs/README.md`](ops/work/runs/README.md) the numbers and the Colab `nohup` requirement,
and [`ops/work/runs/run-192x120.ipynb`](ops/work/runs/run-192x120.ipynb) the notebook version.
