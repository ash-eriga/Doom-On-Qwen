# What this project is built on

This repo is the Doom port. It depends on three things it did not write: a machine, a set of base
weights, and a game. Getting that dependency chain right matters more than the credits list, so
it comes first.

```
Doom-On-Qwen  (this repo, MIT)
├── the guest        Doom + DoomGeneric                        GPL-2.0
│                    ported for LX in ash-eriga/qwenputer-doom
├── the machine      linuxformer, package `lx`                 MIT
│                    LX: RV32IM + Zba. lx.ref.Machine is the
│                    executable spec; lx.fast is the kernel
├── the base weights Qwen/Qwen3.5-4B                           Apache-2.0
│                    widened layers; the circuit is emitted into these
└── the emit method  Qwenputer, by Divyansh Agrawal            MIT
                     `lx/emit.py` says it derives from
                     Qwenputer's `chipformer.q35.emit`
```

**Ours:** the Doom→LX port (`port-lf/`), the native emulator (`native/fastemu-lf.c`), the
verification and run pipeline (`ops/work/`), the measurements in `ops/work/STATE.md`, and the
clips.

**Not ours:** everything in the four rows above. This repo does not contain the Qwenputer model, its
checkpoint, or its source tree.

## What is not loaded

Easy to get wrong, so stated plainly: the Qwenputer checkpoint
(`divagr1925/Qwen3.5-4B-Qwenputer`) is never loaded, and this repo contains neither it nor its
source tree. The circuit is built by `lx.core.build` from **linuxformer** and emitted into **base
Qwen3.5-4B** layers. From Qwenputer this project takes the emit method only.

## Doom

- **Doom** — © 1993-1996 id Software. Source released under **GNU GPL v2** in 1999. Doom is a
  trademark of id Software LLC.
- **DoomGeneric** — © ozkl, <https://github.com/ozkl/doomgeneric>. **GPL-2.0**.
- **Chocolate Doom contributors** — DoomGeneric carries derived code. **GPL-2.0**.
- `doom1.wad` is the **shareware** episode, freely redistributable. It is not commercial game data.

`port-lf/build.sh` links this GPL-2.0 code. Built or redistributed binaries of that port are
governed by the GPL, not by this repo's MIT license.

## The machine

- **linuxformer** — © 2026 the linuxformer authors, **MIT**,
  <https://github.com/itsrealranky/linuxformer>. Provides `lx.core`, `lx.ref.Machine`,
  `lx.fast`, `lx.hf`, `lx.emit`. The RV32IM + Zba spec, the reference machine every frame is
  verified against, and the kernel that runs it on a GPU.

## Models

- **Qwen3.5-4B** — © Alibaba Qwen team, **Apache-2.0**,
  <https://huggingface.co/Qwen/Qwen3.5-4B>. The base weights the circuit is emitted into.
- **Qwenputer** — Divyansh Agrawal, **MIT**, <https://github.com/divagr18/qwenputer>. Source of the
  emit method; its checkpoint is not used here.

## Prior art

- **Bobliuuu/qwen-os** — a Doom demo running in-weights on a different machine. Not a dependency;
  named because `ops/work/STATE.md` §11 compares against it.

## Python and toolchain

| component | license | used for |
|---|---|---|
| `torch` (PyTorch) | BSD-3-Clause | tensor runtime and `torch.compile` |
| `transformers` (Hugging Face) | Apache-2.0 | `Qwen3_5ForCausalLM` and layout plumbing |
| `flash-linear-attention` (`fla`) | MIT | chunked Gated DeltaNet kernels, ~45% of runtime |
| `safetensors` / `huggingface_hub` | Apache-2.0 | weight and artifact transport |
| `picolibc` | BSD-2-Clause-style | RISC-V C library for the guest |
| RISC-V GNU toolchain (GCC) | GPL-3.0 + GCC runtime exception | cross-compiler for the ELF |

## Data

The per-frame key schedule (`sched.txt`) was recorded by `fastemu-lf --autopilot` playing E1M5,
not captured from a human session, and it replays deterministically. The reference frames used as
the verification oracle come from `fastemu-lf` on the same ELF and schedule, so they are
reproducible without any GPU time.

## If you publish the clips

State plainly that the pixels are read out of the reference machine's RAM while it proposes the
trace, and that the circuit is verified to predict that same token stream. The frames are **not** a
reconstruction from the model's own predictions. That distinction is in `ops/work/STATE.md` §10 and
it still holds.
