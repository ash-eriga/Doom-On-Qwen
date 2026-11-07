# Credits and attributions — doom-lx

Everything in this directory builds on other people's work. This file says what, under which
license, and what is ours.

## Ours

The Doom→LX port (`port-lf/`), the native emulator (`native/fastemu-lf.c`), the operational
pipeline (`ops/work/`), the run drivers (`ops/work/runs/`), and the measurements in `STATE.md` are
this project's own work. Repository code is MIT (see `../../LICENSE`), © Divyansh Agrawal.

## Doom

- **Doom** — © 1993-1996 id Software. The source is distributed under the **GNU GPL v2**
  (the 1999 source release). Doom is a trademark of id Software LLC.
- **doomgeneric** — © ozkl, the platform-abstraction port this builds on:
  <https://github.com/ozkl/doomgeneric>. **GPL-2.0**.
- **Chocolate Doom contributors** — doomgeneric carries code derived from their work. **GPL-2.0**.
- `doom1.wad` is the **shareware** episode and is freely redistributable; it is not the commercial
  game data.

`port-lf/build.sh` links this GPL-2.0 code. Building or redistributing the resulting ELF is
subject to the GPL, not to this repository's MIT license. Keep that in mind before publishing a
binary.

## Models

- **Qwen3.5-4B** — © Alibaba Qwen team, **Apache-2.0**.
- **Qwenputer weights** — `divagr1925/Qwen3.5-4B-Qwenputer` and its siblings on Hugging Face and
  ModelScope, by Divyansh Agrawal. The circuit emitted in `STATE.md` is derived from these
  weights.
- **linuxformer / LX machine** — `itsrealranky/linuxformer`, the reference machine (`lx.ref`) and
  the fast kernel layer used to verify the circuit.

## Python and toolchain

| component | license | used for |
|---|---|---|
| `torch` (PyTorch) | BSD-3-Clause | tensor runtime and `torch.compile` |
| `transformers` (Hugging Face) | Apache-2.0 | `Qwen3_5ForCausalLM` and the layout plumbing |
| `flash-linear-attention` (`fla`) | MIT | the chunked Gated DeltaNet kernels, ~45% of runtime |
| `safetensors` / `huggingface_hub` | Apache-2.0 | weight and artifact transport |
| `picolibc` | BSD-2-Clause-style | RISC-V C library for the guest |
| RISC-V GNU toolchain (GCC) | GPL-3.0 + GCC runtime exception | cross-compiler for the ELF |

## Data and schedules

The per-frame key schedule (`sched.txt`) was recorded by `fastemu-lf --autopilot` playing E1M5,
not captured from a human session, and it replays deterministically. The reference frames used as
the verification oracle are produced by `fastemu-lf` from the same ELF and schedule, so they are
reproducible without any of the rented GPU time.

## If you publish the clips

State plainly that the pixels are read out of the reference machine's RAM while it proposes the
trace, and that the circuit is verified to predict that same token stream. The frames are **not** a
reconstruction from the model's own predictions. That distinction is in `STATE.md` §10 and it
still holds.
