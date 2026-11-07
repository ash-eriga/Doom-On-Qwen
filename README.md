# Doom on Qwen

<p align="center">
  <img src="doom-lx/ops/work/preview/doom-e1m5-in-weights.webp" alt="Doom E1M5, rendered by a RV32IM machine running inside Qwen3.5-4B's weights." width="100%">
</p>

Doom, running on a RV32IM CPU compiled into the weights of Qwen3.5-4B.

The CPU is a 55-layer transformer circuit emitted into the base model, with every weight computed
by a compiler rather than trained. Its memory is Qwen's own Gated DeltaNet state, which the delta
rule rewrites one row at a time, so the registers, stack and program counter all live in the model.
A guest instruction costs about 1.6 predicted tokens, and the guest never runs as a process of its
own.

The port targets the LX machine of
[linuxformer](https://github.com/itsrealranky/linuxformer), with DoomGeneric as the engine.
`lx.core.build` emits the machine into the model's layers.

| clip | frames | verified | GPU |
|---|---|---|---|
| **192x120** | 116 | 116/116 byte-exact | A100 40GB, 19,109 tok/s |
| [96x60](doom-lx/ops/work/preview/doom-e1m5-in-weights-96x60.webp) | 116 | 116/116 byte-exact | L40S, 17,700 tok/s |

Both clips are the first 11.6 seconds of E1M5, at two render sizes. A frame is only written once
every token that produced it has verified, and it has to match `fastemu-lf`, a native C emulator
given the same ELF and key schedule, byte for byte. All 116 frames of both clips do. The pixels
come from the reference machine's framebuffer; the circuit's part is predicting the instruction
stream that draws them.

## Run it

Emit the circuit for your build on CPU first, which takes about 11 minutes. It is per-build:
different `.text` bytes are a different weight set.

```bash
python doom-lx/ops/work/kaggle/run_kaggle.py --emit \
    --data <dir with doom-lf.elf> --lxsrc <lxsrc> --work <out>
```

Then rent a GPU. `packed.pt` is 4.08 GiB, so 24 GB is enough.

```bash
colab new -s doom192 -gpu A100
colab exec -s doom192 -f doom-lx/ops/work/runs/colab_setup.py --timeout 1700
colab exec -s doom192 -f doom-lx/ops/work/runs/colab_run.py --timeout 9000 \
    --env HF_TOKEN=... --env HF_DATASET=<owner>/<dataset> \
    --env SET=192 --env RESUME=0 --env HOURS=3.0 --env NTOK=95000000
```

Or use the notebook: [`run-192x120.ipynb`](doom-lx/ops/work/runs/run-192x120.ipynb).

## Where things are

| path | |
|---|---|
| [`doom-lx/`](doom-lx) | the project: port, pipeline, clips |
| [`doom-lx/ops/work/STATE.md`](doom-lx/ops/work/STATE.md) | the decision trail, including what went wrong |
| [`doom-lx/ops/work/runs/`](doom-lx/ops/work/runs) | GPU drivers, and the token and throughput numbers |
| [`doom-lx/CREDITS.md`](doom-lx/CREDITS.md) | licenses and attribution |

The port itself lives in [`ash-eriga/qwenputer-doom`](https://github.com/ash-eriga/qwenputer-doom),
mounted at `doom-lx/port`.

## Credits

Doom and DoomGeneric are **GPL-2.0**; the LX machine from linuxformer is **MIT**; base Qwen3.5-4B is
**Apache-2.0**. The emit technique comes from Qwenputer by Divyansh Agrawal (**MIT**), whose
checkpoint this project does not load.

[`doom-lx/CREDITS.md`](doom-lx/CREDITS.md) has the full chain, including the GPL carve-out for the
port.
