# Doom on Qwen

<p align="center">
  <img src="doom-lx/ops/work/preview/doom-e1m5-in-weights.webp" alt="Doom E1M5, rendered by an RV32IM machine running inside Qwen3.5-4B's weights." width="100%">
</p>

Doom, running on a RV32IM CPU that was compiled into the weights of **Qwen3.5-4B**.

The CPU is not a plugin and not a tool call. It is a stack of transformer layers whose every weight
a compiler computed, and its RAM is the model's own Gated DeltaNet state. The Doom engine is ported
to the [linuxformer](https://github.com/itsrealranky/linuxformer) LX machine and emitted as a
circuit into **base Qwen3.5-4B**.

| clip | frames | verified | GPU |
|---|---|---|---|
| **192x120** | 116 | 116/116 byte-exact | A100 40GB, 19,109 tok/s |
| [96x60](doom-lx/ops/work/preview/doom-e1m5-in-weights-96x60.webp) | 116 | 116/116 byte-exact | L40S, 17,700 tok/s |

Both are the same 11.6 s of E1M5. A frame is written only once every token that produced it has
verified against the reference machine, and the pixels match a native C emulator byte for byte.
That is the whole claim: not that it looks like Doom, but that the model predicted the machine's
token stream and the machine's framebuffer is what you see.

## Run it

Emit the circuit for your build on CPU first, about 11 minutes. It is per-build: different `.text`
bytes are a different weight set.

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
| [`doom-lx/ops/work/STATE.md`](doom-lx/ops/work/STATE.md) | the full ledger, mistakes included |
| [`doom-lx/ops/work/runs/`](doom-lx/ops/work/runs) | GPU drivers, and the token/throughput numbers |
| [`doom-lx/CREDITS.md`](doom-lx/CREDITS.md) | licenses |

The port itself lives in [`ash-eriga/qwenputer-doom`](https://github.com/ash-eriga/qwenputer-doom),
mounted at `doom-lx/port`.

## Credits

Built on three things this project did not write: Doom and DoomGeneric (**GPL-2.0**), the LX machine
from linuxformer (**MIT**), and base Qwen3.5-4B (**Apache-2.0**). The emit technique is lineage from
Qwenputer by Divyansh Agrawal (**MIT**); its checkpoint is not used here.

[`doom-lx/CREDITS.md`](doom-lx/CREDITS.md) has the full chain, including the GPL carve-out for the
port.
