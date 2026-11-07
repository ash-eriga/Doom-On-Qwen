# Doom on Qwen

<p align="center">
  <img src="doom-lx/ops/work/preview/doom-e1m5-in-weights.webp" alt="Doom E1M5 rendered by an RV32IM machine running inside Qwen3.5-4B's weights." width="100%">
</p>

<p align="center">
  <sub><b>E1M5, frames 6-121, 192x120.</b> Running on a RV32IM CPU whose weights were compiled, not trained. Every frame is byte-identical to the reference emulator.</sub>
</p>

The Doom engine, ported to the [linuxformer](https://github.com/itsrealranky/linuxformer) **LX**
machine and emitted as a circuit into **base Qwen3.5-4B** weights.

The CPU is not a plugin and not a tool call. It is a set of transformer layers whose every weight a
compiler computed, appended to the model's own. Its RAM is the model's Gated DeltaNet state, one
row per memory cell, and each guest instruction is one predicted token.

| clip | frames | verified | GPU | tokens |
|---|---|---|---|---|
| **192x120** | 116 (6-121) | **116/116 byte-exact** | A100 40GB, 19,109 tok/s | 106.1M | ~10 CU |
| 96x60 | 116 (6-121) | **116/116 byte-exact** | L40S, 17,700 tok/s | 64.7M |

Both are the same 11.6 s of E1M5 at two render sizes. The 96x60 preview is
[here](doom-lx/ops/work/preview/doom-e1m5-in-weights-96x60.webp).

## What "verified" means

Two machines run the same trace:

- the **reference machine** (`lx.ref.Machine`, the executable spec) proposes tokens and holds the
  ground RAM;
- the **circuit** verifies that stream as chunked prefill, never allowed to drift more than a
  bounded window ahead of the verified frontier.

A frame is written only once every token that produced it has verified. The comparison that
matters is against `fastemu-lf`, a native C emulator that renders from the same ELF and the same
key schedule. Frames are compared byte for byte on the PPM payload, and all 116 in both clips are
exact.

This is not "it looks like Doom". It is "the model predicted the machine's token stream, and the
machine's framebuffer is what you see".

## What is not true

Three claims from earlier sessions were wrong, and are corrected in
[`doom-lx/ops/work/STATE.md`](doom-lx/ops/work/STATE.md):

- **It does not predict every token.** 192x120 misses 450 of 106.1M (~4.2e-6); 96x60 misses 1,023
  of 64.7M (~1.6e-5). Frames are unaffected because the errors are one-value-off pairs
  (`65536`/`65537`), which look like a rounding boundary in the fast path.
- **The A100 is not slower than the L40S.** An intermediate reading said 11,541 tok/s on A100;
  that run had loaded mismatched artifacts. With the correct pair: 19,109 vs 17,700.
- **The clips are not 320x200.** They are 96x60 and 192x120 at `screenblocks=11`, read as a linear
  prefix of the 320x200 framebuffer, not a strided crop.

## Status

The engine runs; it does not play well. A `lightlevel` bug leaves the player in the line of fire,
so the demo dies early. That is a guest-side bug, not a circuit limit.

## Layout

| path | what it is |
|---|---|
| [`doom-lx/`](doom-lx) | the whole project: port, ops pipeline, clips, ledger |
| [`doom-lx/CREDITS.md`](doom-lx/CREDITS.md) | licenses: what is ours, what is GPL, what is not |
| [`doom-lx/ops/work/STATE.md`](doom-lx/ops/work/STATE.md) | the decision trail, sections 1-16, including the mistakes |
| [`doom-lx/ops/work/runs/`](doom-lx/ops/work/runs) | GPU run drivers and a notebook |

The port itself lives in the separate repo
[`ash-eriga/qwenputer-doom`](https://github.com/ash-eriga/qwenputer-doom), mounted at
`doom-lx/port`.

## Reproducing it

The emitted circuit is per-build: different `.text` bytes are a different weight set, and a
checkpoint only resumes on the weights it was saved with. Emit on CPU first, about 11 minutes:

```bash
python doom-lx/ops/work/kaggle/run_kaggle.py --emit \
    --data <dir with doom-lf.elf> --lxsrc <lxsrc> --work <out>
```

Then rent a GPU. `packed.pt` is 4.08 GiB, so a 24 GB card is enough:

```bash
colab new -s doom192 -gpu A100
colab exec -s doom192 -f doom-lx/ops/work/runs/colab_setup.py --timeout 1700
colab exec -s doom192 -f doom-lx/ops/work/runs/colab_run.py --timeout 9000 \
    --env HF_TOKEN=... --env HF_DATASET=<owner>/<dataset> \
    --env SET=192 --env RESUME=0 --env HOURS=3.0 --env NTOK=95000000
```

[`doom-lx/ops/work/runs/run-192x120.ipynb`](doom-lx/ops/work/runs/run-192x120.ipynb) is the same
flow as a notebook, including the `fastemu-lf` comparison. Budget and throughput numbers are in
[`doom-lx/ops/work/runs/README.md`](doom-lx/ops/work/runs/README.md).

## Credits

This project is three borrowed things plus a port: Doom and DoomGeneric (**GPL-2.0**), the LX
machine from linuxformer (**MIT**), and base Qwen3.5-4B (**Apache-2.0**). The emit technique is
lineage from Qwenputer by Divyansh Agrawal (**MIT**); its checkpoint is not used here.

[`doom-lx/CREDITS.md`](doom-lx/CREDITS.md) has the full chain and the GPL carve-out.
