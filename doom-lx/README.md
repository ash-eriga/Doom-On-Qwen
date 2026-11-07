# doom-lx

Index for the Doom→LX work. The overview is in the [top-level README](../README.md).

The machine is **linuxformer**'s LX (`lx.ref.Machine`, `lx.fast`), the weights are **base
Qwen3.5-4B** widened by `lx.core.build`, and the guest is **DoomGeneric** built for LX.

| path | |
|---|---|
| `port/` | the DoomGeneric port for LX (submodule): `port-lf/build.sh`, the native emulator `native/fastemu-lf.c`, the speed patch |
| `ops/work/runs/` | GPU drivers, a notebook, and the measured token/throughput numbers |
| `ops/work/kaggle/` | the verifier: chunked prefill plus a strict per-token check |
| `ops/work/preview/` | clips and contact sheets |
| `ops/work/STATE.md` | the full ledger, sections 1-16 |
| [`CREDITS.md`](CREDITS.md) | what is ours, what is GPL, what is not |

## Verify a frame

`fastemu-lf` renders from the same ELF and the same key schedule, and every frame is compared
against it byte for byte on the PPM payload.

```bash
cc -O3 -o fastemu-lf port/native/fastemu-lf.c -lm
./fastemu-lf doom-lf.elf --wad doom1.wad \
  --fbp-sym DG_ScreenBuffer --fc-sym qwen_frame_count \
  --keys-sym qwen_key_state --demo-sym qwen_demo_enabled \
  --frames 130 --script sched.txt \
  --dump-every 1 --dump-from 6 --dump-to 121 --dump-dir ref --guest-out none
```

## Known issues

About 4.2e-6 of predictions are wrong at 192x120, and 1.6e-5 at 96x60 — always one-value-off pairs,
so the frames are unaffected but the circuit does not predict every token. Separately, a
`lightlevel` bug in the guest leaves the player in the line of fire, so the demo dies early. Both
are in `ops/work/STATE.md`.
