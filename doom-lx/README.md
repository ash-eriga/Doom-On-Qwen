# doom-lx

Index for the Doom port. The overview is in the [top-level README](../README.md).

The machine is **linuxformer**'s LX (`lx.ref.Machine`, `lx.fast`), the weights are **base
Qwen3.5-4B** widened by `lx.core.build`, and the guest is **DoomGeneric** built for LX.

| path | |
|---|---|
| `port/` | the DoomGeneric port for LX (submodule): `port-lf/build.sh`, the native emulator `native/fastemu-lf.c`, the speed patch |
| `ops/work/runs/` | GPU drivers, a notebook, and the measured token and throughput numbers |
| `ops/work/kaggle/` | the verifier: chunked prefill plus a strict check on every token |
| `ops/work/preview/` | clips and contact sheets |
| `ops/work/STATE.md` | the decision trail, sections 1-16 |
| [`CREDITS.md`](CREDITS.md) | licenses and attribution |

## Verify a frame

`fastemu-lf` renders reference frames from the same ELF and key schedule as the run. Each frame is
then compared on the PPM payload, byte for byte.

```bash
cc -O3 -o fastemu-lf port/native/fastemu-lf.c -lm
./fastemu-lf doom-lf.elf --wad doom1.wad \
  --fbp-sym DG_ScreenBuffer --fc-sym qwen_frame_count \
  --keys-sym qwen_key_state --demo-sym qwen_demo_enabled \
  --frames 130 --script sched.txt \
  --dump-every 1 --dump-from 6 --dump-to 121 --dump-dir ref --guest-out none
```

## Known issues

About 4.2e-6 of predictions are wrong at 192x120, and 1.6e-5 at 96x60. They are always
one-value-off pairs, so the frames are unaffected, but the circuit does not predict every token.

The autopilot also lands 0 kills on E1M5: a wall corner blocks the line of fire and the enemy stays
at 30 health. Closing that needs pathfinding over `BLOCKMAP`/`REJECT`, so the clips show walking and
firing rather than a kill.

The prediction defect is in `ops/work/STATE.md` §7, the playthrough in §8, and the measured rates in
the two `RUNLOG-*.md`.
