# Oracle

The ground truth the in-weights run is checked against, for the 192x120 build.

| file | what it is |
|---|---|
| `oracle-192x120.md5` | md5 of the **PPM pixel payload** (the bytes after the `P6` header, not the whole file) of frames 6..130 |
| `sched.txt` | the per-frame key masks, one `frame mask` pair per line (bit0 up, 1 down, 2 left, 3 right, 4 fire, 5 use, 6 enter, 7 escape) |
| `check_frames.py` | hashes the payloads in a directory and compares them with the manifest |

The clip is frames 6..121. The manifest runs to 130 so a longer run can be checked too
(`--last 130`).

Artifacts the manifest was produced from:

| artifact | md5 |
|---|---|
| `doom-lf.elf` (192x120 build) | `ff4ad5641ef0e7a000ed5536a6cddd93` |
| `packed.pt` (emitted circuit, 4,384,117,361 bytes) | `5e3c5affb7f16fbcdb267d992ae63af0` |
| `doom1.wad` (shareware v1.9, 4,196,020 bytes) | `f0cefca49926d00903cf57551d901abe` |

Regenerate the reference frames on CPU (about a second):

```bash
cc -O3 -o fastemu-lf ../port/native/fastemu-lf.c -lm
mkdir ref
./fastemu-lf doom-lf.elf --wad doom1.wad \
  --fbp-sym DG_ScreenBuffer --fc-sym qwen_frame_count \
  --keys-sym qwen_key_state --demo-sym qwen_demo_enabled \
  --res 192x120 --frames 130 --script sched.txt \
  --dump-every 1 --dump-from 6 --dump-to 130 --dump-dir ref --guest-out none
python check_frames.py ref --last 130
```

An in-weights frame counts as genuine only if its payload matches here.
