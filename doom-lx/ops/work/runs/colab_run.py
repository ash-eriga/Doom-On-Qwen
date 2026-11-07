"""Drive the in-weights Doom trace on a Colab GPU, from the Colab CLI.

    colab new -s doom192 -gpu A100
    colab exec -s doom192 -f colab_setup.py --timeout 1700      # exact stack, once per VM
    colab exec -s doom192 -f colab_run.py --timeout 9000 \\
        --env HF_TOKEN=... --env HF_DATASET=<owner>/<dataset> \\
        --env SET=192 --env RESUME=0 --env HOURS=3.0 --env NTOK=95000000

Why a driver and not a notebook cell: the Colab CLI drops the kernel connection after ~30 s
without output, so anything long has to be launched under `nohup` and polled. The run also
outlives a dead session only if frames and the checkpoint leave the VM, so every artifact is
uploaded to the HF dataset as it is produced.

Artifacts the dataset must contain (see README.md in this directory for how they were produced):

    packed-192x120.pt / packed.pt      the emitted circuit for this ELF
    doom-lf-192x120.elf / doom-lf-fs96.elf
    doom1.wad
    lxsrc.zip                          linuxformer `lx` package + run_kaggle.py + colab_bench.py
    sched.txt                          the per-frame key schedule

Environment:

    HF_TOKEN      required, write access to HF_DATASET
    HF_DATASET    required, e.g. "<owner>/<dataset>"
    SET           96 or 192 (which build to run); default 192
    RESUME        1 to continue from results/session-<SET>.pt, 0 to start fresh; default 1
    HOURS         wall-clock cap for the verifier; default 1.6
    NTOK          tokens to verify this job; default 86,000,000
    BENCH         1 to run the warm-up chunk, print the honest tok/s and stop
    FBW / FBH     framebuffer read geometry; default from SET
    FC_OFF etc.   symbol offsets, already correct per SET; override only for a new build
"""
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SRC = Path("/content/lxsrc")

# What to load. "192" is the 192x120 build; "96" is the 96x60 pair, kept because a
# cross-GPU throughput comparison must use identical weights and an identical trace.
SET = os.environ.get("SET", "192")
if SET == "96":
    PACKED_NAME = os.environ.get("PACKED_NAME", "packed.pt")
    ELF_NAME = os.environ.get("ELF_NAME", "doom-lf-fs96.elf")
    FBW, FBH = 96, 60
else:
    PACKED_NAME = os.environ.get("PACKED_NAME", "packed-192x120.pt")
    ELF_NAME = os.environ.get("ELF_NAME", "doom-lf-192x120.elf")
    FBW, FBH = 192, 120
FBW = int(os.environ.get("FBW", FBW))
FBH = int(os.environ.get("FBH", FBH))

REPO = os.environ.get("HF_DATASET", "")
if not REPO:
    raise SystemExit("set HF_DATASET=<owner>/<dataset>; this driver does not guess a destination")
HOURS = os.environ.get("HOURS", "1.6")
NTOK = os.environ.get("NTOK", "86000000")
BENCH = os.environ.get("BENCH", "0") == "1"
RESUME = os.environ.get("RESUME", "1") == "1"

# .text offsets, from `nm` minus RAM_BASE 0x80000000, for the ELF this SET names.
OFFSETS = {
    "96": {"fc": "0x54bf34", "key": "0x54bf30", "demo": "0x4684d8", "fb": "0x48bef8"},
    "192": {"fc": "0x5556b4", "key": "0x5556b0", "demo": "0x467f50", "fb": "0x495678"},
}
OFF = {k: os.environ.get(f"{k.upper()}_OFF", v) for k, v in OFFSETS[SET].items()}

# Known-good ELF hashes. A mismatch is never silent: it is the class of bug that makes a run
# apply one build's offsets to another build's program, which corrupts the guest and looks
# exactly like a broken circuit.
EXPECT = {"96": {"doom-lf.elf": "413ef6762c0421b3a05d92f6e1a37a42"},
          "192": {"doom-lf.elf": "ff4ad5641ef0e7a000ed5536a6cddd93"}}.get(SET, {})

t_start = time.time()


def sh(*cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def log(msg):
    print(f"[{time.time() - t_start:7.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- environment
WANT_TORCH = "2.11.0"
WANT_TF = "5.22.0"


def _ver(mod):
    try:
        return __import__("importlib.metadata", fromlist=["version"]).version(mod)
    except Exception:  # noqa: BLE001
        return None


need = []
if not (_ver("torch") or "").startswith(WANT_TORCH):
    need.append("torch")
if _ver("transformers") != WANT_TF:
    need.append("transformers")
if _ver("fla-core") is None and _ver("flash-linear-attention") is None:
    need.append("fla")

if need:
    # Colab ships a newer torch; installing 2.11 over it leaves 2.14 package files beside 2.11
    # shared objects (_EvalFrameOverride ImportError). Purge first, the way colab_setup.py does.
    log(f"installing (missing/mismatched: {', '.join(need)})")
    if "torch" in need:
        for pkg in ("torch", "torchvision", "torchaudio", "torchtext", "triton"):
            sh(sys.executable, "-m", "pip", "uninstall", "-y", "-q", pkg)
        sh(sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", f"torch=={WANT_TORCH}",
           "--index-url", "https://download.pytorch.org/whl/cu130")
    if need - {"torch"}:
        sh(sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
           f"transformers=={WANT_TF}", "fla-core==0.5.2", "flash-linear-attention==0.5.2",
           "accelerate", "safetensors", "huggingface_hub", "ninja")
    # transformers 5.22 chokes on a Colab-broken torchao ("Invalid version: 'unknown'")
    sh(sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao", "transformer_engine")
else:
    log("torch/transformers/fla already match, skipping install")

# Do not import torch here: the long-lived Colab kernel caches modules, so after an install it
# holds a mix of old and new. Report from a fresh interpreter and let run_kaggle.py (a
# subprocess) import it for real.
_probe = subprocess.run(
    [sys.executable, "-c",
     "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available());"
     "p=torch.cuda.get_device_properties(0);print(torch.cuda.get_device_name(0),p.major,p.minor,"
     "round(p.total_memory/2**30,1))"],
    capture_output=True, text=True)
if _probe.returncode == 0:
    log("torch probe: " + " | ".join(_probe.stdout.split()))
else:
    log("torch probe FAILED: " + _probe.stderr[-800:])
    raise SystemExit("torch does not import in a fresh interpreter; refusing to spend GPU time")

# ---------------------------------------------------------------- artifacts
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=TOKEN)

# One directory per build. A shared directory was a real bug: destinations are fixed names
# (doom-lf.elf, packed.pt) because run_kaggle.py hardcodes them, so a 96x60 run left its ELF and
# weights where the 192x120 run picked them up. Running 192x120's offset table against a 96x60
# program makes the guest read its frame counter from the wrong address, corrupt its RAM and
# call I_Error.
DATA = Path(f"/content/data-{SET}")
DATA.mkdir(parents=True, exist_ok=True)
WORK = Path(f"/content/work-{SET}")
WORK.mkdir(parents=True, exist_ok=True)


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dl(name, local=None):
    """Fetch `name` from the dataset under the fixed filename run_kaggle.py expects."""
    dest = DATA / (local or Path(name).name)
    log(f"fetching {name} -> {dest.name}")
    got = Path(hf_hub_download(REPO, name, repo_type="dataset", local_dir=str(DATA), token=TOKEN))
    if got.resolve() != dest.resolve():
        dest.unlink(missing_ok=True)
        try:
            os.link(got, dest)
        except OSError:
            shutil.copy(got, dest)
    digest = md5(dest)
    log(f"  {dest.name}: {dest.stat().st_size / 2**20:.0f} MiB md5 {digest}")
    want = EXPECT.get(dest.name)
    if want and digest != want:
        raise SystemExit(f"WRONG ARTIFACT: {dest.name} md5 {digest} != expected {want}")
    return dest


packed = dl(PACKED_NAME, "packed.pt")
elf = dl(ELF_NAME, "doom-lf.elf")
dl("doom1.wad")
zip_p = dl("lxsrc.zip")
sched_p = dl("sched.txt")

if not (SRC / "lx").exists():
    shutil.rmtree(SRC, ignore_errors=True)
    SRC.mkdir(parents=True, exist_ok=True)
    sh("unzip", "-q", "-o", str(zip_p), "-d", str(SRC))

# The zip unpacks the repo one level down, and the importable package is <root>/lx/lx.
# run_kaggle.py prepends --lxsrc and --lxsrc/lx to sys.path, so --lxsrc must be the directory
# that CONTAINS the `lx` package directory.
root = SRC
if not (root / "lx" / "lx" / "__init__.py").exists():
    hits = list(SRC.glob("*/lx/lx/__init__.py"))
    if hits:
        root = hits[0].parents[2]
log(f"lxsrc root {root} (package {root / 'lx' / 'lx'})")

# A local copy of the runner wins over whatever the zip shipped.
for f in ("run_kaggle.py", "colab_bench.py"):
    local = Path("/content") / f
    if local.exists():
        shutil.copy(local, root / f)
        log(f"using local {f}")

play = ";".join(l.strip() for l in sched_p.read_text().splitlines()
                if l.strip() and not l.startswith("#"))
log(f"play {play}")

# ---------------------------------------------------------------- resume slot
resume = WORK / "session.pt"
remote_ckpt = f"results/session-{SET}.pt"
if BENCH or not RESUME:
    log(f"fresh state (BENCH={BENCH}, RESUME={RESUME}); ignoring any checkpoint")
else:
    try:
        log("looking for a checkpoint to resume from")
        hf_hub_download(REPO, remote_ckpt, repo_type="dataset", local_dir=str(WORK), token=TOKEN)
        got = WORK / remote_ckpt
        if got.exists():
            shutil.copy(got, resume)
            log(f"resuming from {remote_ckpt} ({got.stat().st_size / 2**20:.0f} MiB)")
    except Exception as exc:  # noqa: BLE001
        log(f"no checkpoint ({type(exc).__name__}), starting fresh")

# ---------------------------------------------------------------- the run
cmd = [sys.executable, str(root / "run_kaggle.py"),
       "--data", str(DATA), "--lxsrc", str(root), "--work", str(WORK),
       "--packed", str(packed), "--outdir", "/content/frames",
       "--ntok", NTOK, "--hours", HOURS, "--dump", "6-121",
       "--play", play, "--chunk", "32768",
       "--head-block", "131072", "--head-vblock", "8192",
       "--fc-off", OFF["fc"], "--key-off", OFF["key"],
       "--demo-off", OFF["demo"], "--fb-off", OFF["fb"],
       "--fb-w", str(FBW), "--fb-h", str(FBH)]
if BENCH:
    cmd += ["--bench", "--fresh"]
elif not RESUME:
    # run_kaggle.py would silently pick up a leftover work/session.pt; be explicit.
    cmd += ["--fresh"]

logp = Path("/content/run.log")
with open(logp, "w") as logf:
    r = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in r.stdout:
        print(line, end="", flush=True)
        logf.write(line)
        logf.flush()
rc = r.wait()
log(f"runner rc={rc}")

# ---------------------------------------------------------------- ship results
if BENCH:
    # --bench stops after the warm-up chunk and writes nothing; only the log is worth keeping.
    out = f"results/bench-{os.environ.get('TAG', 'x')}.log"
    api.upload_file(path_or_fileobj=str(logp), path_in_repo=out, repo_id=REPO, repo_type="dataset")
    log(f"bench log -> {out}")
    print("DONE", flush=True)
    raise SystemExit(0)

frames = sorted(Path("/content/frames").glob("frame-*.ppm"))
log(f"{len(frames)} frames rendered")
res_dir = {"96": "results/96x60", "192": "results/192x120"}.get(SET, f"results/{SET}")
for f in frames:
    api.upload_file(path_or_fileobj=str(f), path_in_repo=f"{res_dir}/{f.name}",
                    repo_id=REPO, repo_type="dataset")
for f in (resume, logp):
    if f.exists():
        name = remote_ckpt if f is resume else f"results/run-{SET}.log"
        api.upload_file(path_or_fileobj=str(f), path_in_repo=name,
                        repo_id=REPO, repo_type="dataset")
log("uploaded frames, checkpoint and log")
print("DONE", flush=True)
