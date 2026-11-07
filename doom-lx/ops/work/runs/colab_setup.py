"""Pin the exact stack on a Colab GPU VM. Run once per VM, before colab_run.py.

    colab exec -s doom192 -f colab_setup.py --timeout 1700

Three things about the Colab image make this necessary:

* it ships torch 2.14.1, and `pip install torch==2.11.0` over it leaves 2.14 package files
  beside 2.11 shared objects -> `ImportError: cannot import name '_EvalFrameOverride'`.
  The only working fix is to remove torch and its companions completely, then install clean.
* `torchao` ships with metadata that `packaging` cannot parse ("Invalid version: 'unknown'"),
  which kills the transformers import even though torchao is never used here.
* `transformers 5.22` reads `importlib.metadata.version("torchcodec")` unconditionally from
  `audio_utils.py`, so a VM without torchcodec cannot import the model at all. Audio is never
  used, so installing its metadata with `--no-deps` is enough (and avoids touching torch).
"""
import glob
import shutil
import subprocess
import sys

PY = "/usr/local/lib/python3.13/dist-packages"

WANT_TORCH = "2.11.0"
WANT_TF = "5.22"


def run(*cmd, check=True):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout[-1500:], flush=True)
    if r.returncode and check:
        print("STDERR", r.stderr[-2000:], flush=True)
    return r


for pkg in ("torch", "torchvision", "torchaudio", "torchtext", "triton"):
    run(sys.executable, "-m", "pip", "uninstall", "-y", "-q", pkg, check=False)

for pattern in ("torch", "torch-*", "torchvision*", "torchaudio*", "torchtext*", "triton",
                "nvidia*", "cuda_*", "cuda-*"):
    for path in glob.glob(f"{PY}/{pattern}"):
        print("removing", path, flush=True)
        if not path.endswith(".pth"):
            shutil.rmtree(path, ignore_errors=True)

for path in glob.glob(f"{PY}/*.dist-info") + glob.glob(f"{PY}/*.egg-info"):
    base = path.split("/")[-1].lower()
    if base.startswith(("torch", "triton", "nvidia", "cuda")):
        print("removing", path, flush=True)
        shutil.rmtree(path, ignore_errors=True)

# transformer_engine and torchao ship compiled extensions / bad metadata against the old torch.
for pattern in ("transformer_engine*", "torchao*"):
    for path in glob.glob(f"{PY}/{pattern}"):
        print("removing", path, flush=True)
        shutil.rmtree(path, ignore_errors=True)

r = run(sys.executable, "-c", "import torch; print('still importable', torch.__version__)",
        check=False)
if r.returncode == 0:
    print("WARNING: torch still importable after purge", flush=True)

run(sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", f"torch=={WANT_TORCH}",
    "--index-url", "https://download.pytorch.org/whl/cu130")
print(f"torch {WANT_TORCH} installed", flush=True)

run(sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", f"transformers=={WANT_TF}",
    "fla-core==0.5.2", "flash-linear-attention==0.5.2",
    "accelerate", "safetensors", "huggingface_hub", "ninja")
print("deps installed", flush=True)

# metadata only: transformers 5.22 needs the entry to exist, nothing imports torchcodec here.
run(sys.executable, "-m", "pip", "install", "-q", "--no-deps", "torchcodec")
print("torchcodec metadata installed", flush=True)

import torch  # noqa: E402
import fla  # noqa: E402
import transformers  # noqa: E402

print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
print("transformers", transformers.__version__, flush=True)
print("fla", getattr(fla, "__version__", "?"), flush=True)
p = torch.cuda.get_device_properties(0)
print("GPU", torch.cuda.get_device_name(0), "sm_%d%d" % (p.major, p.minor), flush=True)
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
b = a @ a
torch.cuda.synchronize()
print("bf16 matmul ok", float(b.float().abs().mean()), flush=True)
print("SETUP OK", flush=True)
