import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent.resolve()
REQ = HERE / "requirements.txt"

def sh(cmd: str, check: bool = True):
    res = subprocess.run(cmd, shell=True)
    if check and res.returncode != 0:
        raise SystemExit(res.returncode)

def pip_install(args: str, check: bool = True):
    sh(f"{sys.executable} -m pip install {args}", check=check)

def read_requirements_excluding(patterns=("mmcv", "mmcv-full")):
    lines = []
    for raw in REQ.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        low = s.lower()
        if any(low.startswith(p) for p in patterns):
            continue
        lines.append(s)
    return lines

def print_torch_env():
    try:
        import torch
        tv_full = torch.__version__
        tv = tv_full.split("+")[0]
        cu = getattr(torch.version, "cuda", None)
        print(f"Torch: {tv_full}  | parsed: {tv}  | CUDA: {cu}")
        return tv, cu
    except Exception:
        print("Install torch first")
        raise

def choose_mmcv_target(torch_ver: str, cuda_ver: str):
    return {"kind": "cuda", "version": "2.0.0rc4", "index": None}

def maybe_skip_existing(target_ver: str):
    try:
        import mmcv
        cur = getattr(mmcv, "__version__", "")
        if cur == target_ver:
            print(f"mmcv/mmcv-full {cur} already installed")
            return True
        else:
            print(f"Reinstall mmcv: {cur} → {target_ver}")
    except ImportError:
        pass
    return False

def install_requirements():
    pip_install("-U openmim")
    pip_install('"pycocotools>=2.0.7"')

    reqs = read_requirements_excluding(("mmcv", "mmcv-full"))
    if reqs:
        tmp = HERE / ".requirements.no_mmcv.txt"
        tmp.write_text("\n".join(reqs) + "\n", encoding="utf-8")
        pip_install(f'-r "{tmp.as_posix()}"')
        tmp.unlink()

def install_mmcv_with_mim(plan):
    sh(f'{sys.executable} -m pip uninstall -y mmcv mmcv-full mmengine', check=False)
    target_mmcv = os.environ.get("MMCV_VERSION", "2.1.0")

    sh(f'{sys.executable} -m mim install "mmcv=={target_mmcv}"')
    sh(f'{sys.executable} -m mim install "mmengine>=0.7.0"')

def install_system_libs():
    cmds = [
        "apt-get update -y",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx",
        "printf '6\\n69\\n' | DEBIAN_FRONTEND=noninteractive apt-get install -y libglib2.0-0"
    ]
    for cmd in cmds:
        subprocess.run(cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)

def main():
    tv, cu = print_torch_env()
    install_requirements()

    plan = choose_mmcv_target(tv, cu)
    target_ver = plan["version"]
    if maybe_skip_existing(target_ver):
        return

    install_mmcv_with_mim(plan)
    install_system_libs()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error during installation: {e}")
        raise

setup(
    name="object-detection",
    version="0.1.0",
    packages=find_packages(exclude=("tests", "examples")),
    install_requires=[],
    python_requires=">=3.8,<3.11",
    author="Boostcamp-ai-tech-OD",
    description="Setting for Baselines (torchvision, mmdetection, detectron2)",
)