"""Install optional runtime packages and pre-download local Qwen/BGE models.

Examples:
  python -m LLM_agent.bootstrap
  python -m LLM_agent.bootstrap --qwen Qwen/Qwen2.5-7B-Instruct --no-bge
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def install_requirements(requirements: Path) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])


def download_models(qwen: str, cache_dir: Path, bge: str | None) -> None:
    from huggingface_hub import snapshot_download
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(qwen, cache_dir=str(cache_dir / "huggingface"), resume_download=True)
    if bge:
        snapshot_download(bge, cache_dir=str(cache_dir / "huggingface"), resume_download=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", default=os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--bge", default=os.getenv("BGE_MODEL", "BAAI/bge-m3"))
    parser.add_argument("--cache-dir", default="LLM_agent/models")
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--no-bge", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if not args.no_install:
        install_requirements(root / "requirements.txt")
    download_models(args.qwen, Path(args.cache_dir), None if args.no_bge else args.bge)
    print("Local model setup complete.")


if __name__ == "__main__":
    main()
