#!/usr/bin/env bash
set -euo pipefail

# First run installs Transformers/Sentence-Transformers and downloads Qwen/BGE.
# Subsequent runs reuse the local cache.
python -m LLM_agent.bootstrap
python -m LLM_agent.run "$@"
