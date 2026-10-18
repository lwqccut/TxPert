"""Single command entrypoint for the complete local agent pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .orchestrator import ResearchAgent
from .schemas import parse_task


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is missing. Run: python -m LLM_agent.bootstrap") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen + KG + BGE-M3 + TxPert agent")
    parser.add_argument("--config", type=Path, default=Path("LLM_agent/config.yaml"))
    parser.add_argument("--input", type=Path, default=Path("LLM_agent/example_request.json"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    task = parse_task(json.loads(args.input.read_text(encoding="utf-8")))
    result = ResearchAgent(config).run(task)
    output = result.to_json()
    destination = args.output or Path(config.get("output", {}).get("dir", "LLM_agent/runs")) / "prediction.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(output + "\n", encoding="utf-8")
    print(f"Saved result to {destination}")


if __name__ == "__main__":
    main()
