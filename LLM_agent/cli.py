"""Command line entrypoint for the standalone LLM agent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import LLMAgent, AgentRequest
from .evidence import JsonlEvidenceStore


class DemoForwardModel:
    """A deterministic smoke-test model; replace with a TxPert checkpoint adapter."""
    def __call__(self, control, perturbations, context):
        shift = 0.01 * sum(perturbations)
        return [float(value) + shift for value in control]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the separate TxPert LLM agent")
    parser.add_argument("--input", type=Path, help="JSON request file")
    parser.add_argument("--kg", type=Path, help="KG evidence JSONL")
    parser.add_argument("--rag", type=Path, help="RAG evidence JSONL")
    parser.add_argument("--output", type=Path, help="Write response JSON to this file")
    args = parser.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8")) if args.input else {
        "control_expression": [0.1, 0.2, 0.3], "perturbation_ids": [2], "cell_type": "K562",
    }
    response = LLMAgent(DemoForwardModel(), kg=JsonlEvidenceStore(args.kg) if args.kg else None,
                        rag=JsonlEvidenceStore(args.rag) if args.rag else None).run(AgentRequest(**raw))
    output = response.to_json()
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
