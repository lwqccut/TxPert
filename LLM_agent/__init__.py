"""Standalone LLM agent for evidence-guided TxPert inference.

This package is deliberately separate from the original training and inference
entrypoints.  It can be run with ``python -m LLM_agent.cli`` from the repository
root and uses a deterministic planner when no LLM endpoint is configured.
"""

from .agent import LLMAgent, AgentRequest, AgentResponse
from .orchestrator import ResearchAgent

__all__ = ["LLMAgent", "AgentRequest", "AgentResponse", "ResearchAgent"]
