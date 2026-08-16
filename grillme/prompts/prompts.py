"""Load prompts from .txt files."""

import os
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(prompt_name: str) -> str:
    """Load a prompt from a .txt file."""
    prompt_path = PROMPTS_DIR / f"{prompt_name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


# Lazy-load prompts on module import
RESUME_ANALYZER_PROMPT = load_prompt("resume_analyzer")
JD_ANALYZER_PROMPT = load_prompt("jd_analyzer")
STRATEGY_AGENT_PROMPT = load_prompt("strategy_agent")
CONDUCTOR_PROMPT = load_prompt("conductor")
EVALUATOR_PROMPT = load_prompt("evaluator")
QUESTION_PLANNER_PROMPT = load_prompt("question_planner")
