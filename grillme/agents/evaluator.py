"""Evaluator Agent — scores user answers."""

from langchain_core.language_models import BaseChatModel

from grillme.config import get_evaluator_llm
from grillme.models.interview import Evaluation
from grillme.prompts.prompts import EVALUATOR_PROMPT


def evaluate_answer(
    question_type: str,
    question_text: str,
    difficulty: str,
    experience_tier: str,
    company: str,
    role: str,
    user_answer: str,
    user_code: str | None = None,
    llm: BaseChatModel | None = None,
) -> Evaluation:
    """
    Evaluator agent — scores the candidate's answer to a question.
    
    Args:
        question_type: Type of question (behavioral, system_design, ml_concepts, coding, etc.)
        question_text: The question that was asked
        difficulty: Interview difficulty level
        experience_tier: Candidate tier (student, junior, senior)
        company: Target company
        role: Target job role
        user_answer: The candidate's answer text
        user_code: Optional code submission (for coding questions)
        llm: Optional LangChain chat model (defaults to cheaper evaluator model)
    
    Returns:
        Evaluation with score, strengths, weaknesses, missed_points, weak_signals
    """
    print(f"[agent:evaluator] ENTRY question_type={question_type} difficulty={difficulty} experience_tier={experience_tier} code_present={bool(user_code)}")
    if llm is None:
        llm = get_evaluator_llm()
    
    if llm is None:
        raise RuntimeError("No LLM configured. Set LLM_API_KEY in .env")
    
    # Format code section if provided
    code_section = ""
    if user_code:
        code_section = f"\n\nCode Submission:\n```\n{user_code}\n```"
    
    # Format the prompt
    prompt = EVALUATOR_PROMPT.format(
        question_type=question_type,
        question_text=question_text,
        difficulty=difficulty,
        experience_tier=experience_tier,
        company=company,
        role=role,
        user_answer=user_answer,
        code_submission=code_section,
    )
    
    # Bind structured output and invoke
    llm_with_output = llm.with_structured_output(Evaluation)
    result = llm_with_output.invoke(prompt)
    try:
        out = result.model_dump_json()
    except Exception:
        out = str(result)
    print(f"[agent:evaluator] EXIT result={out}")
    return result
