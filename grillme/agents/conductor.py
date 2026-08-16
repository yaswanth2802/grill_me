"""Conductor Agent — delivers questions and feedback to user."""

from langchain_core.language_models import BaseChatModel

from grillme.config import get_main_llm
from grillme.models.interview import ConductorResponse
from grillme.models.strategy import InterviewStrategy
from grillme.prompts.prompts import CONDUCTOR_PROMPT


def conduct_turn(
    company: str,
    role: str,
    experience_tier: str,
    difficulty: str,
    strategy: InterviewStrategy,
    question_number: int,
    questions_asked_count: int,
    type_coverage: dict,
    weak_areas: list,
    next_question: str | None = None,
    evaluation_summary: str | None = None,
    tone: str = "balanced",
    llm: BaseChatModel | None = None,
) -> ConductorResponse:
    """
    Conductor agent — delivers interview questions and feedback naturally.
    """
    print(f"[agent:conductor] ENTRY company={company} role={role} question_number={question_number} evaluation_summary_present={bool(evaluation_summary)}")
    if llm is None:
        llm = get_main_llm()
    
    if llm is None:
        raise RuntimeError("No LLM configured. Set LLM_API_KEY in .env")
    
    # Format the prompt with context
    prompt = CONDUCTOR_PROMPT.format(
        company=company,
        role=role,
        experience_tier=experience_tier,
        difficulty=difficulty,
        question_number=question_number,
        strategy_summary=strategy.opening_question_suggestion,
        questions_asked_count=questions_asked_count,
        type_coverage=type_coverage,
        weak_areas=weak_areas,
        tone=tone,
        next_question=next_question or "[No question provided yet]",
        evaluation_summary=evaluation_summary or "[No evaluation yet]",
    )
    
    # Bind the Pydantic model directly to the LLM for guaranteed structured output
    structured_llm = llm.with_structured_output(ConductorResponse)
    
    # Invoke LLM — it will return an instance of ConductorResponse directly
    resp: ConductorResponse = structured_llm.invoke(prompt)
    
    # Fallback safety in case the model returns None or something unexpected
    if not resp:
        resp = ConductorResponse(
            response_text="Let's continue with the interview.",
            next_question=next_question
        )

    try:
        out = resp.model_dump_json()
    except Exception:
        out = str(resp)
        
    print(f"[agent:conductor] EXIT result={out}")
    return resp