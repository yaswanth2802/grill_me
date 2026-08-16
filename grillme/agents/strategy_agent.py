"""Strategy Agent — creates interview strategy and resume advice."""

from langchain_core.language_models import BaseChatModel

from grillme.config import get_main_llm
from grillme.models.jd import JDProfile
from grillme.models.resume import ResumeProfile
from grillme.models.strategy import InterviewStrategy, QuestionHook, ResumeAdvice
from grillme.prompts.prompts import STRATEGY_AGENT_PROMPT


def build_interview_strategy(
    resume: ResumeProfile, jd: JDProfile, llm: BaseChatModel | None = None
) -> InterviewStrategy:
    """
    Analyze resume vs JD and generate interview strategy.
    
    Args:
        resume: Extracted resume profile
        jd: Extracted job description profile
        llm: Optional LangChain chat model. If None, uses get_main_llm()
    
    Returns:
        InterviewStrategy with gaps, hooks, and interview plan
    """
    print(f"[agent:strategy_agent] ENTRY resume_name={getattr(resume, 'name', None)} jd_role={getattr(jd, 'role', None)}")
    if llm is None:
        llm = get_main_llm()
    
    if llm is None:
        raise RuntimeError("No LLM configured. Set LLM_API_KEY in .env")
    
    # Bind structured output to the LLM
    llm_with_output = llm.with_structured_output(InterviewStrategy)
    
    # Format the prompt with both profiles
    prompt = STRATEGY_AGENT_PROMPT.format(
        resume_profile=resume.model_dump_json(indent=2),
        jd_profile=jd.model_dump_json(indent=2),
    )
    
    # Invoke and return structured result
    result = llm_with_output.invoke(prompt)
    try:
        out = result.model_dump_json()
    except Exception:
        out = str(result)
    print(f"[agent:strategy_agent] EXIT result={out}")
    return result
