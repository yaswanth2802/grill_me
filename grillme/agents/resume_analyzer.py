"""Resume Analyzer Agent — extracts structured profile from resume text."""

from langchain_core.language_models import BaseChatModel

from grillme.config import get_main_llm
from grillme.models.resume import ResumeProfile
from grillme.prompts.prompts import RESUME_ANALYZER_PROMPT


def analyze_resume(resume_text: str, llm: BaseChatModel | None = None) -> ResumeProfile:
    """
    Extract structured profile from resume text using the LLM.
    
    Args:
        resume_text: Raw resume text (e.g., from PDF extraction)
        llm: Optional LangChain chat model. If None, uses get_main_llm()
    
    Returns:
        ResumeProfile with extracted information
    """
    print(f"[agent:resume_analyzer] ENTRY resume_text_length={len(resume_text) if resume_text else 0}")
    if llm is None:
        llm = get_main_llm()
    
    if llm is None:
        raise RuntimeError("No LLM configured. Set LLM_API_KEY in .env")
    
    # Bind structured output to the LLM
    llm_with_output = llm.with_structured_output(ResumeProfile)
    
    # Format the prompt with the resume text
    prompt = RESUME_ANALYZER_PROMPT.format(resume_text=resume_text)
    
    # Invoke and return structured result
    result = llm_with_output.invoke(prompt)
    try:
        out = result.model_dump_json()
    except Exception:
        out = str(result)
    print(f"[agent:resume_analyzer] EXIT result={out}")
    return result
