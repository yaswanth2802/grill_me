"""JD Analyzer Agent — extracts structured profile from job description."""

from langchain_core.language_models import BaseChatModel

from grillme.config import get_main_llm
from grillme.models.jd import JDProfile
from grillme.prompts.prompts import JD_ANALYZER_PROMPT


def analyze_jd(jd_text: str, llm: BaseChatModel | None = None) -> JDProfile:
    """
    Extract structured profile from job description text using the LLM.
    
    Args:
        jd_text: Raw job description text
        llm: Optional LangChain chat model. If None, uses get_main_llm()
    
    Returns:
        JDProfile with extracted information
    """
    print(f"[agent:jd_analyzer] ENTRY jd_text_length={len(jd_text) if jd_text else 0}")
    if llm is None:
        llm = get_main_llm()
    
    if llm is None:
        raise RuntimeError("No LLM configured. Set LLM_API_KEY in .env")
    
    # Bind structured output to the LLM
    llm_with_output = llm.with_structured_output(JDProfile)
    
    # Format the prompt with the JD text
    prompt = JD_ANALYZER_PROMPT.format(jd_text=str(jd_text))
    
    # Invoke and return structured result
    result = llm_with_output.invoke(prompt)
    try:
        out = result.model_dump_json()
    except Exception:
        out = str(result)
    print(f"[agent:jd_analyzer] EXIT result={out}")
    return result
