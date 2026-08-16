import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()


def create_llm(provider: str, model: str, api_key: str, temperature: float = 0.7) -> BaseChatModel | None:
    provider = (provider or "").lower()
    if provider == "gemini":
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temperature)
    if provider == "openai":
        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)
    if provider == "anthropic":
        return ChatAnthropic(model=model, api_key=api_key, temperature=temperature)
    return None


def get_main_llm() -> BaseChatModel | None:
    provider = os.getenv("LLM_PROVIDER", "gemini")
    model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    api_key = os.getenv("LLM_API_KEY", "")
    return create_llm(provider, model, api_key)


def get_evaluator_llm() -> BaseChatModel | None:
    provider = os.getenv("LLM_PROVIDER", "gemini")
    model = os.getenv("EVALUATOR_MODEL", os.getenv("LLM_MODEL", "gemini-2.0-flash-lite"))
    api_key = os.getenv("EVALUATOR_API_KEY") or os.getenv("LLM_API_KEY", "")
    return create_llm(provider, model, api_key)
