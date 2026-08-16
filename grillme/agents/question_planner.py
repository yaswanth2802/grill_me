"""Question Planner Agent — adaptively selects next question."""

from langchain_core.language_models import BaseChatModel

from grillme.config import get_main_llm
from grillme.models.interview import QuestionPlan
from grillme.models.strategy import InterviewStrategy
from grillme.prompts.prompts import QUESTION_PLANNER_PROMPT


def plan_next_question(
    questions_asked_count: int,
    type_coverage: dict,
    type_scores: dict,
    weak_areas: list,
    topics_asked: list,
    follow_up_depth: int,
    difficulty: str,
    experience_tier: str,
    strategy: InterviewStrategy,
    company: str,
    llm: BaseChatModel | None = None,
) -> QuestionPlan:
    if llm is None:
        llm = get_main_llm()
    
    if llm is None:
        raise RuntimeError("No LLM configured. Set LLM_API_KEY in .env")
    
    type_scores_avg = {}
    for qtype, scores in type_scores.items():
        if scores:
            type_scores_avg[qtype] = sum(scores) / len(scores)
        else:
            type_scores_avg[qtype] = 0.0
    
    hooks_summary = "\n".join(
        [f"- {hook.detail} ({hook.question_type})" for hook in strategy.question_hooks]
    )
    
    prompt = QUESTION_PLANNER_PROMPT.format(
        questions_asked_count=questions_asked_count,
        type_coverage=type_coverage,
        type_scores=type_scores_avg,
        weak_areas=weak_areas,
        topics_asked=topics_asked,
        follow_up_depth=follow_up_depth,
        difficulty=difficulty,
        experience_tier=experience_tier,
        strategy_summary=strategy.opening_question_suggestion,
        high_priority_topics=strategy.high_priority_topics,
        company_focus_areas=strategy.company_focus_areas,
        question_hooks=hooks_summary,
        candidate_strengths_to_test=strategy.candidate_strengths_to_test,
    )
    
    llm_with_output = llm.with_structured_output(QuestionPlan)
    result = llm_with_output.invoke(prompt)
    return result