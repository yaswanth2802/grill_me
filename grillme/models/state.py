from typing import TypedDict

from .interview import ConductorResponse, Evaluation, QuestionPlan, QuestionRecord
from .jd import JDProfile
from .report import GrindReport
from .resume import ResumeProfile
from .strategy import InterviewStrategy, ResumeAdvice


class GrillMeState(TypedDict, total=False):
    resume_text: str
    jd_text: str
    experience_tier: str
    difficulty: str
    question_types: list[str]
    feedback_mode: str
    interaction_mode: str
    company: str

    resume_profile: ResumeProfile | None
    jd_profile: JDProfile | None
    interview_strategy: InterviewStrategy | None

    conversation_history: list[dict]
    question_records: list[QuestionRecord]
    current_question_type: str | None
    type_coverage: dict[str, int]
    type_scores: dict[str, list[float]]
    weak_areas: list[str]
    topics_asked: list[str]
    follow_up_depth: int
    session_start_time: str
    question_count: int
    should_end: bool

    user_answer: str
    user_code: str | None
    current_evaluation: Evaluation | None
    next_question_plan: QuestionPlan | None
    agent_response: ConductorResponse | None

    report: GrindReport | None
    resume_advice: ResumeAdvice | None
