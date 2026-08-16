from pydantic import BaseModel


class QuestionHook(BaseModel):
    source: str
    detail: str
    question_type: str
    priority: str


class InterviewStrategy(BaseModel):
    matching_skills: list[str] = []
    missing_skills: list[str] = []
    experience_gaps: list[str] = []
    transferable_strengths: list[str] = []
    question_hooks: list[QuestionHook] = []
    opening_question_suggestion: str = ""
    high_priority_topics: list[str] = []
    company_focus_areas: list[str] = []
    candidate_strengths_to_test: list[str] = []


class BulletSuggestion(BaseModel):
    original: str | None = None
    suggested: str
    reason: str


class ResumeAdvice(BaseModel):
    skills_to_add: list[str] = []
    skills_to_highlight: list[str] = []
    projects_to_emphasize: list[str] = []
    gaps_to_address: list[str] = []
    bullet_point_suggestions: list[BulletSuggestion] = []
    overall_fit_score: float = 0.0
    summary: str = ""
