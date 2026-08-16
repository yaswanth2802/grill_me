from pydantic import BaseModel


class QuestionDetail(BaseModel):
    question_text: str
    question_type: str
    user_answer_summary: str
    code_submitted: str | None = None
    score: float = 0.0
    strengths: list[str] = []
    weaknesses: list[str] = []
    missed_points: list[str] = []
    is_follow_up: bool = False


class WeaknessPattern(BaseModel):
    pattern: str
    evidence: list[str] = []
    severity: str


class Recommendation(BaseModel):
    topic: str
    why: str
    suggested_resources: list[str] = []


class GrindReport(BaseModel):
    overall_score: float = 0.0
    category_scores: dict[str, float] = {}
    questions: list[QuestionDetail] = []
    weakness_patterns: list[WeaknessPattern] = []
    recommendations: list[Recommendation] = []
    session_duration_minutes: float = 0.0
    total_questions: int = 0
