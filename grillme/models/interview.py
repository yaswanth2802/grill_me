from pydantic import BaseModel, Field


class CodeFeedback(BaseModel):
    correctness: str
    time_complexity: str
    space_complexity: str


class Evaluation(BaseModel):
    score: float = Field(ge=0, le=10)
    strengths: list[str] = []
    weaknesses: list[str] = []
    missed_points: list[str] = []
    weak_signals: list[str] = []
    code_feedback: CodeFeedback | None = None


class QuestionRecord(BaseModel):
    question_text: str
    question_type: str
    answer_summary: str
    score: float = 0.0
    is_follow_up: bool = False


class QuestionPlan(BaseModel):
    question_type: str
    is_follow_up: bool = False
    topic: str
    directive: str
    resume_hook: str | None = None
    difficulty_adjustment: str = "same"
    reasoning: str
    rationale: str | None = None  # Add this if needed

class ConductorResponse(BaseModel):
    response_text: str
    next_question: str | None = None
