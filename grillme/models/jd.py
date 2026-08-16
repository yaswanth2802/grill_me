from pydantic import BaseModel


class JDProfile(BaseModel):
    company: str
    role: str
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    responsibilities: list[str] = []
    experience_required: str = ""
    domain: str = ""
