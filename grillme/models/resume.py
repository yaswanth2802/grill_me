from pydantic import BaseModel


class ExperienceEntry(BaseModel):
    role: str
    company: str
    duration: str
    highlights: list[str] = []
    technologies: list[str] = []


class Project(BaseModel):
    name: str
    description: str
    metrics: list[str] = []
    technologies: list[str] = []


class ResumeProfile(BaseModel):
    name: str
    skills: list[str] = []
    experience_entries: list[ExperienceEntry] = []
    projects: list[Project] = []
    education: list[str] = []
    certifications: list[str] = []
    total_experience_years: float = 0.0
    domain_expertise: list[str] = []
