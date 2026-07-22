from pydantic import BaseModel, Field
from typing import List, Optional

class SkillAnalytics(BaseModel):
    matched_skills: List[str]
    missing_skills: List[str]
    skill_match_rate: float

class PerformanceMetrics(BaseModel):
    processing_time_seconds: float

class CandidateProfile(BaseModel):
    candidate_id: str
    name: str
    email: str
    phone: str
    linkedin: str
    github: str
    experience_years: float
    education_tier: str
    extracted_skills: List[str]

class MatchResult(BaseModel):
    profile: CandidateProfile
    tfidf_score: float
    embedding_score: float
    final_hybrid_score: float
    explainability: SkillAnalytics
    recommendation: str

class BatchATSResponse(BaseModel):
    job_description_summary: str
    total_processed: int
    duplicates_detected: int
    engine_model_used: str
    leaderboard: List[MatchResult]
    performance_summary: PerformanceMetrics