import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "🛡️ Enterprise Smart ATS CV Matching Core"
    
    # Model Registry Configuration
    EMBEDDING_MODEL_REGISTRY: dict = {
        "v1-mini": "all-MiniLM-L6-v2",
        "v2-bge": "BAAI/bge-small-en-v1.5"
    }
    ACTIVE_MODEL_VERSION: str = "v1-mini"
    
    # Default Sourcing Parameters
    DEFAULT_WEIGHT_EMBEDDING: float = 0.6
    DEFAULT_WEIGHT_TFIDF: float = 0.4
    
    class Config:
        env_file = ".env"

settings = Settings()