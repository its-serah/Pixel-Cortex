"""
Configuration settings for the application
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "sqlite:///./pixel_cortex.db"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    
    # API
    API_V1_STR: str = "/api/v1"
    
    # LLM
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    class Config:
        env_file = ".env"


# Create settings instance
settings = Settings()
