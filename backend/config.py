from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    GROQ_API_KEY: str = Field(..., description="Groq API key for vision and LLM inference")
    VISION_MODEL: str = Field(default="llama-3.2-90b-vision-preview")
    LLM_MODEL: str = Field(default="llama-3.3-70b-versatile")
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")
    WEIGHT_SEMANTIC: float = Field(default=0.4)
    WEIGHT_LLM: float = Field(default=0.6)
    MAX_FILE_SIZE_MB: int = Field(default=10)
    TEMP_DIR: str = Field(default="./temp_uploads")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()