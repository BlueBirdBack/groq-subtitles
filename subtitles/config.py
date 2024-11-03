import os
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, Literal


class WhisperModel(str, Enum):
    LARGE_V3 = "whisper-large-v3"
    LARGE_V3_TURBO = "whisper-large-v3-turbo"
    DISTIL_ENGLISH = "distil-whisper-large-v3-en"


class ResponseFormat(str, Enum):
    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    TEXT = "text"


@dataclass
class Config:
    groq_api_key: str
    model: WhisperModel = WhisperModel.LARGE_V3_TURBO
    output_format: str = "srt"
    response_format: ResponseFormat = ResponseFormat.VERBOSE_JSON
    num_workers: int = 4
    temperature: Optional[float] = 0.0
    language: Optional[str] = None
    recursive: bool = False
    log_file: str = "subtitle_generation.log"
    prompt: Optional[str] = None

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "Config":
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            model=WhisperModel(os.getenv("MODEL", "whisper-large-v3-turbo")),
            output_format=os.getenv("OUTPUT_FORMAT", "srt"),
            num_workers=int(os.getenv("NUM_WORKERS", "4")),
            language=os.getenv("LANGUAGE", "en"),
            recursive=bool(os.getenv("RECURSIVE", "False")),
            log_file=os.getenv("LOG_FILE", "subtitle_generation.log"),
            response_format=ResponseFormat(
                os.getenv("RESPONSE_FORMAT", "verbose_json")
            ),
            temperature=(
                float(os.getenv("TEMPERATURE")) if os.getenv("TEMPERATURE") else None
            ),
            prompt=os.getenv("PROMPT"),
        )
