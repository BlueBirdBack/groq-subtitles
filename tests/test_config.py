import pytest
from pathlib import Path
from subtitles.config import Config, WhisperModel, ResponseFormat


def test_config_from_env(tmp_path):
    # Create temporary .env file
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
        GROQ_API_KEY=test_key
        MODEL=whisper-large-v3
        OUTPUT_FORMAT=srt
        NUM_WORKERS=2
        LANGUAGE=en
        RECURSIVE=True
    """.strip()
    )

    config = Config.from_env(env_file)

    assert config.groq_api_key == "test_key"
    assert config.model == WhisperModel.LARGE_V3
    assert config.output_format == "srt"
    assert config.num_workers == 2
    assert config.language == "en"
    assert config.recursive is True


def test_config_default_values():
    config = Config(groq_api_key="test_key")

    assert config.model == WhisperModel.LARGE_V3_TURBO
    assert config.output_format == "srt"
    assert config.num_workers == 4
    assert config.language is None
    assert config.recursive is False
