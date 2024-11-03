import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from subtitles.transcriber import WhisperTranscriber


@pytest.fixture
def transcriber():
    return WhisperTranscriber("fake_api_key")


@pytest.fixture
def sample_response():
    return {"segments": [{"start": 0.0, "end": 2.5, "text": "Hello world"}]}


@pytest.mark.asyncio
async def test_transcribe_audio(transcriber, sample_response, tmp_path):
    # Create a test audio file
    audio_file = tmp_path / "test.m4a"
    audio_file.write_bytes(b"fake audio content")

    # Mock the API response
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json.return_value = sample_response

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value = mock_response
        result = await transcriber.transcribe_audio(audio_file)

        assert result == sample_response
