import aiohttp
import asyncio
import json
from pathlib import Path
from typing import Dict, Optional, Union
from .config import WhisperModel, ResponseFormat


class WhisperTranscriber:
    """
    A client for the Groq Whisper API that handles audio transcription.

    Supports multiple Whisper models:
    - whisper-large-v3: Best accuracy, multilingual
    - whisper-large-v3-turbo: Fast, good accuracy, multilingual
    - distil-whisper-large-v3-en: Fastest, English-only

    File limitations:
    - Max size: 25 MB
    - Min length: 0.01 seconds (10 seconds minimum billing)
    - Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm
    """

    SUPPORTED_AUDIO_FORMATS = {
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mpga",
        ".m4a",
        ".wav",
        ".webm",
    }
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

    def __init__(
        self,
        api_key: str,
        model: WhisperModel = WhisperModel.LARGE_V3_TURBO,
        language: Optional[str] = None,
        response_format: ResponseFormat = ResponseFormat.VERBOSE_JSON,
        temperature: Optional[float] = None,
        prompt: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.language = language
        self.response_format = response_format
        self.temperature = temperature
        self.prompt = prompt
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe_audio(self, audio_path: Path) -> Dict:
        # Validate file size
        file_size = audio_path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds maximum of {self.MAX_FILE_SIZE} bytes"
            )

        # Validate file format
        if audio_path.suffix.lower() not in self.SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. Supported formats: {self.SUPPORTED_AUDIO_FORMATS}"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        async with aiohttp.ClientSession() as session:
            with open(audio_path, "rb") as audio_file:
                form_data = aiohttp.FormData()
                form_data.add_field(
                    "file",
                    audio_file,
                    filename=audio_path.name,
                    content_type="audio/mpeg",
                )
                form_data.add_field("model", self.model)
                form_data.add_field("response_format", self.response_format)

                if self.language and self.model != WhisperModel.DISTIL_ENGLISH:
                    form_data.add_field("language", self.language)
                if self.temperature is not None:
                    form_data.add_field("temperature", str(self.temperature))
                if self.prompt:
                    form_data.add_field("prompt", self.prompt)

                async with session.post(
                    self.api_url, headers=headers, data=form_data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"Transcription failed (HTTP {response.status}): {error_text}"
                        )

                    if self.response_format == ResponseFormat.TEXT:
                        return {"text": await response.text()}
                    return await response.json()

    def save_subtitles(
        self, transcription: Dict, output_path: Path, format: str = "srt"
    ) -> None:
        if format == "srt":
            self._save_srt(transcription, output_path)
        elif format == "vtt":
            self._save_vtt(transcription, output_path)
        elif format == "txt":
            self._save_txt(transcription, output_path)
        else:
            raise ValueError(f"Unsupported subtitle format: {format}")

    def _save_srt(self, transcription: Dict, output_path: Path) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(transcription["segments"], 1):
                start = self._format_timestamp(segment["start"])
                end = self._format_timestamp(segment["end"])
                f.write(f"{i}\n{start} --> {end}\n{segment['text']}\n\n")

    def _save_vtt(self, transcription: Dict, output_path: Path) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for segment in transcription["segments"]:
                start = self._format_timestamp(segment["start"], vtt=True)
                end = self._format_timestamp(segment["end"], vtt=True)
                f.write(f"{start} --> {end}\n{segment['text']}\n\n")

    def _save_txt(self, transcription: Dict, output_path: Path) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            for segment in transcription["segments"]:
                f.write(f"{segment['text']}\n")

    @staticmethod
    def _format_timestamp(seconds: float, vtt: bool = False) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msecs = int((seconds - int(seconds)) * 1000)

        if vtt:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{msecs:03d}"
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"
