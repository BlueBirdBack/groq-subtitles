import asyncio
import ffmpeg
import logging
from pathlib import Path
from typing import List, Optional, Set
import json
from concurrent.futures import ThreadPoolExecutor
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from .transcriber import WhisperTranscriber
from .config import Config


class VideoProcessor:
    SUPPORTED_FORMATS = {".mp4", ".mkv", ".avi", ".mov"}

    def __init__(self, config: Config):
        self.config = config
        self.transcriber = WhisperTranscriber(config.groq_api_key, config.language)
        self.logger = self._setup_logger()
        self._processed_files: Set[Path] = self._load_progress()

    def process_videos(self, input_path: Path) -> None:
        if input_path.is_file():
            video_files = [input_path] if self._is_supported_video(input_path) else []
        else:
            video_files = self._collect_video_files(input_path)

        if not video_files:
            self.logger.warning("No supported video files found")
            return

        unprocessed_files = [f for f in video_files if f not in self._processed_files]
        if not unprocessed_files:
            self.logger.info("All files have been processed")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(
                "Processing videos...", total=len(unprocessed_files)
            )

            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                loop = asyncio.get_event_loop()
                futures = []

                for video_file in unprocessed_files:
                    future = loop.run_in_executor(
                        executor, self._process_single_video, video_file, progress, task
                    )
                    futures.append(future)

                loop.run_until_complete(asyncio.gather(*futures))

    def _process_single_video(
        self, video_path: Path, progress: Progress, task_id: int
    ) -> None:
        try:
            # Extract audio
            audio_path = video_path.with_suffix(".m4a")
            self._extract_audio(video_path, audio_path)

            # Transcribe
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            transcription = loop.run_until_complete(
                self.transcriber.transcribe_audio(audio_path)
            )

            # Save subtitles
            output_path = video_path.with_suffix(f".{self.config.output_format}")
            self.transcriber.save_subtitles(
                transcription, output_path, self.config.output_format
            )

            # Cleanup
            audio_path.unlink()

            self._processed_files.add(video_path)
            self._save_progress()

            self.logger.info(f"Successfully processed: {video_path}")
            progress.update(task_id, advance=1)

        except Exception as e:
            self.logger.error(f"Failed to process {video_path}: {str(e)}")
            progress.update(task_id, advance=1)

    def _extract_audio(self, video_path: Path, audio_path: Path) -> None:
        """
        Extract and preprocess audio from video using FFmpeg.
        Converts to 16kHz mono audio as required by Groq API.
        """
        try:
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(
                stream,
                str(audio_path),
                acodec="pcm_s16le",  # 16-bit PCM
                ac=1,  # mono
                ar=16000,  # 16kHz sampling rate
                map="0:a:0",  # select first audio track
            )
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            raise Exception(f"FFmpeg error: {e.stderr.decode()}")

    def _collect_video_files(self, directory: Path) -> List[Path]:
        if self.config.recursive:
            return [f for f in directory.rglob("*") if self._is_supported_video(f)]
        return [f for f in directory.iterdir() if self._is_supported_video(f)]

    def _is_supported_video(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in self.SUPPORTED_FORMATS

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("subtitles")
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(self.config.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def _load_progress(self) -> Set[Path]:
        progress_file = Path("subtitle_progress.json")
        if progress_file.exists():
            with open(progress_file, "r") as f:
                return {Path(p) for p in json.load(f)}
        return set()

    def _save_progress(self) -> None:
        with open("subtitle_progress.json", "w") as f:
            json.dump([str(p) for p in self._processed_files], f)
