import typer
from pathlib import Path
from typing import Optional
from rich import print
from .processor import VideoProcessor
from .config import Config, WhisperModel, ResponseFormat

app = typer.Typer()


@app.command()
def process(
    input_path: Path = typer.Argument(..., help="Video file or directory to process"),
    groq_api_key: Optional[str] = typer.Option(
        None, "--groq-api-key", "-k", help="Groq API key"
    ),
    model: WhisperModel = typer.Option(
        WhisperModel.LARGE_V3_TURBO,
        "--model",
        "-m",
        help="Whisper model to use for transcription. Options: "
        "large_v3 (best accuracy), "
        "large_v3_turbo (fast, good accuracy), "
        "distil_english (fastest, English-only)",
    ),
    lang: Optional[str] = typer.Option(
        None,
        "--lang",
        "-l",
        help="Language code for transcription (ISO 639-1, e.g., 'en' for English)",
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Process directories recursively"
    ),
    output_format: str = typer.Option(
        "srt", "--format", "-f", help="Output subtitle format (srt/vtt/txt)"
    ),
    response_format: ResponseFormat = typer.Option(
        ResponseFormat.VERBOSE_JSON,
        "--response",
        "-R",  # Changed from --r to -R to avoid conflict with --recursive
        help="API response format (json/verbose_json/text)",
        case_sensitive=False,
    ),
    workers: int = typer.Option(
        4, "--workers", "-w", help="Number of parallel workers", min=1
    ),
    temperature: Optional[float] = typer.Option(
        None, "--temperature", "-t", help="Model temperature (0-1)", min=0.0, max=1.0
    ),
    prompt: Optional[str] = typer.Option(
        None, "--prompt", "-p", help="Optional prompt for context or spelling guidance"
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env", "-e", help="Path to .env file"
    ),
) -> None:
    """
    Generate subtitles for video files using the Groq Whisper API.
    """
    try:
        config = Config.from_env(env_file)

        config.groq_api_key = groq_api_key or config.groq_api_key
        config.model = model
        config.language = lang or config.language
        config.recursive = recursive
        config.output_format = output_format.lower()  # Ensure lowercase format
        config.response_format = response_format
        config.num_workers = max(1, workers)  # Ensure at least 1 worker
        config.temperature = temperature or config.temperature
        config.prompt = prompt or config.prompt

        # Validate essential configurations
        if not config.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables or CLI arguments"
            )

        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")

        if output_format.lower() not in ["srt", "vtt", "txt"]:
            raise ValueError(f"Unsupported output format: {output_format}")

        processor = VideoProcessor(config)
        processor.process_videos(input_path)

        print("[green]✓ Subtitle generation completed successfully![/green]")

    except Exception as e:
        print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
