import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from subtitles.processor import VideoProcessor
from subtitles.config import Config, WhisperModel, ResponseFormat


@pytest.fixture
def mock_config():
    return Config(
        groq_api_key="test_key",
        model=WhisperModel.LARGE_V3_TURBO,
        output_format="srt",
        response_format=ResponseFormat.VERBOSE_JSON,
        num_workers=1,
        language="en",
    )


@pytest.fixture
def processor(mock_config, tmp_path):
    with patch("subtitles.processor.WhisperTranscriber"):
        processor = VideoProcessor(mock_config)
        # Mock logger to avoid file creation
        processor.logger = Mock()
        return processor


def test_supported_video_formats(processor):
    """Test video format detection"""
    assert processor._is_supported_video(Path("video.mp4"))
    assert processor._is_supported_video(Path("video.mkv"))
    assert processor._is_supported_video(Path("video.avi"))
    assert processor._is_supported_video(Path("video.mov"))
    assert not processor._is_supported_video(Path("video.txt"))
    assert not processor._is_supported_video(Path("video.wmv"))


def test_collect_video_files(processor, tmp_path):
    """Test video file collection"""
    # Create test directory structure
    video1 = tmp_path / "test1.mp4"
    video2 = tmp_path / "test2.mkv"
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    video3 = subdir / "test3.avi"

    # Create empty files
    video1.touch()
    video2.touch()
    video3.touch()

    # Test non-recursive collection
    files = processor._collect_video_files(tmp_path)
    assert len(files) == 2
    assert video1 in files
    assert video2 in files
    assert video3 not in files

    # Test recursive collection
    processor.config.recursive = True
    files = processor._collect_video_files(tmp_path)
    assert len(files) == 3
    assert video1 in files
    assert video2 in files
    assert video3 in files


@pytest.mark.asyncio
async def test_process_single_video(processor, tmp_path):
    """Test single video processing"""
    video_path = tmp_path / "test.mp4"
    video_path.touch()

    # Mock progress tracking
    progress = Mock()
    task_id = 1

    # Mock ffmpeg
    with (
        patch("ffmpeg.run"),
        patch("ffmpeg.output", return_value=Mock()),
        patch("ffmpeg.input", return_value=Mock()),
    ):

        # Mock transcriber response
        mock_transcription = {
            "segments": [{"start": 0, "end": 2, "text": "Test subtitle"}]
        }
        processor.transcriber.transcribe_audio = Mock(return_value=mock_transcription)

        # Process video
        await processor._process_single_video(video_path, progress, task_id)

        # Verify progress was updated
        progress.update.assert_called_with(task_id, advance=1)

        # Verify subtitle file was created
        subtitle_path = video_path.with_suffix(".srt")
        assert subtitle_path.exists()


def test_process_videos_empty_directory(processor, tmp_path):
    """Test handling of empty directory"""
    processor.process_videos(tmp_path)
    processor.logger.warning.assert_called_with("No supported video files found")


def test_process_videos_all_processed(processor, tmp_path):
    """Test handling of already processed files"""
    # Create a test video file
    video_path = tmp_path / "test.mp4"
    video_path.touch()

    # Mark it as processed
    processor._processed_files.add(video_path)

    processor.process_videos(tmp_path)
    processor.logger.info.assert_called_with("All files have been processed")


@pytest.mark.asyncio
async def test_error_handling(processor, tmp_path):
    """Test error handling during processing"""
    video_path = tmp_path / "test.mp4"
    video_path.touch()

    progress = Mock()
    task_id = 1

    # Simulate ffmpeg error
    with patch("ffmpeg.run", side_effect=Exception("FFmpeg error")):
        await processor._process_single_video(video_path, progress, task_id)

        # Verify error was logged
        processor.logger.error.assert_called()
        # Verify progress was still updated
        progress.update.assert_called_with(task_id, advance=1)


def test_load_save_progress(processor, tmp_path):
    """Test progress tracking persistence"""
    # Create test video paths
    video1 = Path("test1.mp4")
    video2 = Path("test2.mp4")

    # Add to processed files
    processor._processed_files.add(video1)
    processor._processed_files.add(video2)

    # Save progress
    processor._save_progress()

    # Create new processor instance
    new_processor = VideoProcessor(processor.config)

    # Verify loaded progress matches saved progress
    assert video1 in new_processor._processed_files
    assert video2 in new_processor._processed_files
