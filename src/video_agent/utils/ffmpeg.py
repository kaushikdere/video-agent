"""ffmpeg utilities — stitch, normalise, extract frames."""
from __future__ import annotations

import subprocess
from pathlib import Path


def stitch_clips(clip_paths: list[str], output_path: str) -> None:
    """
    Concatenate clips using ffmpeg concat demuxer.
    All clips must have same codec/resolution.
    """
    if not clip_paths:
        raise ValueError("No clips to stitch")

    # Build concat list file
    list_path = Path(output_path).with_suffix(".txt")
    with open(list_path, "w") as f:
        for path in clip_paths:
            f.write(f"file '{path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    list_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg stitch failed: {result.stderr.decode()[:300]}")


def add_music_bed(video_path: str, audio_path: str, output_path: str) -> None:
    """Mix a music bed under the video at -20dBFS."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-filter_complex",
        "[1:a]volume=-20dB[music];[0:a][music]amix=inputs=2:duration=first[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg music bed failed: {result.stderr.decode()[:300]}")


def extract_final_frame(clip_path: str, output_path: str) -> None:
    """Extract the very last frame of a clip as PNG."""
    cmd = [
        "ffmpeg", "-y",
        "-sseof", "-0.1",
        "-i", clip_path,
        "-frames:v", "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extract failed: {result.stderr.decode()[:200]}")


def probe_duration(clip_path: str) -> float:
    """Return duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        clip_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=10)
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.decode().strip())
    except ValueError:
        return 0.0
