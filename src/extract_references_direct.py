#!/usr/bin/env python3
"""
Direct Reference Extractor — V10 "Curated Mode"

Instead of speaker matching from full videos, this extracts EXACTLY the
user-specified time ranges (which are already curated/verified to contain
the target speaker), applies the full V10 isolation chain, and stitches them.

This is much more reliable than trying to speaker-match from full videos.
"""

import sys
import json
import subprocess
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import torch

SKILL_DIR = Path(__file__).parent.parent
VENV_BIN = SKILL_DIR / "venv" / "bin"
VENV_PYTHON = VENV_BIN / "python"

sys.path.insert(0, str(SKILL_DIR / "src"))

from downloader import AudioDownloader
from vocal_isolator import VocalIsolator


def extract_clip(source_wav: Path, start: float, end: float, out_path: Path):
    """Extract exact time range from a WAV file using FFmpeg."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source_wav),
        "-ss", str(start),
        "-t", str(duration),
        "-ar", "44100",
        "-ac", "1",
        str(out_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg clip failed: {result.stderr[-300:]}")
    return out_path


def apply_loudnorm(path: Path, target_lufs: float = -16.0) -> Path:
    """Apply EBU R128 loudness normalization."""
    tmp = path.with_suffix("._norm.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(path),
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        "-ar", "24000",
        "-ac", "1",
        str(tmp)
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    shutil.move(str(tmp), str(path))
    return path


def stitch_with_pauses(clips: list, output_path: Path,
                       pause_between_ms: int = 350,
                       sample_rate: int = 24000):
    """Stitch clips together with natural pauses between them."""
    from pydub import AudioSegment
    silence = AudioSegment.silent(duration=pause_between_ms, frame_rate=sample_rate)

    result = None
    for clip_path in clips:
        seg = AudioSegment.from_wav(str(clip_path))
        seg = seg.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
        if result is None:
            result = seg
        else:
            result = result + silence + seg

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.export(str(output_path), format="wav", parameters=["-acodec", "pcm_s16le"])
    return output_path


def main(config_path: str):
    with open(config_path) as f:
        config = json.load(f)

    name = config.get("name", "voice")
    references = config.get("reference_sequences", [])
    output_file = Path(config.get("output_file", "output.wav"))
    crossfade_ms = config.get("crossfade_ms", 50)

    print(f"\n=== DIRECT REFERENCE EXTRACTION: {name} ===")
    print(f"  {len(references)} curated reference segments")

    workspace = Path(tempfile.mkdtemp(prefix=f"direct_extract_{name}_"))
    download_dir = workspace / "downloads"
    isolated_dir = workspace / "isolated"
    clips_dir = workspace / "clips"

    for d in [download_dir, isolated_dir, clips_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n[STEP 1] Downloading {len(set(r['url'] for r in references))} unique sources...")
    downloader = AudioDownloader(output_dir=str(download_dir), sample_rate=44100)

    # Get unique URLs
    unique_urls = list(dict.fromkeys(r["url"] for r in references))
    url_to_path = downloader.download_all(unique_urls)
    print(f"  Downloaded {len(url_to_path)} files")

    print(f"\n[STEP 2] V10 Triple-Pass Isolation on all source files...")
    isolator = VocalIsolator(output_dir=str(isolated_dir))
    isolated_map = {}
    for url, raw_path in url_to_path.items():
        isolated_path = isolator.isolate(raw_path)
        isolated_map[url] = isolated_path
    print(f"  Isolated {len(isolated_map)} files")

    print(f"\n[STEP 3] Extracting exact curated time ranges...")
    extracted_clips = []
    total_duration = 0.0

    for i, ref in enumerate(references, 1):
        url = ref["url"]
        start = float(ref["start_time"])
        end = float(ref["end_time"])
        duration = end - start
        total_duration += duration

        isolated_path = isolated_map.get(url)
        if not isolated_path:
            print(f"  [{i}] WARN: No isolated file for {url}")
            continue

        clip_path = clips_dir / f"clip_{i:02d}_{start:.1f}-{end:.1f}.wav"
        extract_clip(Path(isolated_path), start, end, clip_path)

        # Per-clip loudnorm
        apply_loudnorm(clip_path)
        extracted_clips.append(clip_path)

        print(f"  [{i}] {url.split('/')[-1]} → {start:.1f}s-{end:.1f}s ({duration:.1f}s) ✓")

    print(f"\n  Total curated audio: {total_duration:.1f}s from {len(extracted_clips)} clips")

    print(f"\n[STEP 4] Applying EQ (highpass 100Hz, lowpass 10000Hz) + cross-source equalization...")
    # Stitch first, then apply global EQ + dynaudnorm
    raw_stitched = workspace / "stitched_raw.wav"
    stitch_with_pauses(extracted_clips, raw_stitched, pause_between_ms=350)

    # Global EQ
    eq_out = workspace / "stitched_eq.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(raw_stitched),
        "-af", "highpass=f=100,lowpass=f=10000",
        "-ar", "24000", "-ac", "1", str(eq_out)
    ], capture_output=True, check=True)

    # Cross-source equalization
    eq2_out = workspace / "stitched_eq2.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(eq_out),
        "-af", "dynaudnorm=g=15:f=500:r=0.95,acompressor=threshold=-24dB:ratio=2:attack=20:release=250",
        "-ar", "24000", "-ac", "1", str(eq2_out)
    ], capture_output=True, check=True)

    # Final loudnorm
    apply_loudnorm(eq2_out)

    # Move to final output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(eq2_out), str(output_file))

    import os
    final_size = os.path.getsize(str(output_file))
    print(f"\n[DONE] Output: {output_file}")
    print(f"       Duration: {total_duration:.1f}s | Size: {final_size/1024/1024:.1f}MB")

    # Cleanup
    shutil.rmtree(workspace, ignore_errors=True)
    return str(output_file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
