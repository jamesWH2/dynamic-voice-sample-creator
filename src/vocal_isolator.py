#!/usr/bin/env python3
"""
V10 — Hybrid SOTA Vocal Isolation Pipeline.

Pipeline:
  Pass 1: htdemucs_ft (Demucs v4)       — SDR 10.8, best working vocal extractor
  Pass 2: FFmpeg afftdn                  — Adaptive spectral noise reduction
  Pass 3: Kim_Vocal_2.onnx (MDX-Net)    — Fine-grained vocal clean-up pass

Roformer ckpt models are listed as available but fail to load on Python 3.14
due to PyTorch archive format incompatibility post-beartype upgrade.
Demucs v4 (htdemucs_ft) uses a different model loading path and is fully functional.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List
import torch


SKILL_DIR = Path(__file__).parent.parent
VENV_BIN = SKILL_DIR / "venv" / "bin"
AUDIO_SEP = VENV_BIN / "audio-separator"
MODEL_DIR = Path("/tmp/audio-separator-models")

# V10 Models — in order of pipeline
PASS1_MODEL = "htdemucs_ft.yaml"       # Demucs v4 — vocal stem (SDR 10.8)
PASS3_MODEL = "Kim_Vocal_2.onnx"       # MDX-Net ONNX — fine clean-up (proven working)


def _audio_sep_cmd() -> str:
    if AUDIO_SEP.exists():
        return str(AUDIO_SEP)
    return "audio-separator"


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _run_separator(input_path: Path, model: str, stem_keyword: str, out_dir: Path) -> Path:
    """
    Run one audio-separator pass and return the output stem file.
    stem_keyword: substring to identify the desired output (e.g. 'Vocals', 'vocals')
    """
    device = _get_device()
    cmd = [
        _audio_sep_cmd(),
        str(input_path),
        "-m", model,
        "--model_file_dir", str(MODEL_DIR),
        "--output_dir", str(out_dir),
        "--output_format", "WAV",
        "--sample_rate", "44100",
    ]
    if device == "cpu":
        cmd.append("--cpu")

    print(f"[ISOLATOR] Pass: {model} on {input_path.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

    if result.returncode != 0:
        print(f"[ISOLATOR] STDERR: {result.stderr[-1500:]}")
        raise RuntimeError(f"audio-separator failed [{model}]: {result.stderr[-400:]}")

    # Find the output file by stem keyword
    stem_kw_lower = stem_keyword.lower()
    candidates = sorted(
        [f for f in out_dir.glob("*.wav")
         if stem_kw_lower in f.name.lower() or stem_keyword in f.name],
        key=lambda f: f.stat().st_mtime, reverse=True
    )

    if not candidates:
        available = [f.name for f in out_dir.glob("*.wav")]
        raise RuntimeError(
            f"No '{stem_keyword}' output found for {input_path.name}. Files: {available}"
        )

    best = candidates[0]
    print(f"[ISOLATOR]   → {best.name}")
    return best


def _run_ffmpeg_denoise(input_path: Path, output_path: Path) -> Path:
    """
    Pass 2: Adaptive spectral noise reduction via FFmpeg afftdn.
    afftdn is a real-time adaptive filter — doesn't need a noise profile.
    nr=10: noise reduction strength (0-97 dB), nt=w: wideband noise type
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", "afftdn=nr=10:nf=-25:nt=w",
        "-ar", "44100",
        str(output_path)
    ]
    print(f"[ISOLATOR] Pass 2 (FFmpeg afftdn denoise): {input_path.name} → {output_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"[ISOLATOR] FFmpeg STDERR: {result.stderr[-500:]}")
        raise RuntimeError(f"FFmpeg afftdn failed: {result.stderr[-300:]}")
    return output_path


class VocalIsolator:
    """
    V10 Hybrid Pipeline:
    1) Demucs v4 (htdemucs_ft)   — Best available vocal extraction
    2) FFmpeg afftdn              — Adaptive spectral noise reduction
    3) Kim_Vocal_2 MDX-Net        — Fine-grained vocal isolation pass
    """

    def __init__(self, output_dir: str, model_name: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = _get_device()
        print(f"[ISOLATOR] V10 Hybrid Pipeline — Device: {self.device.upper()}")

    def isolate(self, input_path: str) -> str:
        input_path = Path(input_path)
        output_final = self.output_dir / f"{input_path.stem}_vocals.wav"

        if output_final.exists():
            print(f"[ISOLATOR] Cached: {output_final.name}")
            return str(output_final)

        work_dir = self.output_dir / f"_work_{input_path.stem}"
        work_dir.mkdir(exist_ok=True)

        try:
            # ── PASS 1: Demucs v4 vocal extraction ───────────────────────
            p1_out = _run_separator(input_path, PASS1_MODEL, "Vocals", work_dir)
            # Clean up non-vocal Demucs stems to save space
            for stem_name in ["Bass", "Drums", "Other", "Guitar", "Piano"]:
                for f in work_dir.glob(f"*({stem_name})*.wav"):
                    f.unlink(missing_ok=True)
                for f in work_dir.glob(f"*_{stem_name}_*.wav"):
                    f.unlink(missing_ok=True)

            # ── PASS 2: FFmpeg adaptive spectral denoising ───────────────
            p2_out = work_dir / f"{input_path.stem}_denoised.wav"
            _run_ffmpeg_denoise(p1_out, p2_out)

            # ── PASS 3: Kim_Vocal_2 MDX-Net fine clean-up ───────────────
            p3_dir = work_dir / "p3"
            p3_dir.mkdir(exist_ok=True)
            p3_out = _run_separator(p2_out, PASS3_MODEL, "Vocals", p3_dir)

            # Move final result to canonical output path
            shutil.move(str(p3_out), str(output_final))
            print(f"[ISOLATOR] ✅ V10 studio-grade vocals: {output_final.name}")
            return str(output_final)

        except Exception as e:
            # Fallback: if Pass 3 fails, try returning pass 2 output
            p2_fallback = work_dir / f"{input_path.stem}_denoised.wav"
            if p2_fallback.exists():
                shutil.move(str(p2_fallback), str(output_final))
                print(f"[ISOLATOR] ⚠️ Fallback to Demucs+Denoise (Pass 3 failed): {e}")
                return str(output_final)
            # Last resort: pass 1 only
            p1_candidates = sorted(work_dir.glob("*(Vocals)*.wav"))
            if p1_candidates:
                shutil.move(str(p1_candidates[0]), str(output_final))
                print(f"[ISOLATOR] ⚠️ Fallback to Demucs only (Pass 2+3 failed): {e}")
                return str(output_final)
            raise RuntimeError(f"Vocal isolation completely failed: {e}")
        finally:
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)

    def isolate_all(self, input_paths: List[str]) -> List[str]:
        results = []
        print(f"[ISOLATOR] V10 processing {len(input_paths)} files...")
        for i, path in enumerate(input_paths, 1):
            print(f"\n[ISOLATOR] [{i}/{len(input_paths)}] {Path(path).name}")
            results.append(self.isolate(path))
        return results


def check_audio_separator() -> bool:
    if AUDIO_SEP.exists():
        return True
    try:
        r = subprocess.run(["audio-separator", "--version"], capture_output=True, text=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False
