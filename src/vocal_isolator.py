#!/usr/bin/env python3
"""
Vocal isolation using audio-separator with MDX-Net models.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Tuple
import torch


class VocalIsolator:
    """Isolate vocals from audio using MDX-Net models."""
    
    def __init__(self, output_dir: str, model_name: str = "Kim_Vocal_2.onnx"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ISOLATOR] Device: {self.device.upper()}")
    
    def isolate(self, input_path: str) -> str:
        """
        Isolate vocals from audio file.
        Returns path to vocal-only WAV file.
        """
        input_path = Path(input_path)
        output_vocals = self.output_dir / f"{input_path.stem}_vocals.wav"
        
        if output_vocals.exists():
            print(f"[ISOLATOR] Vocals exist: {output_vocals}")
            return str(output_vocals)
        
        print(f"[ISOLATOR] Processing: {input_path.name}")
        
        # Use audio-separator CLI
        cmd = [
            "audio-separator",
            str(input_path),
            "-m", self.model_name,
            "--output_dir", str(self.output_dir),
            "--output_format", "WAV",
            "--sample_rate", "44100",
        ]
        
        # Add CPU flag if no CUDA
        if self.device == "cpu":
            cmd.append("--cpu")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                print(f"[ISOLATOR] STDERR: {result.stderr}")
                raise RuntimeError(f"audio-separator failed: {result.stderr}")
            
            # Find the vocals file (audio-separator adds suffix)
            vocals_files = list(self.output_dir.glob(f"{input_path.stem}*(Vocals)*.wav"))
            
            if not vocals_files:
                # Try alternative naming
                vocals_files = list(self.output_dir.glob(f"{input_path.stem}_vocals.wav"))
            
            if vocals_files:
                # Rename to consistent name
                actual_vocals = vocals_files[0]
                if actual_vocals != output_vocals:
                    actual_vocals.rename(output_vocals)
                
                # Clean up instrumental file
                instrumental_files = list(self.output_dir.glob(f"{input_path.stem}*(Instrumental)*.wav"))
                for inst_file in instrumental_files:
                    inst_file.unlink()
                    print(f"[ISOLATOR] Removed instrumental: {inst_file.name}")
                
                print(f"[ISOLATOR] Saved vocals: {output_vocals}")
                return str(output_vocals)
            else:
                raise RuntimeError(f"Could not find vocals output for {input_path}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"audio-separator timed out for {input_path}")
        except Exception as e:
            raise RuntimeError(f"Vocal isolation failed: {e}")
    
    def isolate_all(self, input_paths: List[str]) -> List[str]:
        """
        Isolate vocals from all input files.
        Returns list of vocal-only file paths.
        """
        results = []
        print(f"[ISOLATOR] Processing {len(input_paths)} files...")
        
        for i, input_path in enumerate(input_paths, 1):
            print(f"[ISOLATOR] [{i}/{len(input_paths)}] {Path(input_path).name}")
            try:
                vocal_path = self.isolate(input_path)
                results.append(vocal_path)
            except Exception as e:
                print(f"[ISOLATOR] ERROR: {e}")
                raise
        
        return results


def check_audio_separator() -> bool:
    """Check if audio-separator is installed."""
    try:
        result = subprocess.run(
            ["audio-separator", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
