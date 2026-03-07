#!/usr/bin/env python3
"""
Quality validation using DNSMOS (Deep Noise Suppression Mean Opinion Score).

Microsoft's DNSMOS P.835 model predicts human perceptual scores for:
- SIG: Signal quality (voice clarity, distortion)
- BAK: Background quality (noise level)
- OVRL: Overall quality
"""

import os
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import onnxruntime as ort
import librosa
import soundfile as sf
from pydub import AudioSegment


@dataclass
class DNSMOSThresholds:
    """Quality thresholds for segment acceptance."""
    sig_min: float = 3.5    # Voice signal quality
    bak_min: float = 3.0    # Background noise level (higher = less noise)
    ovrl_min: float = 3.5   # Overall quality


@dataclass
class QualityScore:
    """DNSMOS scores for a segment."""
    sig: float
    bak: float
    ovrl: float
    passed: bool
    reason: Optional[str] = None


class QualityValidator:
    """
    Validates audio segments using DNSMOS P.835 model.
    """

    # Model path
    MODEL_PATH = Path(__file__).parent.parent / "pretrained_models" / "dnsmos" / "sig_bak_ovr.onnx"

    # DNSMOS expects 16kHz audio
    TARGET_SR = 16000
    EXPECTED_SAMPLES = 144160  # ~9 seconds at 16kHz

    def __init__(self, thresholds: DNSMOSThresholds = None):
        self.thresholds = thresholds or DNSMOSThresholds()
        self.session = None  # Lazy load ONNX session

    def _load_model(self):
        """Load ONNX model on first use."""
        if self.session is not None:
            return

        if not self.MODEL_PATH.exists():
            raise RuntimeError(
                f"DNSMOS model not found at {self.MODEL_PATH}. "
                "Download from: https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovrl.onnx"
            )

        self.session = ort.InferenceSession(
            str(self.MODEL_PATH),
            providers=['CPUExecutionProvider']
        )
        print(f"[DNSMOS] Loaded model: {self.MODEL_PATH}")

    def _prepare_audio(self, audio_path: str, start: float, end: float) -> np.ndarray:
        """
        Load and prepare audio segment for DNSMOS.

        Args:
            audio_path: Path to audio file
            start: Start time in seconds
            end: End time in seconds

        Returns:
            Audio samples at 16kHz mono
        """
        # Load full audio
        audio, sr = librosa.load(audio_path, sr=None, mono=True)

        # Extract segment
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = audio[start_sample:end_sample]

        # Resample to 16kHz if needed
        if sr != self.TARGET_SR:
            segment = librosa.resample(segment, orig_sr=sr, target_sr=self.TARGET_SR)

        return segment

    def score_segment(self, audio_path: str, start: float, end: float) -> QualityScore:
        """
        Get DNSMOS scores for an audio segment.

        Args:
            audio_path: Path to audio file
            start: Start time in seconds
            end: End time in seconds

        Returns:
            QualityScore with SIG, BAK, OVRL scores and pass/fail status
        """
        self._load_model()

        # Prepare audio
        audio = self._prepare_audio(audio_path, start, end)

        # Pad or truncate to expected length (9 seconds = 144160 samples)
        if len(audio) < self.EXPECTED_SAMPLES:
            # Pad with zeros
            padding = self.EXPECTED_SAMPLES - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        elif len(audio) > self.EXPECTED_SAMPLES:
            # Truncate
            audio = audio[:self.EXPECTED_SAMPLES]

        # DNSMOS expects [batch, samples]
        input_audio = audio[np.newaxis, :].astype(np.float32)

        # Run inference
        inputs = {self.session.get_inputs()[0].name: input_audio}
        outputs = self.session.run(None, inputs)

        # Output is [N, 3] where columns are [SIG, BAK, OVRL]
        scores = outputs[0][0]  # First batch, all 3 scores
        sig = float(scores[0])
        bak = float(scores[1])
        ovrl = float(scores[2])

        # Check thresholds
        passed = True
        reason = None

        if sig < self.thresholds.sig_min:
            passed = False
            reason = f"SIG {sig:.2f} < {self.thresholds.sig_min}"
        elif bak < self.thresholds.bak_min:
            passed = False
            reason = f"BAK {bak:.2f} < {self.thresholds.bak_min}"
        elif ovrl < self.thresholds.ovrl_min:
            passed = False
            reason = f"OVRL {ovrl:.2f} < {self.thresholds.ovrl_min}"

        return QualityScore(
            sig=sig,
            bak=bak,
            ovrl=ovrl,
            passed=passed,
            reason=reason
        )

    def validate_segment(self, segment, source_path_getter=None) -> Tuple[bool, QualityScore]:
        """
        Validate a segment against quality thresholds.

        Args:
            segment: Segment object with source_path, start, end attributes
            source_path_getter: Optional function to resolve source path

        Returns:
            (passed, QualityScore)
        """
        # Get source path
        source = getattr(segment, 'source_path', getattr(segment, 'source', None))
        if source is None and source_path_getter:
            source = source_path_getter(segment)

        if source is None:
            return False, QualityScore(0, 0, 0, False, "No source path")

        # Get scores
        scores = self.score_segment(source, segment.start, segment.end)

        return scores.passed, scores

    def filter_segments(self, segments: List, verbose: bool = True) -> Tuple[List, List, Dict]:
        """
        Filter segments by DNSMOS quality.

        Args:
            segments: List of segment objects
            verbose: Print progress

        Returns:
            (accepted_segments, rejected_segments, stats)
        """
        accepted = []
        rejected = []

        total_sig = 0
        total_bak = 0
        total_ovrl = 0
        rejection_reasons = {'low_sig': 0, 'low_bak': 0, 'low_ovrl': 0}

        if verbose:
            print(f"\n[DNSMOS] Validating {len(segments)} segments...")
            print(f"[DNSMOS] Thresholds: SIG>={self.thresholds.sig_min}, "
                  f"BAK>={self.thresholds.bak_min}, OVRL>={self.thresholds.ovrl_min}")

        for i, seg in enumerate(segments, 1):
            # Get source path
            source = getattr(seg, 'source_path', getattr(seg, 'source', None))
            if source is None:
                if verbose:
                    print(f"  [{i}/{len(segments)}] Skipped (no source path)")
                continue

            # Score segment
            scores = self.score_segment(source, seg.start, seg.end)

            # Track stats
            total_sig += scores.sig
            total_bak += scores.bak
            total_ovrl += scores.ovrl

            # Store scores on segment
            seg.quality_scores = scores

            if scores.passed:
                accepted.append(seg)
                if verbose:
                    print(f"  [{i}/{len(segments)}] ✓ PASS  SIG={scores.sig:.2f} "
                          f"BAK={scores.bak:.2f} OVRL={scores.ovrl:.2f}")
            else:
                rejected.append(seg)
                if 'SIG' in str(scores.reason):
                    rejection_reasons['low_sig'] += 1
                elif 'BAK' in str(scores.reason):
                    rejection_reasons['low_bak'] += 1
                elif 'OVRL' in str(scores.reason):
                    rejection_reasons['low_ovrl'] += 1

                if verbose:
                    print(f"  [{i}/{len(segments)}] ✗ FAIL  {scores.reason}")

        # Calculate averages
        n = len(segments)
        stats = {
            'total_segments': n,
            'accepted': len(accepted),
            'rejected': len(rejected),
            'acceptance_rate': len(accepted) / n if n > 0 else 0,
            'average_scores': {
                'sig': total_sig / n if n > 0 else 0,
                'bak': total_bak / n if n > 0 else 0,
                'ovrl': total_ovrl / n if n > 0 else 0
            },
            'rejection_breakdown': rejection_reasons
        }

        if verbose:
            print(f"\n[DNSMOS] Results: {len(accepted)} accepted, {len(rejected)} rejected")
            print(f"[DNSMOS] Average: SIG={stats['average_scores']['sig']:.2f}, "
                  f"BAK={stats['average_scores']['bak']:.2f}, "
                  f"OVRL={stats['average_scores']['ovrl']:.2f}")

        return accepted, rejected, stats

    def generate_report(self, segments: List, accepted: List, rejected: List,
                       stats: Dict, output_path: str = None) -> Dict:
        """
        Generate quality report JSON.

        Args:
            segments: All segments
            accepted: Accepted segments
            rejected: Rejected segments
            stats: Statistics dict
            output_path: Optional path to save JSON

        Returns:
            Report dict
        """
        import json

        report = {
            'summary': stats,
            'thresholds': {
                'sig_min': self.thresholds.sig_min,
                'bak_min': self.thresholds.bak_min,
                'ovrl_min': self.thresholds.ovrl_min
            },
            'rejected_segments': [
                {
                    'source': Path(getattr(seg, 'source_path', getattr(seg, 'source', 'unknown'))).name,
                    'start': seg.start,
                    'end': seg.end,
                    'duration': seg.end - seg.start,
                    'scores': {
                        'sig': seg.quality_scores.sig,
                        'bak': seg.quality_scores.bak,
                        'ovrl': seg.quality_scores.ovrl
                    },
                    'reason': seg.quality_scores.reason
                }
                for seg in rejected if hasattr(seg, 'quality_scores')
            ]
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"[DNSMOS] Report saved: {output_path}")

        return report


def test_dnsmos():
    """Test DNSMOS on a sample file."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python quality_validator.py <audio.wav> [start] [end]")
        return

    audio_path = sys.argv[1]
    start = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    end = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

    validator = QualityValidator()
    scores = validator.score_segment(audio_path, start, end)

    print(f"\nDNSMOS Scores for {audio_path} [{start}s - {end}s]:")
    print(f"  SIG:  {scores.sig:.2f} {'✓' if scores.sig >= validator.thresholds.sig_min else '✗'}")
    print(f"  BAK:  {scores.bak:.2f} {'✓' if scores.bak >= validator.thresholds.bak_min else '✗'}")
    print(f"  OVRL: {scores.ovrl:.2f} {'✓' if scores.ovrl >= validator.thresholds.ovrl_min else '✗'}")
    print(f"\nResult: {'PASS' if scores.passed else 'FAIL'}")


if __name__ == "__main__":
    test_dnsmos()
