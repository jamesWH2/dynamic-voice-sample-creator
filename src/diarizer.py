#!/usr/bin/env python3
"""
Speaker diarization using pyannote.audio.
"""

import torch
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pyannote.audio import Pipeline


@dataclass
class SpeakerSegment:
    """Represents a speaker segment."""
    start: float
    end: float
    speaker: str
    
    @property
    def duration(self) -> float:
        return self.end - self.start


class SpeakerDiarizer:
    """Perform speaker diarization using pyannote."""
    
    def __init__(self, hf_token: str, model_name: str = "pyannote/speaker-diarization-3.1"):
        self.hf_token = hf_token
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipeline = None
        print(f"[DIARIZER] Device: {self.device.upper()}")
    
    @property
    def pipeline(self):
        """Lazy load pipeline."""
        if self._pipeline is None:
            if not self.hf_token:
                raise ValueError(
                    "HuggingFace token required for pyannote. "
                    "Set HF_TOKEN env var or pass hf_token parameter."
                )
            
            print(f"[DIARIZER] Loading pipeline: {self.model_name}")
            self._pipeline = Pipeline.from_pretrained(
                self.model_name,
                token=self.hf_token
            )
            self._pipeline.to(torch.device(self.device))
        return self._pipeline
    
    def diarize(self, audio_path: str) -> List[SpeakerSegment]:
        """
        Perform diarization on audio file.
        Returns list of SpeakerSegment objects.
        """
        print(f"[DIARIZER] Processing: {Path(audio_path).name}")
        
        try:
            diarization_output = self.pipeline(audio_path)
        except Exception as e:
            raise RuntimeError(f"Diarization failed for {audio_path}: {e}")
        
        segments = []
        
        # pyannote 4.0+ returns DiarizeOutput object
        if hasattr(diarization_output, 'speaker_diarization'):
            # Access the speaker_diarization attribute (not a method)
            diarization = diarization_output.speaker_diarization
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker
                )
                segments.append(segment)
        elif hasattr(diarization_output, 'itertracks'):
            # Old API - direct Annotation object
            for turn, _, speaker in diarization_output.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker
                )
                segments.append(segment)
        else:
            raise RuntimeError(f"Unknown diarization output type: {type(diarization_output)}")
        
        print(f"[DIARIZER] Found {len(segments)} segments from {len(set(s.speaker for s in segments))} speakers")
        return segments
    
    def get_speaker_clusters(self, segments: List[SpeakerSegment]) -> dict:
        """
        Group segments by speaker cluster.
        Returns dict mapping speaker_id -> list of segments.
        """
        clusters = {}
        for seg in segments:
            if seg.speaker not in clusters:
                clusters[seg.speaker] = []
            clusters[seg.speaker].append(seg)
        return clusters
    
    def get_longest_segment(self, segments: List[SpeakerSegment], min_duration: float = 2.0) -> Optional[SpeakerSegment]:
        """
        Get the longest segment from a list, filtered by minimum duration.
        """
        valid_segments = [s for s in segments if s.duration >= min_duration]
        if not valid_segments:
            return None
        return max(valid_segments, key=lambda s: s.duration)
    
    def clear_cache(self):
        """Clear GPU cache if using CUDA."""
        if self.device == "cuda":
            import torch
            del self._pipeline
            self._pipeline = None
            torch.cuda.empty_cache()
            print("[DIARIZER] Cleared GPU cache and unloaded model")
        
        print(f"[DIARIZER] Found {len(segments)} segments from {len(set(s.speaker for s in segments))} speakers")
        return segments
    
    def diarize_all(self, audio_paths: List[str]) -> dict:
        """
        Diarize multiple audio files.
        Returns dict mapping path -> list of SpeakerSegments.
        """
        results = {}
        print(f"[DIARIZER] Processing {len(audio_paths)} files...")
        
        for i, path in enumerate(audio_paths, 1):
            print(f"[DIARIZER] [{i}/{len(audio_paths)}] {Path(path).name}")
            try:
                segments = self.diarize(path)
                results[path] = segments
            except Exception as e:
                print(f"[DIARIZER] ERROR: {e}")
                raise
        
        return results
    
    def clear_cache(self):
        """Clear GPU cache if using CUDA."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print("[DIARIZER] Cleared GPU cache")
