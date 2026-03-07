#!/usr/bin/env python3
"""
Audio stitching using pydub.
"""

from pathlib import Path
from typing import List
from pydub import AudioSegment
from dataclasses import dataclass


@dataclass
class MatchedSegment:
    """A segment that matches the target speaker."""
    source_path: str
    start: float
    end: float
    similarity: float
    audio_segment: AudioSegment = None


class AudioStitcher:
    """Stitch matched audio segments into final output."""
    
    def __init__(self, crossfade_ms: int = 50, output_sample_rate: int = 24000, 
                 min_rms_energy: float = 0.0, min_segment_duration: float = 0.0,
                 per_segment_loudnorm: bool = False, target_lufs: float = -16.0):
        self.crossfade_ms = crossfade_ms
        self.output_sample_rate = output_sample_rate
        self.min_rms_energy = min_rms_energy
        self.min_segment_duration = min_segment_duration
        self.per_segment_loudnorm = per_segment_loudnorm
        self.target_lufs = target_lufs
    
    def calculate_rms_energy(self, audio_segment: AudioSegment) -> float:
        """Calculate RMS energy of an audio segment (normalized 0.0-1.0)."""
        # Convert to mono for energy calculation
        mono = audio_segment.set_channels(1)
        
        # Get raw samples
        samples = mono.get_array_of_samples()
        
        # Calculate RMS
        if len(samples) == 0:
            return 0.0
        
        rms = (sum(s**2 for s in samples) / len(samples)) ** 0.5
        
        # Normalize to 0.0-1.0 (max 16-bit value = 32767)
        return rms / 32767.0
    
    def is_valid_segment(self, audio_segment: AudioSegment) -> bool:
        """Check if segment has sufficient audio energy."""
        if self.min_rms_energy <= 0.0:
            return True
        
        energy = self.calculate_rms_energy(audio_segment)
        return energy >= self.min_rms_energy
    
    def extract_segment(self, source_path: str, start: float, end: float) -> AudioSegment:
        """
        Extract audio segment from source file.
        
        Args:
            source_path: Path to source WAV file
            start: Start time in seconds
            end: End time in seconds
        
        Returns:
            AudioSegment
        """
        audio = AudioSegment.from_wav(source_path)
        segment = audio[int(start * 1000):int(end * 1000)]
        return segment
    
    def normalize_segment(self, audio_seg: AudioSegment, target_lufs: float = -16.0) -> AudioSegment:
        """
        Apply EBU R128 loudness normalization to a segment.
        
        Args:
            audio_seg: AudioSegment to normalize
            target_lufs: Target loudness in LUFS
        
        Returns:
            Normalized AudioSegment
        """
        import subprocess
        import tempfile
        
        # Export to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
            tmp_in_path = tmp_in.name
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
            tmp_out_path = tmp_out.name
        
        try:
            audio_seg.export(tmp_in_path, format='wav')
            
            # Apply loudnorm via FFmpeg
            cmd = [
                'ffmpeg', '-y',
                '-i', tmp_in_path,
                '-af', f'loudnorm=I={target_lufs}:TP=-1.5:LRA=11',
                '-ar', str(self.output_sample_rate),
                '-ac', '1',
                tmp_out_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return AudioSegment.from_wav(tmp_out_path)
            else:
                # Return original if normalization fails
                return audio_seg
        finally:
            Path(tmp_in_path).unlink(missing_ok=True)
            Path(tmp_out_path).unlink(missing_ok=True)
    
    def stitch(self, segments: List, output_path: str, target_duration: float = None) -> str:
        """
        Stitch segments into final output.
        
        Args:
            segments: List of segment objects (MatchedSegment or SpeakerSegment)
                      Must have: source_path, start, end attributes
            output_path: Path for output WAV file
            target_duration: Maximum duration in seconds (None = use all)
        
        Returns:
            Path to output file
        """
        if not segments:
            raise ValueError("No segments to stitch")
        
        print(f"[STITCHER] Stitching {len(segments)} segments...")
        
        # Sort by source path and start time for chronological order
        segments = sorted(segments, key=lambda s: (getattr(s, 'source_path', ''), s.start))
        
        # Extract audio for each segment
        audio_segments = []
        rejected_energy = 0
        rejected_duration = 0
        for seg in segments:
            # Handle both MatchedSegment and SpeakerSegment
            source = getattr(seg, 'source_path', getattr(seg, 'source', None))
            if source is None:
                continue
            
            # Duration filter (check before extracting audio)
            duration = seg.end - seg.start
            if self.min_segment_duration > 0 and duration < self.min_segment_duration:
                rejected_duration += 1
                continue
            
            audio_seg = self.extract_segment(
                source,
                seg.start,
                seg.end
            )
            
            # Per-segment loudnorm (approved v9 workflow)
            if self.per_segment_loudnorm:
                audio_seg = self.normalize_segment(audio_seg, self.target_lufs)
            
            # Energy validation
            if not self.is_valid_segment(audio_seg):
                rejected_energy += 1
                continue
            
            audio_segments.append(audio_seg)
            
            # Check if we've reached target duration
            if target_duration:
                current_duration = sum(len(a) for a in audio_segments) / 1000.0
                if current_duration >= target_duration:
                    print(f"[STITCHER] Reached target duration: {target_duration}s")
                    break
        
        if not audio_segments:
            raise ValueError("No valid audio segments extracted")
        
        if rejected_duration > 0:
            print(f"[STITCHER] Rejected {rejected_duration} short segments (< {self.min_segment_duration}s)")
        if rejected_energy > 0:
            print(f"[STITCHER] Rejected {rejected_energy} low-energy segments")
        
        # Concatenate with crossfade
        output = audio_segments[0]
        
        for i, audio_seg in enumerate(audio_segments[1:], 1):
            duration = len(audio_seg) / 1000.0
            similarity = getattr(seg, 'similarity', 1.0)  # Default to 1.0 if not available
            print(f"[STITCHER] Adding segment {i}/{len(audio_segments)-1} (dur={duration:.1f}s)")
            output = output.append(audio_seg, crossfade=self.crossfade_ms)
        
        # Convert to target format
        output = output.set_frame_rate(self.output_sample_rate)
        output = output.set_channels(1)  # Mono
        output = output.set_sample_width(2)  # 16-bit
        
        # Truncate if target duration specified
        if target_duration:
            max_ms = int(target_duration * 1000)
            if len(output) > max_ms:
                output = output[:max_ms]
        
        # Export
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output.export(
            str(output_path),
            format="wav",
            parameters=["-acodec", "pcm_s16le"]
        )
        
        duration_sec = len(output) / 1000.0
        print(f"[STITCHER] Output: {output_path}")
        print(f"[STITCHER] Duration: {duration_sec:.1f}s ({len(audio_segments)} segments)")
        
        return str(output_path)


def preview_segments(segments: List[MatchedSegment]) -> str:
    """Generate text preview of segments."""
    lines = ["Matched Segments:", "-" * 60]
    
    total_duration = 0
    for i, seg in enumerate(segments, 1):
        duration = seg.end - seg.start
        total_duration += duration
        lines.append(
            f"{i:3d}. {Path(seg.source_path).name:30s} "
            f"{seg.start:6.1f}s - {seg.end:6.1f}s "
            f"(sim={seg.similarity:.3f})"
        )
    
    lines.append("-" * 60)
    lines.append(f"Total: {len(segments)} segments, {total_duration:.1f}s")
    
    return "\n".join(lines)
