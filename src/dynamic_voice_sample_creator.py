#!/usr/bin/env python3
"""
Dynamic Voice Sample Creator

Extracts, isolates, and stitches a specific target voice from YouTube URLs.
"""

import os
import sys
import json
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PipelineConfig, 
    resolve_config, 
    get_hf_token,
    create_test_config,
    ReferenceSequence,
    DNSMOSFilter
)
from downloader import AudioDownloader, validate_ffmpeg
from vocal_isolator import VocalIsolator, check_audio_separator
from embedder import SpeakerEmbedder
from quality_validator import QualityValidator, DNSMOSThresholds
from diarizer import SpeakerDiarizer, SpeakerSegment
from stitcher import AudioStitcher, preview_segments, MatchedSegment


class DynamicVoiceSampleCreator:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig, workspace: str = None):
        self.config = config
        self.workspace = Path(workspace or tempfile.mkdtemp(prefix="voice_sample_"))
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # Track files
        self.downloaded_files: Dict[str, str] = {}
        self.vocal_files: List[str] = []
        self.matched_segments: List[MatchedSegment] = []
        
        print(f"[PIPELINE] Workspace: {self.workspace}")
        print(f"[PIPELINE] Target: {config.name}")
        print(f"[PIPELINE] Similarity threshold: {config.similarity_threshold}")
    
    def validate_prerequisites(self):
        """Check all prerequisites before starting."""
        print("\n[STEP 0] Validating prerequisites...")
        
        # FFmpeg
        if not validate_ffmpeg():
            raise RuntimeError("ffmpeg not found. Install with: sudo apt-get install ffmpeg")
        print("  ✓ ffmpeg")
        
        # HF Token
        if not self.config.hf_token:
            raise RuntimeError(
                "HuggingFace token not found. Set HF_TOKEN env var or add to secure_credentials DB."
            )
        print("  ✓ HF token")
        
        # Python packages (will fail on import if missing)
        try:
            import yt_dlp
            print("  ✓ yt-dlp")
        except ImportError:
            raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp")
        
        try:
            import pydub
            print("  ✓ pydub")
        except ImportError:
            raise RuntimeError("pydub not installed. Run: pip install pydub")
        
        print("[STEP 0] Prerequisites OK\n")
    
    def step1_download(self):
        """Download all audio from YouTube URLs."""
        print("[STEP 1] Downloading audio...")
        
        # Collect all unique URLs (sources + reference URLs)
        all_urls = list(set(
            self.config.sources + 
            [r.url for r in self.config.reference_sequences]
        ))
        
        downloader = AudioDownloader(
            output_dir=str(self.workspace / "downloads"),
            sample_rate=44100
        )
        
        self.downloaded_files = downloader.download_all(all_urls)
        print(f"[STEP 1] Downloaded {len(self.downloaded_files)} files\n")
    
    def step2_isolate_vocals(self):
        """Isolate vocals from all downloaded files."""
        print("[STEP 2] Isolating vocals...")
        
        isolator = VocalIsolator(
            output_dir=str(self.workspace / "vocals"),
            model_name="Kim_Vocal_2.onnx"
        )
        
        self.vocal_files = isolator.isolate_all(list(self.downloaded_files.values()))
        print(f"[STEP 2] Isolated {len(self.vocal_files)} vocal tracks\n")
    
    def step3_generate_embedding(self) -> 'torch.Tensor':
        """Generate master speaker embedding from reference sequences."""
        print("[STEP 3] Generating master embedding...")
        
        embedder = SpeakerEmbedder()
        
        # Build reference list with vocal file paths
        references = []
        for ref in self.config.reference_sequences:
            # Find the vocal file for this URL
            original_path = self.downloaded_files.get(ref.url)
            if not original_path:
                print(f"[WARN] No download found for reference URL: {ref.url}")
                continue
            
            # Find corresponding vocal file
            stem = Path(original_path).stem
            vocal_path = self.workspace / "vocals" / f"{stem}_vocals.wav"
            
            if not vocal_path.exists():
                print(f"[WARN] Vocal file not found: {vocal_path}")
                continue
            
            references.append({
                'path': str(vocal_path),
                'start_time': ref.start_time,
                'end_time': ref.end_time
            })
            print(f"  Reference: {ref.start_time:.1f}s - {ref.end_time:.1f}s ({ref.duration:.1f}s)")
        
        if not references:
            raise RuntimeError("No valid reference sequences found")
        
        master_embedding = embedder.generate_master_embedding(references)
        print(f"[STEP 3] Master embedding generated from {len(references)} references\n")
        
        return master_embedding, embedder
    
    def step4_match_speaker(self, master_embedding, embedder):
        """Identify target speaker using smart cluster verification."""
        print("[STEP 4] Matching speaker clusters...")
        
        diarizer = SpeakerDiarizer(hf_token=self.config.hf_token)
        from stitcher import MatchedSegment  # Import here to avoid circular dependency
        
        # Store all cluster info: {vocal_path: {speaker_id: [segments]}}
        all_clusters = {}
        
        # First pass: Diarize all files
        for vocal_path in self.vocal_files:
            print(f"\n  Processing: {Path(vocal_path).name}")
            
            # Diarize
            segments = diarizer.diarize(vocal_path)
            
            # Group by speaker clusters
            clusters = diarizer.get_speaker_clusters(segments)
            all_clusters[vocal_path] = clusters
            
            print(f"    Found {len(clusters)} speakers")
            for speaker, segs in clusters.items():
                longest = diarizer.get_longest_segment(segs)
                if longest:
                    print(f"      {speaker}: {len(segs)} segments, longest: {longest.duration:.1f}s")
        
        # Clear diarizer from memory
        diarizer.clear_cache()
        
        # Second pass: Verify each cluster against master embedding
        print("\n[STEP 4] Verifying clusters against master reference...")
        matched_segments = []
        
        for vocal_path, clusters in all_clusters.items():
            print(f"\n  Verifying: {Path(vocal_path).name}")
            
            for speaker_id, segments in clusters.items():
                # Get longest representative segment (min 2s for accuracy)
                longest = diarizer.get_longest_segment(segments, min_duration=2.0)
                
                if not longest:
                    print(f"    {speaker_id}: Skipped (no segment >= 2s)")
                    continue
                
                # Generate embedding for this segment
                seg_embedding = embedder.generate_embedding(
                    vocal_path,
                    longest.start,
                    longest.end
                )
                
                # Calculate similarity
                similarity = embedder.cosine_similarity(master_embedding, seg_embedding)
                
                if similarity >= self.config.similarity_threshold:
                    # MATCH! Accept all segments from this cluster
                    print(f"    {speaker_id}: ✓ MATCH (sim={similarity:.3f}) - Accepting {len(segments)} segments")
                    
                    # Convert SpeakerSegments to MatchedSegments for stitcher
                    for seg in segments:
                        matched_segments.append(MatchedSegment(
                            source_path=vocal_path,
                            start=seg.start,
                            end=seg.end,
                            similarity=similarity
                        ))
                else:
                    print(f"    {speaker_id}: ✗ No match (sim={similarity:.3f})")
        
        self.matched_segments = matched_segments
        print(f"\n[STEP 4] Matched {len(matched_segments)} segments from target speaker\n")
    
    def step5_dnsmos_filter(self):
        """Filter segments by DNSMOS quality scores."""
        if not self.config.dnsmos_filter.enabled:
            print("[STEP 5] DNSMOS filter disabled - skipping\n")
            self.quality_segments = self.matched_segments
            return
        
        print("[STEP 5] DNSMOS quality filtering...")
        
        from quality_validator import QualityValidator, DNSMOSThresholds
        
        # Create validator with config thresholds
        thresholds = DNSMOSThresholds(
            sig_min=self.config.dnsmos_filter.sig_min,
            bak_min=self.config.dnsmos_filter.bak_min,
            ovrl_min=self.config.dnsmos_filter.ovrl_min
        )
        
        validator = QualityValidator(thresholds=thresholds)
        
        # Filter segments
        accepted, rejected, stats = validator.filter_segments(self.matched_segments)
        
        if not accepted:
            print(f"[STEP 5] WARNING: All segments rejected by DNSMOS!")
            print(f"[STEP 5] Using all matched segments instead (DNSMOS bypassed)")
            self.quality_segments = self.matched_segments
            self.dnsmos_stats = {'bypassed': True, 'all_rejected': True}
            return
        
        self.quality_segments = accepted
        self.dnsmos_stats = stats
        
        # Generate report
        report_path = Path(self.config.output_file).with_suffix('.quality_report.json')
        validator.generate_report(
            self.matched_segments,
            accepted,
            rejected,
            stats,
            str(report_path)
        )
        
        print(f"[STEP 5] Quality filter complete: {len(accepted)}/{len(self.matched_segments)} segments passed\n")
    
    def step6_stitch(self):
        """Stitch quality-validated segments into final output."""
        print("[STEP 6] Stitching final output...")
        
        segments_to_stitch = getattr(self, 'quality_segments', self.matched_segments)
        
        if not segments_to_stitch:
            raise RuntimeError("No segments to stitch")
        
        # Preview
        print(preview_segments(segments_to_stitch))
        
        # Check if per-segment loudnorm is enabled
        per_segment_loudnorm = getattr(self.config, 'loudnorm', None) and self.config.loudnorm.enabled
        target_lufs = getattr(self.config.loudnorm, 'target_lufs', -16.0) if per_segment_loudnorm else -16.0
        
        stitcher = AudioStitcher(
            crossfade_ms=self.config.crossfade_ms,
            output_sample_rate=self.config.output_sample_rate,
            min_rms_energy=self.config.audio_energy_filter.min_rms_energy if self.config.audio_energy_filter.enabled else 0.0,
            min_segment_duration=self.config.min_segment_duration,
            per_segment_loudnorm=per_segment_loudnorm,
            target_lufs=target_lufs
        )
        
        if per_segment_loudnorm:
            print(f"[STEP 6] Per-segment loudnorm enabled (target: {target_lufs} LUFS)")
        
        # Don't truncate - use all available audio
        output_path = stitcher.stitch(
            segments_to_stitch,
            self.config.output_file,
            target_duration=None  # Use all segments
        )
        
        print(f"[STEP 6] Complete: {output_path}\n")
        return output_path
    
    def step7_apply_eq(self, input_path: str) -> str:
        """Apply EQ filtering (highpass + lowpass) to stitched audio."""
        if not getattr(self.config, 'noise_reduction', None) or not self.config.noise_reduction.enabled:
            print("[STEP 7] EQ filtering disabled - skipping\n")
            return input_path
        
        print("[STEP 7] Applying EQ filter (highpass 100Hz, lowpass 7500Hz)...")
        
        import subprocess
        
        # Build FFmpeg command for EQ
        temp_path = input_path.replace('.wav', '_eq.wav')
        
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-af', 'highpass=f=100,lowpass=f=7500',
            '-ar', str(self.config.output_sample_rate),
            '-ac', '1',
            temp_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[STEP 7] Warning: EQ failed, using original")
            return input_path
        
        # Replace original with processed
        import shutil
        shutil.move(temp_path, input_path)
        
        print(f"[STEP 7] EQ applied (100Hz-7500Hz)\n")
        return input_path
    
    def step8_apply_nlm(self, input_path: str) -> str:
        """Apply NLM noise reduction to stitched audio."""
        if not getattr(self.config, 'noise_reduction', None) or not self.config.noise_reduction.enabled:
            print("[STEP 8] NLM noise reduction disabled - skipping\n")
            return input_path
        
        print("[STEP 8] Applying NLM noise reduction...")
        
        strength = getattr(self.config.noise_reduction, 'strength', 50)
        
        # Build FFmpeg command
        import subprocess
        temp_path = input_path.replace('.wav', '_nlm.wav')
        
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-af', f'anlmdn=s={strength}',
            '-ar', str(self.config.output_sample_rate),
            '-ac', '1',
            temp_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[STEP 8] Warning: NLM failed, using original")
            return input_path
        
        # Replace original with processed
        import shutil
        shutil.move(temp_path, input_path)
        
        print(f"[STEP 8] NLM applied (strength={strength})\n")
        return input_path
    
    def step9_final_loudnorm(self, input_path: str) -> str:
        """Apply final EBU R128 loudness normalization."""
        if not getattr(self.config, 'loudnorm', None) or not self.config.loudnorm.enabled:
            print("[STEP 9] Final loudnorm disabled - skipping\n")
            return input_path
        
        print("[STEP 9] Applying final loudnorm...")
        
        target_lufs = getattr(self.config.loudnorm, 'target_lufs', -16.0)
        true_peak = getattr(self.config.loudnorm, 'true_peak', -1.5)
        
        # Build FFmpeg command
        import subprocess
        temp_path = input_path.replace('.wav', '_norm.wav')
        
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-af', f'loudnorm=I={target_lufs}:TP={true_peak}:LRA=11',
            '-ar', str(self.config.output_sample_rate),
            '-ac', '1',
            temp_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[STEP 8] Warning: Loudnorm failed, using original")
            return input_path
        
        # Replace original with processed
        import shutil
        shutil.move(temp_path, input_path)
        
        print(f"[STEP 9] Final loudnorm applied ({target_lufs} LUFS)\n")
        return input_path
    
    def run(self) -> str:
        """Execute full pipeline."""
        start_time = datetime.now()
        
        try:
            self.validate_prerequisites()
            self.step1_download()
            self.step2_isolate_vocals()
            master_embedding, embedder = self.step3_generate_embedding()
            self.step4_match_speaker(master_embedding, embedder)
            self.step5_dnsmos_filter()
            output_path = self.step6_stitch()
            output_path = self.step7_apply_eq(output_path)
            output_path = self.step8_apply_nlm(output_path)
            output_path = self.step9_final_loudnorm(output_path)
            
            elapsed = datetime.now() - start_time
            print(f"[PIPELINE] ✓ Complete in {elapsed}")
            print(f"[PIPELINE] Output: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"[PIPELINE] ✗ Failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic Voice Sample Creator"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to JSON config file"
    )
    parser.add_argument(
        "--name", "-n",
        help="Target speaker name"
    )
    parser.add_argument(
        "--sources", "-s",
        nargs="+",
        help="YouTube source URLs"
    )
    parser.add_argument(
        "--references", "-r",
        nargs="+",
        help="Reference sequences (url,start_sec,end_sec)"
    )
    parser.add_argument(
        "--output", "-o",
        default="output.wav",
        help="Output file path"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.75,
        help="Similarity threshold (default: 0.75)"
    )
    parser.add_argument(
        "--crossfade",
        type=int,
        default=50,
        help="Crossfade duration in ms (default: 50)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run with test config (Jimmy)"
    )
    parser.add_argument(
        "--workspace",
        help="Workspace directory (default: temp)"
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep workspace after completion"
    )
    
    args = parser.parse_args()
    
    # Handle test mode
    if args.test:
        print("[INFO] Running in test mode with Jimmy config")
        test_config = create_test_config()
        config = PipelineConfig.from_dict(test_config)
        config.hf_token = get_hf_token()
    else:
        config = resolve_config(args)
    
    # Create pipeline
    pipeline = DynamicVoiceSampleCreator(
        config=config,
        workspace=args.workspace
    )
    
    # Run
    try:
        output_path = pipeline.run()
        print(f"\n✓ Success: {output_path}")
        return 0
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        return 1
    finally:
        # Cleanup
        if not args.keep_workspace and not args.workspace:
            if pipeline.workspace.exists():
                shutil.rmtree(pipeline.workspace)
                print(f"[CLEANUP] Removed workspace: {pipeline.workspace}")


if __name__ == "__main__":
    sys.exit(main())
