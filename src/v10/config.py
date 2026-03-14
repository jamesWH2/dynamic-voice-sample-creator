#!/usr/bin/env python3
"""
Configuration handling for Dynamic Voice Sample Creator.
"""

import json
import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import psycopg2


@dataclass
class ReferenceSequence:
    url: str
    start_time: float
    end_time: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class AudioEnergyFilter:
    """Audio energy validation settings."""
    enabled: bool = False
    min_rms_energy: float = 0.02


@dataclass
class DNSMOSFilter:
    """DNSMOS quality validation settings."""
    enabled: bool = False
    sig_min: float = 3.5    # Signal quality threshold
    bak_min: float = 3.0    # Background quality threshold
    ovrl_min: float = 3.5   # Overall quality threshold


@dataclass
class NoiseReduction:
    """Noise reduction settings."""
    enabled: bool = False
    method: str = "nlm"     # NLM (anlmdn) only - tested and proven
    strength: int = 50      # NLM strength (1-100)


@dataclass
class Loudnorm:
    """EBU R128 loudness normalization settings."""
    enabled: bool = False
    target_lufs: float = -16.0    # Target loudness (broadcast standard)
    true_peak: float = -1.5       # True peak limit (dBTP)


@dataclass
class PipelineConfig:
    name: str
    sources: List[str]
    reference_sequences: List[ReferenceSequence]
    output_file: str = "output.wav"
    similarity_threshold: float = 0.75
    crossfade_ms: int = 50
    output_sample_rate: int = 24000
    hf_token: Optional[str] = None
    audio_energy_filter: AudioEnergyFilter = field(default_factory=AudioEnergyFilter)
    dnsmos_filter: DNSMOSFilter = field(default_factory=DNSMOSFilter)
    noise_reduction: NoiseReduction = field(default_factory=NoiseReduction)
    loudnorm: Loudnorm = field(default_factory=Loudnorm)
    min_segment_duration: float = 0.0  # Minimum segment duration in seconds (0 = disabled)
    
    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        refs = [
            ReferenceSequence(
                url=r["url"],
                start_time=r["start_time"],
                end_time=r["end_time"]
            )
            for r in data.get("reference_sequences", [])
        ]
        
        # Parse audio energy filter
        energy_filter = AudioEnergyFilter()
        if "audio_energy_filter" in data:
            ef = data["audio_energy_filter"]
            energy_filter = AudioEnergyFilter(
                enabled=ef.get("enabled", False),
                min_rms_energy=ef.get("min_rms_energy", 0.02)
            )

        # Parse DNSMOS filter
        dnsmos_filter = DNSMOSFilter()
        if "dnsmos_filter" in data:
            df = data["dnsmos_filter"]
            dnsmos_filter = DNSMOSFilter(
                enabled=df.get("enabled", False),
                sig_min=df.get("sig_min", 3.5),
                bak_min=df.get("bak_min", 3.0),
                ovrl_min=df.get("ovrl_min", 3.5)
            )

        # Parse noise reduction
        noise_reduction = NoiseReduction()
        if "noise_reduction" in data:
            nr = data["noise_reduction"]
            noise_reduction = NoiseReduction(
                enabled=nr.get("enabled", False),
                method=nr.get("method", "nlm"),
                strength=nr.get("strength", 50)
            )

        # Parse loudnorm
        loudnorm = Loudnorm()
        if "loudnorm" in data:
            ln = data["loudnorm"]
            loudnorm = Loudnorm(
                enabled=ln.get("enabled", False),
                target_lufs=ln.get("target_lufs", -16.0),
                true_peak=ln.get("true_peak", -1.5)
            )
        
        return cls(
            name=data.get("name", "unknown"),
            sources=data.get("sources", []),
            reference_sequences=refs,
            output_file=data.get("output_file", "output.wav"),
            similarity_threshold=data.get("similarity_threshold", 0.75),
            crossfade_ms=data.get("crossfade_ms", 50),
            output_sample_rate=data.get("output_sample_rate", 24000),
            hf_token=data.get("hf_token"),
            audio_energy_filter=energy_filter,
            dnsmos_filter=dnsmos_filter,
            noise_reduction=noise_reduction,
            loudnorm=loudnorm,
            min_segment_duration=data.get("min_segment_duration", 0.0),
        )
    
    @classmethod
    def from_json_file(cls, path: str) -> "PipelineConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_hf_token_from_db() -> Optional[str]:
    """Retrieve HuggingFace token from secure_credentials database."""
    try:
        # Get DB password from environment
        db_pass = os.environ.get("DASHBOARD_DB_PASS")
        if not db_pass:
            return None
            
        conn = psycopg2.connect(
            host="localhost",
            database="clawd_main",
            user="dataling",
            password=db_pass
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT api_keys->>'read_token' FROM secure_credentials WHERE service = %s",
            ("huggingface",)
        )
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        if result and result[0]:
            return result[0]
    except Exception as e:
        print(f"[WARN] Could not retrieve HF token from DB: {e}")
    
    return None


def get_hf_token() -> Optional[str]:
    """Get HF token from DB, env, bashrc, or return None."""
    # Try database first
    token = get_hf_token_from_db()
    if token:
        return token
    
    # Try environment variables (multiple names)
    for var in ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_API_KEY", "HF_API_KEY"]:
        token = os.environ.get(var)
        if token:
            print(f"[CONFIG] Found HF token from {var}")
            return token
    
    # Try reading from .bashrc
    bashrc_path = Path.home() / ".bashrc"
    if bashrc_path.exists():
        import re
        content = bashrc_path.read_text()
        match = re.search(r'export\s+(?:HF_TOKEN|HUGGINGFACE_TOKEN|HUGGINGFACE_API_KEY|HF_API_KEY)=["\']?([^"\'\s]+)["\']?', content)
        if match:
            print(f"[CONFIG] Found HF token from .bashrc")
            return match.group(1)
    
    return None


def resolve_config(args: argparse.Namespace) -> PipelineConfig:
    """Resolve configuration from CLI args or config file."""
    if args.config:
        config = PipelineConfig.from_json_file(args.config)
    else:
        # Build from CLI args
        refs = []
        for i, ref_str in enumerate(args.references or []):
            parts = ref_str.split(",")
            if len(parts) == 3:
                refs.append(ReferenceSequence(
                    url=parts[0],
                    start_time=float(parts[1]),
                    end_time=float(parts[2])
                ))
        
        config = PipelineConfig(
            name=args.name or "unknown",
            sources=args.sources or [],
            reference_sequences=refs,
            output_file=args.output or "output.wav",
            similarity_threshold=args.threshold or 0.75,
            crossfade_ms=args.crossfade or 50,
        )
    
    # Resolve HF token if not in config
    if not config.hf_token:
        config.hf_token = get_hf_token()
    
    return config


def create_test_config() -> dict:
    """Create test configuration for Jimmy."""
    return {
        "name": "Jimmy",
        "sources": [
            "https://www.youtube.com/watch?v=16NSPOMDGss",
            "https://www.youtube.com/watch?v=PFWsyzIigMo"
        ],
        "reference_sequences": [
            {
                "url": "https://www.youtube.com/watch?v=16NSPOMDGss",
                "start_time": 15.0,
                "end_time": 25.5
            },
            {
                "url": "https://www.youtube.com/watch?v=PFWsyzIigMo",
                "start_time": 34.0,
                "end_time": 80.0
            }
        ],
        "similarity_threshold": 0.75,
        "crossfade_ms": 50,
        "output_file": "jimmy_voice_samples.wav"
    }
