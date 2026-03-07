# Dynamic Voice Sample Creator

Production-ready voice sample extraction pipeline from YouTube URLs with advanced audio processing.

## Features

- **Speaker-Specific Extraction** - Dynamically generates embeddings from reference audio
- **DNSMOS Quality Filtering** - Automatic rejection of low-quality segments
- **EBU R128 Loudness Normalization** - Broadcast-standard audio levels
- **NLM Noise Reduction** - Non-Local Means denoising for clean voice

## Pipeline Workflow

```
1. Download (yt-dlp)
2. Vocal Isolation (Kim_Vocal_2)
3. Embedding Generation (ECAPA-TDNN)
4. Speaker Matching (pyannote 3.1)
5. DNSMOS Filter (SIG≥3.5, BAK≥3.0, OVRL≥3.5)
6. Per-Segment Loudnorm (-16 LUFS)
7. Stitch (50ms crossfade)
8. EQ (100Hz-7500Hz)
9. NLM Noise Reduction (strength=50)
10. Final Loudnorm (-16 LUFS)
```

## Installation

```bash
git clone https://github.com/clawd-ai/dynamic-voice-sample-creator.git
cd dynamic-voice-sample-creator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Prerequisites

- Python 3.10+
- FFmpeg
- CUDA-capable GPU (recommended)
- Hugging Face token

## Usage

```bash
source venv/bin/activate
python src/dynamic_voice_sample_creator.py --config config/example.json
```

## Configuration

```json
{
  "name": "SpeakerName",
  "sources": ["youtube_url_1", "youtube_url_2"],
  "reference_sequences": [
    {"url": "youtube_url_1", "start_time": 10.0, "end_time": 20.0}
  ],
  "similarity_threshold": 0.90,
  "crossfade_ms": 50,
  "output_file": "output.wav",
  "dnsmos_filter": {
    "enabled": true,
    "sig_min": 3.5,
    "bak_min": 3.0,
    "ovrl_min": 3.5
  },
  "noise_reduction": {
    "enabled": true,
    "method": "nlm",
    "strength": 50
  },
  "loudnorm": {
    "enabled": true,
    "target_lufs": -16.0,
    "true_peak": -1.5
  }
}
```

## Key Parameters

- **similarity_threshold**: Speaker matching confidence (0.0-1.0)
  - 0.90+ = strict (recommended for TTS)
  - 0.85 = balanced
  - 0.75 = lenient (more false positives)

- **DNSMOS Thresholds**: Audio quality gates
  - SIG (Signal Quality) ≥ 3.5
  - BAK (Background Quality) ≥ 3.0
  - OVRL (Overall Quality) ≥ 3.5

- **NLM Strength**: Noise reduction intensity
  - 30 = light
  - 50 = balanced (recommended)
  - 70+ = aggressive

## Output

- **Format**: WAV, 24kHz, 16-bit, mono
- **Loudness**: -16 LUFS (EBU R128)
- **Frequency**: 100Hz - 7500Hz

## Requirements

- PyTorch 2.10+
- SpeechBrain 1.0.3
- pyannote.audio 4.0+
- pydub
- yt-dlp
- audio-separator
- FFmpeg

## License

MIT

## Author

Clawd AI Team
