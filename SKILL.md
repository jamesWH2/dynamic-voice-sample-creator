# Dynamic Voice Sample Creator

**Skill ID:** `dynamic-voice-sample-creator`

**Version:** 2.0 (Production Ready) ✅

Extracts, isolates, and stitches a specific target voice from YouTube URLs using:
- **DNSMOS quality filtering** (Microsoft P.835)
- **NLM noise reduction** (Non-Local Means)
- **EBU R128 loudness normalization**

---

## What It Does

1. Downloads audio from YouTube URLs
2. Isolates vocals using MDX-Net (Kim_Vocal_2)
3. Generates speaker embedding from reference timestamps
4. Identifies target speaker via diarization + cosine similarity
5. **Filters segments by DNSMOS quality scores**
6. **Normalizes loudness per segment (-16 LUFS)**
7. Stitches matching segments with crossfade
8. **Applies NLM noise reduction (strength=50)**
9. **Final loudness normalization (-16 LUFS)**

---

## Prerequisites

```bash
# System
sudo apt-get install ffmpeg

# Python
pip install yt-dlp pydub audio-separator[cpu] pyannote.audio speechbrain torch torchaudio onnxruntime librosa soundfile

# DNSMOS model (auto-downloaded on first run)
# NLM is built into FFmpeg
```

## Hugging Face Token

Pyannote diarization requires a HF token with accepted user conditions:
1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1
2. Accept user conditions
3. Generate a read token at https://huggingface.co/settings/tokens
4. Store in database:
```sql
INSERT INTO secure_credentials (service, api_keys) 
VALUES ('huggingface', '{"read_token": "hf_xxx..."}');
```

---

## Usage

### CLI
```bash
python src/dynamic_voice_sample_creator.py --config config/jimmy_v9_production.json
```

### Config File Structure (v9)
```json
{
  "name": "Jimmy",
  "sources": [
    "https://www.youtube.com/watch?v=VIDEO1",
    "https://www.youtube.com/watch?v=VIDEO2"
  ],
  "reference_sequences": [
    {"url": "https://www.youtube.com/watch?v=VIDEO1", "start_time": 15.0, "end_time": 25.5}
  ],
  "similarity_threshold": 0.90,
  "crossfade_ms": 50,
  "output_file": "output.wav",
  "audio_energy_filter": {
    "enabled": true,
    "min_rms_energy": 0.02
  },
  "min_segment_duration": 2.0,
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

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| sources | [string] | required | YouTube URLs to process |
| reference_sequences | [object] | required | Timestamp sequences for embedding |
| similarity_threshold | float | 0.90 | Cosine similarity threshold |
| dnsmos_filter.sig_min | float | 3.5 | Voice signal quality (1-5) |
| dnsmos_filter.bak_min | float | 3.0 | Background quality (1-5) |
| dnsmos_filter.ovrl_min | float | 3.5 | Overall quality (1-5) |
| noise_reduction.method | string | "nlm" | Noise reduction method |
| noise_reduction.strength | int | 50 | NLM strength (1-100) |
| loudnorm.target_lufs | float | -16.0 | Target loudness (LUFS) |
| loudnorm.true_peak | float | -1.5 | True peak limit (dBTP) |

---

## Pipeline Steps

1. **Download** - yt-dlp extracts audio at 44.1kHz
2. **Vocal Isolation** - audio-separator with Kim_Vocal_2 model
3. **Embedding Generation** - speechbrain ECAPA-TDNN, averaged across references
4. **Diarization** - pyannote 3.1 identifies speaker segments
5. **DNSMOS Filtering** - Quality validation (SIG/BAK/OVRL)
6. **Per-Segment Loudnorm** - Normalize each segment to -16 LUFS
7. **Stitching** - Chronological concatenation with 50ms crossfade
8. **NLM Noise Reduction** - Non-Local Means (strength=50)
9. **Final Loudnorm** - EBU R128 final pass to -16 LUFS

---

## Output

- 24kHz, 16-bit mono WAV
- **DNSMOS quality-validated** (SIG≥3.5, BAK≥3.0, OVRL≥3.5)
- **Loudness normalized** (-16 LUFS, TP -1.5 dBTP)
- **Noise reduced** (NLM strength=50)
- **Production-ready** for voice cloning/TTS

---

## Quality Scores Reference

### DNSMOS (1-5 scale)

| Score | Quality | Description |
|-------|---------|-------------|
| 4.5-5.0 | Excellent | Broadcast quality |
| 4.0-4.5 | Good | Clear voice, minimal noise |
| 3.5-4.0 | Fair | Acceptable, slight issues |
| 3.0-3.5 | Poor | Noticeable degradation |
| < 3.0 | Bad | Significant problems |

### What Each Metric Measures

- **SIG (Signal)**: Voice clarity, distortion, clipping
- **BAK (Background)**: Background noise level (higher = less noise)
- **OVRL (Overall)**: Composite quality score

---

## Noise Reduction: Why NLM?

**Tested approaches:**

| Method | Result | Issue |
|--------|--------|-------|
| FFT denoiser (afftdn) | ❌ Failed | Can't handle non-stationary noise |
| Spectral gating | ❌ Failed | Destroys voice quality |
| EQ-only | ❌ Failed | Insufficient noise removal |
| **NLM (anlmdn)** | ✅ **Success** | Pattern-matching handles voice+noise overlap |

**Why NLM works:**
- Finds similar patterns in audio
- Averages them to reduce noise
- Handles scratching, wind, crumpling sounds
- Preserves voice better than FFT

---

## Best Practices

### ✅ DO
- Use **multiple sources** (DNSMOS filters bad segments)
- Keep **dnsmos_filter enabled** (quality gate)
- Keep **noise_reduction enabled** with NLM
- Keep **loudnorm enabled** (broadcast standard)

### ❌ DON'T
- Use FFT denoiser (afftdn) - fails on non-stationary noise
- Use spectral gating - destroys voice
- Apply aggressive noise reduction before stitching
- Skip DNSMOS filtering (quality will be inconsistent)

---

## Integration

```python
# From OpenClaw agent
import subprocess

result = subprocess.run([
    "python", 
    "~/.openclaw/workspace/skills/dynamic-voice-sample-creator/src/dynamic_voice_sample_creator.py",
    "--config", config_path
], capture_output=True, text=True)
```

---

## Documentation

- `docs/workflow_v9_NLM.txt` - Complete workflow ASCII diagram
- `docs/workflow_v9_NLM_diagram.png` - Visual workflow diagram
- `config/jimmy_v9_production.json` - Production config example

---

## Version History

| Version | Changes |
|---------|---------|
| 2.0 (v9) | + DNSMOS filtering, NLM noise reduction, EBU R128 loudnorm |
| 1.0 | Basic pipeline with vocal isolation and speaker matching |

---

*Last updated: 2026-03-07*
*Version: 2.0*
*Status: Production Ready ✅*
