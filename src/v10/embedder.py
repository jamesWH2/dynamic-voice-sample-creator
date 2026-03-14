#!/usr/bin/env python3
"""
Speaker embedding generation using SpeechBrain ECAPA-TDML.
"""

import os
import torch
import torchaudio
from pathlib import Path
from typing import List, Optional
from pydub import AudioSegment
import numpy as np


class SpeakerEmbedder:
    """Generate speaker embeddings using ECAPA-TDML."""
    
    def __init__(self, model_name: str = "speechbrain/spkrec-ecapa-voxceleb"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        print(f"[EMBEDDER] Device: {self.device.upper()}")
    
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            print(f"[EMBEDDER] Loading model: {self.model_name}")
            from speechbrain.inference.speaker import EncoderClassifier
            from huggingface_hub import hf_hub_download
            
            model_dir = Path("pretrained_models/spkrec-ecapa-voxceleb")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Manually download required files to avoid custom.py 404
            files = ["hyperparams.yaml", "embedding_model.ckpt", "mean_var_norm_emb.ckpt"]
            for f in files:
                if not (model_dir / f).exists():
                    print(f"[EMBEDDER] Downloading {f}...")
                    hf_hub_download(
                        repo_id=self.model_name,
                        filename=f,
                        local_dir=str(model_dir),
                        token=os.environ.get("HUGGINGFACE_API_KEY")
                    )
            
            self._model = EncoderClassifier.from_hparams(
                source=str(model_dir),
                run_opts={"device": self.device}
            )
        return self._model
    
    def load_audio_segment(self, filepath: str, start_sec: float, end_sec: float) -> torch.Tensor:
        """
        Load audio segment from file.
        Returns tensor ready for embedding.
        """
        print(f"[EMBEDDER] Loading segment: {start_sec:.1f}s - {end_sec:.1f}s")
        
        audio = AudioSegment.from_wav(filepath)
        segment = audio[int(start_sec * 1000):int(end_sec * 1000)]
        
        # Convert to tensor
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        samples = samples / (2 ** 15)  # Normalize to [-1, 1]
        
        # Reshape for model (batch, time)
        tensor = torch.from_numpy(samples).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def generate_embedding(self, audio_path: str, start_sec: Optional[float] = None, end_sec: Optional[float] = None) -> torch.Tensor:
        """
        Generate speaker embedding from audio file.
        If start/end provided, only use that segment.
        Returns embedding tensor (1, 192) for ECAPA.
        """
        if start_sec is not None and end_sec is not None:
            audio_tensor = self.load_audio_segment(audio_path, start_sec, end_sec)
        else:
            # Load full file
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            audio_tensor = waveform.to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model.encode_batch(audio_tensor)
        
        # Squeeze to remove extra dimensions: [1, 1, 192] -> [1, 192]
        embedding = embedding.squeeze(1)
        
        print(f"[EMBEDDER] Generated embedding shape: {embedding.shape}")
        return embedding
    
    def generate_master_embedding(self, references: List[dict]) -> torch.Tensor:
        """
        Generate averaged master embedding from multiple references.
        
        Args:
            references: List of dicts with 'path', 'start_time', 'end_time'
        
        Returns:
            Averaged embedding tensor
        """
        embeddings = []
        
        for i, ref in enumerate(references, 1):
            print(f"[EMBEDDER] Reference {i}/{len(references)}: {Path(ref['path']).name}")
            emb = self.generate_embedding(
                ref['path'],
                ref.get('start_time'),
                ref.get('end_time')
            )
            embeddings.append(emb)
        
        # Average embeddings
        master = torch.stack(embeddings).mean(dim=0)
        print(f"[EMBEDDER] Master embedding shape: {master.shape}")
        
        return master
    
    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        Calculate cosine similarity between two embeddings.
        Returns float in range [-1, 1].
        """
        # Normalize
        emb1_norm = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2_norm = torch.nn.functional.normalize(emb2, p=2, dim=1)
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1_norm, emb2_norm)
        
        return similarity.item()
    
    def clear_cache(self):
        """Clear GPU cache if using CUDA."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print("[EMBEDDER] Cleared GPU cache")
