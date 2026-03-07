#!/usr/bin/env python3
"""
Audio downloader using yt-dlp.
"""

import os
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, Optional
import yt_dlp


class AudioDownloader:
    """Download audio from YouTube URLs."""
    
    def __init__(self, output_dir: str, sample_rate: int = 44100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.cache: Dict[str, str] = {}  # url -> filepath
    
    def _get_video_id(self, url: str) -> str:
        """Extract video ID from URL for filename."""
        # Use URL hash as unique identifier
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        return url_hash
    
    def download(self, url: str) -> str:
        """
        Download audio from URL.
        Returns path to downloaded WAV file.
        """
        # Check cache
        if url in self.cache:
            print(f"[DOWNLOADER] Using cached: {self.cache[url]}")
            return self.cache[url]
        
        video_id = self._get_video_id(url)
        output_path = self.output_dir / f"{video_id}.wav"
        
        if output_path.exists():
            print(f"[DOWNLOADER] File exists: {output_path}")
            self.cache[url] = str(output_path)
            return str(output_path)
        
        print(f"[DOWNLOADER] Downloading: {url}")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(self.output_dir / f'{video_id}.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'postprocessor_args': [
                '-ar', str(self.sample_rate),
                '-ac', '1'  # Mono
            ],
            'quiet': False,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            if output_path.exists():
                self.cache[url] = str(output_path)
                print(f"[DOWNLOADER] Saved: {output_path}")
                return str(output_path)
            else:
                raise RuntimeError(f"Download completed but file not found: {output_path}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to download {url}: {e}")
    
    def download_all(self, urls: list) -> Dict[str, str]:
        """
        Download all unique URLs.
        Returns dict mapping url -> filepath.
        """
        results = {}
        unique_urls = list(set(urls))  # Remove duplicates
        
        print(f"[DOWNLOADER] Downloading {len(unique_urls)} unique URLs...")
        
        for url in unique_urls:
            try:
                filepath = self.download(url)
                results[url] = filepath
            except Exception as e:
                print(f"[DOWNLOADER] ERROR: {e}")
                raise
        
        return results


def validate_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
