#!/usr/bin/env python3
"""
Minimal DeepFilterNet audio denoising script
Usage: python denoise_audio.py input_file.wav [output_file.wav]
"""

import sys
import os
import torch
from df import config
from df.enhance import enhance, init_df, load_audio, save_audio

def denoise_audio(input_path: str, output_path: str = None):
    """Denoise an audio file using DeepFilterNet2"""
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False
    
    # Generate output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_denoised{ext}"
    
    # Initialize model
    print("Loading DeepFilterNet2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use model name or path from environment variable, default to DeepFilterNet2
    model_name_or_path = os.environ.get("DEEPFILTERNET_MODEL", "DeepFilterNet2")
    
    try:
        # init_df can take a model name (like "DeepFilterNet2") or a path
        model, df, _ = init_df(model_name_or_path, config_allow_defaults=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"\nTried to load: {model_name_or_path}")
        print("You can set DEEPFILTERNET_MODEL environment variable to:")
        print("  - A model name: 'DeepFilterNet', 'DeepFilterNet2', or 'DeepFilterNet3'")
        print("  - A path to a model directory")
        return False
    model = model.to(device=device).eval()
    
    # Load audio
    sr = config("sr", 48000, int, section="df")
    print(f"Loading audio from {input_path}...")
    sample, meta = load_audio(input_path, sr)
    
    # Convert to mono if needed
    if sample.dim() > 1 and sample.shape[0] > 1:
        sample = sample.mean(dim=0, keepdim=True)
    
    print(f"Audio shape: {sample.shape}, Sample rate: {sr}Hz")
    
    # Denoise
    print("Denoising audio...")
    enhanced = enhance(model, df, sample)
    
    # Apply fade-in to avoid clicks
    lim = torch.linspace(0.0, 1.0, int(sr * 0.15)).unsqueeze(0)
    if lim.shape[1] < enhanced.shape[1]:
        lim = torch.cat((lim, torch.ones(1, enhanced.shape[1] - lim.shape[1])), dim=1)
    enhanced = enhanced * lim
    
    # Resample if needed
    if meta.sample_rate != sr:
        from df.io import resample
        enhanced = resample(enhanced, sr, meta.sample_rate)
        sr = meta.sample_rate
    
    # Save
    print(f"Saving denoised audio to {output_path}...")
    save_audio(output_path, enhanced, sr)
    
    print(f"Done! Denoised audio saved to: {output_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python denoise_audio.py input_file.wav [output_file.wav]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = denoise_audio(input_file, output_file)
    sys.exit(0 if success else 1)

