#!/usr/bin/env python3
"""Run denoising on one file. Exit 0 on success, 1 on failure.
Used as subprocess to isolate C++ crashes (e.g. std::length_error in libsndfile).

Imports df (DeepFilterNet) directly instead of pipeline.utils to avoid pulling
in heavy/incompatible transitive deps (wtpsplit -> skops -> yt_dlp -> no_Cryptodome)."""
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: denoise_one.py <input.wav> <output.wav>", file=sys.stderr)
        sys.exit(2)
    inp, out = sys.argv[1], sys.argv[2]
    try:
        import torch
        import torchaudio
        # Stub for torchaudio 2.8+ compat (same as utils.py)
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile", "ffmpeg", "sox"]

        from df import config, utils as df_utils
        from df.enhance import enhance, init_df, load_audio, save_audio

        print("Loading DeepFilterNet2 model...")
        model, df, _ = init_df()

        sr = config("sr", 48000, int, section="df")
        print(f"Loading audio from {inp}...")
        sample, meta = load_audio(inp, sr)

        if sample.dim() > 1 and sample.shape[0] > 1:
            sample = sample.mean(dim=0, keepdim=True)

        print("Denoising audio...")
        sample = sample.cpu()
        _orig = df_utils.get_device
        df_utils.get_device = lambda: torch.device("cpu")
        try:
            enhanced = enhance(model, df, sample)
        finally:
            df_utils.get_device = _orig

        # Fade-in to avoid clicks
        lim = torch.linspace(0.0, 1.0, int(sr * 0.15)).unsqueeze(0)
        if lim.shape[1] < enhanced.shape[1]:
            lim = torch.cat((lim, torch.ones(1, enhanced.shape[1] - lim.shape[1])), dim=1)
        enhanced = enhanced * lim

        if meta.sample_rate != sr:
            from df.io import resample
            enhanced = resample(enhanced, sr, meta.sample_rate)
            sr = meta.sample_rate

        print(f"Saving denoised audio to {out}...")
        save_audio(out, enhanced.cpu(), sr)
        print("Done!")
        sys.exit(0)
    except Exception as e:
        print(f"Denoise failed: {e}", file=sys.stderr)
        sys.exit(1)
