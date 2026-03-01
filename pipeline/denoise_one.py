#!/usr/bin/env python3
"""Run denoising on one file. Exit 0 on success, 1 on failure.
Used as subprocess to isolate C++ crashes (e.g. std::length_error in libsndfile)."""
import os
import sys

# Ensure pipeline is importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)  # project root (parent of pipeline/)
for p in [_root, "/root", "/root/pipeline"]:
    if p and os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: denoise_one.py <input.wav> <output.wav>", file=sys.stderr)
        sys.exit(2)
    inp, out = sys.argv[1], sys.argv[2]
    try:
        from pipeline.utils import denoise_audio, get_denoiser
        model, df = get_denoiser()
        ok = denoise_audio(inp, out, model, df)
        sys.exit(0 if ok else 1)
    except Exception as e:
        print(f"Denoise failed: {e}", file=sys.stderr)
        sys.exit(1)
