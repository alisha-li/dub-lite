#!/usr/bin/env python3
"""Create torchaudio.backend stub and patch torchaudio.info for torchaudio 2.8+."""
import pathlib
from dataclasses import dataclass

# Define AudioMetaData - removed from torchaudio 2.8
@dataclass
class AudioMetaData:
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str


def _info_soundfile(path, **kwargs):
    """Fallback info() using soundfile for torchaudio 2.8+ (ta.info removed)."""
    import soundfile as sf
    i = sf.info(path)
    # Map soundfile subtype to bits_per_sample (PCM_16->16, PCM_24->24, etc.)
    bps = 0
    if "PCM" in (i.subtype or ""):
        for n in (8, 16, 24, 32):
            if str(n) in i.subtype:
                bps = n
                break
    return AudioMetaData(
        sample_rate=int(i.samplerate),
        num_frames=int(i.frames),
        num_channels=int(i.channels),
        bits_per_sample=bps,
        encoding=i.format or "PCM_S",
    )


ta = __import__("torchaudio")
p = pathlib.Path(ta.__file__).parent / "backend"
p.mkdir(exist_ok=True)
(p / "__init__.py").touch()
(p / "common.py").write_text(
    "from dataclasses import dataclass\n\n"
    "@dataclass\n"
    "class AudioMetaData:\n"
    "    sample_rate: int\n"
    "    num_frames: int\n"
    "    num_channels: int\n"
    "    bits_per_sample: int\n"
    "    encoding: str\n\n"
    '__all__ = ["AudioMetaData"]\n'
)

# Patch torchaudio.info if missing (removed in torchaudio 2.8)
if not hasattr(ta, "info") or ta.info is None:
    ta.info = _info_soundfile
    print("Patched torchaudio.info with soundfile fallback")
print("Created torchaudio.backend stub at", p)
