"""
Audio-only dubbing pipeline for Chrome extension.
Audio in, dubbed audio out — no video processing.

Deploy:  modal deploy pipeline/modal_stream.py
Test:    modal run pipeline/modal_stream.py
"""
import modal
import os
import sys
import traceback

app = modal.App("dub-stream")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "curl", "build-essential", "rubberband-cli")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .pip_install("torch")
    .pip_install("torchaudio", "torchvision")
    .pip_install("transformers", "faster-whisper")
    .pip_install("speechbrain", "coqui-tts")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("audio-separator", "DeepFilterNet")
    .pip_install("wtpsplit", "pycryptodome")
    .run_commands("python3 -c \"import nltk; nltk.download('punkt_tab')\"")
    .add_local_dir("pipeline", "/root/pipeline", ignore=[".DS_Store", "**/.DS_Store", "CosyVoice", "pretrained_models"], copy=True)
    .run_commands("pip install 'setuptools<78' && pip install --no-build-isolation --no-deps openai-whisper==20231117 && pip install tiktoken")
    .run_commands("pip install --force-reinstall 'mistralai==1.12.4'")
    .run_commands("python3 /root/pipeline/patch_torchaudio_backend.py")
)

vol = modal.Volume.from_name("dub-lite-volume")

NUM_TTS_WORKERS = 4


def _setup_env():
    os.environ["TORCH_HOME"] = "/models/torch"
    os.environ["HF_HOME"] = "/models/huggingface"
    os.environ["COQUI_TOS_AGREED"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/pipeline")


@app.function(
    image=image,
    gpu="A10G",
    memory=65536,
    timeout=1800,
    volumes={"/models": vol},
    secrets=[modal.Secret.from_name("dub-env")],
)
def tts_worker(batch: dict) -> dict:
    """Process a batch of TTS segments on a dedicated GPU. Returns WAV bytes."""
    import torch
    from pydub import AudioSegment

    _setup_env()
    from pipeline.utils import tts_segment

    segments = batch["segments"]
    speaker_wavs = batch["speaker_wavs"]
    targ = batch["targ"]

    os.makedirs("/tmp/tts_speakers", exist_ok=True)
    os.makedirs("/tmp/tts_audio_chunks", exist_ok=True)
    for speaker_id, wav_bytes in speaker_wavs.items():
        with open(f"/tmp/tts_speakers/{speaker_id}.wav", "wb") as f:
            f.write(wav_bytes)

    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())

    results = {}
    for seg in segments:
        i = seg["index"]
        text = seg["text"]
        emotion = seg["emotion"]
        speaker = seg["speaker"]
        chunk_path = f"/tmp/tts_audio_chunks/{i}.wav"

        if not text or text.strip() == "":
            dur_ms = max(1, int((seg["end"] - seg["start"]) * 1000))
            AudioSegment.silent(duration=dur_ms).export(chunk_path, format="wav")
        else:
            tts_segment(tts, text, i, f"/tmp/tts_speakers/{speaker}.wav",
                        targ, emotion, output_dir="/tmp/tts_audio_chunks")

        with open(chunk_path, "rb") as f:
            results[str(i)] = f.read()

    return {"wavs": results}


@app.function(
    image=image,
    gpu="A10G",
    memory=65536,
    timeout=3600,
    volumes={"/models": vol},
    secrets=[modal.Secret.from_name("dub-env")],
)
def run_dub_audio(
    audio_bytes: bytes,
    targ: str,
    mistral_api: str = None,
    groq_api: str = None,
    groq_model: str = None,
    gemini_api: str = None,
    gemini_model: str = None,
) -> dict:
    """Audio-only dubbing on Modal GPU. Calls dub_audio() from main.py."""
    _setup_env()
    from pipeline.main import YTDubPipeline

    if not (mistral_api or "").strip():
        mistral_api = (os.environ.get("MISTRAL_API_KEY") or "").strip() or None

    try:
        pipeline = YTDubPipeline()
        return pipeline.dub_audio(
            audio_bytes=audio_bytes,
            targ=targ,
            mistral_api=mistral_api,
            groq_api=groq_api,
            groq_model=groq_model,
            gemini_api=gemini_api,
            gemini_model=gemini_model,
            tts_worker_fn=tts_worker,
            num_tts_workers=NUM_TTS_WORKERS,
        )
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(f"{e}\n\nTraceback (actual failure):\n{tb}") from None


@app.local_entrypoint()
def test():
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    test_audio = Path(__file__).resolve().parent / "temp" / "orig_audio.wav"
    if not test_audio.exists():
        print(f"No test audio at {test_audio}.")
        print("Run the main pipeline first so temp/orig_audio.wav exists, or place an audio file there.")
        return

    with open(test_audio, "rb") as f:
        audio_bytes = f.read()

    print(f"Sending {len(audio_bytes) / 1024 / 1024:.1f} MB of audio...")
    result = run_dub_audio.remote(
        audio_bytes=audio_bytes,
        targ="zh",
        mistral_api=os.environ.get("MISTRAL_API_KEY"),
    )

    print(f"Got {len(result['chunks'])} chunks back, source language: {result['src_lang']}")

    # Save combined audio for listening
    out_dir = Path(__file__).resolve().parent / "temp"
    combined_path = out_dir / "dubbed_audio_only.wav"
    with open(combined_path, "wb") as f:
        f.write(result["combined_audio"])
    print(f"Combined audio saved to {combined_path}")
