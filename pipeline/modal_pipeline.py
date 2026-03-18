import modal
import os
import sys
import traceback
import boto3
from botocore.config import Config

app = modal.App("dub-lite")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "curl", "build-essential", "rubberband-cli", "fonts-noto", "fonts-noto-cjk", "libass-dev")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .pip_install("torch")          # Layer 1: just torch
    .pip_install("torchaudio", "torchvision")  # Layer 2
    .pip_install("transformers", "faster-whisper")  # Layer 3
    .pip_install("speechbrain", "coqui-tts")  # Layer 4
    # .pip_install("pyannote-audio", "pyannote-pipeline")  # Layer 5
    .pip_install_from_requirements("requirements.txt")  # Layer 6: remaining
    .pip_install("audio-separator", "DeepFilterNet")  # Layer 7
    .pip_install("wtpsplit", "pycryptodome")  # Layer 8
    .run_commands("python3 -c \"import nltk; nltk.download('punkt_tab')\"")
    .add_local_dir("pipeline", "/root/pipeline", ignore=[".DS_Store", "**/.DS_Store", "CosyVoice", "pretrained_models"], copy=True)
    .add_local_dir("pipeline/CosyVoice", "/root/CosyVoice", ignore=[".DS_Store", "**/.DS_Store", ".git", "pretrained_models", "asset", "examples", "**/requirements.txt"], copy=True)
    .pip_install(
        # CosyVoice + Matcha-TTS deps (not already in earlier layers)
        "conformer", "omegaconf", "hydra-core", "HyperPyYAML", "modelscope",
        "x-transformers", "pyworld", "diffusers", "wetext", "inflect",
        "gdown", "wget", "deepspeed", "lightning", "pyarrow",
        "onnxruntime-gpu", "grpcio", "grpcio-tools",
        "einops", "Unidecode", "phonemizer", "rootutils",
        "hydra-colorlog", "hydra-optuna-sweeper",
    )
    .run_commands("pip install 'setuptools<78' && pip install --no-build-isolation --no-deps openai-whisper==20231117 && pip install tiktoken")
    .run_commands("pip install --force-reinstall 'mistralai==1.12.4'")  # force correct version
    .run_commands("python3 /root/pipeline/patch_torchaudio_backend.py")
)

vol = modal.Volume.from_name("dub-lite-volume")
progress_dict = modal.Dict.from_name("dub-lite-progress", create_if_missing=True)


@app.function(image=image, gpu="L4", memory=65536, volumes={"/models": vol})
def debug_imports():
    """Run: modal run pipeline/modal_pipeline.py::debug_imports"""
    import os
    import sys

    os.environ["TORCH_HOME"] = "/models/torch"
    os.environ["HF_HOME"] = "/models/huggingface"
    os.environ["COQUI_TOS_AGREED"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/pipeline")

    steps = [
        ("torch", lambda: __import__("torch")),
        ("torchaudio", lambda: __import__("torchaudio")),
        ("transformers", lambda: __import__("transformers")),
        ("faster_whisper", lambda: __import__("faster_whisper")),
        ("TTS", lambda: __import__("TTS")),
        # ("pyannote.audio", lambda: __import__("pyannote.audio")),  # commented out – using Mistral for diarization
        ("df", lambda: __import__("df")),
        ("pipeline.main", lambda: __import__("pipeline.main")),
    ]
    for name, fn in steps:
        try:
            fn()
            print(f"OK: {name}", flush=True)
        except Exception as e:
            print(f"FAIL: {name} - {e}", flush=True)
            raise
    return "all ok"


NUM_TTS_WORKERS = 4  # number of parallel GPU containers for TTS


@app.function(
    image=image,
    gpu="A10G",
    memory=65536,
    timeout=1800,
    volumes={"/models": vol},
    secrets=[modal.Secret.from_name("dub-env")],
)
def tts_worker(batch: dict) -> dict:
    """Process a batch of TTS segments on a dedicated GPU container.

    Input: {
        "segments": [{"index": int, "text": str, "emotion": str, "speaker": str,
                       "start": float, "end": float}],
        "speaker_wavs": {speaker_id: bytes},  # WAV bytes per speaker
        "targ": str,  # target language code
        "tts_engine": str,
    }
    Returns: {"wavs": {str(index): bytes}}  # generated WAV bytes keyed by segment index
    """
    import io
    import torch
    from pydub import AudioSegment

    os.environ["TORCH_HOME"] = "/models/torch"
    os.environ["HF_HOME"] = "/models/huggingface"
    os.environ["COQUI_TOS_AGREED"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/pipeline")

    from pipeline.utils import tts_segment, tts_segment_cosyvoice, load_cosyvoice

    segments = batch["segments"]
    speaker_wavs = batch["speaker_wavs"]
    targ = batch["targ"]
    tts_engine = batch.get("tts_engine", "xtts")

    # Write speaker WAVs to temp files so TTS can read them
    os.makedirs("/tmp/tts_speakers", exist_ok=True)
    os.makedirs("/tmp/tts_audio_chunks", exist_ok=True)
    for speaker_id, wav_bytes in speaker_wavs.items():
        with open(f"/tmp/tts_speakers/{speaker_id}.wav", "wb") as f:
            f.write(wav_bytes)

    # Load TTS model once for this container
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cosyvoice = tts_engine == "cosyvoice"
    if use_cosyvoice:
        cosyvoice_model = load_cosyvoice(batch.get("cosyvoice_model_dir", "/root/pretrained_models/Fun-CosyVoice3-0.5B"))
    else:
        from TTS.api import TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())

    results = {}
    for seg in segments:
        i = seg["index"]
        text = seg["text"]
        emotion = seg["emotion"]
        speaker = seg["speaker"]
        speaker_wav_path = f"/tmp/tts_speakers/{speaker}.wav"
        chunk_path = f"/tmp/tts_audio_chunks/{i}.wav"

        if not text or text.strip() == "":
            dur_ms = max(1, int((seg["end"] - seg["start"]) * 1000))
            AudioSegment.silent(duration=dur_ms).export(chunk_path, format="wav")
        elif use_cosyvoice:
            tts_segment_cosyvoice(cosyvoice_model, text, i, speaker_wav_path, emotion,
                                   output_dir="/tmp/tts_audio_chunks")
        else:
            tts_segment(tts, text, i, speaker_wav_path, targ, emotion,
                        output_dir="/tmp/tts_audio_chunks")

        # Read generated WAV back as bytes
        with open(chunk_path, "rb") as f:
            results[str(i)] = f.read()

    return {"wavs": results}


def _upload_to_spaces(local_path: str, job_id: str) -> str:
    """Upload the dubbed video to Spaces and return the object key."""
    client = boto3.client(
        "s3",
        region_name=os.environ.get("SPACES_REGION"),
        endpoint_url=os.environ.get("SPACES_ENDPOINT"),
        aws_access_key_id=os.environ.get("SPACES_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("SPACES_SECRET_KEY"),
        config=Config(signature_version="s3v4"),
    )
    bucket = os.environ.get("SPACES_BUCKET")
    object_key = f"outputs/{job_id}/dubbed.mp4"
    client.upload_file(local_path, bucket, object_key, ExtraArgs={"ContentType": "video/mp4"})
    return object_key


@app.function(
    image=image,
    gpu="A10G",
    memory=65536,
    timeout = 3600,
    volumes = {"/models": vol},
    secrets=[modal.Secret.from_name("dub-env"), modal.Secret.from_name("dub-spaces")],
)
def run_dubbing_pipeline(
    job_id: str,
    src: str,
    targ: str,
    hf_token: str,
    pyannote_key: str = None,
    gemini_api: str = None,
    groq_api: str = None,
    groq_model: str = None,
    gemini_model: str = None,
    mistral_api: str = None,
    speakerTurnsPkl: bool = False,
    segmentsPkl: bool = False,
    finalSentencesPkl: bool = False,
    tts_engine: str = "xtts",
    cosyvoice_model_dir: str = "/root/pretrained_models/Fun-CosyVoice3-0.5B",
):
    """Runs the full dubbing pipeline on GPU.

    src accepts: presigned Spaces URL, YouTube URL, or local path.
    """

    os.environ["TORCH_HOME"] = "/models/torch"
    os.environ["HF_HOME"] = "/models/huggingface"
    os.environ["COQUI_TOS_AGREED"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    sys.path.append("/root")
    sys.path.append("/root/pipeline")
    from pipeline.main import YTDubPipeline

    # read from dub-env secret if not passed
    if not (mistral_api or "").strip():
        mistral_api = (os.environ.get("MISTRAL_API_KEY") or "").strip() or None

    def report_progress(stage: str, percent: int):
        progress_dict[job_id] = {"stage": stage, "progress": percent}

    try:
        report_progress("Starting...", 0)
        pipeline = YTDubPipeline()
        output_path = pipeline.dub(
            src=src,
            targ=targ,
            hf_token=hf_token,
            pyannote_key=pyannote_key,
            gemini_api=gemini_api,
            groq_api=groq_api,
            groq_model=groq_model,
            gemini_model=gemini_model,
            mistral_api=mistral_api,
            speakerTurnsPkl=speakerTurnsPkl,
            segmentsPkl=segmentsPkl,
            finalSentencesPkl=finalSentencesPkl,
            tts_engine=tts_engine,
            cosyvoice_model_dir=cosyvoice_model_dir,
            progress_callback=report_progress,
            tts_worker_fn=tts_worker,
            num_tts_workers=NUM_TTS_WORKERS,
        )
        report_progress("Uploading to Spaces...", 95)
        output_key = _upload_to_spaces(output_path, job_id)
        report_progress("Done", 100)
        return output_key
    except Exception as e:
        # Re-raise as simple RuntimeError so Modal can serialize it; include full traceback so you can see which line in main/utils failed
        tb = traceback.format_exc()
        raise RuntimeError(f"{e}\n\nTraceback (actual failure):\n{tb}") from None


@app.local_entrypoint()
def test():
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    client = boto3.client(
        "s3",
        region_name=os.environ.get("SPACES_REGION"),
        endpoint_url=os.environ.get("SPACES_ENDPOINT"),
        aws_access_key_id=os.environ.get("SPACES_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("SPACES_SECRET_KEY"),
        config=Config(signature_version="s3v4"),
    )
    presigned_url = client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": os.environ.get("SPACES_BUCKET"),
            "Key": "uploads/ff6d603c-e025-4fd0-8753-748218e7499d/Why Vail Resorts Is Losing Skiers in a Growing Industry ｜ WSJ [GlcWwAcrsfI].mp4",
        },
        ExpiresIn=3600,
    )

    result = run_dubbing_pipeline.remote(
        job_id="test-local",
        src=presigned_url,
        targ="zh",
        hf_token=os.environ["HF_TOKEN"],
        mistral_api=os.environ.get("MISTRAL_API_KEY"),
        tts_engine="xtts",
    )
    print(f"Result: {result}")