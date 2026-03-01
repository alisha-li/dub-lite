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
    .pip_install("wtpsplit")  # Layer 8
    .pip_install("mistralai")  # Layer 9
    .run_commands("python3 -c \"import nltk; nltk.download('punkt_tab')\"")
    .add_local_dir("pipeline", "/root/pipeline", ignore=[".DS_Store", "**/.DS_Store"], copy=True)
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
            progress_callback=report_progress,
        )
        report_progress("Uploading to Spaces...", 95)
        output_key = _upload_to_spaces(output_path, job_id)
        report_progress("Done", 100)
        return output_key
    except Exception as e:
        # Re-raise as simple RuntimeError so Modal can serialize it; include full traceback so you can see which line in main/utils failed
        tb = traceback.format_exc()
        raise RuntimeError(f"{e}\n\nTraceback (actual failure):\n{tb}") from None