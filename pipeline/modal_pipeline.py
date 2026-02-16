import modal
import os
import sys
import traceback

app = modal.App("dub-lite")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "curl", "build-essential", "rubberband-cli")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .pip_install("torch")          # Layer 1: just torch
    .pip_install("torchaudio", "torchvision")  # Layer 2
    .pip_install("transformers", "faster-whisper")  # Layer 3
    .pip_install("speechbrain", "coqui-tts")  # Layer 4
    .pip_install("pyannote-audio", "pyannote-pipeline")  # Layer 5
    .pip_install_from_requirements("requirements.txt")  # Layer 6: remaining
    .pip_install("audio-separator", "DeepFilterNet")  # Layer 7
    .run_commands("python3 -c \"import nltk; nltk.download('punkt_tab')\"")
    .add_local_dir("pipeline", "/root/pipeline")
)

vol = modal.Volume.from_name("dub-lite-volume")
progress_dict = modal.Dict.from_name("dub-lite-progress", create_if_missing=True)

@app.function(
    image=image,
    gpu="T4",
    timeout = 3600,
    volumes = {"/models": vol},
    secrets=[modal.Secret.from_name("dub-env")],
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
    speakerTurnsPkl: bool = False,
    segmentsPkl: bool = False,
    finalSentencesPkl: bool = False,
    file_bytes: bytes = None,
    file_name: str = None,
):
    """Runs the full dubbing pipeline on GPU"""

    os.environ["TORCH_HOME"] = "/models/torch"
    os.environ["HF_HOME"] = "/models/huggingface"
    os.environ["COQUI_TOS_AGREED"] = "1"  # Accept TTS terms non-interactively (no stdin in Modal)
    sys.path.append("/root")
    sys.path.append("/root/pipeline")
    from pipeline.main import YTDubPipeline

    # Progress callback: writes to modal.Dict so the API can read it
    def report_progress(stage: str, percent: int):
        progress_dict[job_id] = {"stage": stage, "progress": percent}

    # If file bytes were sent, write them to a local file in the container
    if file_bytes and file_name:
        os.makedirs("temp", exist_ok=True)
        local_path = f"temp/{file_name}"
        with open(local_path, "wb") as f:
            f.write(file_bytes)
        src = local_path

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
            speakerTurnsPkl=speakerTurnsPkl,
            segmentsPkl=segmentsPkl,
            finalSentencesPkl=finalSentencesPkl,
            progress_callback=report_progress,
        )
        with open(output_path, "rb") as f:
            video_bytes = f.read()
        return video_bytes
    except Exception as e:
        # Re-raise as simple RuntimeError so Modal can serialize it; include full traceback so you can see which line in main/utils failed
        tb = traceback.format_exc()
        raise RuntimeError(f"{e}\n\nTraceback (actual failure):\n{tb}") from None