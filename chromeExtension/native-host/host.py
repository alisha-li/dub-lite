#!/Users/alishali/.pyenv/versions/3.12.0/bin/python3
"""
Native Messaging host for Dub Lite Chrome extension.
Chrome launches this script, sends a JSON message via stdin,
and reads the JSON response from stdout.

Chrome's native messaging protocol:
- Each message is prefixed with a 4-byte unsigned int (little-endian) indicating the message length
- Followed by the JSON message as UTF-8 bytes

Message types:
- ping → {pong: true}
- dub-url {url, target_lang} → {success, job_id} (spawns Modal job)
- poll-job {job_id} → {status, progress, stage, output_url?}
- transcribe-url {url, language?} → {success, transcript, audio_path} (local whisper.cpp, de-scoped)
- download-audio {url, target_lang, api_base} → legacy API server path
"""

import sys
import json
import struct
import subprocess
import tempfile
import os
import uuid
import urllib.request
import urllib.error

# Chrome launches this script with a minimal PATH that doesn't include
# pyenv/homebrew/etc. Add common paths so we can find yt-dlp.
extra_paths = [
    os.path.expanduser("~/.pyenv/shims"),
    os.path.expanduser("~/.pyenv/versions/3.12.0/bin"),
    os.path.expanduser("~/.local/bin"),
    "/usr/local/bin",
    "/opt/homebrew/bin",
]
os.environ["PATH"] = os.pathsep.join(extra_paths) + os.pathsep + os.environ.get("PATH", "")

# Cross-call state: each native messaging invocation is one-shot, so we persist
# job_id -> modal_call_id mapping on disk between calls.
STATE_DIR = os.path.expanduser("~/Library/Application Support/dub-lite/jobs")
os.makedirs(STATE_DIR, exist_ok=True)

# Project root for loading .env. Two levels up from this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

WHISPER_BIN = os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = os.path.expanduser("~/whisper.cpp/models/ggml-medium.bin")

MODAL_APP_NAME = "dub-lite"
MODAL_FN_NAME = "run_audio_dubbing_pipeline"
MODAL_PROGRESS_DICT = "dub-lite-progress"


def read_message():
    raw_length = sys.stdin.buffer.read(4)
    if not raw_length:
        return None
    length = struct.unpack('<I', raw_length)[0]
    raw_message = sys.stdin.buffer.read(length)
    return json.loads(raw_message)


def send_message(message):
    encoded = json.dumps(message).encode('utf-8')
    sys.stdout.buffer.write(struct.pack('<I', len(encoded)))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def load_project_env():
    """Load .env from the dub-lite project root into os.environ."""
    from dotenv import load_dotenv
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(env_path):
        raise Exception(f".env not found at {env_path}")
    load_dotenv(env_path)


def save_job_state(job_id, data):
    with open(os.path.join(STATE_DIR, f"{job_id}.json"), "w") as f:
        json.dump(data, f)


def load_job_state(job_id):
    path = os.path.join(STATE_DIR, f"{job_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def download_audio(url, whisper_ready=False):
    """Use yt-dlp to download audio. Returns the file path.

    If whisper_ready=True, output is 16kHz mono WAV.
    Otherwise, mp3.
    """
    output_dir = tempfile.mkdtemp(prefix="dub-lite-")
    output_path = os.path.join(output_dir, "audio.%(ext)s")

    if whisper_ready:
        cmd = [
            "yt-dlp",
            "-f", "bestaudio",
            "-x",
            "--audio-format", "wav",
            "--postprocessor-args", "ExtractAudio:-ar 16000 -ac 1",
            "-o", output_path,
            "--no-playlist",
            url,
        ]
    else:
        cmd = [
            "yt-dlp",
            "-f", "bestaudio",
            "-x",
            "--audio-format", "mp3",
            "-o", output_path,
            "--no-playlist",
            url,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise Exception(f"yt-dlp failed: {result.stderr}")

    for f in os.listdir(output_dir):
        return os.path.join(output_dir, f)

    raise Exception("yt-dlp produced no output file")


def transcribe(wav_path, language=None):
    if not os.path.exists(WHISPER_BIN):
        raise Exception(f"whisper.cpp binary not found at {WHISPER_BIN}")
    if not os.path.exists(WHISPER_MODEL):
        raise Exception(f"whisper model not found at {WHISPER_MODEL}")

    out_prefix = wav_path.rsplit(".", 1)[0]
    cmd = [
        WHISPER_BIN,
        "-m", WHISPER_MODEL,
        "-f", wav_path,
        "-oj",
        "-of", out_prefix,
        "--print-progress",
    ]
    if language:
        cmd += ["-l", language]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise Exception(f"whisper.cpp failed: {result.stderr}")

    json_path = out_prefix + ".json"
    with open(json_path) as f:
        return json.load(f)


def upload_to_api(audio_path, target_lang, api_base):
    """Legacy path: POST audio to API server. Returns job_id."""
    boundary = "----DubLiteBoundary" + os.urandom(8).hex()
    filename = os.path.basename(audio_path)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    body = b""
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    body += b"Content-Type: audio/mpeg\r\n\r\n"
    body += audio_bytes
    body += b"\r\n"
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="target_language"\r\n\r\n'
    body += target_lang.encode()
    body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{api_base}/api/jobs/audio",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return result["job_id"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        raise Exception(f"API error {e.code}: {error_body}")


def _spaces_client():
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3",
        region_name=os.environ["SPACES_REGION"],
        endpoint_url=os.environ["SPACES_ENDPOINT"],
        aws_access_key_id=os.environ["SPACES_ACCESS_KEY"],
        aws_secret_access_key=os.environ["SPACES_SECRET_KEY"],
        config=Config(signature_version="s3v4"),
    )


def upload_audio_to_spaces(audio_path, job_id):
    """Upload audio to Spaces under uploads/<job_id>/. Returns object_key."""
    client = _spaces_client()
    bucket = os.environ["SPACES_BUCKET"]
    object_key = f"uploads/{job_id}/{os.path.basename(audio_path)}"
    with open(audio_path, "rb") as f:
        client.put_object(Bucket=bucket, Key=object_key, Body=f.read(), ContentType="audio/mpeg")
    return object_key


def spaces_presigned_get(object_key, expires_in=3600):
    client = _spaces_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": os.environ["SPACES_BUCKET"], "Key": object_key},
        ExpiresIn=expires_in,
    )


def spawn_modal_dub(audio_url, target_lang, job_id, provider="groq", groq_api_override=None, gemini_api_override=None):
    """Spawn the Modal audio dubbing pipeline. Returns modal call object_id.

    provider: 'groq' | 'gemini' | 'helsinki'. Selects translation backend.
    *_override: optional API keys from caller (mac app config); fall back to env.
    """
    import modal
    run_pipeline = modal.Function.from_name(MODAL_APP_NAME, MODAL_FN_NAME)

    groq_api = (groq_api_override or os.environ.get("GROQ_API_KEY") or "").strip() or None
    gemini_api = (gemini_api_override or os.environ.get("GEMINI_API_KEY") or "").strip() or None
    mistral_api = (os.environ.get("MISTRAL_API_KEY") or "").strip() or None

    if provider == "groq":
        gemini_api = None  # force Groq path
    elif provider == "gemini":
        groq_api = None  # force Gemini path
    elif provider == "helsinki":
        groq_api = None
        gemini_api = None  # both None → pipeline falls back to MarianMT/Helsinki

    call = run_pipeline.spawn(
        job_id=job_id,
        audio_url=audio_url,
        targ=target_lang,
        mistral_api=mistral_api,
        groq_api=groq_api,
        gemini_api=gemini_api,
        groq_model="openai/gpt-oss-120b" if groq_api else None,
        translation_mode="full_transcript",
    )
    return call.object_id


def get_modal_job_status(modal_call_id, job_id):
    """Poll Modal. Returns dict with status + progress + (when done) output_url."""
    import modal
    progress_dict = modal.Dict.from_name(MODAL_PROGRESS_DICT, create_if_missing=True)
    call = modal.FunctionCall.from_id(modal_call_id)

    # First, see if it finished
    try:
        output_key = call.get(timeout=0)
    except TimeoutError:
        # Still running — read live progress from Modal Dict
        prog = None
        try:
            prog = progress_dict.get(job_id)
        except Exception:
            pass
        return {
            "status": "processing",
            "progress": (prog or {}).get("progress", 0),
            "stage": (prog or {}).get("stage", "Pipeline running on GPU..."),
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}

    # Completed — return presigned download URL
    return {
        "status": "completed",
        "progress": 100,
        "stage": "Done",
        "output_url": spaces_presigned_get(output_key, expires_in=86400),
    }


def handle_dub_url(message):
    """Main flow: download → upload to Spaces → spawn Modal → return job_id."""
    url = message.get("url")
    target_lang = message.get("target_lang", "zh")
    provider = message.get("provider", "groq")
    groq_api_override = message.get("groq_api")
    gemini_api_override = message.get("gemini_api")
    if not url:
        return {"error": "no URL provided"}

    load_project_env()

    job_id = str(uuid.uuid4())
    audio_path = None
    try:
        # 1. yt-dlp local
        audio_path = download_audio(url, whisper_ready=False)
        # 2. Upload audio bytes to Spaces
        object_key = upload_audio_to_spaces(audio_path, job_id)
        # 3. Presigned GET URL for Modal to download
        audio_url = spaces_presigned_get(object_key, expires_in=3600)
        # 4. Spawn Modal pipeline
        modal_call_id = spawn_modal_dub(
            audio_url, target_lang, job_id,
            provider=provider,
            groq_api_override=groq_api_override,
            gemini_api_override=gemini_api_override,
        )
        # 5. Save state so subsequent polls can find this
        save_job_state(job_id, {
            "modal_call_id": modal_call_id,
            "target_lang": target_lang,
            "source_url": url,
        })
        return {"success": True, "job_id": job_id, "modal_call_id": modal_call_id}
    finally:
        # Clean up local audio file
        if audio_path and os.path.exists(audio_path):
            try: os.remove(audio_path)
            except OSError: pass
            try: os.rmdir(os.path.dirname(audio_path))
            except OSError: pass


def handle_poll_job(message):
    """Poll a job by job_id. Returns current status."""
    job_id = message.get("job_id")
    if not job_id:
        return {"error": "no job_id provided"}

    state = load_job_state(job_id)
    if not state:
        return {"error": f"unknown job_id: {job_id}"}

    load_project_env()
    return get_modal_job_status(state["modal_call_id"], job_id)


def main():
    message = read_message()
    if not message:
        return

    msg_type = message.get("type")

    try:
        if msg_type == "ping":
            send_message({"pong": True})

        elif msg_type == "dub-url":
            send_message(handle_dub_url(message))

        elif msg_type == "poll-job":
            send_message(handle_poll_job(message))

        elif msg_type == "transcribe-url":
            url = message.get("url")
            language = message.get("language")
            if not url:
                send_message({"error": "no URL provided"})
                return
            wav_path = download_audio(url, whisper_ready=True)
            transcript = transcribe(wav_path, language=language)
            send_message({
                "success": True,
                "transcript": transcript,
                "audio_path": wav_path,
            })

        elif msg_type == "download-audio":
            # Legacy API-server path. Still supported but new clients should use dub-url.
            url = message.get("url")
            target_lang = message.get("target_lang", "zh")
            api_base = message.get("api_base", "http://localhost:8000")
            if not url:
                send_message({"error": "no URL provided"})
                return
            audio_path = download_audio(url)
            file_size = os.path.getsize(audio_path)
            job_id = upload_to_api(audio_path, target_lang, api_base)
            os.remove(audio_path)
            os.rmdir(os.path.dirname(audio_path))
            send_message({"success": True, "job_id": job_id, "size": file_size})

        else:
            send_message({"error": f"unknown message type: {msg_type}"})

    except Exception as e:
        import traceback
        send_message({"error": str(e), "traceback": traceback.format_exc()})


if __name__ == "__main__":
    main()
