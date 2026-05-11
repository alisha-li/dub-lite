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
- dub-url {url, target_lang} → {success, job_id} (calls droplet API, no Modal SDK locally)
- poll-job {job_id} → {status, progress, stage, output_url?}
- transcribe-url {url, language?} → {success, transcript, audio_path} (local whisper.cpp, de-scoped)
"""

import sys
import json
import struct
import subprocess
import tempfile
import os
import urllib.request
import urllib.parse
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

# Mac app config (api_base, default_target_lang). Single source of truth.
CONFIG_PATH = os.path.expanduser("~/Library/Application Support/dub-lite/config.json")
DEFAULT_API_BASE = "http://159.89.182.232"

# Per-call state persists job_id metadata so subsequent polls can find it.
STATE_DIR = os.path.expanduser("~/Library/Application Support/dub-lite/jobs")
os.makedirs(STATE_DIR, exist_ok=True)

WHISPER_BIN = os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = os.path.expanduser("~/whisper.cpp/models/ggml-medium.bin")


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


def load_app_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def get_api_base():
    cfg = load_app_config()
    base = (cfg.get("api_base") or DEFAULT_API_BASE).rstrip("/")
    return base


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


# --- API server HTTP helpers ---

def _http_post_form(url, form_fields, timeout=30):
    """POST application/x-www-form-urlencoded to url. Returns parsed JSON."""
    body = urllib.parse.urlencode(form_fields).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise Exception(f"POST {url} → {e.code}: {e.read().decode(errors='replace')}")


def _http_get_json(url, timeout=30):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise Exception(f"GET {url} → {e.code}: {e.read().decode(errors='replace')}")


def _http_put_file(presigned_url, file_path, content_type, timeout=300):
    """PUT file bytes to a presigned URL."""
    with open(file_path, "rb") as f:
        body = f.read()
    req = urllib.request.Request(
        presigned_url,
        data=body,
        headers={"Content-Type": content_type, "Content-Length": str(len(body))},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        raise Exception(f"PUT spaces → {e.code}: {e.read().decode(errors='replace')}")


def handle_dub_url(message):
    """yt-dlp local → presigned PUT to Spaces → POST /api/jobs/audio → job_id."""
    url = message.get("url")
    target_lang = message.get("target_lang", "zh")
    if not url:
        return {"error": "no URL provided"}

    api_base = get_api_base()
    audio_path = None
    try:
        # 1. yt-dlp local → /tmp/<random>/audio.mp3
        audio_path = download_audio(url, whisper_ready=False)
        filename = os.path.basename(audio_path)

        # 2. Get presigned PUT URL from API server
        presigned = _http_post_form(
            f"{api_base}/api/upload-url",
            {"filename": filename},
        )
        upload_url = presigned["upload_url"]
        object_key = presigned["object_key"]
        content_type = presigned.get("content_type", "audio/mpeg")

        # 3. PUT audio bytes directly to Spaces
        _http_put_file(upload_url, audio_path, content_type)

        # 4. Create job at API server (server spawns Modal with its own keys)
        job_resp = _http_post_form(
            f"{api_base}/api/jobs/audio",
            {
                "spaces_object_key": object_key,
                "target_language": target_lang,
            },
        )
        job_id = job_resp["job_id"]

        # 5. Persist enough to poll later (api_base, source url for retry)
        save_job_state(job_id, {
            "api_base": api_base,
            "target_lang": target_lang,
            "source_url": url,
            "object_key": object_key,
        })
        return {"success": True, "job_id": job_id}
    finally:
        if audio_path and os.path.exists(audio_path):
            try: os.remove(audio_path)
            except OSError: pass
            try: os.rmdir(os.path.dirname(audio_path))
            except OSError: pass


def handle_poll_job(message):
    """GET /api/jobs/{id} on the API server. Returns its JSON response."""
    job_id = message.get("job_id")
    if not job_id:
        return {"error": "no job_id provided"}

    state = load_job_state(job_id)
    api_base = (state or {}).get("api_base") or get_api_base()
    return _http_get_json(f"{api_base}/api/jobs/{job_id}")


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

        else:
            send_message({"error": f"unknown message type: {msg_type}"})

    except Exception as e:
        import traceback
        send_message({"error": str(e), "traceback": traceback.format_exc()})


if __name__ == "__main__":
    main()
