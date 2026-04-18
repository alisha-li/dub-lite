#!/usr/bin/env python3
"""
Native Messaging host for Dub Lite Chrome extension.
Chrome launches this script, sends a JSON message via stdin,
and reads the JSON response from stdout.

Chrome's native messaging protocol:
- Each message is prefixed with a 4-byte unsigned int (little-endian) indicating the message length
- Followed by the JSON message as UTF-8 bytes
"""

import sys
import json
import struct
import subprocess
import tempfile
import os
import base64
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


def read_message():
    """Read a native messaging message from stdin."""
    # Read the 4-byte length prefix
    raw_length = sys.stdin.buffer.read(4)
    if not raw_length:
        return None
    length = struct.unpack('<I', raw_length)[0]

    # Read the JSON message
    raw_message = sys.stdin.buffer.read(length)
    return json.loads(raw_message)


def send_message(message):
    """Send a native messaging message to stdout."""
    encoded = json.dumps(message).encode('utf-8')
    # Write the 4-byte length prefix, then the message
    sys.stdout.buffer.write(struct.pack('<I', len(encoded)))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def download_audio(url):
    """Use yt-dlp to download audio-only from a YouTube URL. Returns the file path."""
    output_dir = tempfile.mkdtemp(prefix="dub-lite-")
    output_path = os.path.join(output_dir, "audio.%(ext)s")

    result = subprocess.run(
        [
            "yt-dlp",
            "-f", "bestaudio",        # audio only — much smaller/faster than full video
            "-x",                      # extract audio
            "--audio-format", "mp3",   # convert to mp3
            "-o", output_path,         # output path
            "--no-playlist",           # single video only
            url,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise Exception(f"yt-dlp failed: {result.stderr}")

    # Find the output file (yt-dlp may name it differently)
    for f in os.listdir(output_dir):
        return os.path.join(output_dir, f)

    raise Exception("yt-dlp produced no output file")


def upload_to_api(audio_path, target_lang, api_base):
    """POST the audio file to the API and return a job_id."""
    # Build a multipart/form-data request using only stdlib (no requests dependency)
    boundary = "----DubLiteBoundary" + os.urandom(8).hex()
    filename = os.path.basename(audio_path)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # Construct multipart body
    body = b""
    # Audio file field
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    body += b"Content-Type: audio/mpeg\r\n\r\n"
    body += audio_bytes
    body += b"\r\n"
    # Target language field
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


def main():
    message = read_message()
    if not message:
        return

    msg_type = message.get("type")

    if msg_type == "download-audio":
        url = message.get("url")
        target_lang = message.get("target_lang", "zh")
        api_base = message.get("api_base", "http://localhost:8000")
        if not url:
            send_message({"error": "no URL provided"})
            return

        try:
            # Step 1: Download audio from YouTube via yt-dlp
            audio_path = download_audio(url)
            file_size = os.path.getsize(audio_path)

            # Step 2: POST the audio file to our API to create a dubbing job
            job_id = upload_to_api(audio_path, target_lang, api_base)

            # Step 3: Clean up the temp file
            os.remove(audio_path)
            os.rmdir(os.path.dirname(audio_path))

            send_message({
                "success": True,
                "job_id": job_id,
                "size": file_size,
            })
        except Exception as e:
            send_message({"error": str(e)})

    elif msg_type == "ping":
        send_message({"pong": True})

    else:
        send_message({"error": f"unknown message type: {msg_type}"})


if __name__ == "__main__":
    main()
