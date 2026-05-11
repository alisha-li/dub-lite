# Dub Lite — Native App Pivot Milestones

Strategy: **Path C — all pipeline stays on Modal. Mac app + Chrome ext are thin clients.** Quality is non-negotiable, speed is the priority, and the local-split approach (whisper.cpp on user hardware) gave up T4 speed advantage for negligible cost savings. Keep Modal doing what it does well.

## Architecture

```
[Chrome extension] ── native messaging ──┐
                                          ▼
                                   [Mac local app]
                                          │
                                          │ HTTPS (audio bytes or URL)
                                          ▼
                                   [Modal pipeline] — unchanged
                                          │  PyAnnote, whisper, Groq translate,
                                          │  emotion, XTTS, demucs, stitch, encode
                                          ▼
                                   [DigitalOcean Spaces]
                                          │ presigned GET URL
                                          ▼
                                   Mac app plays dubbed audio
```

What dies vs stays:

**Dies:**
- DO droplet API server (eventually — Mac app can call Modal directly)
- Frontend webapp (Vite/React) when Mac app reaches parity
- Redis (poll Modal Dict directly from Mac app)

**Stays:**
- Modal pipeline (whisper + diarize + emotion + XTTS + stitch + encode all on GPU)
- Spaces (audio + video transit)
- Groq (translation, called from Modal as before)
- HuggingFace + PyAnnote API (Modal uses them)

## Milestones

### ✅ Milestone 1 — Native messaging hello world
Chrome extension → local Python script via stdio. Built and verified.

- `chromeExtension/native-host/host.py` — Python stdin/stdout loop
- `chromeExtension/native-host/com.dub_lite.host.json` — Chrome manifest
- `chromeExtension/native-host/install.sh` — Manifest installer
- `chromeExtension/background.js` — Routes ping + transcribe-url
- Installed for Chrome, Arc, Brave
- Ping/pong tested ✓

### ✅ Milestone 2 — Local whisper.cpp (built but de-scoped)
Built whisper.cpp Metal locally + integrated into native messaging. Works. **Now de-scoped** because Modal T4 is faster (~10x realtime vs M2 ~6x).

- Keep the install for future "transcribe only / offline" features
- `host.py` `transcribe-url` handler stays but won't be used in main dub flow
- `~/whisper.cpp/build/bin/whisper-cli` + `~/whisper.cpp/models/ggml-medium.bin` stay on disk
- May re-enable later if launching standalone "free transcription" mode

### ✅ Milestone 3 — Mac app calls Modal directly (skip API server)
**Done.** Native host now spawns Modal jobs directly. No API server in the loop. Tested end-to-end with a 19s YouTube clip → produced dubbed WAV at /tmp/dub-lite-test-output.wav.

**Earlier plan (kept for reference):**
Currently: Chrome ext → API server → Modal. Goal: Chrome ext → Mac native app → Modal directly.

**Steps:**
1. Add Modal Python SDK to native host: `pip install modal`
2. User runs `modal token new` once (BYO Modal token, paste into app)
3. New `dub-url` handler in `host.py`:
   - Receives YouTube URL + target language from chrome ext
   - Calls existing `run_audio_dubbing_pipeline` Modal function via `modal.Function.from_name(...)` → `.spawn(...)`
   - Polls `modal.Dict.from_name("dub-lite-progress")` for status updates
   - Streams progress back to chrome ext via native messaging
   - On complete, returns presigned Spaces URL
4. Chrome ext UI: progress bar + play final audio
5. Remove `upload_to_api()` path from `host.py`

**No Modal pipeline changes.** All GPU stages stay where they are.

**Human checkpoint:** dubbed audio plays — voice matches speaker? Lip sync within ±200ms? Emotion authentic?

### Milestone 4 — Cold-start optimization
Goal: cut Modal cold start from ~30s to <10s.

**Approaches:**
- Strip Modal image: remove unused libs (TensorFlow, scipy variants, etc.)
- Lazy-load models inside function (XTTS only loaded if TTS stage runs)
- Use `@app.function(scaledown_window=900)` to keep container warm 15 min after last call
- Optional: cron ping every 4 min via separate tiny `@app.function` to keep container warm cheap
- Avoid full `keep_warm=1` ($430/mo on T4) until revenue justifies

**Measure:** time first-job-after-idle vs job-during-warm. Aim for ~10s cold delta.

### ~~Milestone 5 — Whisper turbo swap~~ — N/A
Pipeline uses **Mistral Voxtral** for transcription (`voxtral-mini-2602`, with built-in diarization), not Faster-Whisper. The `from faster_whisper import WhisperModel` line in pipeline/main.py:16 is dead code — model never instantiated. Whisper turbo swap moot. Voxtral is the transcription engine.

If transcription quality becomes a concern: evaluate Voxtral vs alternatives, not Whisper model size.

### Milestone 6 — Mac app shell
- Electron + Python sidecar (Python sidecar = current native host)
- Menu bar icon
- Floating progress window
- Settings UI for Modal token, target language, voice options
- Hotkey support (global ⌘⇧D = "dub current YouTube tab")
- Detects YouTube URLs from clipboard or chrome ext

### Milestone 7 — Packaging + distribution
- Apple Developer cert + notarization ($99/yr)
- DMG installer
- Installer registers native messaging manifest for Chrome/Arc/Brave
- First-run: prompts user to paste Modal token
- Auto-update via electron-updater

## Quality gates (human required)

LLM agents cannot judge:
- ✗ Voice naturalness in TTS output
- ✗ Voice clone fidelity vs source speaker
- ✗ Emotion authenticity
- ✗ Lip sync timing precision (±200ms tolerance)
- ✗ Background music balance vs dub volume
- ✗ Translation register / cultural appropriateness

LLM auto-handles:
- ✓ Modal SDK integration + retry logic
- ✓ Native messaging protocol bugs
- ✓ Image/dep slimming for cold start
- ✓ JSON parsing + structural validation
- ✓ Stack trace debugging
- ✓ ffmpeg / yt-dlp invocations

## Open decisions

- **Modal auth model:** BYO user token v1, proxy server v2 for friction-free signup
- **First language target:** zh only or all from start? Default zh
- **Whisper model:** N/A — pipeline uses Mistral Voxtral, not Whisper
- **Distribution:** DMG manual install (Mac App Store sandbox kills yt-dlp)
- **Audio capture for non-URL sources (Netflix, etc.):** ScreenCaptureKit live tap, deferred post-v1
- **Pricing model:** free with BYO Modal+Groq keys? Subscription with bundled compute? Deferred

## Current state

- DO droplet still serves old webapp + chrome ext path (production fallback)
- Modal pipeline runs unchanged on production
- Chrome ext `q` key now uses Mac-app-direct path (host.py → Modal) — bypasses API server
- `host.py` shebang hardcoded to pyenv python (`/Users/alishali/.pyenv/versions/3.12.0/bin/python3`) so Chrome-launched subprocess finds installed deps. Will need fix for distribution.
- Local whisper.cpp installed but de-scoped — `transcribe-url` handler kept for future
- Verified end-to-end with 19s "Me at the zoo" YouTube clip — dubbed WAV at `outputs/dub-test-zoo.wav`
- Mute fix (volume=0 + repeated re-mute) applied to defeat YouTube re-muting on play events

## Cleanup eventually

- `pipeline/test_full_transcript.py` — scratch test, delete when comfortable
- `pipeline/pipeline.log` truncated (was 11MB with leaked Groq key); **rotate key at console.groq.com**
- `api/pipeline.log` (empty) — delete
- API server + frontend webapp die when Mac app shell reaches parity (post-Milestone 6)
- `outputs/dub-test-zoo.wav` — test artifact, delete or git-ignore
