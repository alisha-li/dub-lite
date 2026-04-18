"""Standalone test of translate_full_transcript logic without importing utils.py
(utils has heavy deps that break on local Mac). Copies the relevant functions
verbatim so behavior matches production."""

import os
import json
import pickle
import time
import logging

from dotenv import load_dotenv
load_dotenv("../.env")

from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LANG_CODE_TO_NAME = {
    "en": "English", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "ru": "Russian", "zh": "Chinese",
}


def _build_full_transcript_prompt(chunk_sents, offset, src, targ):
    source_lang = LANG_CODE_TO_NAME.get(src, src)
    target_lang = LANG_CODE_TO_NAME.get(targ, targ)

    unique_speakers = {s.get('speaker') for s in chunk_sents if s.get('speaker')}
    multi_speaker = len(unique_speakers) > 1

    lines = []
    for i, s in enumerate(chunk_sents):
        idx = offset + i
        st = s.get('start')
        et = s.get('end')
        text = (s.get('sentence') or '').strip().replace('\n', ' ')
        ts_part = (
            f"{st:.1f}s-{et:.1f}s"
            if isinstance(st, (int, float)) and isinstance(et, (int, float))
            else ""
        )
        if multi_speaker:
            spk = s.get('speaker') or 'Unknown'
            header = f"{idx} | {ts_part} | {spk}" if ts_part else f"{idx} | {spk}"
        else:
            header = f"{idx} | {ts_part}" if ts_part else f"{idx}"
        lines.append(f"[{header}] {text}")
    transcript = "\n".join(lines)

    if multi_speaker:
        line_format = "[index | timestamp | Speaker] text"
        style_hint = (
            "Use context from surrounding lines to resolve pronouns, slang, and ambiguity. "
            "Lines from the same speaker label should use a consistent register and word choice."
        )
    else:
        line_format = "[index | timestamp] text"
        style_hint = (
            "Use context from surrounding lines to resolve pronouns, slang, and ambiguity. "
            "Keep register and word choice consistent across the whole transcript."
        )

    first_idx = offset
    second_idx = offset + 1 if len(chunk_sents) > 1 else offset
    return (
        f"You are translating a transcript from {source_lang} to {target_lang}.\n\n"
        f"Each line has the format: {line_format}\n\n"
        f"Translate every line into {target_lang}. {style_hint}\n\n"
        f"Output rules:\n"
        f"- Respond with a single JSON object, nothing else.\n"
        f"- Keys are the indices as strings (\"0\", \"1\", ...).\n"
        f"- Values are the translated text ONLY — no index, no timestamp, no speaker label.\n"
        f"- Include EVERY index shown in the transcript. Do not skip any.\n"
        f"- Do not wrap in markdown fences or add commentary.\n\n"
        f"Transcript:\n{transcript}\n\n"
        f'Example output shape: {{"{first_idx}": "translation", "{second_idx}": "translation"}}'
    )


def translate_full_transcript(sentences, src, targ, groq_api, groq_model=None):
    if not groq_api:
        raise ValueError("groq_api required")
    if not sentences:
        return sentences
    model = groq_model or "openai/gpt-oss-120b"

    MAX_CHARS_PER_CHUNK = 30_000
    MAX_SENTENCES_PER_CHUNK = 200

    chunks = []
    cur_start = 0
    cur_chars = 0
    for i, s in enumerate(sentences):
        s_len = len(s.get('sentence') or '')
        over_sent = (i - cur_start) >= MAX_SENTENCES_PER_CHUNK
        over_char = (cur_chars + s_len) > MAX_CHARS_PER_CHUNK
        if (over_sent or over_char) and i > cur_start:
            chunks.append((cur_start, i))
            cur_start = i
            cur_chars = 0
        cur_chars += s_len
    chunks.append((cur_start, len(sentences)))
    logger.info(f"Full-transcript translate: {len(sentences)} sentences → {len(chunks)} chunk(s)")

    translated_by_idx = {}
    client = Groq(api_key=groq_api)

    for chunk_i, (start, end) in enumerate(chunks):
        chunk_sents = sentences[start:end]
        prompt = _build_full_transcript_prompt(chunk_sents, start, src, targ)

        parsed = None
        for attempt in range(3):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                raw = (completion.choices[0].message.content or "").strip()
                parsed = json.loads(raw)
                break
            except Exception as e:
                logger.warning(f"Chunk {chunk_i+1}/{len(chunks)} attempt {attempt+1}/3: {e}")
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))

        if not isinstance(parsed, dict):
            logger.warning(f"Chunk {chunk_i+1} produced no dict")
            continue

        added = 0
        for k, v in parsed.items():
            try:
                idx = int(k)
            except (ValueError, TypeError):
                continue
            if isinstance(v, str) and v.strip():
                translated_by_idx[idx] = v.strip()
                added += 1
        logger.info(f"Chunk {chunk_i+1}/{len(chunks)}: {added}/{end-start} translations")

    for i, s in enumerate(sentences):
        s['translation'] = translated_by_idx.get(i, "")

    return sentences


with open("temp/sentences.pkl", "rb") as f:
    sentences = pickle.load(f)

print(f"Loaded {len(sentences)} sentences")
print("Sample sentence keys:", list(sentences[0].keys()) if sentences else "(empty)")
print()

# Show the prompt for inspection (first chunk only)
if sentences:
    sample_prompt = _build_full_transcript_prompt(sentences[:min(5, len(sentences))], 0, "en", "zh")
    print("=== SAMPLE PROMPT (first 5 sentences) ===")
    print(sample_prompt)
    print("=" * 40, "\n")

result = translate_full_transcript(
    sentences, src="en", targ="zh",
    groq_api=os.getenv("GROQ_API_KEY"),
)

print("\n=== RESULTS (first 15) ===")
for i, s in enumerate(result[:15]):
    src = s.get("sentence", "")[:60]
    tr = s.get("translation", "")[:60]
    print(f"[{i}] {src}")
    print(f"     → {tr}")

missing = sum(1 for s in result if not s.get("translation"))
print(f"\nTotal missing: {missing}/{len(result)}")
