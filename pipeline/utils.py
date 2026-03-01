# download_video_and_extract_audio
import os
import re
import requests
import yt_dlp
from pydub import AudioSegment

_YT_DLP_DOMAINS = ("youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "twitch.tv")

# diarize_audio
# from pyannoteai.sdk import Client
# from pyannote.audio import Pipeline as PyannotePipeline

# denoise_audio
from df import config
from df.enhance import enhance, init_df, load_audio, save_audio
from df import utils as df_utils
import torch
import sys

# create_sentences
from collections import defaultdict
# from nltk.tokenize import sent_tokenize
from wtpsplit import SaT
# import pysbd
import logging
logger = logging.getLogger(__name__)

# translate
from groq import Groq
from google import genai as gemini
from transformers import MarianMTModel, MarianTokenizer
from transformers import pipeline

# classify_emotion
# torchaudio 2.8+ removed list_audio_backends – stub for SpeechBrain
# torchaudio 2.8+ removed info – stub for DeepFilterNet df.io.load_audio
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile", "ffmpeg", "sox"]


def _torchaudio_info_soundfile(path, **kwargs):
    """Fallback for torchaudio.info (removed in 2.8). DeepFilterNet df.io needs it."""
    from dataclasses import dataclass
    import soundfile as sf
    i = sf.info(path)
    bps = 0
    if i.subtype and "PCM" in i.subtype:
        for n in (8, 16, 24, 32):
            if str(n) in i.subtype:
                bps = n
                break

    @dataclass
    class AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str

    return AudioMetaData(
        sample_rate=int(i.samplerate),
        num_frames=int(i.frames),
        num_channels=int(i.channels),
        bits_per_sample=bps,
        encoding=i.format or "PCM_S",
    )


if not hasattr(torchaudio, "info") or torchaudio.info is None:
    torchaudio.info = _torchaudio_info_soundfile
from speechbrain.inference.interfaces import foreign_class

# adjust_audio
import librosa
import soundfile as sf
import pyrubberband as pyrb
import numpy as np

# combine_audio_with_background
import subprocess
from types import SimpleNamespace

_helsinki_cache = {}
_translategemma_cache = {}  # (model, processor) for text-only path


def _normalize_speaker_to_pyannote(speaker_id: str) -> str:
    """Convert Mistral speaker_id (e.g. speaker_1, SPEAKER_0) to PyAnnote format (SPEAKER_00, SPEAKER_01)."""
    if not speaker_id:
        return "SPEAKER_00"
    m = re.search(r"(\d+)", str(speaker_id))
    idx = int(m.group(1)) if m else 0
    return f"SPEAKER_{idx:02d}"


def mistral_segments_to_pipeline(mistral_segments):
    """Convert Mistral TranscriptionSegmentChunk list to pipeline format.
    Adds 'speaker' (from speaker_id, normalized to PyAnnote SPEAKER_00 format) and synthesized 'words'.
    """
    if not mistral_segments:
        return []
    result = []
    for seg in mistral_segments:
        text = getattr(seg, "text", None) or (seg.get("text", "") if isinstance(seg, dict) else "")
        start = getattr(seg, "start", None)
        if start is None and isinstance(seg, dict):
            start = seg.get("start", 0)
        end = getattr(seg, "end", None)
        if end is None and isinstance(seg, dict):
            end = seg.get("end", 0)
        speaker_id = getattr(seg, "speaker_id", None)
        if speaker_id is None and isinstance(seg, dict):
            speaker_id = seg.get("speaker_id")
        speaker = _normalize_speaker_to_pyannote(speaker_id)
        words = [
            SimpleNamespace(word=w, start=float(start), end=float(end))
            for w in (text or "").split()
        ]
        result.append({
            "text": text or "",
            "start": float(start),
            "end": float(end),
            "speaker": speaker,
            "words": words,
        })
    return result

def download_video_and_extract_audio(
    source, 
    video_path="temp/orig_video.mp4", 
    audio_path="temp/orig_audio.wav"
):
    """
    Downloads a video from URL or uses a local file, then extracts audio as WAV.
    
    Args:
        source: YouTube/video URL OR local file path
        video_path: Path to save the downloaded video (ignored if source is local file)
        audio_path: Path to save the extracted audio
        
    Returns:
        tuple: (video_path, audio_path, AudioSegment object)
    """
    # Ensure temp directory exists
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    
    # Check if source is a local file or URL
    if os.path.isfile(source):
        print(f"Using local video file: {source}")
        video_path = source
    elif isinstance(source, str) and source.startswith(("http://", "https://")):
        source_lower = source.lower()
        if any(domain in source_lower for domain in _YT_DLP_DOMAINS):
            print(f"Downloading from streaming site via yt-dlp: {source}")
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            cookies_path = os.path.join(os.path.dirname(__file__), "cookies.txt")
            ydl_opts = {"format": "best", "outtmpl": video_path}
            if os.path.isfile(cookies_path):
                ydl_opts["cookiefile"] = cookies_path
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([source])
        else:
            print(f"Downloading from direct URL: {source[:80]}...")
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            resp = requests.get(source, stream=True, timeout=600)
            resp.raise_for_status()
            with open(video_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        print(f"Downloading video from URL: {source}")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        cookies_path = os.path.join(os.path.dirname(__file__), "cookies.txt")
        ydl_opts = {"format": "best", "outtmpl": video_path}
        if os.path.isfile(cookies_path):
            ydl_opts["cookiefile"] = cookies_path
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([source])
    
    # Extract audio
    print(f"Extracting audio to: {audio_path}")
    orig_audio = AudioSegment.from_file(video_path, format="mp4")
    orig_audio.export(audio_path, format="wav")
    
    return video_path, audio_path, orig_audio


def segments_to_speaker_turns(segments: list) -> dict:
    """Derive speaker turns from segments (e.g. Mistral) that have start, end, speaker.
    Returns {(start, end): speaker} dict for split_speakers_and_denoise.
    """
    speaker_turns = {}
    for seg in segments:
        start = seg.get("start")
        end = seg.get("end")
        speaker = seg.get("speaker")
        if start is not None and end is not None and speaker and abs(end - start) > 0.2:
            speaker_turns[(float(start), float(end))] = speaker
    return speaker_turns


# Pyannote diarization (commented out – use Mistral transcription with diarize=True instead)
# def diarize_audio(audio_path: str, pyannote_key: str, hf_token: str):
#     if pyannote_key: # paid
#         client = Client(pyannote_key)
#         orig_audio_url = client.upload(audio_path)
#         diarization_job = client.diarize(orig_audio_url, transcription=True)
#         diarization = client.retrieve(diarization_job)
#
#         turns = diarization['output']['turnLevelTranscription']
#         speaker_turns = {}
#
#         for turn in turns:
#             start = turn['start']
#             end = turn['end']
#             speaker = turn['speaker']
#
#             if abs(end - start) > 0.2:
#                 speaker_turns[(start, end)] = speaker
#                 logger.info(f"Speaker {speaker}: from {start}s to {end}s")
#
#     else:  #free
#         logger.info("Running speaker diarization (this may take several minutes)...")
#         diarizationPipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=hf_token)
#
#         if torch.cuda.is_available():
#             diarizationPipeline = diarizationPipeline.to(torch.device("cuda"))
#
#         output = diarizationPipeline(audio_path)
#         speaker_turns = {}
#
#         for turn, speaker in output.speaker_diarization:
#             if abs(turn.end - turn.start) > .2:
#                 logger.info(f"Speaker {speaker}: from {turn.start}s to {turn.end}s")
#                 speaker_turns[(turn.start, turn.end)] = speaker
#
#         logger.info("Diarization completed")
#
#     return speaker_turns


def get_denoiser():
     # Initialize model
    print("Loading DeepFilterNet2 model...")
    try:
        model, df, _ = init_df()
    except Exception as e:
        raise e
    return model, df


def denoise_audio(input_path: str, output_path: str = None, model=None, df=None):
    """Denoise an audio file using DeepFilterNet2"""
    if model is None or df is None:
        model, df = get_denoiser()

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False
    
    # Generate output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_denoised{ext}"
    
    # Load audio
    sr = config("sr", 48000, int, section="df")
    print(f"Loading audio from {input_path}...")
    sample, meta = load_audio(input_path, sr)
    
    # Convert to mono if needed
    if sample.dim() > 1 and sample.shape[0] > 1:
        sample = sample.mean(dim=0, keepdim=True)
    
    print(f"Audio shape: {sample.shape}, Sample rate: {sr}Hz")
    
    # Denoise — force CPU on GPU hosts (Modal): DeepFilterNet uses get_device() for features;
    # patch it so model and features stay on CPU and match.
    print("Denoising audio...")
    sample = sample.cpu()
    _orig = df_utils.get_device
    df_utils.get_device = lambda: torch.device("cpu")
    try:
        enhanced = enhance(model, df, sample)
    finally:
        df_utils.get_device = _orig
    
    # Apply fade-in to avoid clicks
    lim = torch.linspace(0.0, 1.0, int(sr * 0.15)).unsqueeze(0)
    if lim.shape[1] < enhanced.shape[1]:
        lim = torch.cat((lim, torch.ones(1, enhanced.shape[1] - lim.shape[1])), dim=1)
    enhanced = enhanced * lim
    
    # Resample if needed
    if meta.sample_rate != sr:
        from df.io import resample
        enhanced = resample(enhanced, sr, meta.sample_rate)
        sr = meta.sample_rate
    
    # Save
    print(f"Saving denoised audio to {output_path}...")
    save_audio(output_path, enhanced.cpu(), sr)
    
    print(f"Done! Denoised audio saved to: {output_path}")
    return True


# Max speaker audio duration (seconds) to attempt denoising. Longer files can trigger
# std::length_error in libsndfile/torchaudio when loading.
MAX_DENOISE_DURATION_SEC = 300  # 5 minutes


def split_speakers_and_denoise(audio: AudioSegment, speaker_turns: dict, output_dir: str = "temp/speakers_audio"):
    # For voice cloning later
    speakers = set(speaker_turns.values())
    # Compute duration per speaker (seconds)
    speaker_durations = {s: 0.0 for s in speakers}
    for (start, end), spk in speaker_turns.items():
        speaker_durations[spk] += float(end) - float(start)

    for speaker in speakers:
        speaker_audio = AudioSegment.empty()
        for key, value in speaker_turns.items():
            if speaker == value:
                start = int(key[0])*1000  # Convert seconds to milliseconds
                end = int(key[1])*1000
                speaker_audio += audio[start:end]  # Extract this speaker's audio segments
        wav_path = f"{output_dir}/{speaker}.wav"
        speaker_audio.export(wav_path, format="wav")

        dur = speaker_durations[speaker]
        if dur > MAX_DENOISE_DURATION_SEC:
            logger.info("Skipping denoising for %s (%.1f min > %.1f min limit)", speaker, dur / 60, MAX_DENOISE_DURATION_SEC / 60)
            continue

        # Run denoise in subprocess to isolate C++ crashes (e.g. std::length_error in load_audio)
        denoise_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "denoise_one.py")
        try:
            result = subprocess.run(
                [sys.executable, denoise_script, wav_path, wav_path],
                capture_output=True,
                timeout=600,
                cwd="/root" if os.path.exists("/root") else os.getcwd(),
            )
            if result.returncode == 0:
                logger.info("Using denoised audio for voice cloning")
            else:
                logger.warning("Denoising failed (exit %d), using original audio", result.returncode)
        except subprocess.TimeoutExpired:
            logger.warning("Denoising timed out, using original audio")
        except Exception as e:
            logger.warning("Denoising failed: %s, using original audio", e)
    return output_dir


def merge_close_segments(segments_with_speakers, max_gap=0.5):
    """Merge segments if gap between them is < max_gap seconds and shared speaker"""
    merged = []
    current = segments_with_speakers[0]
    
    for next_seg in segments_with_speakers[1:]:
        gap = next_seg['start'] - current['end']
        if gap < max_gap and current['speaker'] == next_seg['speaker']:
            # Merge: extend current segment
            current['end'] = next_seg['end']
            current['text'] += " " + next_seg['text']
            current['words'].extend(next_seg['words'])
        else:
            merged.append(current)
            current = next_seg
    merged.append(current)
    return merged


def assign_speakers_to_segments(segments: list, speaker_turns: dict):
    time_stamped = []
    wordCount = 0
    segments_with_speakers = []
    
    # Get the last speaker (least frequent, grainy) as fallback for unmatched segments
    fallback_speaker = list(speaker_turns.values())[-1] if speaker_turns else None
    
    for segment in segments:
        first_word_idx = wordCount
        for word in segment.words:
            wordCount += 1
            time_stamped.append([word.word, word.start, word.end]) # used later when determining start and end times for sentences
        last_word_idx = wordCount - 1

        start = time_stamped[first_word_idx][1]
        end = time_stamped[last_word_idx][2]

        resSpeaker = None
        max_overlap = 0
        for times, speaker in speaker_turns.items():
            speaker_start = times[0]  # Keep as float!
            speaker_end = times[1]    # Keep as float!
            if speaker_end < start:
                continue
            elif speaker_start > end:
                break
                
            overlap_start = max(start, speaker_start)
            overlap_end = min(end, speaker_end)
            overlap = overlap_end - overlap_start
        
            # Update speaker if this overlap is greater than previous ones
            if overlap > max_overlap:
                max_overlap = overlap
                resSpeaker = speaker
        
        # If no overlap found, use fallback speaker (last/least frequent)
        if resSpeaker is None:
            resSpeaker = fallback_speaker
            logger.warning(f"No speaker overlap for segment at {start:.2f}-{end:.2f}s, "
                          f"using fallback speaker: {resSpeaker}")
                
        segments_with_speakers.append({
            'text': segment.text,
            'words': segment.words,
            'speaker': resSpeaker,
            'start': start,
            'end': end
        })
    return segments_with_speakers

def create_sentences(segments_with_speakers: list):
    """
    Creates a list of sentence objects (sentence, start, end, speaker) 
    from segments with speakers, sorted by start
    """
    # Group segments by speaker before sentence tokenization
    speakers_with_segments = get_speakers_with_segments(segments_with_speakers)

    speakers_with_sentences = defaultdict(list)            
    for speaker, segments in speakers_with_segments.items():
        fullTextList = []
        for i, segment in segments:
            for word in segment['words']:
                fullTextList.append([word.word, word.start, word.end])
        fullTextStr = " ".join([word[0] for word in fullTextList])
        # sentences = sent_tokenize(fullTextStr)
        # seg = pysbd.Segmenter(language="en")
        # sentences = seg.segment(fullTextStr)

        sat = SaT("sat-12l-sm")
        # sat.half().to("cuda")
        sentences = sat.split(fullTextStr) # can even try lora if i find a video that needs it
        
        word_idx = 0
        for sentence in sentences:
            num_words = len(sentence.split())
            sentence_words = fullTextList[word_idx:word_idx+num_words]
            speakers_with_sentences[speaker].append({
                'speaker': speaker, # this might not be necessary? since we already have speaker key.
                'sentence': sentence,
                'start': sentence_words[0][1], # first word's start
                'end': sentence_words[-1][2] # last word's end
            })
            word_idx += num_words

    all_sentences = []
    for speaker, sentences in speakers_with_sentences.items():
        all_sentences.extend(sentences)  # Add all sentences from this speaker

    # Sort by start time
    sorted_sentences = sorted(all_sentences, key=lambda x: x['start'])
    return sorted_sentences

LANG_CODE_TO_NAME = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ru": "Russian",
    "zh": "Chinese"
}

def translate(sentence, before_context, after_context, src: str, targ: str, groq_api: str = None, groq_model: str = "openai/gpt-oss-120b", gemini_api: str = None, gemini_model: str = "gemini-3-flash-preview"):
    import time
    prompt = f"""Context: "{before_context} {sentence} {after_context}" Correct transcription errors, if any. ONLY output {targ} translation of '{sentence}'."""
    if groq_api:
        logger.info(f"Translating with Groq API: {groq_api}")
        client = Groq(api_key=groq_api)
        model = groq_model or "openai/gpt-oss-120b"
        last_err = None
        for attempt in range(3):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                translation = completion.choices[0].message.content.strip()
                time.sleep(.3)  # avoid rate limits
                return translation
            except Exception as e:
                last_err = e
                logger.warning(f"Groq attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
        if last_err is not None:
            raise RuntimeError(last_err) from last_err
        raise RuntimeError("Groq translation failed for an unknown reason")
    elif gemini_api:
        model = gemini_model or "gemini-3-flash-preview"
        logger.info(f"Translating with Gemini API: model={model}")
        client = gemini.Client(api_key=gemini_api)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text
    else:
        # Direct init (bypasses pipeline image preprocessing bug with text-only input)
        # https://huggingface.co/google/translategemma-4b-it
        logger.info("Translating with TranslateGemma (direct init)")
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_id = "google/translategemma-4b-it"
        if "gemma" not in _translategemma_cache:
            _device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(_device)
            _translategemma_cache["gemma"] = (model, processor, _device)

        model, processor, _device = _translategemma_cache["gemma"]
        source_lang = LANG_CODE_TO_NAME.get(src, src)
        target_lang = LANG_CODE_TO_NAME.get(targ, targ)

        text = sentence
# 
# f"""You are a professional {source_lang} ({src}) to {target_lang} ({targ}) translator.
# Your goal is to accurately convey the meaning and nuances of the original {source_lang} text while adhering to {target_lang} grammar, vocabulary, and cultural sensitivities.
# Produce ONLY the {target_lang} translation of the <CURRENT> sentence, without any additional explanations or commentary.
# Use the PREVIOUS and NEXT sentences only for context.

# PREVIOUS:
# {before_context}

# CURRENT:
# {sentence}

# NEXT:
# {after_context}"""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": src,
                        "target_lang_code": targ,
                        "text": text,
                    }
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(_device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        generation = generation[0][input_len:]
        return processor.decode(generation, skip_special_tokens=True)

        # logger.info("Translating with Helsinki")
        # model_name = f"Helsinki-NLP/opus-mt-{src}-{targ}"
        # cache_key = (src, targ)
        # if cache_key not in _helsinki_cache:
        #     _helsinki_cache[cache_key] = (
        #         MarianTokenizer.from_pretrained(model_name),
        #         MarianMTModel.from_pretrained(model_name).to('cpu'),
        #     )
        # tokenizer, model = _helsinki_cache[cache_key]

        # def translate_chunk(text):
        #     inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to('cpu')
        #     out_ids = model.generate(**inputs, max_new_tokens=256, )
        #     return tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # words = sentence.split()
        # if len(words) > 200:
        #     parts = [" ".join(words[i:i+200]) for i in range(0, len(words), 200)]
        #     outputs = [translate_chunk(p) for p in parts]
        #     return " ".join(outputs)
        # else:
        #     return translate_chunk(sentence)

def classify_emotion(audio_path: str, classifier: foreign_class):
    try:
        out_prob, score, index, text_lab = classifier.classify_file(audio_path)
        return text_lab[0]
    except Exception as e:
        logger.warning(f"Emotion classification failed: {e}, using 'neutral'")
        return "neutral"

def assign_sentences_to_segments(sorted_sentences: list, segments_with_speakers: list):
    """
    Adds segments property with list of segments to each sentence. 
    'segments' --> (segment_index, proportion_of_sentence_in_segment)

    Assumes sentences are sorted by start time
    """

    speakers_with_segments = get_speakers_with_segments(segments_with_speakers)
    final_sentences = []
    for speaker, speaker_segments in speakers_with_segments.items():
        speaker_sentences = [sentence for sentence in sorted_sentences if sentence['speaker'] == speaker]

        # within speaker indices
        i_seg = 0
        i_seg_word = 0
        i_sent = 0
        # index of segment among all segments (across all speakers)
        i_seg_global, cur_segment = speaker_segments[i_seg]
            
        while i_sent < len(speaker_sentences):
            i_sent_word = 0
            sentence = speaker_sentences[i_sent]
            sentence_words = sentence['sentence'].split()
            speaker_sentences[i_sent]['segments'] = []
            segment_start_word = 0  # for correct proportion later
            
            while i_sent_word < len(sentence_words):
                if i_seg_word < len(cur_segment['words']):
                    i_seg_word += 1
                    i_sent_word += 1
                else:
                    words_in_segment = i_sent_word - segment_start_word # number of words of curSentence in curSegment
                    prop = words_in_segment / len(sentence_words)
                    speaker_sentences[i_sent]['segments'].append((i_seg_global, prop))
                    
                    i_seg += 1
                    i_seg_word = 0
                    i_seg_global, cur_segment = speaker_segments[i_seg]
                    segment_start_word = i_sent_word 

            # if didn't finish a segment, then sentence is part of that segment
            words_in_segment = i_sent_word - segment_start_word
            if words_in_segment > 0:
                prop = words_in_segment / len(sentence_words)
                speaker_sentences[i_sent]['segments'].append((i_seg_global, prop))
            
            if len(speaker_sentences[i_sent]['segments']) == 0:
                speaker_sentences[i_sent]['segments'].append((i_seg_global, 1))
            i_sent += 1
        
        final_sentences.extend(speaker_sentences)
    return sorted(final_sentences, key=lambda x: x['start'])


def map_translated_sentences_to_segments(sentences: list, segments: list):
    """
    Maps translated sentences to segments based on word proportion.
    Returns: segments with translated text
    """
    for i, segment in enumerate(segments):
        segment['translation'] = []

    for i, sentence_obj in enumerate(sentences):
        translation_text = sentence_obj.get('translation') or ""
        
        # For languages without spaces (Chinese, Japanese), split by characters
        # For languages with spaces (English, Spanish), split by words
        if translation_text and ' ' in translation_text and len(translation_text.split()) > 1:
            # Has spaces - split by words
            words = translation_text.split()
            join_with = " "
        else:
            # No spaces or single word - split by characters
            words = list(translation_text)
            join_with = ""
        
        total_words = len(words)
        word_idx = 0

        for j, (i_seg_global, prop) in enumerate(sentence_obj['segments']):
            if j == len(sentence_obj['segments'])-1:
                num_words = total_words - word_idx # last segment gets remaining words of sentence
            else:
                num_words = round(total_words*prop)
            segment_words = words[word_idx:word_idx + num_words]
            segments[i_seg_global]['translation'].append(join_with.join(segment_words))
            word_idx += num_words
        
        # will delete in cleanup
        if not sentence_obj['segments']:
            logger.warning(f"Sentence {i} has no segments: {sentence_obj['sentence']}")

    for segment in segments:
        segment['translation'] = " ".join(segment['translation'])
    return segments


def adjust_audio(segments, MIN_SPEED, MAX_SPEED, orig_audio_len):
    """
    Adjusts translated segment audio to fit in roughly the same time slot of original segment
    Returns: adjAudio
    """
    curDuration = 0
    for i, segment in enumerate(segments):
            print(f"Audio adjusting segment {i}")            
            audio = AudioSegment.from_wav(f"temp/audio_chunks/{i}.wav")
            prev_silence, next_silence, usable_prev_silence, usable_next_silence = calculate_silences(segment, i, segments, orig_audio_len)

            translated_dur = len(audio)
            orig_dur = (segment['end'] - segment['start']) * 1000

            # Handle zero duration case
            if orig_dur <= 0:
                print(f"Warning: Sentence {i} has zero/negative duration ({orig_dur}ms), skipping adjustment")
                audio.export(f"temp/adjAudio_chunks/{i}.wav", format="wav")
                continue
            
            # Calculate accumulated drift (how far behind schedule we are)
            drift_ms = max(0, curDuration - (segment['start'] * 1000))
            logger.info(f"curDuration: {curDuration}")
            logger.info(f"segment start: {segment['start']}")
            logger.info(f"drift_ms: {drift_ms}")
            target_dur = max(orig_dur + usable_prev_silence + usable_next_silence - drift_ms, .01)
            
            if (translated_dur == orig_dur and drift_ms == 0):
                # Exact match and on schedule - just copy
                adjAudio = AudioSegment.silent(duration=usable_prev_silence) + audio + AudioSegment.silent(duration=usable_next_silence)
                speed_factor = 1.0
        
            elif (orig_dur < translated_dur < target_dur):
                speed_factor = 1.0
                adjAudio = audio
                total_audio = usable_prev_silence + translated_dur
                remaining = target_dur - total_audio
                adjAudio = AudioSegment.silent(duration=usable_prev_silence) + audio + AudioSegment.silent(duration=max(0, remaining))

            # elif (orig_dur < translated_dur < target_dur):
            #     logger.info("orig_dur < translated_dur < target_dur")
            #     range_size = target_dur - orig_dur
            #     distance_from_orig = translated_dur - orig_dur
            #     ratio = distance_from_orig / range_size
            #     speed_factor = 1.0 + (ratio * (MAX_SPEED - 1.0))
            #     adjAudio = audio.speedup(playback_speed=speed_factor)
                
            #     current_total = usable_prev_silence + len(adjAudio)
            #     usableNextSilence_needed = target_dur - current_total
            #     if usableNextSilence_needed >= 0:
            #         adjAudio = AudioSegment.silent(duration=usable_prev_silence) + adjAudio + AudioSegment.silent(duration=usableNextSilence_needed)
            #     else:
            #         leftover_prevSilence = usable_prev_silence + usableNextSilence_needed
            #         adjAudio = AudioSegment.silent(duration=leftover_prevSilence) + adjAudio

            elif translated_dur >= target_dur:
                logger.info("translated_dur >= target_dur")
                # Need to speed up
                speed_factor = translated_dur / target_dur
                if speed_factor > MAX_SPEED:
                    speed_factor = MAX_SPEED
                adjAudio = audio.speedup(playback_speed=speed_factor)
                
            # elif translated_dur < orig_dur:
            #     logger.info("translated_dur < orig_dur")
            #     # Need to slow down 
            #     speed_factor = translated_dur / orig_dur
            #     if speed_factor < MIN_SPEED: #translated_dur signif shorter than target_dur
            #         speed_factor = MIN_SPEED
                
            #     # 1. Load the audio file
            #     # y: audio time series, sr: sampling rate
            #     y, sr = librosa.load(f"temp/audio_chunks/{i}.wav", sr=None) # sr=None preserves original sample rate

            #     # 2. Define the stretch factor for slowing down
            #     # rate < 1.0 slows down; rate > 1.0 speeds up
            #     slow_rate = speed_factor # Slows down by 25%
            #     # slow_rate = 0.5 # Halves the speed

            #     # 3. Apply time stretching
            #     y_slow = librosa.effects.time_stretch(y, rate=slow_rate)

            #     # 4. Save the slowed-down audio
            #     sf.write(f"temp/audio_chunks/{i}_slowed.wav", y_slow, sr) # {Link: soundfile.write https://pypi.org/project/soundfile/}
            #     adjAudio = AudioSegment.from_wav(f"temp/audio_chunks/{i}_slowed.wav")

            #     if os.path.exists(f"temp/audio_chunks/{i}_slowed.wav"):
            #         logger.info(f"Removing temp/audio_chunks/{i}_slowed.wav")
            #         os.remove(f"temp/audio_chunks/{i}_slowed.wav")


            #     # If still shorter after slowing, pad with silence
            #     slowed_dur = len(adjAudio)
            #     if slowed_dur < target_dur:
            #         silence_budget = target_dur - slowed_dur
            #         if silence_budget >= usable_prev_silence:
            #             # Room for prev silence and some after
            #             next_silence = silence_budget - usable_prev_silence
            #             adjAudio = AudioSegment.silent(duration=usable_prev_silence) + adjAudio + AudioSegment.silent(duration=next_silence)
            #         else:
            #             # Only room for partial prev silence
            #             adjAudio = AudioSegment.silent(duration=silence_budget) + adjAudio
            elif translated_dur < orig_dur:
                logger.info("translated_dur < orig_dur")

                # Check if slowdown would be too extreme
                natural_speed_factor = translated_dur / orig_dur
                
                if natural_speed_factor < 0.9:
                    # Add initial padding to avoid extreme slowdown
                    logger.info(f"Speed factor {natural_speed_factor:.2f}x < 0.9, adding initial padding")
                    gap = orig_dur - translated_dur
                    initial_padding = min(300, gap)

                    adjAudio = AudioSegment.silent(duration=initial_padding) + audio
                    
                    # If still shorter than orig_dur, slow down the audio portion
                    if len(adjAudio) < orig_dur:
                        speed_factor = translated_dur / (orig_dur - initial_padding)
                        if speed_factor < MIN_SPEED:
                            speed_factor = MIN_SPEED
                        
                        adjAudio_slowed = stretch_audio(f"temp/audio_chunks/{i}.wav", speed_factor)
                        
                        adjAudio = AudioSegment.silent(duration=initial_padding) + adjAudio_slowed
                        
                    else:
                        speed_factor = 1.0  # Padding was more than enough
                
                else:
                    logger.info(f"Speed factor {natural_speed_factor:.2f}x is acceptable, slowing normally")
                    speed_factor = natural_speed_factor
                    if speed_factor < MIN_SPEED:
                        speed_factor = MIN_SPEED
                    
                    adjAudio = stretch_audio(f"temp/audio_chunks/{i}.wav", speed_factor)
                
                # Pad to target_dur if needed
                if len(adjAudio) < target_dur:
                    remaining = target_dur - len(adjAudio)
                    adjAudio = adjAudio + AudioSegment.silent(duration=remaining)
            # TODO:
                # if next segment is longer than orig_dur, dont slow down as much. This is more complicated
                # since the usable_prev_silence would be dynamic.
            logger.info(f"SEGMENT {i} AUDIO ADJUSTMENT DETAILS:")
            logger.info(f"speed factor: {speed_factor}")
            logger.info(f"target dur: {target_dur}")
            logger.info(f"adjusted audio dur: {len(adjAudio)}")
            logger.info(f"usable_prev_silence: {usable_prev_silence}")
            logger.info(f"usable_next_silence: {usable_next_silence}")
            logger.info(f"next_silence: {next_silence}")
            logger.info(f"prev_silence: {prev_silence}")
            logger.info(f"orig duration: {orig_dur}")
            logger.info(f"translated duration: {translated_dur}")
            logger.info("-"*80)

            if i == 0:
                adjAudio = AudioSegment.silent(duration = prev_silence-usable_prev_silence) + adjAudio
            elif i == len(segments) - 1:
                adjAudio = adjAudio + AudioSegment.silent(duration = next_silence-usable_next_silence)
            adjAudio.export(f"temp/adjAudio_chunks/{i}.wav", format="wav")
            curDuration += len(adjAudio)


def stitch_chunks(segments):
    logger.info("Stitching chunks together")
    final_audio = AudioSegment.empty()
    for i in range(len(segments)):
        adjAudio = AudioSegment.from_wav(f"temp/adjAudio_chunks/{i}.wav")
        final_audio += adjAudio
        logger.info(f"Added chunk {i} ({len(adjAudio)}ms), total so far: {len(final_audio)}ms")

    final_audio.export(f"temp/final_audio.wav", format="wav")
    logger.info(f"Final audio length: {len(final_audio)}ms ({len(final_audio)/1000:.2f}s)")

def overlay_audios(audio1, audio2):
    if len(audio1) > len(audio2):
        audio2 = audio2 + AudioSegment.silent(duration=len(audio1) - len(audio2))
    elif len(audio1) < len(audio2):
        audio1 = audio1 + AudioSegment.silent(duration=len(audio2) - len(audio1))
    combined_audio = audio1.overlay(audio2)
    combined_audio.export("temp/combined_audio.wav", format="wav")
    return "temp/combined_audio.wav"

def get_video_resolution(video_path: str):
    """Use ffprobe to get video width and height."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0:s=x',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"ffprobe failed: {result.stderr}, defaulting to 1920x1080")
        return 1920, 1080
    parts = result.stdout.strip().split('x')
    return int(parts[0]), int(parts[1])


def _add_pinyin(text):
    """Add pinyin above Chinese text. Returns 'pinyin\\Ncharacters' format for ASS."""
    try:
        from pypinyin import lazy_pinyin, Style
        pinyin = ' '.join(lazy_pinyin(text, style=Style.TONE))
        return pinyin
    except ImportError:
        logger.warning("pypinyin not installed, skipping pinyin generation")
        return None


def create_subtitle_chunks_from_segments(final_segments, target_lang=None):
    """Build subtitle chunks directly from final segments.

    One chunk per segment, using each segment's start/end and translation.
    Aligns subtitles with when each dubbed segment actually plays.
    """
    MIN_DURATION = 0.3

    raw_chunks = []
    for seg in final_segments:
        original = (seg.get('text') or '').strip()
        translation = (seg.get('translation') or '').strip()
        start = seg.get('start', 0)
        end = seg.get('end', 0)

        if not translation and not original:
            continue
        if end - start < 0.2:
            continue

        if end - start < MIN_DURATION:
            end = start + MIN_DURATION

        raw_chunks.append({
            'start': start,
            'end': end,
            'original': original,
            'translation': translation,
        })

    chunks = raw_chunks

    if target_lang and target_lang.lower() in ('zh', 'zh-cn', 'zh-tw', 'chinese'):
        for chunk in chunks:
            if chunk['translation']:
                pinyin = _add_pinyin(chunk['translation'])
                if pinyin:
                    chunk['pinyin'] = pinyin

    return chunks


def create_subtitle_chunks(sentences, segments=None, target_lang=None):
    """Convert sentences into subtitle display events.

    One chunk per sentence. Uses segment boundaries for timing (avoids dependency
    on word-level timestamps). If segments is provided, derives start/end from
    the first/last segment each sentence spans; otherwise falls back to sent['start']/['end'].
    Sentences have 'sentence' (original), 'translation', and 'segments' (list of (seg_idx, prop)).
    """
    MIN_DURATION = 1.0

    raw_chunks = []
    for sent in sentences:
        original = (sent.get('sentence') or '').strip()
        translation = (sent.get('translation') or '').strip()
        seg_refs = sent.get('segments') or []
        if segments and seg_refs:
            first_idx = seg_refs[0][0]
            last_idx = seg_refs[-1][0]
            start = segments[first_idx]['start']
            end = segments[last_idx]['end']
        else:
            start = sent.get('start', 0)
            end = sent.get('end', 0)

        if not original or end - start < 0.2:
            continue

        # Ensure minimum display time
        if end - start < MIN_DURATION:
            end = start + MIN_DURATION

        raw_chunks.append({
            'start': start,
            'end': end,
            'original': original,
            'translation': translation,
        })

    # Sort and resolve overlaps: trim previous chunk to end when next starts
    GAP = 0.05
    raw_chunks.sort(key=lambda c: c['start'])
    for i in range(len(raw_chunks) - 1):
        if raw_chunks[i]['end'] > raw_chunks[i + 1]['start'] - GAP:
            raw_chunks[i]['end'] = raw_chunks[i + 1]['start'] - GAP

    # Filter out chunks that became too short after overlap trimming
    chunks = [c for c in raw_chunks if c['end'] - c['start'] >= 0.3]

    # Add pinyin for Chinese target languages
    if target_lang and target_lang.lower() in ('zh', 'zh-cn', 'zh-tw', 'chinese'):
        for chunk in chunks:
            if chunk['translation']:
                pinyin = _add_pinyin(chunk['translation'])
                if pinyin:
                    chunk['pinyin'] = pinyin

    return chunks


def _ass_escape(text):
    """Escape special ASS characters in text."""
    return text.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')


def _format_ass_time(seconds):
    """Format seconds as ASS timestamp H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def generate_subtitles(chunks, video_width, video_height, output_path="temp/subtitles.ass"):
    """Write an ASS subtitle file with all lines in a single block per chunk.

    All text (translation, pinyin, original) is combined into one Dialogue event
    using \\N line breaks and inline style overrides, so they share one background box.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Scale font sizes to video resolution
    trans_font_size = max(20, int(video_height * 0.05))
    orig_font_size = max(14, int(video_height * 0.03))
    pinyin_font_size = max(12, int(video_height * 0.022))
    margin_v = max(15, int(video_height * 0.025))

    # Single style — the base style handles the background box.
    # We use inline overrides {\fsN} to change font size per line.
    header = f"""[Script Info]
Title: Dual Subtitles
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,Noto Sans,{trans_font_size},&H00FFFFFF,&H000000FF,&H00000000,&HA0000000,-1,0,0,0,100,100,0,0,4,1,0,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # One block per chunk: translation (top), pinyin, original (bottom)
    pinyin_prefix = f"\\N{{\\fs{pinyin_font_size}\\b0}}"
    orig_prefix = f"\\N{{\\fs{orig_font_size}\\b0}}"

    lines = []
    for chunk in chunks:
        start = _format_ass_time(chunk['start'])
        end = _format_ass_time(chunk['end'])
        parts = []
        if chunk.get('translation'):
            parts.append(_ass_escape(chunk['translation']))
        if chunk.get('pinyin'):
            parts.append(f"{pinyin_prefix}{_ass_escape(chunk['pinyin'])}")
        if chunk.get('original'):
            parts.append(f"{orig_prefix}{_ass_escape(chunk['original'])}")
        text = "".join(parts)
        lines.append(f"Dialogue: 0,{start},{end},Sub,,0,0,{margin_v},,{text}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write('\n'.join(lines))
        f.write('\n')

    logger.info(f"Generated {len(chunks)} subtitle chunks to {output_path}")
    return output_path


def combine_audio_with_video(audio_path: str, video_path: str, subtitle_path: str = None):
    # Combine new audio with original video, optionally burning in subtitles
    if subtitle_path:
        # Re-encode video to burn in ASS subtitles
        command = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-vf', f"ass={subtitle_path}",
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '192k',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            '-y',
            'temp/output_video.mp4'
        ]
    else:
        command = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            '-y',
            'temp/output_video.mp4'
        ]
    logger.info("Combining audio with video using ffmpeg...")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        raise Exception(f"Failed to combine audio with video: {result.stderr}")
    logger.info("Video with dubbed audio saved to temp/output_video.mp4")
    return "temp/output_video.mp4"


############################################################
#                      Helper Functions                    #
############################################################
def get_speakers_with_segments(segments_with_speakers: list):
    speakers_with_segments = defaultdict(list)
    for i, segment in enumerate(segments_with_speakers):
        speakers_with_segments[segment['speaker']].append((i, segment))
    return speakers_with_segments

def calculate_silences(sentence_obj, idx, sentences, orig_audio_len):
    """
    Calculate available silence before/after a sentence.
    Returns: prev_silence, next_silence, usable_prev_silence, usable_next_silence
    """
    if idx == 0:
        prev_silence = sentence_obj['start'] * 1000
        next_silence = (sentences[idx+1]['start'] * 1000) - (sentence_obj['end'] * 1000)
    elif idx == len(sentences) - 1:
        prev_silence = (sentence_obj['start'] * 1000) - (sentences[idx-1]['end'] * 1000)
        next_silence = orig_audio_len - sentence_obj['end'] * 1000
    else:
        prev_silence = (sentence_obj['start'] * 1000) - (sentences[idx-1]['end'] * 1000)
        next_silence = (sentences[idx+1]['start'] * 1000) - (sentence_obj['end'] * 1000)

    usable_prev_silence = min(300, max(prev_silence, 0)) # don't start > 300ms before orig start
    usable_next_silence = max(next_silence - 300, 0) # allocate 300ms for audio after

    return prev_silence, next_silence, usable_prev_silence, usable_next_silence

    # putting old code here in case things go wrong
    # if i == 0:
    #             prev_silence = sentence_obj['start']*1000
    #             next_silence = (sorted_sentences[i+1]['start']*1000) - (sentence_obj['end']*1000)
    #         elif i == len(sorted_sentences) - 1:
    #             prev_silence = (sentence_obj['start']*1000) - (sorted_sentences[i-1]['end']*1000)
    #             next_silence = len(orig_audio) - sentence_obj['end']*1000
    #         else:
    #             prev_silence = (sentence_obj['start']*1000) - (sorted_sentences[i-1]['end']*1000)
    #             next_silence = (sorted_sentences[i+1]['start']*1000) - (sentence_obj['end']*1000)

    #         usable_prev_silence = min(300, max(prev_silence, 0)) # don't start > 300ms before orig start
    #         usable_next_silence = max(next_silence - 300, 0) # allocate 300ms for audio after


def stretch_audio(audio_path: str, speed_factor: float):
    y, sr = sf.read(audio_path)
    y_stretch = pyrb.time_stretch(y, sr, speed_factor)
    
    # Convert to 16-bit PCM
    y_stretch_int16 = (y_stretch * 32767).astype(np.int16)
    
    # Create AudioSegment from numpy array
    audio_segment = AudioSegment(
        y_stretch_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,  # 2 bytes for 16-bit audio
        channels=1 if len(y_stretch.shape) == 1 else y_stretch.shape[1]
    )
    
    return audio_segment


MAX_TTS_CHARS = 250  # stay well under XTTS's 400-token limit


def split_text(text: str, max_chars: int = MAX_TTS_CHARS) -> list[str]:
    """Split text into chunks that fit within the TTS token limit,
    preferring sentence boundaries, then clause boundaries, then word boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        split_idx = -1
        for sep in ['. ', '! ', '? ', '。', '！', '？']:
            idx = remaining.rfind(sep, 0, max_chars)
            if idx != -1 and idx > split_idx:
                split_idx = idx + len(sep)
        if split_idx == -1:
            for sep in [', ', '; ', '، ']:
                idx = remaining.rfind(sep, 0, max_chars)
                if idx != -1:
                    split_idx = idx + len(sep)
                    break
        if split_idx == -1:
            split_idx = remaining.rfind(' ', 0, max_chars)
            if split_idx != -1:
                split_idx += 1
        if split_idx <= 0:
            split_idx = max_chars
        chunks.append(remaining[:split_idx].strip())
        remaining = remaining[split_idx:].strip()
    return [c for c in chunks if c]


def _tts_to_file_safe(tts, text: str, file_path: str, speaker_wav: str, language: str, emotion: str, speed: float = 1.0):
    """Call tts.tts_to_file, falling back to silence on ZeroDivisionError (TTS bug when audio is 0-duration)."""
    try:
        tts.tts_to_file(text=text, file_path=file_path, speaker_wav=speaker_wav, language=language, emotion=emotion, speed=speed)
    except ZeroDivisionError:
        logger.warning(f"TTS produced 0-duration audio for text={repr(text[:50])}..., writing 100ms silence")
        AudioSegment.silent(duration=100).export(file_path, format="wav")


def tts_segment(tts, text: str, seg_index: int,
                       speaker_wav: str, language: str, emotion: str):
    """Synthesize a segment, splitting long text into chunks and concatenating."""
    text = (text or "").strip()
    if not text:
        AudioSegment.silent(duration=100).export(f"temp/audio_chunks/{seg_index}.wav", format="wav")
        return

    chunks = [c for c in split_text(text) if c.strip()]
    if not chunks:
        AudioSegment.silent(duration=100).export(f"temp/audio_chunks/{seg_index}.wav", format="wav")
        return

    if len(chunks) == 1:
        _tts_to_file_safe(tts, text, f"temp/audio_chunks/{seg_index}.wav", speaker_wav, language, emotion, 1.0)
        return

    logger.info(f"Segment {seg_index}: splitting into {len(chunks)} chunks")
    combined = AudioSegment.empty()
    for ci, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        chunk_path = f"temp/audio_chunks/{seg_index}_part{ci}.wav"
        _tts_to_file_safe(tts, chunk, chunk_path, speaker_wav, language, emotion, 1.0)
        combined += AudioSegment.from_file(chunk_path)
        os.remove(chunk_path)
    combined.export(f"temp/audio_chunks/{seg_index}.wav", format="wav")