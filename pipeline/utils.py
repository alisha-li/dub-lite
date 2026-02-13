# download_video_and_extract_audio
import os
import yt_dlp
from pydub import AudioSegment

# diarize_audio
from pyannoteai.sdk import Client
from pyannote.audio import Pipeline as PyannotePipeline

# denoise_audio
from df import config
from df.enhance import enhance, init_df, load_audio, save_audio
import torch
import sys

# create_sentences
from collections import defaultdict
from nltk.tokenize import sent_tokenize
import logging
logger = logging.getLogger(__name__)

# translate
from groq import Groq
from google import genai as gemini
from transformers import MarianMTModel, MarianTokenizer

# classify_emotion
from speechbrain.inference.interfaces import foreign_class

# adjust_audio
import librosa
import soundfile as sf
import pyrubberband as pyrb
import numpy as np

# combine_audio_with_background
import subprocess

_helsinki_cache = {}

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
    else:
        print(f"Downloading video from URL: {source}")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        ydl_opts = {
            'format': 'best',
            'outtmpl': video_path,
            # 'cookiesfrombrowser': ('chrome',) # will still work for most videos, but not age-restricted, private, or member-only
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([source])
    
    # Extract audio
    print(f"Extracting audio to: {audio_path}")
    orig_audio = AudioSegment.from_file(video_path, format="mp4")
    orig_audio.export(audio_path, format="wav")
    
    return video_path, audio_path, orig_audio


def diarize_audio(audio_path: str, pyannote_key: str, hf_token: str):
    if pyannote_key: # paid
        client = Client(pyannote_key)
        orig_audio_url = client.upload(audio_path)
        diarization_job = client.diarize(orig_audio_url, transcription=True)
        diarization = client.retrieve(diarization_job)

        turns = diarization['output']['turnLevelTranscription']
        speaker_turns = {}

        for turn in turns:
            start = turn['start']
            end = turn['end']
            speaker = turn['speaker']
            
            if abs(end - start) > 0.2:
                speaker_turns[(start, end)] = speaker
                logger.info(f"Speaker {speaker}: from {start}s to {end}s")

    else:  #free
        logger.info("Running speaker diarization (this may take several minutes)...")
        diarizationPipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=hf_token)
        output = diarizationPipeline(audio_path)
        speaker_turns = {}

        # now test this
        for turn, speaker in output.speaker_diarization:
            if abs(turn.end - turn.start) > .2:
                logger.info(f"Speaker {speaker}: from {turn.start}s to {turn.end}s")
                speaker_turns[(turn.start, turn.end)] = speaker
    
        # for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
        #     if abs(speech_turn.end - speech_turn.start) > .2:
        #         logger.info(f"Speaker {speaker}: from {speech_turn.start}s to {speech_turn.end}s")
        #         speaker_turns[(speech_turn.start, speech_turn.end)] = speaker
        
        logger.info("Diarization completed")
    
    return speaker_turns


def get_denoiser():
     # Initialize model
    print("Loading DeepFilterNet2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use model name or path from environment variable, default to DeepFilterNet2
    model_name_or_path = os.environ.get("DEEPFILTERNET_MODEL", "DeepFilterNet2")
    
    try:
        # init_df can take a model name (like "DeepFilterNet2") or a path
        model, df, _ = init_df(model_name_or_path, config_allow_defaults=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"\nTried to load: {model_name_or_path}")
        print("You can set DEEPFILTERNET_MODEL environment variable to:")
        print("  - A model name: 'DeepFilterNet', 'DeepFilterNet2', or 'DeepFilterNet3'")
        print("  - A path to a model directory")
        raise e
    model = model.to(device=device).eval()

    return model, df, device


def denoise_audio(input_path: str, output_path: str = None, model=None, df=None, device=None):
    """Denoise an audio file using DeepFilterNet2"""
    if model is None or df is None or device is None:
        model, df, device = get_denoiser()

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
    
    # Denoise
    print("Denoising audio...")
    sample = sample.to(device)
    enhanced = enhance(model, df, sample)
    
    # Apply fade-in to avoid clicks
    lim = torch.linspace(0.0, 1.0, int(sr * 0.15), device=device).unsqueeze(0)
    if lim.shape[1] < enhanced.shape[1]:
        lim = torch.cat((lim, torch.ones(1, enhanced.shape[1] - lim.shape[1], device=device)), dim=1)
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


def split_speakers_and_denoise(audio: AudioSegment, speaker_turns: dict, output_dir: str = "temp/speakers_audio"):
    # For voice cloning later
    speakers = set(speaker_turns.values())
    model, df, device = get_denoiser()
    for speaker in speakers:
        speaker_audio = AudioSegment.empty()
        for key, value in speaker_turns.items():
            if speaker == value:
                start = int(key[0])*1000  # Convert seconds to milliseconds
                end = int(key[1])*1000
                speaker_audio += audio[start:end]  # Extract this speaker's audio segments
        speaker_audio.export(f"{output_dir}/{speaker}.wav", format="wav")
        if denoise_audio(f"{output_dir}/{speaker}.wav", f"{output_dir}/{speaker}.wav", model, df, device):
            logger.info("Using denoised audio for voice cloning")
        else:
            logger.info("Warning: Denoising failed, using original audio")
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
        sentences = sent_tokenize(fullTextStr)
        
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
                logger.warning(f"Groq attempt {attempt + 1}/3 failed: {_sanitize_api_error(e)}")
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
        if last_err is not None:
            raise RuntimeError(_sanitize_api_error(last_err)) from last_err
        raise RuntimeError("Groq translation failed for an unknown reason")
    elif gemini_api:
        logger.info(f"Translating with Gemini API: {gemini_api}")
        client = gemini.Client(api_key=gemini_api)
        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
        )
        return response.text
    else:
        logger.info("Translating with Helsinki")
        model_name = f"Helsinki-NLP/opus-mt-{src}-{targ}"
        cache_key = (src, targ)
        if cache_key not in _helsinki_cache:
            _helsinki_cache[cache_key] = (
                MarianTokenizer.from_pretrained(model_name),
                MarianMTModel.from_pretrained(model_name).to('cpu'),
            )
        tokenizer, model = _helsinki_cache[cache_key]

        def translate_chunk(text):
            inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to('cpu')
            out_ids = model.generate(**inputs, max_new_tokens=256)
            return tokenizer.decode(out_ids[0], skip_special_tokens=True)

        words = sentence.split()
        if len(words) > 200:
            parts = [" ".join(words[i:i+200]) for i in range(0, len(words), 200)]
            outputs = [translate_chunk(p) for p in parts]
            return " ".join(outputs)
        else:
            return translate_chunk(sentence)

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
        translation_text = sentence_obj['translation']
        
        # For languages without spaces (Chinese, Japanese), split by characters
        # For languages with spaces (English, Spanish), split by words
        if ' ' in translation_text and len(translation_text.split()) > 1:
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

def combine_audio_with_video(audio_path:str, video_path:str):
    # Combine new audio with original video
    command = [
        'ffmpeg',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',  # Copy video stream without re-encoding (fast)
        '-map', '0:v:0',  # Use video from first input
        '-map', '1:a:0',  # Use audio from second input
        '-shortest',  # End when shortest stream ends
        '-y',  # Overwrite output file if it exists
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