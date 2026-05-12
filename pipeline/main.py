# 1. YT-DLP downloads video
# 2. Speaker Diarization and Transcription
# 3. translate
# 4. text to speech
# 5. separate out background sounds from orig audio
# 6. 1 overlay with dubbed audio
# 7.  ffmpeg combine audio with video

# test video: https://www.youtube.com/watch?v=jIZkKsf6VYo
# easier test video: https://www.youtube.com/watch?v=YgxyLrnxCH4
import os
from dotenv import load_dotenv
load_dotenv()

from pydub import AudioSegment
import pickle
import utils
# Heavy ML libs (TTS, audio_separator, speechbrain, mistralai) lazy-imported inside
# the methods that use them — keeps cold-container handler entry fast.
from utils import (
    download_video_and_extract_audio,
    # diarize_audio,  # commented out – using Mistral for diarization
    segments_to_speaker_turns,
    get_denoiser,
    split_speakers_and_denoise,
    merge_close_segments,
    assign_speakers_to_segments,
    mistral_segments_to_pipeline,
    create_sentences,
    classify_emotion,
    assign_sentences_to_segments,
    adjust_audio,
    map_translated_sentences_to_segments,
    stitch_chunks,
    overlay_audios,
    combine_audio_with_video,
    tts_segment,
    tts_segment_cosyvoice,
    load_cosyvoice,
    get_video_resolution,
    create_subtitle_chunks,
    create_subtitle_chunks_from_segments,
    generate_subtitles,
    translate_full_transcript,
)
from log import setup_logging
import logging
import torch

setup_logging()
logger = logging.getLogger(__name__)

class YTDubPipeline:
    def __init__(self):
        os.makedirs("temp", exist_ok=True)
        os.makedirs("temp/speakers_audio", exist_ok=True)

    def dub(self,
            src: str,
            targ: str,
            hf_token: str,
            pyannote_key: str = None,
            gemini_api: str = None,
            groq_api: str = None,
            mistral_api: str = None,
            groq_model: str = None,
            gemini_model: str = None,
            speakerTurnsPkl: bool = False,
            segmentsPkl: bool = False,
            finalSentencesPkl: bool = False,
            tts_engine: str = "xtts",
            cosyvoice_model_dir: str = "pretrained_models/Fun-CosyVoice3-0.5B",
            progress_callback=None,
            tts_worker_fn=None,
            num_tts_workers: int = 4,
            translation_mode: str = "per_sentence"):
        """progress_callback(stage: str, percent: int) is called at each pipeline stage."""
        def report(stage: str, percent: int):
            if progress_callback:
                progress_callback(stage, percent)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        compute_type = "float16" if device.type == "cuda" else "int8"

        # Clean up old temp files from previous runs
        report("Starting...", 0)
        cleanup_dirs = [
            "temp/speakers_audio",
            "temp/audio_chunks", 
            "temp/adj_audio_chunks",
            "temp/emotions_audio"
        ]
        for dir_path in cleanup_dirs:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    if file.endswith(".wav"):
                        os.remove(os.path.join(dir_path, file))
        logger.info("Cleaned up old temp audio files")

        # 1. Download video & extract audio
        report("Downloading video", 3)
        logger.info(f"Starting dubbing pipeline for: {src}")
        video_path, orig_audio_path, orig_audio = download_video_and_extract_audio(src)
        report("Extracting audio", 7)

        # Kick off UVR background-music separation in a thread NOW. UVR only
        # reads orig_audio_path (immutable input), so it runs concurrently with
        # transcription / translation / emotion / TTS. Saves ~26s on a 14-min
        # video by hiding the separation pass behind other stages.
        import concurrent.futures as _cf
        _separator_pool = _cf.ThreadPoolExecutor(max_workers=1)

        def _separate_background():
            from audio_separator.separator import Separator
            sep = Separator(model_file_dir="/root/baked_models/audio_separator")
            sep.load_model(model_filename='2_HP-UVR.pth')
            return AudioSegment.from_file(sep.separate(orig_audio_path)[0])

        background_audio_future = _separator_pool.submit(_separate_background)

        # 2. Transcription & diarization
        report("Transcribing & diarizing", 10)
        segments_from_diarization = None
        if speakerTurnsPkl:
            logger.info("Loading pyannote pickle...")
            with open("temp/speaker_turns.pkl", "rb") as f:
                speaker_turns = pickle.load(f)
            logger.info(f"Loaded {len(speaker_turns)} speaker turns from file!")
            # Also load segments so we don't re-transcribe in step 4
            if os.path.exists("temp/segments.pkl"):
                with open("temp/segments.pkl", "rb") as f:
                    data = pickle.load(f)
                    segments_from_diarization = (data["segments"], data["language"])
                logger.info("Also loaded segments from pickle")
        else:
            if segmentsPkl:
                logger.info("Loading segments pickle...")
                with open("temp/segments.pkl", "rb") as f:
                    data = pickle.load(f)
                    segments_from_diarization = (data["segments"], data["language"])
            else:
                if not mistral_api:
                    raise ValueError("mistral_api is required for Mistral transcription")
                logger.info("Running Mistral Transcription (with diarization)...")
                # Compress audio to MP3 for faster upload to Mistral API
                mp3_path = orig_audio_path.replace(".wav", "_mistral.mp3")
                orig_audio.export(mp3_path, format="mp3", bitrate="64k")
                logger.info(f"Compressed audio for Mistral: {os.path.getsize(mp3_path) / 1024 / 1024:.1f} MB")
                from mistralai import Mistral
                client = Mistral(api_key=mistral_api)
                with open(mp3_path, "rb") as f:
                    transcription_response = client.audio.transcriptions.complete(
                        model="voxtral-mini-2602",
                        file={
                            "content": f,
                            "file_name": "audio.mp3",
                        },
                        diarize=True,
                        timestamp_granularities=["segment"],
                    )
                os.remove(mp3_path)
                segs = mistral_segments_to_pipeline(transcription_response.segments)
                src = transcription_response.language or "en"
                segments_from_diarization = (segs, src)
                with open("temp/segments.pkl", "wb") as f:
                    pickle.dump({"segments": segs, "language": src}, f)
                logger.info(f"Transcription completed! Found {len(segs)} segments")
            
            speaker_turns = segments_to_speaker_turns(segments_from_diarization[0])
            with open("temp/speaker_turns.pkl", "wb") as f:
                pickle.dump(speaker_turns, f)
        report("Transcription complete", 18)

        # 3. Extract & denoise speaker audio (for voice cloning)
        report("Extracting speaker audio", 20)
        split_speakers_and_denoise(orig_audio, speaker_turns, "temp/speakers_audio")
        report("Speaker audio ready", 24)

        # 4. Process transcription segments
        report("Processing segments", 26)
        segments, src_lang = segments_from_diarization

        segments_with_speakers = merge_close_segments(segments)
        with open("temp/segments_merged.pkl", "wb") as f:
            pickle.dump(segments_with_speakers, f)
        logger.info(f"Saved {len(segments_with_speakers)} merged segments")

        # 5. Build sentences
        report("Building sentences", 30)
        sentences = create_sentences(segments_with_speakers)
        with open("temp/sentences.pkl", "wb") as f:
            pickle.dump(sentences, f)
        logger.info(f"Saved {len(sentences)} sentences")
        sentences = assign_sentences_to_segments(sentences, segments_with_speakers)

        # 6. Translation + 7a. Emotion classify (run concurrently — share
        # segments_with_speakers but touch different fields; mapping function
        # below only adds translation/orig, leaves emotion intact).
        import concurrent.futures

        os.makedirs("temp/audio_chunks", exist_ok=True)
        os.makedirs("temp/emotions_audio", exist_ok=True)

        def _translation_task():
            if finalSentencesPkl:
                report("Translating", 35)
                logger.info("Loading existing final sentences from file...")
                with open("temp/final_sentences.pkl", "rb") as f:
                    return pickle.load(f)
            if translation_mode == "full_transcript" and groq_api:
                report("Translating (full transcript)", 35)
                logger.info("Using full-transcript translation mode (Groq bulk call)")
                out = translate_full_transcript(
                    sentences, src_lang, targ,
                    groq_api=groq_api, groq_model=groq_model,
                    progress_callback=progress_callback,
                    progress_start=35, progress_end=58,
                )
                with open("temp/final_sentences.pkl", "wb") as f:
                    pickle.dump(out, f)
                return out
            report("Translating", 35)
            if translation_mode == "full_transcript":
                logger.warning("translation_mode=full_transcript requested but no groq_api; falling back to per-sentence")
            n_sentences = len(sentences)
            for i, sentence_obj in enumerate(sentences):
                if n_sentences > 0:
                    report("Translating", 35 + int(20 * (i + 1) / n_sentences))
                sentence = sentence_obj['sentence']

                if i == 0:
                    before_context = ""
                    after_context = sentences[i+1]['sentence'] if len(sentences) > 1 else ""
                elif i == len(sentences) - 1:
                    before_context = sentences[i-1]['sentence']
                    after_context = ""
                else:
                    before_context = sentences[i-1]['sentence']
                    after_context = sentences[i+1]['sentence']

                translation = utils.translate(sentence,
                                              before_context,
                                              after_context,
                                              src_lang,
                                              targ,
                                              groq_api=groq_api,
                                              groq_model=groq_model,
                                              gemini_api=gemini_api,
                                              gemini_model=gemini_model)
                sentence_obj['translation'] = translation
            with open("temp/final_sentences.pkl", "wb") as f:
                pickle.dump(sentences, f)
            return sentences

        def _emotion_task():
            from speechbrain.inference.interfaces import foreign_class
            classifier = foreign_class(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                savedir="/root/baked_models/speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                run_opts={"device": device.type})
            for seg in segments_with_speakers:
                if seg.get('speaker') is None:
                    seg['emotion'] = 'neutral'
                    continue
                start = seg['start'] * 1000
                end = seg['end'] * 1000
                tmp = f"temp/emotions_audio/emotion_{int(start)}_{int(end)}.wav"
                orig_audio[start:end].export(tmp, format="wav")
                seg['emotion'] = classify_emotion(tmp, classifier)
                os.remove(tmp)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            t_future = ex.submit(_translation_task)
            e_future = ex.submit(_emotion_task)
            sentences = t_future.result()
            e_future.result()
        report("Translation complete", 58)

        # Map translated sentences to segments (mutates segments_with_speakers
        # in place — emotion field set by the parallel task persists).
        final_segments = map_translated_sentences_to_segments(sentences, segments_with_speakers)
        with open("temp/final_segments.pkl", "wb") as f:
            pickle.dump(final_segments, f)
        logger.info(f"Saved {len(final_segments)} segments to final_segments.pkl")

        video_width, video_height = get_video_resolution(video_path)

        # Debug: Check segment translations
        logger.info("\n=== FINAL SEGMENTS CHECK ===")
        for i, seg in enumerate(final_segments):
            print(f"Segment {i}:")
            print(f"  Speaker: {seg.get('speaker')}")
            print(f"  Original text: '{seg.get('text', 'MISSING')}'")
            print(f"  Translation: '{seg.get('translation', 'MISSING')}'")
            print(f"  Emotion: '{seg.get('emotion', 'MISSING')}'")
            print(f"  Start: {seg.get('start')}, End: {seg.get('end')}")
        logger.info("="*40 + "\n")

        # 7. Text to Speech
        report("Generating speech", 60)
        n_segments = len(final_segments)

        # 7b. Generate TTS — parallel workers or local
        # (emotion already classified above in parallel with translation)
        if tts_worker_fn is not None:
            # --- Parallel TTS across multiple GPU containers ---
            logger.info(f"Using {num_tts_workers} parallel TTS workers")

            # Read speaker WAV files into memory so we can send them to workers
            speaker_ids = set(seg['speaker'] for seg in final_segments if seg['speaker'] is not None)
            speaker_wavs = {}
            for spk in speaker_ids:
                spk_path = f"temp/speakers_audio/{spk}.wav"
                if os.path.exists(spk_path):
                    with open(spk_path, "rb") as f:
                        speaker_wavs[spk] = f.read()

            # Build segment data for workers
            tts_segments = []
            for i, segment in enumerate(final_segments):
                if segment['speaker'] is None:
                    # Write silence locally for speakerless segments
                    dur_ms = max(1, int((segment['end'] - segment['start']) * 1000))
                    AudioSegment.silent(duration=dur_ms).export(f"temp/audio_chunks/{i}.wav", format="wav")
                    continue
                tts_segments.append({
                    "index": i,
                    "text": segment.get('translation', ''),
                    "emotion": segment.get('emotion', 'neutral'),
                    "speaker": segment['speaker'],
                    "start": segment['start'],
                    "end": segment['end'],
                })

            # Dynamically choose worker count: at least 5 segments per worker
            MIN_SEGMENTS_PER_WORKER = 5
            actual_workers = max(1, min(num_tts_workers, len(tts_segments) // MIN_SEGMENTS_PER_WORKER))
            logger.info(f"{len(tts_segments)} segments → {actual_workers} TTS workers")

            # Split segments into batches, one per worker
            batches = [[] for _ in range(actual_workers)]
            for idx, seg_data in enumerate(tts_segments):
                batches[idx % actual_workers].append(seg_data)
            # Remove empty batches
            batches = [b for b in batches if b]

            batch_inputs = []
            for batch in batches:
                # Only include speaker WAVs needed by this batch
                batch_speakers = set(s["speaker"] for s in batch)
                batch_inputs.append({
                    "segments": batch,
                    "speaker_wavs": {spk: speaker_wavs[spk] for spk in batch_speakers if spk in speaker_wavs},
                    "targ": targ,
                    "tts_engine": tts_engine,
                    "cosyvoice_model_dir": cosyvoice_model_dir,
                })

            # Fan out to parallel containers and collect results
            report("Generating speech (parallel)", 62)
            for result_idx, result in enumerate(tts_worker_fn.map(batch_inputs)):
                report("Generating speech (parallel)", 62 + int(16 * (result_idx + 1) / len(batch_inputs)))
                for seg_idx_str, wav_bytes in result["wavs"].items():
                    with open(f"temp/audio_chunks/{seg_idx_str}.wav", "wb") as f:
                        f.write(wav_bytes)
            logger.info("Parallel TTS complete")

        else:
            # --- Local single-GPU TTS (original path) ---
            use_cosyvoice = tts_engine == "cosyvoice"
            if use_cosyvoice:
                logger.info("Using CosyVoice TTS engine")
                cosyvoice_model = load_cosyvoice(cosyvoice_model_dir)
            else:
                logger.info("Using XTTS TTS engine")
                from TTS.api import TTS
                tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())

            for i, segment in enumerate(final_segments):
                if n_segments > 0:
                    report("Generating speech", 60 + int(18 * (i + 1) / n_segments))
                if segment['speaker'] is None:
                    logger.warning(f"Segment {i} has no speaker, skipping")
                    continue

                logger.info(f"TTS-ing segment {i}")

                if not segment['translation'] or segment['translation'].strip() == "":
                    logger.warning(f"Segment {i} has empty translation, generating silence")
                    dur_ms = max(1, int((segment['end'] - segment['start']) * 1000))
                    AudioSegment.silent(duration=dur_ms).export(f"temp/audio_chunks/{i}.wav", format="wav")
                    continue

                if use_cosyvoice:
                    tts_segment_cosyvoice(cosyvoice_model, segment['translation'], i,
                                           f"temp/speakers_audio/{segment['speaker']}.wav",
                                           segment['emotion'])
                else:
                    tts_segment(tts, segment['translation'], i,
                                f"temp/speakers_audio/{segment['speaker']}.wav",
                                targ, segment['emotion'])

        # 8. Adjust audio timing
        report("Adjusting audio timing", 80)
        os.makedirs("temp/adj_audio_chunks", exist_ok=True)
        adjust_audio(final_segments, MIN_SPEED=0.85, MAX_SPEED=2, orig_audio_len=len(orig_audio))

        # 9. Generate subtitles (using adjusted audio durations)
        report("Generating subtitles", 84)
        cursor = 0.0
        for i, seg in enumerate(final_segments):
            adj_chunk = AudioSegment.from_wav(f"temp/adj_audio_chunks/{i}.wav")
            adj_dur = len(adj_chunk) / 1000.0
            # Use raw TTS duration to determine when speech actually ends
            raw_chunk = AudioSegment.from_wav(f"temp/audio_chunks/{i}.wav")
            speech_dur = len(raw_chunk) / 1000.0
            seg['adj_start'] = cursor
            # Subtitle ends 1s after speech ends, but not beyond the chunk
            seg['adj_end'] = cursor + min(speech_dur + 1.0, adj_dur)
            cursor += adj_dur
        subtitle_chunks = create_subtitle_chunks_from_segments(final_segments, target_lang=targ)
        subtitle_path = generate_subtitles(subtitle_chunks, video_width, video_height)
        logger.info(f"Generated {len(subtitle_chunks)} subtitle chunks at {subtitle_path}")

        # 10. Stitch audio chunks
        report("Stitching audio", 87)
        stitch_chunks(final_segments)

        # 11. Wait for the background-separation thread kicked off near the
        # start. By the time we reach here it has usually already finished.
        report("Separating background audio", 90)
        background_audio = background_audio_future.result()
        _separator_pool.shutdown(wait=True)
        dubbed_audio = AudioSegment.from_file("temp/final_audio.wav")
        logger.info(f"dubbed_audio length: {len(dubbed_audio)}")
        logger.info(f"background_audio length: {len(background_audio)}")

        # 12. Overlay dubbed speech with background
        report("Combining speech and background audio", 94)
        combined_audio_path = overlay_audios(dubbed_audio, background_audio)

        # 13. Combine audio with video & burn subtitles
        report("Combining audio with video", 97)
        output_video_path = combine_audio_with_video(combined_audio_path, video_path, subtitle_path)
        report("Done", 100)
        return output_video_path

    def dub_audio(self,
                  audio_bytes: bytes,
                  targ: str,
                  mistral_api: str = None,
                  gemini_api: str = None,
                  groq_api: str = None,
                  groq_model: str = None,
                  gemini_model: str = None,
                  tts_engine: str = "xtts",
                  cosyvoice_model_dir: str = "pretrained_models/Fun-CosyVoice3-0.5B",
                  progress_callback=None,
                  tts_worker_fn=None,
                  num_tts_workers: int = 4,
                  translation_mode: str = "per_sentence"):
        """Audio-only dubbing: same quality as dub(), but no video processing.

        Input: raw audio bytes (mp3/wav/etc)
        Returns: {
            "chunks": [{"index": int, "start": float, "end": float,
                         "speaker": str, "wav": bytes}],
            "src_lang": str,
        }
        """
        import io

        def report(stage: str, percent: int):
            if progress_callback:
                progress_callback(stage, percent)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Clean up old temp files from previous runs
        report("Starting...", 0)
        cleanup_dirs = [
            "temp/speakers_audio",
            "temp/audio_chunks",
            "temp/adj_audio_chunks",
            "temp/emotions_audio"
        ]
        for dir_path in cleanup_dirs:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    if file.endswith(".wav"):
                        os.remove(os.path.join(dir_path, file))
        logger.info("Cleaned up old temp audio files")

        # 1. Load audio from bytes
        report("Loading audio", 3)
        orig_audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        orig_audio_path = "temp/original_audio.wav"
        orig_audio.export(orig_audio_path, format="wav")
        report("Audio loaded", 7)

        # Kick off UVR background-music separation in a thread NOW. UVR only
        # reads orig_audio_path (immutable input), so it can run concurrently
        # with transcription / translation / emotion / TTS. Saves ~26s on a
        # 14-min video by hiding the separation pass behind other stages.
        import concurrent.futures as _cf
        _separator_pool = _cf.ThreadPoolExecutor(max_workers=1)

        def _separate_background():
            from audio_separator.separator import Separator
            sep = Separator(model_file_dir="/root/baked_models/audio_separator")
            sep.load_model(model_filename='2_HP-UVR.pth')
            return AudioSegment.from_file(sep.separate(orig_audio_path)[0])

        background_audio_future = _separator_pool.submit(_separate_background)

        # 2. Transcription & diarization
        report("Transcribing & diarizing", 10)
        if not mistral_api:
            raise ValueError("mistral_api is required for Mistral transcription")
        logger.info("Running Mistral Transcription (with diarization)...")
        mp3_path = "temp/original_audio_mistral.mp3"
        orig_audio.export(mp3_path, format="mp3", bitrate="64k")
        logger.info(f"Compressed audio for Mistral: {os.path.getsize(mp3_path) / 1024 / 1024:.1f} MB")
        from mistralai import Mistral
        client = Mistral(api_key=mistral_api)
        with open(mp3_path, "rb") as f:
            transcription_response = client.audio.transcriptions.complete(
                model="voxtral-mini-2602",
                file={
                    "content": f,
                    "file_name": "audio.mp3",
                },
                diarize=True,
                timestamp_granularities=["segment"],
            )
        os.remove(mp3_path)
        segments = mistral_segments_to_pipeline(transcription_response.segments)
        src_lang = transcription_response.language or "en"
        logger.info(f"Transcription completed! Found {len(segments)} segments")

        speaker_turns = segments_to_speaker_turns(segments)
        report("Transcription complete", 18)

        # 3. Extract & denoise speaker audio (for voice cloning)
        report("Extracting speaker audio", 20)
        split_speakers_and_denoise(orig_audio, speaker_turns, "temp/speakers_audio")
        report("Speaker audio ready", 24)

        # 4. Process transcription segments
        report("Processing segments", 26)
        segments_with_speakers = merge_close_segments(segments)
        logger.info(f"Merged into {len(segments_with_speakers)} segments")

        # 5. Build sentences
        report("Building sentences", 30)
        sentences = create_sentences(segments_with_speakers)
        logger.info(f"Created {len(sentences)} sentences")
        sentences = assign_sentences_to_segments(sentences, segments_with_speakers)

        # 6. Translation + 7a. Emotion classify (run concurrently — they share
        # segments_with_speakers but touch different fields; mapping function
        # below mutates same list to add translation/orig without clobbering
        # the emotion field that the emotion thread sets).
        import concurrent.futures

        os.makedirs("temp/audio_chunks", exist_ok=True)
        os.makedirs("temp/emotions_audio", exist_ok=True)

        def _translation_task():
            if translation_mode == "full_transcript" and groq_api:
                report("Translating (full transcript)", 35)
                logger.info("Using full-transcript translation mode (Groq bulk call)")
                return translate_full_transcript(
                    sentences, src_lang, targ,
                    groq_api=groq_api, groq_model=groq_model,
                    progress_callback=progress_callback,
                    progress_start=35, progress_end=58,
                )
            report("Translating", 35)
            if translation_mode == "full_transcript":
                logger.warning("translation_mode=full_transcript requested but no groq_api; falling back to per-sentence")
            n_sentences = len(sentences)
            for i, sentence_obj in enumerate(sentences):
                if n_sentences > 0:
                    report("Translating", 35 + int(20 * (i + 1) / n_sentences))
                sentence = sentence_obj['sentence']

                if i == 0:
                    before_context = ""
                    after_context = sentences[i+1]['sentence'] if len(sentences) > 1 else ""
                elif i == len(sentences) - 1:
                    before_context = sentences[i-1]['sentence']
                    after_context = ""
                else:
                    before_context = sentences[i-1]['sentence']
                    after_context = sentences[i+1]['sentence']

                translation = utils.translate(sentence,
                                              before_context,
                                              after_context,
                                              src_lang,
                                              targ,
                                              groq_api=groq_api,
                                              groq_model=groq_model,
                                              gemini_api=gemini_api,
                                              gemini_model=gemini_model)
                sentence_obj['translation'] = translation
            return sentences

        def _emotion_task():
            from speechbrain.inference.interfaces import foreign_class
            classifier = foreign_class(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                savedir="/root/baked_models/speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                run_opts={"device": device.type})
            for seg in segments_with_speakers:
                if seg.get('speaker') is None:
                    seg['emotion'] = 'neutral'
                    continue
                start = seg['start'] * 1000
                end = seg['end'] * 1000
                # Per-segment temp file so we never collide with the other thread.
                tmp = f"temp/emotions_audio/emotion_{int(start)}_{int(end)}.wav"
                orig_audio[start:end].export(tmp, format="wav")
                seg['emotion'] = classify_emotion(tmp, classifier)
                os.remove(tmp)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            t_future = ex.submit(_translation_task)
            e_future = ex.submit(_emotion_task)
            sentences = t_future.result()
            e_future.result()
        report("Translation complete", 58)

        # Map translated sentences to segments (mutates segments_with_speakers
        # in place — emotion field set by the parallel task persists).
        final_segments = map_translated_sentences_to_segments(sentences, segments_with_speakers)
        logger.info(f"Mapped translations to {len(final_segments)} segments")

        # Debug: Check segment translations
        logger.info("\n=== FINAL SEGMENTS CHECK ===")
        for i, seg in enumerate(final_segments):
            print(f"Segment {i}:")
            print(f"  Speaker: {seg.get('speaker')}")
            print(f"  Original text: '{seg.get('text', 'MISSING')}'")
            print(f"  Translation: '{seg.get('translation', 'MISSING')}'")
            print(f"  Emotion: '{seg.get('emotion', 'MISSING')}'")
            print(f"  Start: {seg.get('start')}, End: {seg.get('end')}")
        logger.info("="*40 + "\n")

        # 7. Text to Speech
        report("Generating speech", 60)
        n_segments = len(final_segments)

        # 7b. Generate TTS — parallel workers or local
        if tts_worker_fn is not None:
            # --- Parallel TTS across multiple GPU containers ---
            logger.info(f"Using {num_tts_workers} parallel TTS workers")

            speaker_ids = set(seg['speaker'] for seg in final_segments if seg['speaker'] is not None)
            speaker_wavs = {}
            for spk in speaker_ids:
                spk_path = f"temp/speakers_audio/{spk}.wav"
                if os.path.exists(spk_path):
                    with open(spk_path, "rb") as f:
                        speaker_wavs[spk] = f.read()

            tts_segments = []
            for i, segment in enumerate(final_segments):
                if segment['speaker'] is None:
                    dur_ms = max(1, int((segment['end'] - segment['start']) * 1000))
                    AudioSegment.silent(duration=dur_ms).export(f"temp/audio_chunks/{i}.wav", format="wav")
                    continue
                tts_segments.append({
                    "index": i,
                    "text": segment.get('translation', ''),
                    "emotion": segment.get('emotion', 'neutral'),
                    "speaker": segment['speaker'],
                    "start": segment['start'],
                    "end": segment['end'],
                })

            MIN_SEGMENTS_PER_WORKER = 5
            actual_workers = max(1, min(num_tts_workers, len(tts_segments) // MIN_SEGMENTS_PER_WORKER))
            logger.info(f"{len(tts_segments)} segments → {actual_workers} TTS workers")

            batches = [[] for _ in range(actual_workers)]
            for idx, seg_data in enumerate(tts_segments):
                batches[idx % actual_workers].append(seg_data)
            batches = [b for b in batches if b]

            batch_inputs = []
            for batch in batches:
                batch_speakers = set(s["speaker"] for s in batch)
                batch_inputs.append({
                    "segments": batch,
                    "speaker_wavs": {spk: speaker_wavs[spk] for spk in batch_speakers if spk in speaker_wavs},
                    "targ": targ,
                    "tts_engine": tts_engine,
                    "cosyvoice_model_dir": cosyvoice_model_dir,
                })

            report("Generating speech (parallel)", 62)
            for result_idx, result in enumerate(tts_worker_fn.map(batch_inputs)):
                report("Generating speech (parallel)", 62 + int(16 * (result_idx + 1) / len(batch_inputs)))
                for seg_idx_str, wav_bytes in result["wavs"].items():
                    with open(f"temp/audio_chunks/{seg_idx_str}.wav", "wb") as f:
                        f.write(wav_bytes)
            logger.info("Parallel TTS complete")

        else:
            # --- Local single-GPU TTS (original path) ---
            use_cosyvoice = tts_engine == "cosyvoice"
            if use_cosyvoice:
                logger.info("Using CosyVoice TTS engine")
                cosyvoice_model = load_cosyvoice(cosyvoice_model_dir)
            else:
                logger.info("Using XTTS TTS engine")
                from TTS.api import TTS
                tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())

            for i, segment in enumerate(final_segments):
                if n_segments > 0:
                    report("Generating speech", 60 + int(18 * (i + 1) / n_segments))
                if segment['speaker'] is None:
                    logger.warning(f"Segment {i} has no speaker, skipping")
                    continue

                logger.info(f"TTS-ing segment {i}")

                if not segment['translation'] or segment['translation'].strip() == "":
                    logger.warning(f"Segment {i} has empty translation, generating silence")
                    dur_ms = max(1, int((segment['end'] - segment['start']) * 1000))
                    AudioSegment.silent(duration=dur_ms).export(f"temp/audio_chunks/{i}.wav", format="wav")
                    continue

                if use_cosyvoice:
                    tts_segment_cosyvoice(cosyvoice_model, segment['translation'], i,
                                           f"temp/speakers_audio/{segment['speaker']}.wav",
                                           segment['emotion'])
                else:
                    tts_segment(tts, segment['translation'], i,
                                f"temp/speakers_audio/{segment['speaker']}.wav",
                                targ, segment['emotion'])

        # 8. Adjust audio timing
        report("Adjusting audio timing", 80)
        os.makedirs("temp/adj_audio_chunks", exist_ok=True)
        adjust_audio(final_segments, MIN_SPEED=0.85, MAX_SPEED=2, orig_audio_len=len(orig_audio))

        # 9. Stitch audio chunks
        report("Stitching audio", 87)
        stitch_chunks(final_segments)

        # 10. Wait for the background-separation thread kicked off near the
        # start. By the time we reach here it has usually already finished.
        report("Separating background audio", 90)
        background_audio = background_audio_future.result()
        _separator_pool.shutdown(wait=True)
        dubbed_audio = AudioSegment.from_file("temp/final_audio.wav")
        logger.info(f"dubbed_audio length: {len(dubbed_audio)}")
        logger.info(f"background_audio length: {len(background_audio)}")

        # 11. Overlay dubbed speech with background
        report("Combining speech and background audio", 94)
        combined_audio_path = overlay_audios(dubbed_audio, background_audio)

        # 12. Return chunks with timing info
        report("Building response", 97)
        chunks = []
        for i, seg in enumerate(final_segments):
            adj_path = f"temp/adj_audio_chunks/{i}.wav"
            if os.path.exists(adj_path):
                with open(adj_path, "rb") as f:
                    wav_bytes = f.read()
                chunks.append({
                    "index": i,
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": seg.get("speaker"),
                    "wav": wav_bytes,
                })

        # Also return the final combined audio (dubbed + background)
        with open(combined_audio_path, "rb") as f:
            combined_wav_bytes = f.read()

        report("Done", 100)
        return {
            "chunks": chunks,
            "combined_audio": combined_wav_bytes,
            "src_lang": src_lang,
        }

# TODO:
 # 1. Check through functions again to ensure smooth transitions
 # 2. Maybe hardcode paths up top?

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['video', 'audio'], default='video')
    args = parser.parse_args()

    pipeline = YTDubPipeline()

    if args.mode == "audio":
        with open("temp/orig_audio.wav", "rb") as f:
            audio_bytes = f.read()
        result = pipeline.dub_audio(
            audio_bytes=audio_bytes,
            targ="zh",
            mistral_api=os.getenv('MISTRAL_API_KEY'),
            tts_engine="xtts",
        )
        logger.info(f"Got {len(result['chunks'])} chunks, src_lang: {result['src_lang']}")
        with open("temp/dubbed_audio_only.wav", "wb") as f:
            f.write(result["combined_audio"])
        logger.info("Saved combined dubbed audio to temp/dubbed_audio_only.wav")
    else:
        result = pipeline.dub(
            src="test_inputs/345mile.mp4",
            targ="zh",
            hf_token=os.getenv('HF_TOKEN'),
            mistral_api=os.getenv('MISTRAL_API_KEY'),
            groq_api=os.getenv('GROQ_API_KEY'),
            tts_engine="xtts",
            speakerTurnsPkl=True,
            segmentsPkl=True,
            finalSentencesPkl=True,
        )
        logger.info(f"Dubbed video path: {result}")