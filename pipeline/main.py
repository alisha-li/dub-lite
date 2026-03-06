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
from faster_whisper import WhisperModel
import pickle
from TTS.api import TTS
from audio_separator.separator import Separator
import utils
from speechbrain.inference.interfaces import foreign_class
from mistralai import Mistral
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
    get_video_resolution,
    create_subtitle_chunks,
    create_subtitle_chunks_from_segments,
    generate_subtitles,
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
            progress_callback=None):
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
            "temp/adjAudio_chunks",
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

        # 6. Translation
        report("Translating", 35)
        if finalSentencesPkl:
            logger.info("Loading existing final sentences from file...")
            with open("temp/final_sentences.pkl", "rb") as f:
                sentences = pickle.load(f)
        else:
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
        report("Translation complete", 58)

        # Map translated sentences to segments
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
            print(f"  Start: {seg.get('start')}, End: {seg.get('end')}")
        logger.info("="*40 + "\n")

        # 7. Text to Speech
        report("Generating speech", 60)
        os.makedirs("temp/audio_chunks", exist_ok=True)
        os.makedirs("temp/emotions_audio", exist_ok=True)
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
        n_segments = len(final_segments)
        classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                                   pymodule_file="custom_interface.py",
                                   classname="CustomEncoderWav2vec2Classifier",
                                   run_opts={"device": device.type})
        for i, segment in enumerate(final_segments):
            if n_segments > 0:
                report("Generating speech", 60 + int(18 * (i + 1) / n_segments))
            if segment['speaker'] is None:
                logger.warning(f"Segment {i} has no speaker, skipping")
                continue
            start = segment['start']*1000
            end = segment['end']*1000
            orig_audio[start:end].export("temp/emotions_audio/emotions.wav", format="wav")
            segment['emotion'] = classify_emotion("temp/emotions_audio/emotions.wav", classifier)
            os.remove("temp/emotions_audio/emotions.wav")

            logger.info(f"TTS-ing segment {i}")

            if not segment['translation'] or segment['translation'].strip() == "":
                logger.warning(f"Segment {i} has empty translation, generating silence")
                dur_ms = max(1, int((segment['end'] - segment['start']) * 1000))
                AudioSegment.silent(duration=dur_ms).export(f"temp/audio_chunks/{i}.wav", format="wav")
                continue

            tts_segment(tts, segment['translation'], i,
                        f"temp/speakers_audio/{segment['speaker']}.wav",
                        targ, segment['emotion'])

        # 8. Adjust audio timing
        report("Adjusting audio timing", 80)
        os.makedirs("temp/adjAudio_chunks", exist_ok=True)
        adjust_audio(final_segments, MIN_SPEED=0.85, MAX_SPEED=2, orig_audio_len=len(orig_audio))

        # 9. Generate subtitles (using adjusted audio durations)
        report("Generating subtitles", 84)
        cursor = 0.0
        for i, seg in enumerate(final_segments):
            adj_chunk = AudioSegment.from_wav(f"temp/adjAudio_chunks/{i}.wav")
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

        # 11. Separate background audio
        report("Separating background audio", 90)
        separator = Separator()
        separator.load_model(model_filename='2_HP-UVR.pth')
        background_audio = AudioSegment.from_file(separator.separate(orig_audio_path)[0])
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

# TODO:
 # 1. Check through functions again to ensure smooth transitions
 # 2. Maybe hardcode paths up top?

if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--url', required=True)
    # args = parser.parse_args()
    pipeline = YTDubPipeline()
    result = pipeline.dub( 
        src="temp/Orig-NYT.mp4", 
        targ="zh", 
        # gemini_api = os.getenv('GEMINI_API'),
        # gemini_model = os.getenv('GEMINI_MODEL'),
        hf_token = os.getenv('HF_TOKEN'), 
        mistral_api = os.getenv('MISTRAL_API_KEY'),
        speakerTurnsPkl = True, 
        segmentsPkl = True, 
        finalSentencesPkl = False,
    )
    logger.info(f"Dubbed video path: {result}")