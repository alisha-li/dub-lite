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
from utils import (
    download_video_and_extract_audio,
    diarize_audio,
    split_speakers_and_denoise,
    merge_close_segments,
    assign_speakers_to_segments,
    create_sentences,
    classify_emotion,
    assign_sentences_to_segments,
    adjust_audio,
    map_translated_sentences_to_segments,
    stitch_chunks,
    overlay_audios,
    combine_audio_with_video,
)
from log import setup_logging
import logging

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

        # 1. Download video using yt-dlp
        report("Downloading video & extracting audio", 5)
        logger.info(f"Starting dubbing pipeline for: {src}")
        video_path, orig_audio_path, orig_audio = download_video_and_extract_audio(src)
        report("Download complete", 10)

        # 2. Speaker Diarization and Transcription
        report("Speaker diarization", 15)
        if speakerTurnsPkl:
            logger.info("Loading pyannote pickle...")
            with open("temp/speaker_turns.pkl", "rb") as f:
                speaker_turns = pickle.load(f)
            logger.info(f"Loaded {len(speaker_turns)} speaker turns from file!")
        else:
            speaker_turns = diarize_audio(orig_audio_path, pyannote_key, hf_token)
            with open("temp/speaker_turns.pkl", "wb") as f:
                pickle.dump(speaker_turns, f)
        report("Diarization complete", 22)

        # Extract speaker audios and denoise (for voice cloning later)
        split_speakers_and_denoise(orig_audio, speaker_turns, "temp/speakers_audio")
        report("Transcription", 25)
        # 2.2.2 Transcription
        if segmentsPkl:
            logger.info("Loading segments pickle...")
            with open("temp/segments.pkl", "rb") as f:
                data = pickle.load(f)
                segments = data["segments"]
                src_lang = data["language"]
                
        else:
            logger.info("Running Whisper Transcription...")
            model = WhisperModel("medium", device="cpu", compute_type="int8")
            segments, info = model.transcribe(orig_audio_path, word_timestamps=True)
            segments = list(segments) 
            src_lang = info.language
            logger.info(f"Transcription completed! Found {len(segments)} segments")
            with open("temp/segments.pkl", "wb") as f:
                pickle.dump({"segments": segments, "language": src_lang}, f)
        report("Transcription complete", 35)

        segments_with_speakers = assign_speakers_to_segments(segments, speaker_turns)
        
        # Save segments with speakers for inspection
        with open("temp/segments_with_speakers.pkl", "wb") as f:
            pickle.dump(segments_with_speakers, f)
        logger.info(f"Saved {len(segments_with_speakers)} segments with speakers")

        segments_with_speakers = merge_close_segments(segments_with_speakers)

        # Save merged segments for inspection
        with open("temp/segments_merged.pkl", "wb") as f:
            pickle.dump(segments_with_speakers, f)
        logger.info(f"Saved {len(segments_with_speakers)} merged segments")
        report("Assigning speakers", 38)

        # List of sentence objects (sentence, start, end, speaker), sorted by start
        sentences = create_sentences(segments_with_speakers)
        sentences = assign_sentences_to_segments(sentences, segments_with_speakers)

        # 3. Translate and extract emotions
        report("Translation", 40)
        if finalSentencesPkl:
            logger.info("Loading existing final sentences from file...")
            with open("temp/final_sentences.pkl", "rb") as f:
                sentences = pickle.load(f)
        else:
            n_sentences = len(sentences)
            for i, sentence_obj in enumerate(sentences):
                # 40% -> 62% over translation
                if n_sentences > 0:
                    report("Translation", 40 + int(22 * (i + 1) / n_sentences))
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
                
                # Translate with context
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
        report("Translation complete", 65)

        # Map translated sentences to segments
        final_segments = map_translated_sentences_to_segments(sentences, segments_with_speakers)
        
        # Save final_segments for inspection
        with open("temp/final_segments.pkl", "wb") as f:
            pickle.dump(final_segments, f)
        logger.info(f"Saved {len(final_segments)} segments to final_segments.pkl")
        
        # Debug: Check segment translations
        logger.info("\n=== FINAL SEGMENTS CHECK ===")
        for i, seg in enumerate(final_segments[:5]):  # Show first 5
            print(f"Segment {i}:")
            print(f"  Speaker: {seg.get('speaker')}")
            print(f"  Translation: '{seg.get('translation', 'MISSING')}'")
            print(f"  Start: {seg.get('start')}, End: {seg.get('end')}")
        logger.info("="*40 + "\n")
        
        # At this point, segments looks like:
        # [
        #     {
        #         'text': 'Hello, how',
        #         'words': ['Hello', 'how'],
        #         'speaker': 01,
        #         'start': 0,
        #         'end': 1000,
        #         'translation': '你好，你',
        #         'emotion': 'happy'
        #     },
        # ]

        # 4. Text to Speech
        report("Text-to-speech", 68)
        os.makedirs("temp/audio_chunks", exist_ok=True)
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        n_segments = len(final_segments)
        for i, segment in enumerate(final_segments):  # this should be by segment, not sentence
            if n_segments > 0:
                report("Text-to-speech", 68 + int(20 * (i + 1) / n_segments))
            if segment['speaker'] is None:
                logger.warning(f"Segment {i} has no speaker, skipping")
                continue
            os.makedirs("temp/emotions_audio", exist_ok=True)
            start = segment['start']*1000
            end = segment['end']*1000
            orig_audio[start:end].export("temp/emotions_audio/emotions.wav", format="wav")      
            segment['emotion'] = classify_emotion("temp/emotions_audio/emotions.wav")
            os.remove("temp/emotions_audio/emotions.wav")
            
            logger.info(f"TTS-ing segment {i}")
            logger.info(f"Translation text: '{segment['translation']}'")
            logger.info(f"Language: {targ}, Emotion: {segment['emotion']}")
            
            # Skip if translation is empty
            if not segment['translation'] or segment['translation'].strip() == "":
                logger.warning(f"Segment {i} has empty translation, skipping TTS")
                continue
            
            tts.tts_to_file(text=segment['translation'],
                            file_path=f"temp/audio_chunks/{i}.wav",
                            speaker_wav=f"temp/speakers_audio/{segment['speaker']}.wav",
                            language=targ,
                            emotion=segment['emotion'],
                            speed=1.0)

        # 5. Adjust audio (speed/slow, pad/trim)
        report("Adjusting audio", 90)
        os.makedirs("temp/adjAudio_chunks", exist_ok=True)
        adjust_audio(final_segments, MIN_SPEED=0.85, MAX_SPEED=1.6, orig_audio_len=len(orig_audio))

        # 6. Stich adjusted audio chunks together
        report("Stitching audio", 93)
        stitch_chunks(final_segments)

        # 7. Overlay dubbed speech, background sounds, and video
        report("Finalizing video", 96)
        # 7.1 Separate out background sounds
        separator = Separator()
        separator.load_model(model_filename='2_HP-UVR.pth')
        background_audio = AudioSegment.from_file(separator.separate(orig_audio_path)[0])
        dubbed_audio = AudioSegment.from_file("temp/final_audio.wav")
        logger.info(f"dubbed_audio length: {len(dubbed_audio)}")
        logger.info(f"background_audio length: {len(background_audio)}")

        # 7.2 Overlay dubbed speech and background sounds
        combined_audio_path = overlay_audios(dubbed_audio, background_audio)

        # 7.3 Combine with Video
        output_video_path = combine_audio_with_video(combined_audio_path, video_path)
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
        src="https://www.youtube.com/watch?v=aJwdWeiSc8c", 
        targ="zh", 
        gemini_api = os.getenv('GEMINI_API'),
        gemini_model = os.getenv('GEMINI_MODEL'),
        hf_token = os.getenv('HF_TOKEN'), 
        speakerTurnsPkl = False, 
        segmentsPkl = False, 
        finalSentencesPkl = False,
    )
    logger.info(f"Dubbed video path: {result}")