# 1. YT-DLP downloads video
# 2. Speaker Diarization and Transcription
# 3. translate
# 4. text to speech
# 5. separate out background sounds from orig audio
# 6. 1 overlay with dubbed audio
# 7.  ffmpeg combine audio with video

 # TODO:
 # fix download to override current orig_video
 # Add your other pipeline steps here
            # 2. Speaker Diarization
            # 3. Speech to text
            # 4. Translate
            # 5. Text to speech
            # 6. Background sound separation
            # 7. Combine audio with video


# test video: https://www.youtube.com/watch?v=jIZkKsf6VYo
# easier test video: https://www.youtube.com/watch?v=YgxyLrnxCH4
import os
from re import A
import subprocess
from dotenv import load_dotenv
load_dotenv()

from pydub import AudioSegment
from faster_whisper import WhisperModel
import pickle
from TTS.api import TTS
from audio_separator.separator import Separator
from denoise_audio import denoise_audio
import utils
from utils import (
    download_video_and_extract_audio,
    diarize_audio,
    split_speakers_and_denoise,
    assign_speakers_to_segments,
    create_sentences,
    classify_emotion,
    assign_sentences_to_segments,
    adjust_audio,
    map_translated_sentences_to_segments,
    stitch_chunks
)
from log import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)


class YTDubPipeline:
    def __init__(self):
        os.makedirs("temp", exist_ok=True)
        os.makedirs("temp/speakers_audio", exist_ok=True)

    def dub(self,src: str, targ: str, hf_token: str, pyannote_key: bool, gemini_api: str, groq_api: str, speakerTurnsPkl: bool, segmentsPkl: bool, finalSentencesPkl: bool):
        # 1. Download video using yt-dlp  
        print(f"Starting dubbing pipeline for: {src}")
        video_path, orig_audio_path, orig_audio = download_video_and_extract_audio(src)

        # 2. Speaker Diarization and Transcription
        if speakerTurnsPkl:
            logger.info("Loading pyannote pickle...")
            with open("temp/speaker_turns.pkl", "rb") as f:
                speaker_turns = pickle.load(f)
            logger.info(f"Loaded {len(speaker_turns)} speaker turns from file!")
        else:
            speaker_turns, speakers = diarize_audio(orig_audio_path, pyannote_key, hf_token)
            with open("temp/speaker_turns.pkl", "wb") as f:
                pickle.dump(speaker_turns, f)
        
        # Extract speaker audios and denoise (for voice cloning later)
        split_speakers_and_denoise(orig_audio, speaker_turns, "temp/speakers_audio")
                
        # 2.2.2 Transcription
        if segmentsPkl:
            logger.info("Loading segments pickle...")
            with open("temp/segments.pkl", "rb") as f:
                segments = pickle.load(f)
        else:
            logger.info("Running Whisper Transcription...")
            model = WhisperModel("medium", device="cpu", compute_type="int8")
            segments, info = model.transcribe(orig_audio_path, word_timestamps=True)
            segments = list(segments) 
            logger.info(f"Transcription completed! Found {len(segments)} segments")
            with open("temp/segments.pkl", "wb") as f:
                pickle.dump(segments, f)
            
        segments_with_speakers = assign_speakers_to_segments(segments, speaker_turns)

        # list of sentence objects (sentence, start, end, speaker) sorted by start
        sorted_sentences = create_sentences(segments_with_speakers)

        # adds segments prop with list of segments each sentence belongs to
        sorted_sentences = assign_sentences_to_segments(sorted_sentences, segments)

        # 3. Translate and extract emotions
        if finalSentencesPkl:
            print("Loading existing final sentences from file...")
            with open("temp/final_sentences.pkl", "rb") as f:
                sorted_sentences = pickle.load(f)
        else:
            for i, sentence_obj in enumerate(sorted_sentences):
                sentence = sentence_obj['sentence']
                
                if i == 0:
                    before_context = ""
                    after_context = sorted_sentences[i+1]['sentence'] if len(sorted_sentences) > 1 else ""
                elif i == len(sorted_sentences) - 1:
                    before_context = sorted_sentences[i-1]['sentence']
                    after_context = ""
                else:
                    before_context = sorted_sentences[i-1]['sentence']
                    after_context = sorted_sentences[i+1]['sentence']
                
                # Translate with context
                translation = utils.translate(sentence, before_context, after_context, targ)
                sentence_obj['translation'] = translation
               
                os.makedirs("temp/emotions_audio", exist_ok=True)
                start = sentence_obj['start']*1000
                end = sentence_obj['end']*1000
                orig_audio[start:end].export("temp/emotions_audio/emotions.wav", format="wav")      
                sentence_obj['emotion'] = classify_emotion("temp/emotions_audio/emotions.wav")
                os.remove("temp/emotions_audio/emotions.wav")
                
                with open("temp/final_sentences.pkl", "wb") as f:
                    pickle.dump(sorted_sentences, f)
        

        # Map translated sentences to segments
        final_segments = map_translated_sentences_to_segments(sorted_sentences, segments_with_speakers)
        
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
        os.makedirs("temp/audio_chunks", exist_ok=True)
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        for i, segment in enumerate(final_segments): # this should be by segment, not sentence
            print(f"TTS-ing segment {i}")
            tts.tts_to_file(text=segment['translation'],
                            file_path=f"temp/audio_chunks/{i}.wav",
                            speaker_wav=f"temp/speakers_audio/{segment['speaker']}.wav",
                            language=targ,
                            emotion=segment['emotion'],
                            speed=1.0) 
    
        MIN_SPEED = .8
        MAX_SPEED = 1.7
        
        os.makedirs("temp/adjAudio_chunks", exist_ok=True)

        adjust_audio(final_segments, MIN_SPEED, MAX_SPEED, len(orig_audio))
        stitch_chunks(final_segments)

        separator = Separator()
        separator.load_model(model_filename='2_HP-UVR.pth')
        background_path = separator.separate(orig_audio_path)[0]
        print("background path: ", background_path)
        audio1 = AudioSegment.from_file("temp/final_audio.wav")
        audio2 = AudioSegment.from_file(background_path)
        print("audio1 length: ", len(audio1))
        print("audio2 length: ", len(audio2))
        if len(audio1) > len(audio2):
            audio2 = audio2 + AudioSegment.silent(duration=len(audio1) - len(audio2))
        elif len(audio1) < len(audio2):
            audio1 = audio1 + AudioSegment.silent(duration=len(audio2) - len(audio1))
        combined_audio = audio1.overlay(audio2)
        combined_audio.export("temp/combined_audio.wav", format="wav")
        
        # Combine new audio with original video
        command = [
            'ffmpeg',
            '-i', 'temp/orig_video.mp4',
            '-i', 'temp/combined_audio.wav',
            '-c:v', 'copy',  # Copy video stream without re-encoding (fast)
            '-map', '0:v:0',  # Use video from first input
            '-map', '1:a:0',  # Use audio from second input
            '-shortest',  # End when shortest stream ends
            '-y',  # Overwrite output file if it exists
            'temp/output_video.mp4'
        ]
        print("Combining audio with video using ffmpeg...")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            raise Exception(f"Failed to combine audio with video: {result.stderr}")
        print("Video with dubbed audio saved to temp/output_video.mp4")

        return 'temp/output_video.mp4'  
    


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--url', required=True)
    # args = parser.parse_args()

    pipeline = YTDubPipeline()
    result = pipeline.dub(
        "https://www.youtube.com/watch?v=XXPISZI_big", "en", "zh", 
        hf_token = os.getenv('HF_TOKEN'), 
        speakerTurnsPkl = False, 
        segmentsPkl = False, 
        pyannote_key=os.getenv('PYANNOTE_API_KEY'))
    print(f"dubbed video path: {result}")