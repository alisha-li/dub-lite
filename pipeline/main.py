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
import subprocess
from dotenv import load_dotenv
load_dotenv()
from collections import defaultdict

import yt_dlp
from speechbrain.inference.interfaces import foreign_class
from pydub import AudioSegment
from faster_whisper import WhisperModel
import nltk
from nltk.tokenize import sent_tokenize
import pickle
from transformers import MarianTokenizer, MarianMTModel
from groq import Groq
from TTS.api import TTS
from audio_separator.separator import Separator
import librosa
import soundfile as sf
import numpy as np
from denoise_audio import denoise_audio
from utils import download_video_and_extract_audio, diarize_audio, split_speakers_and_denoise, assign_speakers_to_segments, create_sentences
from log import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)


class YTDubPipeline:
    def __init__(self):
        os.makedirs("temp", exist_ok=True)
        os.makedirs("temp/speakers_audio", exist_ok=True)

    def dub(self,src: str, targ: str, hf_token: str, speakerTurnsPkl: bool, segmentsPkl: bool, pyannote_key: bool, gemini_api: str, groq_api: str):
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
            speaker_turns, speakers = diarize_audio(orig_audio_path, pyannote_key)
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
        
        # After translation, we need a method to assign translated sentences back to segments

        # 3. Translate and extract emotions
        if groq_api:
            def translate(sentence, before_context, after_context, target_language):
                client = Groq(api_key=os.environ.get("GROQ_API_KEY"),)
                completion = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "user",
                            "content": f"""{before_context} {sentence} {after_context} Correct any typos and ONLY output {targ} translation of '{sentence}'. Do not output any thing else."""
                        }
                    ]
                )
                translation = completion.choices[0].message.content.strip()
                print(translation)
                return translation
        elif gemini_api:
            ...
        else:
            ...
        if turnsFinalPkl:
            print("Loading existing final turns from file...")
            with open("temp/turnsFinal.pkl", "rb") as f:
                turns = pickle.load(f)
        else:
            classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
            os.makedirs("temp/emotions_audio", exist_ok=True)
            for i, sentence_obj in enumerate(all_sentences_sorted):
                sentence = sentence_obj['sentence']
                
                if i == 0:
                    before_context = ""
                    after_context = all_sentences_sorted[i+1]['sentence'] if len(all_sentences_sorted) > 1 else ""
                elif i == len(all_sentences_sorted) - 1:
                    before_context = all_sentences_sorted[i-1]['sentence']
                    after_context = ""
                else:
                    before_context = all_sentences_sorted[i-1]['sentence']
                    after_context = all_sentences_sorted[i+1]['sentence']
                
                # Translate with context
                translation = translate(sentence, before_context, after_context, targ)
                sentence_obj['translation'] = translation
            # for i, turn in enumerate(turns):
            #     sentence = turn['text']
            #     start = turn['start']*1000
            #     end = turn['end']*1000

            #     if i == 0:
            #         translatedSentence = translate(sentence, "", turns[i+1]['text'], targ)
            #     elif i == len(turns) - 1:
            #         translatedSentence = translate(sentence, turns[i-1]['text'], "", targ)
            #     else:
            #         translatedSentence = translate(sentence, turns[i-1]['text'], turns[i+1]['text'], targ)
            #     turn['translation'] = translatedSentence
    
            #     orig_audio[start:end].export("temp/emotions_audio/emotions.wav", format="wav")      
            #     out_prob, score, index, text_lab = classifier.classify_file("temp/emotions_audio/emotions.wav")
            #     os.remove("temp/emotions_audio/emotions.wav")
                
            #     turn['emotion'] = text_lab[0]
            #     # Save turns to pickle file
            # with open("temp/turnsFinal.pkl", "wb") as f:
            #     pickle.dump(turns, f)

        # 4. Text to Speech and Audio Adjustments
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        print("TTS model initialized!")
        MIN_SPEED = .8
        MAX_SPEED = 1.7
        
        os.makedirs("temp/audio_chunks", exist_ok=True)
        os.makedirs("temp/adjAudio_chunks", exist_ok=True)

        curDuration = 0

        # TTS, speed/slow audio, and add silences
        for i, turn in enumerate(turns):
            # TTS
            print(f"working on turn: {i}")
            tts.tts_to_file(text=turn['translation'],
                            file_path=f"temp/audio_chunks/{i}.wav",
                            speaker_wav=f"temp/speakers_audio/{turn['speaker']}.wav",
                            language=targ,
                            emotion=turn['emotion'],
                            speed=1.0) 
            audio = AudioSegment.from_wav(f"temp/audio_chunks/{i}.wav")
            if i == 0:
                prev_silence = turn['start']*1000
                next_silence = (turns[i+1]['start']*1000) - (turn['end']*1000)
            elif i == len(turns) - 1:
                prev_silence = (turn['start']*1000) - (turns[i-1]['end']*1000)
                next_silence = len(orig_audio) - turn['end']*1000
            else:
                prev_silence = (turn['start']*1000) - (turns[i-1]['end']*1000)
                next_silence = (turns[i+1]['start']*1000) - (turn['end']*1000)

            usable_prev_silence = min(300, max(prev_silence, 0)) # don't start > 300ms before orig start
            usable_next_silence = max(next_silence - 300, 0) # allocate 300ms for audio after
            
            translated_dur = len(audio)
            orig_dur = (turn['end'] - turn['start']) * 1000
            adjAudioFile = f"temp/adjAudio_chunks/{i}.wav"

            # 4.1. speed/slow audio
            # Handle zero duration case
            if orig_dur <= 0:
                print(f"Warning: Sentence {i} has zero/negative duration ({orig_dur}ms), skipping adjustment")
                audio.export(adjAudioFile, format="wav")
                continue
            
            # Calculate accumulated drift (how far behind schedule we are)
            drift_ms = max(0, curDuration - (turn['start'] * 1000))
            logger.info(f"curDuration: {curDuration}")
            logger.info(f"turn start: {turn['start']}")
            logger.info(f"drift_ms: {drift_ms}")
            target_dur = max(orig_dur + usable_prev_silence + usable_next_silence - drift_ms, .01)
            
            if (translated_dur == orig_dur and drift_ms == 0):
                # Exact match and on schedule - just copy
                adjAudio = AudioSegment.silent(duration=usable_prev_silence) + audio + AudioSegment.silent(duration=usable_next_silence)
        
            elif (orig_dur < translated_dur < target_dur):
                logger.info("orig_dur < translated_dur < target_dur")
                range_size = target_dur - orig_dur
                distance_from_orig = translated_dur - orig_dur
                ratio = distance_from_orig / range_size
                speed_factor = 1.0 + (ratio * (MAX_SPEED - 1.0))
                adjAudio = audio.speedup(playback_speed=speed_factor)
                
                current_total = usable_prev_silence + len(adjAudio)
                usableNextSilence_needed = target_dur - current_total
                if usableNextSilence_needed >= 0:
                    adjAudio = AudioSegment.silent(duration=usable_prev_silence) + adjAudio + AudioSegment.silent(duration=usableNextSilence_needed)
                else:
                    leftover_prevSilence = usable_prev_silence + usableNextSilence_needed
                    adjAudio = AudioSegment.silent(duration=leftover_prevSilence) + adjAudio

            elif translated_dur >= target_dur:
                logger.info("translated_dur >= target_dur")
                # Need to speed up
                speed_factor = translated_dur / target_dur
                if speed_factor > MAX_SPEED:
                    speed_factor = MAX_SPEED
                adjAudio = audio.speedup(playback_speed=speed_factor)
                # command = f"ffmpeg -i temp/audio_chunks/{i}.wav -filter:a 'atempo={speed_factor}' -vn {adjAudioFile}"
                # process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)    
                # if process.returncode != 0:
                #     print(f"FFmpeg error: {process.stderr}")
                #     raise Exception(f"Failed to speed up audio: {process.stderr}")
                # adjAudio = AudioSegment.from_wav(adjAudioFile)
                
            elif translated_dur < orig_dur:
                logger.info("translated_dur < orig_dur")
                # Need to slow down 
                speed_factor = translated_dur / orig_dur
                if speed_factor < MIN_SPEED: #translated_dur signif shorter than target_dur
                    speed_factor = MIN_SPEED
                
                # Slow down using frame rate manipulation
                # new_frame_rate = int(audio.frame_rate * speed_factor)
                # adjAudio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})
                # adjAudio = adjAudio.set_frame_rate(audio.frame_rate)

                # 1. Load the audio file
                # y: audio time series, sr: sampling rate
                y, sr = librosa.load(f"temp/audio_chunks/{i}.wav", sr=None) # sr=None preserves original sample rate

                # 2. Define the stretch factor for slowing down
                # rate < 1.0 slows down; rate > 1.0 speeds up
                slow_rate = speed_factor # Slows down by 25%
                # slow_rate = 0.5 # Halves the speed

                # 3. Apply time stretching
                y_slow = librosa.effects.time_stretch(y, rate=slow_rate)

                # 4. Save the slowed-down audio
                sf.write(f"temp/audio_chunks/{i}_slowed.wav", y_slow, sr) # {Link: soundfile.write https://pypi.org/project/soundfile/}
                adjAudio = AudioSegment.from_wav(f"temp/audio_chunks/{i}_slowed.wav")

                if os.path.exists("temp/audio_chunks/{i}_slowed.wav"):
                    logger.info(f"Removing temp/audio_chunks/{i}_slowed.wav")
                    os.remove("temp/audio_chunks/{i}_slowed.wav")

                # if os.path.exists(temp_output):
                #     logger.info(f"Removing {temp_output}")
                #     os.remove(temp_output)
                # If still shorter after slowing, pad with silence
                if len(adjAudio) < orig_dur:
                    adjAudio = adjAudio + AudioSegment.silent(duration=int(orig_dur - len(adjAudio)))
                adjAudio = AudioSegment.silent(duration=usable_prev_silence) + adjAudio + AudioSegment.silent(duration=usable_next_silence)

            logger.info(f"speed factor: {speed_factor}")
            logger.info(f"target dur: {target_dur}")
            logger.info(f"adjusted audio length: {len(adjAudio)}")
            logger.info(f"adjAudio duration: {len(adjAudio)}")
            logger.info(f"usable_prev_silence: {usable_prev_silence}")
            logger.info(f"usable_next_silence: {usable_next_silence}")
            logger.info(f"next_silence: {next_silence}")
            logger.info(f"prev_silence: {prev_silence}")
        
            if i == 0:
                adjAudio = AudioSegment.silent(duration = prev_silence-usable_prev_silence) + adjAudio
            elif i == len(turns) - 1:
                adjAudio = adjAudio + AudioSegment.silent(duration = next_silence-usable_next_silence)
            adjAudio.export(adjAudioFile, format="wav")
            curDuration += len(adjAudio)
            
            logger.info(f"durations for turn {i}:")
            logger.info(f"orig duration: {orig_dur}")
            logger.info(f"translated duration: {translated_dur}")
            logger.info(f"transformed duration: {len(adjAudio)}")


        # Stitch chunks together
        logger.info("Stitching chunks together")
        final_audio = AudioSegment.empty()
        for i in range(len(turns)):
            adjAudioFile = f"temp/adjAudio_chunks/{i}.wav"
            adjAudio = AudioSegment.from_wav(adjAudioFile)
            final_audio += adjAudio
            logger.info(f"Added chunk {i} ({len(adjAudio)}ms), total so far: {len(final_audio)}ms")

        final_audio.export("temp/final_audio.wav", format="wav")
        logger.info(f"Final audio length: {len(final_audio)}ms ({len(final_audio)/1000:.2f}s)")

        # Stitch chunks together
        # print("Stitching chunks together")
        # final_audio = AudioSegment.empty()
        # curPos = 0
        # for i, turn in enumerate(turns):
        #     adjAudioFile = f"temp/adjAudio_chunks/{i}.wav" 
        #     adjAudio = AudioSegment.from_wav(adjAudioFile)
        #     orig_start = turn['start'] * 1000
        #     pad = orig_start - curPos
        #     if pad > 0:
        #         final_audio += AudioSegment.silent(duration=pad)
        #         print(f"padding: {pad}ms")
        #     else:
        #         print(f"Warning: Chunk {i} would overlap previous chunk by {-pad}ms. No silence added.")
        #     final_audio += adjAudio
        #     curPos = orig_start + len(adjAudio)
        
        # final_audio.export("temp/final_audio.wav", format="wav")

        separator = Separator()

        # Load a model
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
    result = pipeline.dub("https://www.youtube.com/watch?v=XXPISZI_big", "en", "zh", os.getenv('HF_TOKEN'), False, False, pyannote_key=os.getenv('PYANNOTE_API_KEY'))
    print(f"dubbed video path: {result}")