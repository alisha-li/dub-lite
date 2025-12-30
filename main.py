# 1. YT-DLP downloads video
# 2. Speaker Diarization and Transcription
# 3. translate
# 4. text to speech
# 5. separate out background sounds from orig audio
# 6. 1 overlay with dubbed audio
# 7.  ffmpeg combine audio with video

# ***REMOVED***

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

import yt_dlp
from speechbrain.inference.interfaces import foreign_class
from pyannote.audio import Pipeline as PyannotePipeline
from pydub import AudioSegment
from faster_whisper import WhisperModel
import nltk
from nltk.tokenize import sent_tokenize
import pickle
from transformers import MarianTokenizer, MarianMTModel
from groq import Groq
from TTS.api import TTS
from audio_separator.separator import Separator
import pypinyin
from pyannoteai.sdk import Client


class YTDubPipeline:
    def __init__(self):
        os.makedirs("temp", exist_ok=True)
        os.makedirs("temp/speakers_audio", exist_ok=True)

    def dub(self, url: str, src: str, targ: str, hf_token: str, pyannotePkl: bool, turnsFinalPkl: bool):
        # 1. Download video using yt-dlp package
        try:
            print(f"Starting dubbing pipeline for: {url}")
            ydl_opts = {
                'format' : 'best',
                'outtmpl' : 'temp/orig_video.mp4',
                'cookiesfrombrowser' : ('chrome',)
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            orig_audio = AudioSegment.from_file("temp/orig_video.mp4", format="mp4")
            orig_audio_path = "temp/orig_audio.wav"
            orig_audio.export(orig_audio_path, format="wav")
            
        except Exception as e:
            raise Exception(f"Video download failed: {str(e)}")

        # 1.5. Denoise audio (optional but recommended for better transcription/cloning)
        # try:
        #     print("Denoising audio for better transcription and voice cloning...")
        #     from denoise_audio import denoise_audio
        #     denoised_path = "temp/orig_audio_denoised.wav"
        #     if denoise_audio(orig_audio_path, denoised_path):
        #         orig_audio_path = denoised_path  # Use denoised audio for rest of pipeline
        #         orig_audio = AudioSegment.from_file(orig_audio_path, format="wav")  # Reload denoised audio
        #         print("Using denoised audio for speaker diarization and transcription")
        #     else:
        #         print("Warning: Denoising failed, using original audio")
        # except Exception as e:
        #     print(f"Warning: Denoising step failed ({e}), continuing with original audio")

        # 2. Speaker Diarization
        try:
            if pyannotePkl:
                print("Loading existing data from file...")
                with open("temp/orchestration.pkl", "rb") as f:
                    turns = pickle.load(f)
                
                # delete speakers part for speed
                print(f"Loaded {len(turns)} turns from file!")
                
            else:
                client = Client("***REMOVED***")
                orig_audio_url = client.upload(orig_audio_path)
                orchestration_job = client.diarize(orig_audio_url, transcription=True)
                orchestration = client.retrieve(orchestration_job)
                print("FIRST 10 RECORDS:")
                for turn in orchestration['output']['turnLevelTranscription'][:20]:
                    print(f"{turn['speaker']} [{turn['start']:6.3f}s — {turn['end']:6.3f}s] {turn['text']}")

                turns = orchestration['output']['turnLevelTranscription']

                with open("temp/orchestration.pkl", "wb") as f:
                    pickle.dump(orchestration['output']['turnLevelTranscription'], f)
            
            # Extract speaker audio segments
            print("Extracting speaker audio segments...")
            from denoise_audio import denoise_audio
            unique_speakers = set(turn['speaker'] for turn in turns)
            for speaker in unique_speakers:
                speaker_audio = AudioSegment.empty()
                for turn in turns:
                    if turn['speaker'] == speaker:
                        start_ms = int(turn['start'] * 1000)  # Convert seconds to milliseconds
                        end_ms = int(turn['end'] * 1000)
                        speaker_audio += orig_audio[start_ms:end_ms]
                speaker_file = f"temp/speakers_audio/{speaker}.wav"
                speaker_audio.export(speaker_file, format="wav")
                if denoise_audio(speaker_file, speaker_file):
                    print("Using denoised audio for voice cloning")
                else:
                    print("Warning: Denoising failed, using original audio")
                print(f"Extracted {len(speaker_audio)}ms of audio for {speaker}")

            print(f"Extracted audio for {len(unique_speakers)} speakers")

        except Exception as e:
            raise Exception(f"Speaker Diarization failed: {str(e)}")

        # 3. Translate and classify emotions
        def translate(sentence, before_context, after_context, target_language):
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"),)
            completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": f"""{before_context} {sentence} {after_context} ONLY output {targ} translation of '{sentence}'. Do not output any thing else."""
                    }
                ]
            )
            translation = completion.choices[0].message.content.strip()
            print(translation)
            return translation
    
        try:
            if turnsFinalPkl:
                print("Loading existing final turns from file...")
                with open("temp/turnsFinal.pkl", "rb") as f:
                    turns = pickle.load(f)
            else:
                classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
                os.makedirs("temp/emotions_audio", exist_ok=True)
                for i, turn in enumerate(turns):
                    sentence = turn['text']
                    start = turn['start']*1000
                    end = turn['end']*1000

                    if i == 0:
                        translatedSentence = translate(sentence, "", turns[i+1]['text'], targ)
                    elif i == len(turns) - 1:
                        translatedSentence = translate(sentence, turns[i-1]['text'], "", targ)
                    else:
                        translatedSentence = translate(sentence, turns[i-1]['text'], turns[i+1]['text'], targ)
                    turn['translation'] = translatedSentence
        
                    orig_audio[start:end].export("temp/emotions_audio/emotions.wav", format="wav")      
                    out_prob, score, index, text_lab = classifier.classify_file("temp/emotions_audio/emotions.wav")
                    os.remove("temp/emotions_audio/emotions.wav")
                    
                    turn['emotion'] = text_lab[0]
                    # Save turns to pickle file
                with open("temp/turnsFinal.pkl", "wb") as f:
                    pickle.dump(turns, f)

        except Exception as e:
            raise Exception(f"Translation/Emotion Labeling Failed: {str(e)}")

        #4. Text to Speech
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        MIN_SPEED = .8
        MAX_SPEED = 1.7
        
        os.makedirs("temp/audio_chunks", exist_ok=True)
        os.makedirs("temp/adjAudio_chunks", exist_ok=True)

        # TTS, speed/slow audio, and add silences
        for i, turn in enumerate(turns):
            # TTS
            tts.tts_to_file(text=turn['translation'],
                            file_path=f"temp/audio_chunks/{i}.wav",
                            speaker_wav=f"temp/speakers_audio/{turn['speaker']}.wav",
                            language=targ,
                            emotion=turn['emotion'],
                            speed=1.0) 
            audio = AudioSegment.from_wav(f"temp/audio_chunks/{i}.wav")

           
            adjAudioFile = f"temp/adjAudio_chunks/{i}.wav"

            translated_dur = len(audio)
            orig_dur = (turn['end'] - turn['start']) * 1000

            # speed/slow audio
            # Handle zero duration case
            if orig_dur <= 0:
                print(f"Warning: Sentence {i} has zero/negative duration ({orig_dur}ms), skipping adjustment")
                audio.export(adjAudioFile, format="wav")
                continue
            
            if translated_dur < orig_dur:
                speed_factor = translated_dur / orig_dur
                if speed_factor < MIN_SPEED: #translated_dur significantly shorter than orig_dur
                    speed_factor = MIN_SPEED
                
                # Slow down using frame rate manipulation
                new_frame_rate = int(audio.frame_rate * speed_factor)
                adjAudio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})
                adjAudio = adjAudio.set_frame_rate(audio.frame_rate)
                
                # If still shorter after slowing, pad with silence
                if len(adjAudio) < orig_dur:
                    adjAudio = adjAudio + AudioSegment.silent(duration=int(orig_dur - len(adjAudio)))
                adjAudio.export(adjAudioFile, format="wav")
            else:
                # Start with the original audio
                adjAudio = audio
                TRIM_MS = 300
                trim_amount = min(TRIM_MS, len(adjAudio) // 4)
                if trim_amount > 0:
                    adjAudio = adjAudio[trim_amount:-trim_amount]
                translated_dur = len(adjAudio)
                if translated_dur > orig_dur:
                    speed_factor = translated_dur / orig_dur
                    if speed_factor > MAX_SPEED:
                        speed_factor = MAX_SPEED
                    adjAudio = adjAudio.speedup(playback_speed=speed_factor)
                adjAudio.export(adjAudioFile, format="wav")

            
            
            print(f"durations for turn {i}:")
            print("orig duration: ", orig_dur)
            print("translated duration: ", translated_dur)
            print("transformed duration: ", len(adjAudio))

        # Stitch chunks together
        print("Stitching chunks together")
        final_audio = AudioSegment.empty()
        curPos = 0
        for i, turn in enumerate(turns):
            adjAudioFile = f"temp/adjAudio_chunks/{i}.wav" 
            adjAudio = AudioSegment.from_wav(adjAudioFile)
            print(f"turn {i}: {turn}")
            orig_start = turn['start'] * 1000
            pad = orig_start - curPos
            if pad > 0:
                final_audio += AudioSegment.silent(duration=pad)
                print(f"padding: {pad}ms")
            else:
                print(f"Warning: Chunk {i} would overlap previous chunk by {-pad}ms. No silence added.")
            final_audio += adjAudio
            curPos = orig_start + len(adjAudio)
        
        final_audio.export("temp/final_audio.wav", format="wav")

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
    result = pipeline.dub("https://www.youtube.com/watch?v=aJwdWeiSc8c&t=640s", "en", "zh", os.getenv('HF_TOKEN'), False, False)
    print(f"dubbed video path: {result}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 


