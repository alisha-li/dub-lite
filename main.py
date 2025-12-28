# 1. YT-DLP downloads video
# 2. Speaker Diarization
#     2.1. create separate audio files for each speaker
# 3. speech to text
#     3.1 segment text into sentences
# 4. translate
# 5. text to speech
# 6. separate out background sounds from orig audio
#     6.1 overlay with dubbed audio
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

    def dub(self, url: str, src: str, targ: str, hf_token: str, segmentsPickle: bool, timeSentencePickle: bool):
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
            
            
        except Exception as e:
            raise Exception(f"Video download failed: {str(e)}")

        client = Client("***REMOVED***")
        audio_url = client.upload("temp/orig_audio.wav")
        orchestration_job = client.diarize(audio_url, transcription=True)
        orchestration = client.retrieve(orchestration_job)
        for turn in orchestration['output']['turnLevelTranscription'][:10]:
            print(f"{turn['speaker']} [{turn['start']:6.3f}s — {turn['end']:6.3f}s] {turn['text']}")
        with open("temp/orchestration.json", "wb") as f:
            pickle.dump(orchestration['output']['turnLevelTranscription'], f)


        # 2. Speaker Diarization
        try:
            # Check if we already have segments saved
            segments_file = "temp/segments.pkl"
            if segmentsPickle:
                print("Loading existing data from file...")
                with open(segments_file, "rb") as f:
                    saved_data = pickle.load(f)
                
                segments = saved_data['segments']
                speakers_rolls = saved_data['speakers_rolls']
                speakers = saved_data['speakers']
                print(f"Loaded {len(segments)} segments and {len(speakers)} speakers from file!")
                
                # Still need to set up audio file
                audio = AudioSegment.from_file("temp/orig_video.mp4", format="mp4")
                audio_file = "temp/orig_audio.wav"
                audio.export(audio_file, format="wav")
            else:
                print("Running speaker diarization (this may take several minutes)...")
                
                diarizationPipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)

                audio = AudioSegment.from_file("temp/orig_video.mp4", format="mp4")
                audio_file = "temp/orig_audio.wav"
                audio.export(audio_file, format="wav")

                diarization = diarizationPipeline(audio_file)
                speakers_rolls ={}
            
                for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
                    if abs(speech_turn.end - speech_turn.start) > 1.5:
                        print(f"Speaker {speaker}: from {speech_turn.start}s to {speech_turn.end}s")
                        speakers_rolls[(speech_turn.start, speech_turn.end)] = speaker
                
                speakers = set(list(speakers_rolls.values()))  # Get unique speaker IDs

                print("done diarization")

        except Exception as e:
            raise Exception(f"Speaker Diarization failed: {str(e)}")
        
        # 3. Transcribe
        try:
            audio = AudioSegment.from_file(audio_file, format="mp4")

            for speaker in speakers:
                speaker_audio = AudioSegment.empty()
                for key, value in speakers_rolls.items():
                    if speaker == value:
                        start = int(key[0])*1000  # Convert seconds to milliseconds
                        end = int(key[1])*1000
                        
                        speaker_audio += audio[start:end]  # Extract this speaker's audio segments
                
                speaker_audio.export(f"temp/speakers_audio/{speaker}.wav", format="wav")
            
            model = WhisperModel("medium", device="cpu", compute_type="int8")
            segments, info = model.transcribe("temp/orig_audio.wav", word_timestamps=True)
            segments = list(segments) 

            print(f"Transcription completed! Found {len(segments)} segments")
            
            # Save all data to file for future use
            print("Saving segments and speaker data to file...")
            data_to_save = {
                'segments': segments,
                'speakers_rolls': speakers_rolls,
                'speakers': speakers
            }
            with open(segments_file, "wb") as f:
                pickle.dump(data_to_save, f)
            print(f"All data saved to {segments_file}")

            print("Processing segments...")
            print(f"Processing {len(segments)} segments...")
            
            # Process in smaller chunks to avoid memory issues
            time_stamped = []
            full_text = []
            
            for i, segment in enumerate(segments):
                if i % 10 == 0:  # Progress update every 10 segments
                    print(f"Processing segment {i+1}/{len(segments)}")
                
                for word in segment.words:
                    time_stamped.append([word.word, word.start, word.end])
                    full_text.append(word.word)
            
            print("Joining text...")
            full_text = "".join(full_text)  # Use space instead of empty string
            print(f"Total words processed: {len(time_stamped)}")       
            # Decompose Long Sentences
            
            # Tokenize the text into sentences
            sentences = sent_tokenize(full_text)
            
            print(sentences)

            # After creating sentences
            print("Mapping sentences to timestamps...")
            timeSentences = []
            wordCount = 0

            for sentence in sentences:
                sentence_words = sentence.split()  # Split sentence into words
                sentence_words = [word.strip('.,!?;:') for word in sentence_words]
                
              
                first_word_idx = wordCount  
                wordCount += len(sentence_words)
                last_word_idx = wordCount - 1  
                
                # Start time = first word's start, End time = last word's end
                timeSentences.append([sentence, time_stamped[first_word_idx][1], time_stamped[last_word_idx][2]])

            print("First 5 timeSentences:")
            for i, item in enumerate(timeSentences[:5]):
                print(f"  {i+1}: {item}")
            
        except Exception as e:
              raise Exception(f"STT Failed: {str(e)}")

        print("\nFirst 5 speakers_rolls:")
        for i, (key, value) in enumerate(list(speakers_rolls.items())[:5]):
            print(f"  {i+1}: {key} -> {value}")


        # 4. Translate
        # Oopsies, should translate by speaker not by sentence
        # Wait idk

        def translate(sentence, before_context, after_context, target_language):
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"),)
            completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Translate to {target_language}. Translate only: {sentence}. Context: {before_context} ... {after_context}"""
                    }
                ]
            )
            translation = completion.choices[0].message.content.strip()
            print(translation)
            return translation
        
        # def translate(chunk):
        #         #calculate overlap first

        #         tokenizer = MarianTokenizer.from_pretrained(model_name)
        #         model = MarianMTModel.from_pretrained(model_name).to('cpu')

        #         def translate_chunk(text):
        #             inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to('cpu')
        #             out_ids = model.generate(**inputs, max_new_tokens=256)
        #             return tokenizer.decode(out_ids[0], skip_special_tokens=True)

        #         words = sentence.split()
        #         if len(words) > 200:
        #             parts = [" ".join(words[i:i+200]) for i in range(0, len(words), 200)]
        #             outputs = [translate_chunk(p) for p in parts]
        #             return " ".join(outputs)
        #         else:
        #             return translate_chunk(sentence)
        try:
            # model_name = f"Helsinki-NLP/opus-mt-{src}-{targ}"

            # tokenizer = MarianTokenizer.from_pretrained(model_name)
            # model = MarianMTModel.from_pretrained(model_name).to('cpu')


            if timeSentencePickle:
                with open("temp/timeSentences.pkl", "rb") as f:
                    timeSentences = pickle.load(f)
            else:
                for i in range(len(timeSentences)):
                    sentence, start, end = timeSentences[i]
                    
                    sentence_speaker = None
                    max_overlap = 0
                    # Check overlap with each speaker's time range
                    for times, speaker in speakers_rolls.items():
                        speaker_start =  int(times[0])
                        speaker_end = int(times[1])
                        if speaker_start > end or speaker_end < start:
                            continue
                            
                        overlap_start = max(start, speaker_start)
                        overlap_end = min(end, speaker_end)
                        overlap = overlap_end - overlap_start
                        
                        # Update speaker if this overlap is greater than previous ones
                        if overlap > max_overlap:
                            max_overlap = overlap
                            sentence_speaker = speaker

                    timeSentences[i].append(sentence_speaker)

                    if i == 0:
                        translatedSentence = translate(timeSentences[i][0], "", timeSentences[i+1][0], targ)
                    elif i == len(timeSentences) - 1:
                        translatedSentence = translate(timeSentences[i][0], timeSentences[i-1][0], "", targ)
                    else:
                        translatedSentence = translate(timeSentences[i][0], timeSentences[i-1][0], timeSentences[i+1][0], targ)
                    
                    timeSentences[i].append(translatedSentence)
            
                    classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")

                    start=start*1000
                    end=end*1000

                    try:
                        os.makedirs("temp/emotions_audio", exist_ok=True)
                        audio[start:end].export("temp/emotions_audio/emotions.wav", format="wav")      
                        out_prob, score, index, text_lab = classifier.classify_file("temp/emotions_audio/emotions.wav")
                        os.remove("temp/emotions_audio/emotions.wav")
                    except Exception as e:
                        print(f"Error classifying emotions: {str(e)}")
                        text_lab = ['None']
                    
                    timeSentences[i].append(text_lab[0])
                
                # Save timeSentences to pickle file
                with open("temp/timeSentences.pkl", "wb") as f:
                    pickle.dump(timeSentences, f)

            print("First 20 sentences with emotions")
            for i in range(20):
                print(str(timeSentences[i]) + "\n")

        except Exception as e:
            raise Exception(f"Translation Failed: {str(e)}")

        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        MIN_SPEED = .8
        MAX_SPEED = 1.5
        
        os.makedirs("temp/audio_chunks", exist_ok=True)
        os.makedirs("temp/adjAudio_chunks", exist_ok=True)

        # TTS, speed/slow audio, and add silences
        for i in range(len(timeSentences)):
            # TTS
            tts.tts_to_file(text=timeSentences[i][4],
                            file_path=f"temp/audio_chunks/{i}.wav",
                            speaker_wav=f"temp/speakers_audio/{timeSentences[i][3]}.wav",
                            language=targ,
                            emotion=timeSentences[i][5],
                            speed=1.0)  # Generate at normal speed
            audio = AudioSegment.from_wav(f"temp/audio_chunks/{i}.wav")

            # speed/slow audio
            adjAudioFile = f"temp/adjAudio_chunks/{i}.wav"
            trans_dur = len(audio)
            orig_dur = (timeSentences[i][2] - timeSentences[i][1]) * 1000
            
            # Handle zero duration case
            if orig_dur <= 0:
                print(f"Warning: Sentence {i} has zero/negative duration ({orig_dur}ms), skipping adjustment")
                audio.export(adjAudioFile, format="wav")
                continue
            
            speed_factor = trans_dur / orig_dur
            
            if trans_dur < orig_dur:
                if speed_factor < MIN_SPEED: #trans_dur significantly shorter than orig_dur
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
                if speed_factor > MAX_SPEED:
                    speed_factor = MAX_SPEED
                adjAudio = audio.speedup(playback_speed=speed_factor)
                adjAudio.export(adjAudioFile, format="wav")

        # Stitch chunks together
        print("Stitching chunks together")
        final_audio = AudioSegment.empty()
        curPos = 0
        for i in range(len(timeSentences)):
            adjAudioFile = f"temp/adjAudio_chunks/{i}.wav" 
            adjAudio = AudioSegment.from_wav(adjAudioFile)
            orig_start = timeSentences[i][1] * 1000
            pad = orig_start - curPos
            if pad > 0:
                final_audio += AudioSegment.silent(duration=pad)
                print(f"padding: {len(AudioSegment.silent(duration=pad))}ms")
                curPos = orig_start
            final_audio += adjAudio
            curPos += len(adjAudio)
        
        final_audio.export("temp/final_audio.wav", format="wav")

        separator = Separator()

        # Load a model
        separator.load_model(model_filename='2_HP-UVR.pth')
        background_path = separator.separate("temp/orig_video.mp4")[0]
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
    result = pipeline.dub("https://www.youtube.com/watch?v=jIZkKsf6VYo", "en", "zh", os.getenv('HF_TOKEN'), True, False)
    print(f"dubbed video path: {result}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 


