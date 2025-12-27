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


class YTDubPipeline:
    def __init__(self):
        os.makedirs("temp", exist_ok=True)
        os.makedirs("temp/speakers_audio", exist_ok=True)

    def dub(self, url: str, src: str, targ: str, hf_token: str):
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

        # 2. Speaker Diarization
        try:
            # Check if we already have segments saved
            segments_file = "temp/segments.pkl"
            if os.path.exists(segments_file):
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
                wordCount += len(sentence_words)
                
                timeSentences.append([sentence, time_stamped[wordCount-1][1], time_stamped[wordCount-1][2]])

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

            for i in range(len(timeSentences)):
                sentence, start, end = timeSentences[i]
                
                # Check overlap with each speaker's time range
                for times, speaker in speakers_rolls.items():
                    speaker_start =  int(times[0])
                    speaker_end = int(times[1])
                    if speaker_start > end or speaker_end < start:
                        break
                        
                    max_overlap = 0
                    overlap_start = max(start, speaker_start)
                    overlap_end = min(end, speaker_end)
                    overlap = overlap_end - overlap_start
                    
                    # Update speaker if this overlap is greater than previous ones
                    if overlap > max_overlap:
                        max_overlap = overlap
                        timeSentences[i].append(speaker)

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


            # if os.path.exists("temp/timeSentences.pkl"):
            #     with open("temp/timeSentences.pkl", "rb") as f:
            #         timeSentences = pickle.load(f)
            # else:
            #     for i in range(len(timeSentences)):
            #         sentence, start, end = timeSentences[i]
                    
            #         # Check overlap with each speaker's time range
            #         for times, speaker in speakers_rolls.items():
            #             speaker_start =  int(times[0])
            #             speaker_end = int(times[1])
            #             if speaker_start > end or speaker_end < start:
            #                 break
                            
            #             max_overlap = 0
            #             overlap_start = max(start, speaker_start)
            #             overlap_end = min(end, speaker_end)
            #             overlap = overlap_end - overlap_start
                        
            #             # Update speaker if this overlap is greater than previous ones
            #             if overlap > max_overlap:
            #                 max_overlap = overlap
            #                 timeSentences[i].append(speaker)

            #         if i == 0:
            #             translatedSentence = translate(timeSentences[i][0], "", timeSentences[i+1][0], targ)
            #         elif i == len(timeSentences) - 1:
            #             translatedSentence = translate(timeSentences[i][0], timeSentences[i-1][0], "", targ)
            #         else:
            #             translatedSentence = translate(timeSentences[i][0], timeSentences[i-1][0], timeSentences[i+1][0], targ)
                    
            #         timeSentences[i].append(translatedSentence)
            
            #         classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")

            #         start=start*1000
            #         end=end*1000

            #         try:
            #             os.makedirs("temp/emotions_audio", exist_ok=True)
            #             audio[start:end].export("temp/emotions_audio/emotions.wav", format="wav")      
            #             out_prob, score, index, text_lab = classifier.classify_file("temp/emotions_audio/emotions.wav")
            #             os.remove("temp/emotions_audio/emotions.wav")
            #         except Exception as e:
            #             print(f"Error classifying emotions: {str(e)}")
            #             text_lab = ['None']
                    
            #         timeSentences[i].append(text_lab[0])
                
            #     # Save timeSentences to pickle file
            #     with open("temp/timeSentences.pkl", "wb") as f:
            #         pickle.dump(timeSentences, f)

            # print("First 20 sentences with emotions")
            # for i in range(20):
            #     print(str(timeSentences[i]) + "\n")

        except Exception as e:
            raise Exception(f"Translation Failed: {str(e)}")

        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        tts.tts_to_file(text=timeSentences[i][0],
                        file_path=f"audio_chunks/{i}.wav",
                        speaker_wav=f"speakers_audio/SPEAKER_0{timeSentences[i][4]}.wav",
                        language=targ,
                        emotion=timeSentences[i][5],
                        speed=2)

        return 'temp/orig_video.mp4'  
    


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--url', required=True)
    # args = parser.parse_args()

    pipeline = YTDubPipeline()
    result = pipeline.dub("https://www.youtube.com/watch?v=jIZkKsf6VYo", "en", "zh", os.getenv('HF_TOKEN'))
    print(f"dubbed video path: {result}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 


