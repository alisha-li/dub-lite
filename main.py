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
    
import os
import yt_dlp
from pyannote.audio import Pipeline as PyannotePipeline
from pydub import AudioSegment
from faster_whisper import WhisperModel
import nltk
from nltk.tokenize import sent_tokenize
import pickle
from transformers import MarianTokenizer, MarianMTModel


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
                print("Loading existing segments from file...")
                with open(segments_file, "rb") as f:
                    segments = pickle.load(f)
                print(f"Loaded {len(segments)} segments from file!")
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
        
        # 3. 
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
            
            # Save segments to file for future use
            print("Saving segments to file...")
            with open(segments_file, "wb") as f:
                pickle.dump(segments, f)
            print(f"Segments saved to {segments_file}")

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

            print(timeSentences)
            
        except Exception as e:
              raise Exception(f"STT Failed: {str(e)}")

        # 4. Translate
        try:
            model_name = f"Helsinki-NLP/opus-mt-{self.src}-{self.targ}"

            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name).to('cpu')

            def translate(sentence):
                if self.source_language == 'tr':
                    model_name = f"Helsinki-NLP/opus-mt-trk-{self.target_language}"
                elif self.target_language == 'tr':
                    model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-trk"
                elif self.source_language == 'zh-cn':
                    model_name = f"Helsinki-NLP/opus-mt-zh-{self.target_language}"
                elif self.target_language == 'zh-cn':
                    model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-zh"
                else:
                    model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-{self.target_language}"

                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name).to('cpu')

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
        except Exception as e:
            raise Exception(f"Translation Failed: {str(e)}")

        return 'temp/orig_video.mp4'  # For now, just return the downloaded video
    


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--url', required=True)
    # args = parser.parse_args()

    pipeline = YTDubPipeline()
    result = pipeline.dub("https://www.youtube.com/watch?v=jIZkKsf6VYo", "es", os.getenv('HF_TOKEN'))
    print(f"dubbed video path: {result}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 


