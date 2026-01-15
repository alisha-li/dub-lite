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
from collections import defaultdict

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
from pyannoteai.sdk import Client
import librosa
import soundfile as sf
import numpy as np
from denoise_audio import denoise_audio
from utils import download_video_and_extract_audio
import logging
log_path = os.path.join("temp", "pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode = 'w'),
        logging.StreamHandler()
    ]
)

class YTDubPipeline:
    def __init__(self):
        os.makedirs("temp", exist_ok=True)
        os.makedirs("temp/speakers_audio", exist_ok=True)

    def dub(self, url: str, src: str, targ: str, hf_token: str, speakerRollsPkl: bool, turnsFinalPkl: bool, segmentsPickle: bool, timeSentencePickle: bool, pyannote: bool, gemini_api: str, groq_api: str):
        # 1. Download video using yt-dlp  
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

        # 1.1 Denoise audio (Produced worse results when I was testing)
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

        # 2. Speaker Diarization and Transcription
        if speakerRollsPkl:
            logging.info("Loading pyannote pickle...")
            with open("temp/speaker_rolls.pkl", "rb") as f:
                speaker_rolls = pickle.load(f)
            logging.info(f"Loaded {len(speaker_rolls)} speaker rolls from file!")
        elif pyannote: # paid
            client = Client("***REMOVED***")
            orig_audio_url = client.upload(orig_audio_path)
            diarization_job = client.diarize(orig_audio_url, transcription=True)
            diarization = client.retrieve(diarization_job)

            turns = diarization['output']['turnLevelTranscription']
            speaker_rolls = {}

            for turn in turns:
                start = turn['start']
                end = turn['end']
                speaker = turn['speaker']
                
                if abs(end - start) > 0.2:
                    speaker_rolls[(start, end)] = speaker
                    logging.info(f"Speaker {speaker}: from {start}s to {end}s")

        else:  #free
            logging.info("Running speaker diarization (this may take several minutes)...")
            diarizationPipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
            diarization = diarizationPipeline(audio_file)
            speaker_rolls = {}
        
            for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
                if abs(speech_turn.end - speech_turn.start) > .2:
                    print(f"Speaker {speaker}: from {speech_turn.start}s to {speech_turn.end}s")
                    speaker_rolls[(speech_turn.start, speech_turn.end)] = speaker
            
            speakers = set(list(speaker_rolls.values()))  # Get unique speaker IDs
            logging.info("Diarization completed")

        

        with open("temp/speaker_rolls.pkl", "wb") as f:
            pickle.dump(speaker_rolls, f)
        # just reorganizing to have pyannote in one section and whisper in another, with pickles at least for pyannote maybe whisper too
        
         # Extract speaker audios and denoise (for voice cloning later)
        logging.info("Extracting speaker audios and denoising...")
        unique_speakers = set(turn['speaker'] for turn in turns)
        for speaker in unique_speakers:
            speaker_audio = AudioSegment.empty()
            for turn in turns:
                if turn['speaker'] == speaker:
                    start_ms = int(turn['start'] * 1000)  # Convert to milliseconds
                    end_ms = int(turn['end'] * 1000)
                    speaker_audio += orig_audio[start_ms:end_ms]
            speaker_file = f"temp/speakers_audio/{speaker}.wav"
            speaker_audio.export(speaker_file, format="wav")
            if denoise_audio(speaker_file, speaker_file):
                logging.info("Using denoised audio for voice cloning")
            else:
                logging.info("Warning: Denoising failed, using original audio")
            logging.info(f"Extracted {len(speaker_audio)}ms of audio for {speaker}")
        logging.info(f"Extracted audio for {len(unique_speakers)} speakers")

        if pyannote:
            # 2.1 Paid Pyannote Orchestration (Diarization + Transcription)
            if pyannotePkl:
                logging.info("Loading pyannote pickle...")
                with open("temp/orchestration.pkl", "rb") as f:
                    turns = pickle.load(f)
                logging.info(f"Loaded {len(turns)} turns from file!")
            else:
                # 2.1.1 Orchestration
                client = Client("***REMOVED***")
                orig_audio_url = client.upload(orig_audio_path)
                diarization_job = client.diarize(orig_audio_url, transcription=True)
                diarization = client.retrieve(diarization_job)

                turns = diarization['output']['turnLevelTranscription']
                speakers_rolls = {}

                for turn in turns:
                    start = turn['start']
                    end = turn['end']
                    speaker = turn['speaker']
                    
                    if abs(end - start) > 0.2:
                        speakers_rolls[(start, end)] = speaker
                        logging.info(f"Speaker {speaker}: from {start}s to {end}s")

                # just thinking about how to pickle this. . . should it be shared?
                with open("temp/diarization.pkl", "wb") as f:
                    pickle.dump(turns, f)
            
                # Extract speaker audios and denoise (for voice cloning later)
                logging.info("Extracting speaker audios and denoising...")
                unique_speakers = set(turn['speaker'] for turn in turns)
                for speaker in unique_speakers:
                    speaker_audio = AudioSegment.empty()
                    for turn in turns:
                        if turn['speaker'] == speaker:
                            start_ms = int(turn['start'] * 1000)  # Convert to milliseconds
                            end_ms = int(turn['end'] * 1000)
                            speaker_audio += orig_audio[start_ms:end_ms]
                    speaker_file = f"temp/speakers_audio/{speaker}.wav"
                    speaker_audio.export(speaker_file, format="wav")
                    if denoise_audio(speaker_file, speaker_file):
                        logging.info("Using denoised audio for voice cloning")
                    else:
                        logging.info("Warning: Denoising failed, using original audio")
                    logging.info(f"Extracted {len(speaker_audio)}ms of audio for {speaker}")
                logging.info(f"Extracted audio for {len(unique_speakers)} speakers")
            
            sentenceDict = defaultdict(list)
            for turn in turns:
                #hmm i dont know the start and end times of each word here, only the start and end times of the segments....

        else: 
            # 2.2 Free Pyannote Diarization
            segments_file = "temp/segments.pkl"
            audio_file = "temp/orig_audio.wav"
            if segmentsPickle:
                logging.info("Loading segments pickle...")
                with open(segments_file, "rb") as f:
                    saved_data = pickle.load(f)
                segments = saved_data['segments']
                speakers_rolls = saved_data['speakers_rolls']
                speakers = saved_data['speakers']
                logging.info(f"Loaded {len(segments)} segments and {len(speakers)} speakers from file!")
            else:
                # 2.2.1 Diarization
                logging.info("Running speaker diarization (this may take several minutes)...")
                diarizationPipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
                diarization = diarizationPipeline(audio_file)
                speakers_rolls = {}
            
                for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
                    if abs(speech_turn.end - speech_turn.start) > .2:
                        print(f"Speaker {speaker}: from {speech_turn.start}s to {speech_turn.end}s")
                        speakers_rolls[(speech_turn.start, speech_turn.end)] = speaker
                
                speakers = set(list(speakers_rolls.values()))  # Get unique speaker IDs

                logging.info("Diarization completed")


                
                audio = AudioSegment.from_file("temp/orig_video.mp4", format="mp4")
                audio.export(audio_file, format="wav")

                # Extract speaker audios and denoise (for voice cloning later)
                for speaker in speakers:
                    speaker_audio = AudioSegment.empty()
                    for key, value in speakers_rolls.items():
                        if speaker == value:
                            start = int(key[0])*1000  # Convert seconds to milliseconds
                            end = int(key[1])*1000
                            
                            speaker_audio += audio[start:end]  # Extract this speaker's audio segments
                    speaker_audio.export(f"temp/speakers_audio/{speaker}.wav", format="wav")
                    if denoise_audio(f"temp/speakers_audio/{speaker}.wav", f"temp/speakers_audio/{speaker}.wav"):
                        logging.info("Using denoised audio for voice cloning")
                    else:
                        logging.info("Warning: Denoising failed, using original audio")
                
                # 2.2.2 Transcription
                logging.info("Running Whisper Transcription...")
                model = WhisperModel("medium", device="cpu", compute_type="int8")
                segments, info = model.transcribe("temp/orig_audio.wav", word_timestamps=True)
                segments = list(segments) 
                logging.info(f"Transcription completed! Found {len(segments)} segments")

                # Save all data to file for future use
                logging.info("Saving segments and speaker data to file...")
                data_to_save = {
                    'segments': segments,
                    'speakers_rolls': speakers_rolls,
                    'speakers': speakers
                }
                with open(segments_file, "wb") as f:
                    pickle.dump(data_to_save, f)
                logging.info(f"All data saved to {segments_file}")
            

            # Assign speaker to each segment
            time_stamped = []
            wordCount = 0
            newSegments = []
            for i, segment in enumerate(segments):
                first_word_idx = wordCount
                for i, word in enumerate(segment.words):
                    wordCount += 1
                    time_stamped.append([word.word, word.start, word.end]) #will be usedlater when determining start and end times for sentences
                last_word_idx = wordCount - 1

                start = time_stamped[first_word_idx][1]
                end = time_stamped[last_word_idx][2]

                resSpeaker = None
                max_overlap = 0
                for times, speaker in speakers_rolls.items():
                    speaker_start =  int(times[0])
                    speaker_end = int(times[1])
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
                     
                newSegments.append({
                    'speaker': resSpeaker,
                    'start': start,
                    'end': end,
                    'text': segment.text
                })

            # combine segments by speaker
            speakerSegments = defaultdict(list)
            for segment in newSegments:
                speakerSegments[segment['speaker']].append(segment)

            sentenceDict = defaultdict(list)            
            for speaker, segments in speakerSegments.items():
                fullTextList = []
                for segment in segments:
                    for word in segment.words:
                        fullTextList.append([word.word, word.start, word.end])
                fullTextStr = " ".join(fullTextList)
                sentences = sent_tokenize(fullTextStr)
                
                word_idx = 0
                for sentence in sentences:
                    num_words = len(sentence.split())
                    sentence_words = fullTextList[word_idx:word_idx+num_words]
                    sentenceDict[speaker].append({
                        'speaker': speaker,
                        'sentence': sentence,
                        'start': sentence_words[0][1], # first word's start
                        'end': sentence_words[-1][2] # last word's end
                    })
                    word_idx += num_words

        all_sentences = []
        for speaker, sentences in sentenceDict.items():
            all_sentences.extend(sentences)  # Add all sentences from this speaker

        # Sort by start time
        all_sentences_sorted = sorted(all_sentences, key=lambda x: x['start'])

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
            logging.info(f"curDuration: {curDuration}")
            logging.info(f"turn start: {turn['start']}")
            logging.info(f"drift_ms: {drift_ms}")
            target_dur = max(orig_dur + usable_prev_silence + usable_next_silence - drift_ms, .01)
            
            if (translated_dur == orig_dur and drift_ms == 0):
                # Exact match and on schedule - just copy
                adjAudio = AudioSegment.silent(duration=usable_prev_silence) + audio + AudioSegment.silent(duration=usable_next_silence)
        
            elif (orig_dur < translated_dur < target_dur):
                logging.info("orig_dur < translated_dur < target_dur")
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
                logging.info("translated_dur >= target_dur")
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
                logging.info("translated_dur < orig_dur")
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
                    os.remove("temp/audio_chunks/{i}_slowed.wav")

                # if os.path.exists(temp_output):
                #     os.remove(temp_output)
                # If still shorter after slowing, pad with silence
                if len(adjAudio) < orig_dur:
                    adjAudio = adjAudio + AudioSegment.silent(duration=int(orig_dur - len(adjAudio)))
                adjAudio = AudioSegment.silent(duration=usable_prev_silence) + adjAudio + AudioSegment.silent(duration=usable_next_silence)

            logging.info(f"speed factor: {speed_factor}")
            logging.info(f"target dur: {target_dur}")
            logging.info(f"adjusted audio length: {len(adjAudio)}")
            logging.info(f"adjAudio duration: {len(adjAudio)}")
            logging.info(f"usable_prev_silence: {usable_prev_silence}")
            logging.info(f"usable_next_silence: {usable_next_silence}")
            logging.info(f"next_silence: {next_silence}")
            logging.info(f"prev_silence: {prev_silence}")
        
            if i == 0:
                adjAudio = AudioSegment.silent(duration = prev_silence-usable_prev_silence) + adjAudio
            elif i == len(turns) - 1:
                adjAudio = adjAudio + AudioSegment.silent(duration = next_silence-usable_next_silence)
            adjAudio.export(adjAudioFile, format="wav")
            curDuration += len(adjAudio)
            
            logging.info(f"durations for turn {i}:")
            logging.info(f"orig duration: {orig_dur}")
            logging.info(f"translated duration: {translated_dur}")
            logging.info(f"transformed duration: {len(adjAudio)}")


        # Stitch chunks together
        logging.info("Stitching chunks together")
        final_audio = AudioSegment.empty()
        for i in range(len(turns)):
            adjAudioFile = f"temp/adjAudio_chunks/{i}.wav"
            adjAudio = AudioSegment.from_wav(adjAudioFile)
            final_audio += adjAudio
            logging.info(f"Added chunk {i} ({len(adjAudio)}ms), total so far: {len(final_audio)}ms")

        final_audio.export("temp/final_audio.wav", format="wav")
        logging.info(f"Final audio length: {len(final_audio)}ms ({len(final_audio)/1000:.2f}s)")

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
    result = pipeline.dub("https://www.youtube.com/watch?v=XXPISZI_big", "en", "zh", os.getenv('HF_TOKEN'), False, False)
    print(f"dubbed video path: {result}")