# download_video_and_extract_audio
import os
import yt_dlp
from pydub import AudioSegment

# diarize_audio
from pyannoteai.sdk import Client
from pyannote.audio import Pipeline as PyannotePipeline

# split_speakers_and_denoise
from denoise_audio import denoise_audio

# create_sentences
from collections import defaultdict
from nltk.tokenize import sent_tokenize
import logging
logger = logging.getLogger(__name__)

# translate
from groq import Groq

# classify_emotion
from speechbrain.inference.interfaces import foreign_class

# adjust_audio
import librosa
import soundfile as sf

# combine_audio_with_background
import subprocess

def download_video_and_extract_audio(
    source, 
    video_path="temp/orig_video.mp4", 
    audio_path="temp/orig_audio.wav"
):
    """
    Downloads a video from URL or uses a local file, then extracts audio as WAV.
    
    Args:
        source: YouTube/video URL OR local file path
        video_path: Path to save the downloaded video (ignored if source is local file)
        audio_path: Path to save the extracted audio
        
    Returns:
        tuple: (video_path, audio_path, AudioSegment object)
    """
    # Ensure temp directory exists
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    
    # Check if source is a local file or URL
    if os.path.isfile(source):
        print(f"Using local video file: {source}")
        video_path = source
    else:
        print(f"Downloading video from URL: {source}")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        ydl_opts = {
            'format': 'best',
            'outtmpl': video_path,
            # 'cookiesfrombrowser': ('chrome',) # will still work for most videos, but not age-restricted, private, or member-only
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([source])
    
    # Extract audio
    print(f"Extracting audio to: {audio_path}")
    orig_audio = AudioSegment.from_file(video_path, format="mp4")
    orig_audio.export(audio_path, format="wav")
    
    return video_path, audio_path, orig_audio


def diarize_audio(audio_path: str, pyannote_key: str, hf_token: str):
    if pyannote_key: # paid
        client = Client(pyannote_key)
        orig_audio_url = client.upload(audio_path)
        diarization_job = client.diarize(orig_audio_url, transcription=True)
        diarization = client.retrieve(diarization_job)

        turns = diarization['output']['turnLevelTranscription']
        speaker_turns = {}

        for turn in turns:
            start = turn['start']
            end = turn['end']
            speaker = turn['speaker']
            
            if abs(end - start) > 0.2:
                speaker_turns[(start, end)] = speaker
                logger.info(f"Speaker {speaker}: from {start}s to {end}s")

    else:  #free
        logger.info("Running speaker diarization (this may take several minutes)...")
        diarizationPipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        diarization = diarizationPipeline(audio_path)
        speaker_turns = {}
    
        for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
            if abs(speech_turn.end - speech_turn.start) > .2:
                logger.info(f"Speaker {speaker}: from {speech_turn.start}s to {speech_turn.end}s")
                speaker_turns[(speech_turn.start, speech_turn.end)] = speaker
        
        logger.info("Diarization completed")
    
    return speaker_turns


def split_speakers_and_denoise(audio: AudioSegment, speaker_turns: dict, output_dir: str = "temp/speakers_audio"):
    # For voice cloning later
    speakers = set(speaker_turns.values())
    for speaker in speakers:
        speaker_audio = AudioSegment.empty()
        for key, value in speaker_turns.items():
            if speaker == value:
                start = int(key[0])*1000  # Convert seconds to milliseconds
                end = int(key[1])*1000
                speaker_audio += audio[start:end]  # Extract this speaker's audio segments
        speaker_audio.export(f"{output_dir}/{speaker}.wav", format="wav")
        if denoise_audio(f"{output_dir}/{speaker}.wav", f"{output_dir}/{speaker}.wav"):
            logger.info("Using denoised audio for voice cloning")
        else:
            logger.info("Warning: Denoising failed, using original audio")
    return output_dir


def assign_speakers_to_segments(segments: list, speaker_turns: dict):
    time_stamped = []
    wordCount = 0
    segments_with_speakers = []
    for segment in segments:
        first_word_idx = wordCount
        for word in segment.words:
            wordCount += 1
            time_stamped.append([word.word, word.start, word.end]) # used later when determining start and end times for sentences
        last_word_idx = wordCount - 1

        start = time_stamped[first_word_idx][1]
        end = time_stamped[last_word_idx][2]

        resSpeaker = None
        max_overlap = 0
        for times, speaker in speaker_turns.items():
            speaker_start = times[0]  # Keep as float!
            speaker_end = times[1]    # Keep as float!
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
                
        segments_with_speakers.append({
            'text': segment.text,
            'words': segment.words,
            'speaker': resSpeaker,
            'start': start,
            'end': end
        })
    return segments_with_speakers

def create_sentences(segments_with_speakers: list):
    """
    Creates a list of sentence objects (sentence, start, end, speaker) 
    from segments with speakers, sorted by start
    """
    # Group segments by speaker before sentence tokenization
    speakers_with_segments = get_speakers_with_segments(segments_with_speakers)

    speakers_with_sentences = defaultdict(list)            
    for speaker, segments in speakers_with_segments.items():
        fullTextList = []
        for i, segment in segments:
            for word in segment['words']:
                fullTextList.append([word.word, word.start, word.end])
        fullTextStr = " ".join([word[0] for word in fullTextList])
        sentences = sent_tokenize(fullTextStr)
        
        word_idx = 0
        for sentence in sentences:
            num_words = len(sentence.split())
            sentence_words = fullTextList[word_idx:word_idx+num_words]
            speakers_with_sentences[speaker].append({
                'speaker': speaker, # this might not be necessary? since we already have speaker key.
                'sentence': sentence,
                'start': sentence_words[0][1], # first word's start
                'end': sentence_words[-1][2] # last word's end
            })
            word_idx += num_words

    all_sentences = []
    for speaker, sentences in speakers_with_sentences.items():
        all_sentences.extend(sentences)  # Add all sentences from this speaker

    # Sort by start time
    sorted_sentences = sorted(all_sentences, key=lambda x: x['start'])
    return sorted_sentences

def translate(sentence, before_context, after_context, targ: str, groq_api: str = None, gemini_api: str = None):
    import time
    if groq_api:
        client = Groq(api_key=groq_api,)
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
        time.sleep(2)  # Wait 2 seconds between requests to avoid rate limits
        return translation
    elif gemini_api:
        return "Nothing yet, gemini api not implemented"
    else:
        return "Nothing yet, no api key provided"

def classify_emotion(audio_path: str):
    try:
        classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
        out_prob, score, index, text_lab = classifier.classify_file(audio_path)
        return text_lab[0]
    except Exception as e:
        logger.warning(f"Emotion classification failed: {e}, using 'neutral'")
        return "neutral"

def assign_sentences_to_segments(sorted_sentences: list, segments_with_speakers: list):
    """
    Adds segments property with list of segments to each sentence. 
    'segments' --> (segment_index, proportion_of_sentence_in_segment)
    """

    speakers_with_segments = get_speakers_with_segments(segments_with_speakers)
    final_sentences = []
    for speaker, speaker_segments in speakers_with_segments.items():
        speaker_sentences = [sentence for sentence in sorted_sentences if sentence['speaker'] == speaker]

        # within speaker indices
        i_seg = 0
        i_seg_word = 0
        i_sent = 0
        # index of segment among all segments (across all speakers)
        i_seg_global, cur_segment = speaker_segments[i_seg]
            
        while i_sent < len(speaker_sentences):
            i_sent_word = 0
            sentence = speaker_sentences[i_sent]
            sentence_words = sentence['sentence'].split()
            speaker_sentences[i_sent]['segments'] = []
            segment_start_word = 0  # for correct proportion later
            
            while i_sent_word < len(sentence_words):
                if i_seg_word < len(cur_segment['words']):
                    i_seg_word += 1
                    i_sent_word += 1
                else:
                    words_in_segment = i_sent_word - segment_start_word # number of words of curSentence in curSegment
                    prop = words_in_segment / len(sentence_words)
                    speaker_sentences[i_sent]['segments'].append((i_seg_global, prop))
                    
                    i_seg += 1
                    i_seg_word = 0
                    i_seg_global, cur_segment = speaker_segments[i_seg]
                    segment_start_word = i_sent_word 

            # if didn't finish a segment, then sentence is part of that segment
            words_in_segment = i_sent_word - segment_start_word
            if words_in_segment > 0:
                prop = words_in_segment / len(sentence_words)
                speaker_sentences[i_sent]['segments'].append((i_seg_global, prop))
            
            if len(speaker_sentences[i_sent]['segments']) == 0:
                speaker_sentences[i_sent]['segments'].append((i_seg_global, 1))
            i_sent += 1
        
        final_sentences.extend(speaker_sentences)
    return sorted(final_sentences, key=lambda x: x['start'])


def map_translated_sentences_to_segments(sorted_sentences: list, segments: list):
    """
    Maps translated sentences to segments based on word proportion.
    Returns: segments with translated text
    """
    for i, segment in enumerate(segments):
        segment['translation'] = []

    for i, sentence_obj in enumerate(sorted_sentences):
        translation_text = sentence_obj['translation']
        
        # For languages without spaces (Chinese, Japanese), split by characters
        # For languages with spaces (English, Spanish), split by words
        if ' ' in translation_text and len(translation_text.split()) > 1:
            # Has spaces - split by words
            words = translation_text.split()
            join_with = " "
        else:
            # No spaces or single word - split by characters
            words = list(translation_text)
            join_with = ""
        
        total_words = len(words)
        word_idx = 0

        for j, (i_seg_global, prop) in enumerate(sentence_obj['segments']):
            if j == len(sentence_obj['segments'])-1:
                num_words = total_words - word_idx # last segment gets remaining words of sentence
            else:
                num_words = round(total_words*prop)
            segment_words = words[word_idx:word_idx + num_words]
            segments[i_seg_global]['translation'].append(" ".join(segment_words))
            word_idx += num_words
        
        # will delete in cleanup
        if not sentence_obj['segments']:
            logger.warning(f"Sentence {i} has no segments: {sentence_obj['sentence']}")

    for segment in segments:
        segment['translation'] = " ".join(segment['translation'])
    return segments


def adjust_audio(segments, MIN_SPEED, MAX_SPEED, orig_audio_len):
    """
    Adjusts translated segment audio to fit in roughly the same time slot of original segment
    Returns: adjAudio
    """
    curDuration = 0
    for i, segment in enumerate(segments):
            print(f"Audio adjusting segment {i}")            
            audio = AudioSegment.from_wav(f"temp/audio_chunks/{i}.wav")
            prev_silence, next_silence, usable_prev_silence, usable_next_silence = calculate_silences(segment, i, segments, orig_audio_len)

            translated_dur = len(audio)
            orig_dur = (segment['end'] - segment['start']) * 1000

            # Handle zero duration case
            if orig_dur <= 0:
                print(f"Warning: Sentence {i} has zero/negative duration ({orig_dur}ms), skipping adjustment")
                audio.export(f"temp/adjAudio_chunks/{i}.wav", format="wav")
                continue
            
            # Calculate accumulated drift (how far behind schedule we are)
            drift_ms = max(0, curDuration - (segment['start'] * 1000))
            logger.info(f"curDuration: {curDuration}")
            logger.info(f"segment start: {segment['start']}")
            logger.info(f"drift_ms: {drift_ms}")
            target_dur = max(orig_dur + usable_prev_silence + usable_next_silence - drift_ms, .01)
            
            if (translated_dur == orig_dur and drift_ms == 0):
                # Exact match and on schedule - just copy
                adjAudio = AudioSegment.silent(duration=usable_prev_silence) + audio + AudioSegment.silent(duration=usable_next_silence)
                speed_factor = 1.0
        
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
                
            elif translated_dur < orig_dur:
                logger.info("translated_dur < orig_dur")
                # Need to slow down 
                speed_factor = translated_dur / orig_dur
                if speed_factor < MIN_SPEED: #translated_dur signif shorter than target_dur
                    speed_factor = MIN_SPEED
                
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

                if os.path.exists(f"temp/audio_chunks/{i}_slowed.wav"):
                    logger.info(f"Removing temp/audio_chunks/{i}_slowed.wav")
                    os.remove(f"temp/audio_chunks/{i}_slowed.wav")


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
            elif i == len(segments) - 1:
                adjAudio = adjAudio + AudioSegment.silent(duration = next_silence-usable_next_silence)
            adjAudio.export(f"temp/adjAudio_chunks/{i}.wav", format="wav")
            curDuration += len(adjAudio)
            
            logger.info(f"durations for segment {i}:")
            logger.info(f"orig duration: {orig_dur}")
            logger.info(f"translated duration: {translated_dur}")
            logger.info(f"transformed duration: {len(adjAudio)}")


def stitch_chunks(segments):
    logger.info("Stitching chunks together")
    final_audio = AudioSegment.empty()
    for i in range(len(segments)):
        adjAudio = AudioSegment.from_wav(f"temp/adjAudio_chunks/{i}.wav")
        final_audio += adjAudio
        logger.info(f"Added chunk {i} ({len(adjAudio)}ms), total so far: {len(final_audio)}ms")

    final_audio.export(f"temp/final_audio.wav", format="wav")
    logger.info(f"Final audio length: {len(final_audio)}ms ({len(final_audio)/1000:.2f}s)")

def overlay_audios(audio1, audio2):
    if len(audio1) > len(audio2):
        audio2 = audio2 + AudioSegment.silent(duration=len(audio1) - len(audio2))
    elif len(audio1) < len(audio2):
        audio1 = audio1 + AudioSegment.silent(duration=len(audio2) - len(audio1))
    combined_audio = audio1.overlay(audio2)
    combined_audio.export("temp/combined_audio.wav", format="wav")
    return "temp/combined_audio.wav"

def combine_audio_with_video(audio_path:str, video_path:str):
    # Combine new audio with original video
    command = [
        'ffmpeg',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',  # Copy video stream without re-encoding (fast)
        '-map', '0:v:0',  # Use video from first input
        '-map', '1:a:0',  # Use audio from second input
        '-shortest',  # End when shortest stream ends
        '-y',  # Overwrite output file if it exists
        'temp/output_video.mp4'
    ]
    logger.info("Combining audio with video using ffmpeg...")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        raise Exception(f"Failed to combine audio with video: {result.stderr}")
    logger.info("Video with dubbed audio saved to temp/output_video.mp4")
    return "temp/output_video.mp4"


############################################################
#                      Helper Functions                    #
############################################################
def get_speakers_with_segments(segments_with_speakers: list):
    speakers_with_segments = defaultdict(list)
    for i, segment in enumerate(segments_with_speakers):
        speakers_with_segments[segment['speaker']].append((i, segment))
    return speakers_with_segments

def calculate_silences(sentence_obj, idx, sentences, orig_audio_len):
    """
    Calculate available silence before/after a sentence.
    Returns: prev_silence, next_silence, usable_prev_silence, usable_next_silence
    """
    if idx == 0:
        prev_silence = sentence_obj['start'] * 1000
        next_silence = (sentences[idx+1]['start'] * 1000) - (sentence_obj['end'] * 1000)
    elif idx == len(sentences) - 1:
        prev_silence = (sentence_obj['start'] * 1000) - (sentences[idx-1]['end'] * 1000)
        next_silence = orig_audio_len - sentence_obj['end'] * 1000
    else:
        prev_silence = (sentence_obj['start'] * 1000) - (sentences[idx-1]['end'] * 1000)
        next_silence = (sentences[idx+1]['start'] * 1000) - (sentence_obj['end'] * 1000)

    usable_prev_silence = min(300, max(prev_silence, 0)) # don't start > 300ms before orig start
    usable_next_silence = max(next_silence - 300, 0) # allocate 300ms for audio after

    return prev_silence, next_silence, usable_prev_silence, usable_next_silence

    # putting old code here in case things go wrong
    # if i == 0:
    #             prev_silence = sentence_obj['start']*1000
    #             next_silence = (sorted_sentences[i+1]['start']*1000) - (sentence_obj['end']*1000)
    #         elif i == len(sorted_sentences) - 1:
    #             prev_silence = (sentence_obj['start']*1000) - (sorted_sentences[i-1]['end']*1000)
    #             next_silence = len(orig_audio) - sentence_obj['end']*1000
    #         else:
    #             prev_silence = (sentence_obj['start']*1000) - (sorted_sentences[i-1]['end']*1000)
    #             next_silence = (sorted_sentences[i+1]['start']*1000) - (sentence_obj['end']*1000)

    #         usable_prev_silence = min(300, max(prev_silence, 0)) # don't start > 300ms before orig start
    #         usable_next_silence = max(next_silence - 300, 0) # allocate 300ms for audio after