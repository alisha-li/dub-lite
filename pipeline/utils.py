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


def diarize_audio(audio_path: str, pyannote_key: str):
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
    speakers = set(list(speaker_turns.values()))
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
        for times, speaker in speaker_turns.items():
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
                
        segments_with_speakers.append({
            'text': segment.text,
            'speaker': resSpeaker,
            'start': start,
            'end': end
        })
    return segments_with_speakers

def create_sentences(segments_with_speakers: list):
    '''
    Creates a list of sentence objects (sentence, start, end, speaker) 
    from segments with speakers, sorted by start
    '''
    # Group segments by speaker before sentence tokenization
    speakers_with_segments = defaultdict(list)
    for segment in segments_with_speakers:
        speakers_with_segments[segment['speaker']].append(segment)

    speakers_with_sentences = defaultdict(list)            
    for speaker, segments in speakers_with_segments.items():
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
    sentences_sorted = sorted(all_sentences, key=lambda x: x['start'])
    return sentences_sorted