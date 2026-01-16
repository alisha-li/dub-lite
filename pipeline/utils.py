import yt_dlp
from pydub import AudioSegment
import os

from pyannoteai.sdk import Client
from pyannote.audio import Pipeline as PyannotePipeline

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
    
    return speaker_rolls