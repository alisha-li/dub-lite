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
            'words': segment.words,
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
    speakers_with_segments = get_speakers_with_segments(segments_with_speakers)

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
    sorted_sentences = sorted(all_sentences, key=lambda x: x['start'])
    return sorted_sentences

def translate(sentence, before_context, after_context, targ, groq_api: str, gemini_api: str):
    if groq_api:
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
        return translation
    elif gemini_api:
        return "Nothing yet, gemini api not implemented"
    else:
        return "Nothing yet, no api key provided"

def classify_emotion(audio_path: str):
    classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
    out_prob, score, index, text_lab = classifier.classify_file(audio_path)
    return text_lab[0]

def assign_sentences_to_segments(sorted_sentences: list, segments: list):
    '''
    Adds segments property with list of segments to each sentence. 
    'segments' --> (segment_index, proportion_of_sentence_in_segment)
    '''

    speakers_with_segments = get_speakers_with_segments(segments)
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


#### Helper Functions ####
def get_speakers_with_segments(segments_with_speakers: list):
    speakers_with_segments = defaultdict(list)
    for i, segment in enumerate(segments_with_speakers):
        speakers_with_segments[segment['speaker']].append((i, segment))
    return speakers_with_segments