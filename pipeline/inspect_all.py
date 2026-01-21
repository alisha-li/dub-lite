#!/usr/bin/env python3
"""Quick script to inspect all pickle files"""
import pickle
import json

print("\n" + "="*80)
print("SPEAKER TURNS (first 10)")
print("="*80)
try:
    with open("temp/speaker_turns.pkl", "rb") as f:
        speaker_turns = pickle.load(f)
    
    print(f"Total turns: {len(speaker_turns)}\n")
    for i, (times, speaker) in enumerate(speaker_turns.items()):
        print(f"{speaker}: {times[0]:.2f}s - {times[1]:.2f}s")
except FileNotFoundError:
    print("speaker_turns.pkl not found")

print("\n" + "="*80)
print("WHISPER SEGMENTS (first 10)")
print("="*80)
try:
    with open("temp/segments.pkl", "rb") as f:
        segments = pickle.load(f)
    
    print(f"Total segments: {len(segments)}\n")
    for i, seg in enumerate(segments):
        if i >= 10:
            print(f"... and {len(segments) - 10} more segments")
            break
        if hasattr(seg, 'text'):
            # Whisper segment object - print text and timing
            print(f"Segment {i}:")
            print(f"  text: {seg.text}")
            print(f"  start: {seg.start}")
            print(f"  end: {seg.end}")
except FileNotFoundError:
    print("segments.pkl not found")

print("\n" + "="*80)
print("FINAL SENTENCES (first 10)")
print("="*80)
try:
    with open("temp/final_sentences.pkl", "rb") as f:
        sentences = pickle.load(f)
    
    print(f"Total sentences: {len(sentences)}\n")
    for i, sent in enumerate(sentences):
        if i >= 20:
            print(f"... and {len(sentences) - 10} more sentences")
            break
        print(f"Sentence {i}:")
        for key, value in sent.items():
            print(f"  {key}: {value}")
        print()
except FileNotFoundError:
    print("final_sentences.pkl not found")

print("\n" + "="*80)
print("FINAL SEGMENTS (first 10)")
print("="*80)
try:
    with open("temp/final_segments.pkl", "rb") as f:
        segments = pickle.load(f)
    
    print(f"Total segments: {len(segments)}\n")
    
    # Find segments with issues (no speaker or empty translation)
    no_speaker = [i for i, s in enumerate(segments) if s.get('speaker') is None]
    empty_trans = [i for i, s in enumerate(segments) if not s.get('translation') or s.get('translation').strip() == ""]
    
    if no_speaker:
        print(f"⚠️  Segments with NO SPEAKER: {no_speaker[:10]}")
    if empty_trans:
        print(f"⚠️  Segments with EMPTY TRANSLATION: {empty_trans[:10]}")
    print()
    
    for i, seg in enumerate(segments[:20]):
        print(f"Segment {i}")
        for key, value in seg.items():
            if key == 'words':
                continue
            print(f"  {key}: {value}")
        print()
except FileNotFoundError:
    print("final_segments.pkl not found")
