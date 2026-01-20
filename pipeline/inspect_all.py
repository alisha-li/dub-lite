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
        if i >= 10:
            print(f"\n... and {len(speaker_turns) - 10} more turns")
            break
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
            print(f"Segment {i}: {seg.text}")
        else:
            print(f"Segment {i}:")
            for key, value in seg.items():
                if key != 'words':  # Don't print word objects
                    print(f"  {key}: {value}")
        print()
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
        if i >= 10:
            print(f"... and {len(sentences) - 10} more sentences")
            break
        print(f"Sentence {i}:")
        for key, value in sent.items():
            print(f"  {key}: {value}")
        print()
except FileNotFoundError:
    print("final_sentences.pkl not found")
