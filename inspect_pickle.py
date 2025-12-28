#!/usr/bin/env python3
"""Quick script to inspect pickled data"""
import pickle
import pprint

# Load the segments pickle
with open('temp/segments.pkl', 'rb') as f:
    data = pickle.load(f)

print("=" * 60)
print("Available keys:", list(data.keys()))
print("=" * 60)

# Inspect speakers_rolls
print("\n📢 speakers_rolls (first 20 entries):")
print("-" * 60)
speakers_rolls = data['speakers_rolls']
for i, (times, speaker) in enumerate(list(speakers_rolls.items())[:20]):
    print(f"{i+1}. {times} -> {speaker}")
print(f"\nTotal speakers_rolls entries: {len(speakers_rolls)}")

# Inspect speakers
print("\n👥 speakers:")
print("-" * 60)
speakers = data['speakers']
print(f"Speakers: {speakers}")
print(f"Total unique speakers: {len(speakers)}")

# Inspect segments (first few)
print("\n📝 segments (first 3):")
print("-" * 60)
segments = data['segments']
for i, segment in enumerate(segments[:3]):
    print(f"\nSegment {i+1}:")
    print(f"  Start: {segment.start:.2f}s, End: {segment.end:.2f}s")
    print(f"  Text: {segment.text}")
    if hasattr(segment, 'words'):
        print(f"  Words: {len(segment.words)}")

print(f"\nTotal segments: {len(segments)}")

