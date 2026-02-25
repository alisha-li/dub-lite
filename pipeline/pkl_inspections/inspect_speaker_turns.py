#!/usr/bin/env python3
"""Inspect temp/speaker_turns.pkl - prints everything in the pickle."""
import pickle


def main():
    try:
        with open("temp/speaker_turns.pkl", "rb") as f:
            data = pickle.load(f)

        print("=" * 80)
        print("speaker_turns.pkl")
        print("=" * 80)
        print(f"Type: {type(data)}")
        print(f"Total turns: {len(data)}\n")

        for i, (times, speaker) in enumerate(data.items()):
            print(f"Turn {i}: {speaker}  |  {times[0]:.2f}s - {times[1]:.2f}s")
    except FileNotFoundError:
        print("temp/speaker_turns.pkl not found. Run the pipeline first.")


if __name__ == "__main__":
    main()
