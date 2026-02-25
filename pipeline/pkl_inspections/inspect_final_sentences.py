#!/usr/bin/env python3
"""Inspect temp/final_sentences.pkl - prints everything in the pickle."""
import pickle


def main():
    try:
        with open("temp/final_sentences.pkl", "rb") as f:
            data = pickle.load(f)

        print("=" * 80)
        print("final_sentences.pkl")
        print("=" * 80)
        print(f"Type: {type(data)}, Total sentences: {len(data)}\n")

        for i, sent in enumerate(data):
            print(f"Sentence {i}:")
            if isinstance(sent, dict):
                for k, v in sent.items():
                    if k == "segments" and v:
                        print(f"  segments: {v}")
                    else:
                        print(f"  {k}: {v}")
            else:
                print(f"  {sent}")
            print()
    except FileNotFoundError:
        print("temp/final_sentences.pkl not found. Run the pipeline first.")


if __name__ == "__main__":
    main()
