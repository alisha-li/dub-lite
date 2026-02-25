#!/usr/bin/env python3
"""Inspect temp/segments_merged.pkl - prints everything in the pickle."""
import pickle


def _format_word(w):
    if hasattr(w, "word"):
        return {"word": w.word, "start": getattr(w, "start", None), "end": getattr(w, "end", None)}
    return str(w)


def main():
    try:
        with open("temp/segments_merged.pkl", "rb") as f:
            data = pickle.load(f)

        print("=" * 80)
        print("segments_merged.pkl")
        print("=" * 80)
        print(f"Type: {type(data)}, Total segments: {len(data)}\n")

        for i, seg in enumerate(data):
            print(f"Segment {i}:")
            if isinstance(seg, dict):
                for k, v in seg.items():
                    if k == "words" and v:
                        print(f"  words: [{len(v)} items]")
                        for j, w in enumerate(v[:5]):
                            print(f"    [{j}] {_format_word(w)}")
                        if len(v) > 5:
                            print(f"    ... and {len(v) - 5} more")
                    else:
                        print(f"  {k}: {v}")
            else:
                print(f"  {seg}")
            print()
    except FileNotFoundError:
        print("temp/segments_merged.pkl not found. Run the pipeline first.")


if __name__ == "__main__":
    main()
