#!/usr/bin/env python3
"""Inspect temp/segments.pkl - prints everything in the pickle."""
import pickle


def _format_word(w):
    if hasattr(w, "word"):
        return {"word": w.word, "start": getattr(w, "start", None), "end": getattr(w, "end", None)}
    return str(w)


def _format_segment(seg):
    if hasattr(seg, "text"):
        d = {"text": seg.text, "start": seg.start, "end": seg.end}
        if hasattr(seg, "words") and seg.words:
            d["words"] = [_format_word(w) for w in seg.words]
        return d
    return str(seg)


def main():
    try:
        with open("temp/segments.pkl", "rb") as f:
            data = pickle.load(f)

        print("=" * 80)
        print("segments.pkl")
        print("=" * 80)
        print(f"Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}\n")

        if isinstance(data, dict):
            if "language" in data:
                print(f"Language: {data['language']}\n")
            segments = data.get("segments", [])
        else:
            segments = data

        print(f"Total segments: {len(segments)}\n")
        for i, seg in enumerate(segments):
            print(f"Segment {i}:")
            formatted = _format_segment(seg)
            for k, v in formatted.items():
                print(f"  {k}: {v}")
            print()
    except FileNotFoundError:
        print("temp/segments.pkl not found. Run the pipeline first.")


if __name__ == "__main__":
    main()
