#!/usr/bin/env python3
"""Run all inspect_*.py scripts and write output to corresponding .txt files."""
import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PIPELINE_DIR = SCRIPT_DIR.parent

INSPECTIONS = [
    "inspect_speaker_turns",
    "inspect_segments",
    "inspect_segments_with_speakers",
    "inspect_segments_merged",
    "inspect_sentences",
    "inspect_final_sentences",
    "inspect_final_segments",
]

def main():
    os.chdir(PIPELINE_DIR)
    for name in INSPECTIONS:
        script = SCRIPT_DIR / f"{name}.py"
        out = SCRIPT_DIR / f"{name.replace('inspect_', '')}.txt"
        if not script.exists():
            print(f"Skip (not found): {script}")
            continue
        print(f"Running {name}.py -> {out.name}")
        with open(out, "w") as f:
            subprocess.run(
                ["python", str(script)],
                cwd=PIPELINE_DIR,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
    print("Done.")

if __name__ == "__main__":
    main()
