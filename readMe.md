Work in progress
## To Do
- [] Figure out servers and how to connect to GPU to speed things up

## Some Design Decisions
- translated sentences are mapped back to segments via word proportion. That is, if a segment had 25% of sentence A's words and 30% of sentence B's words, after the translation map back, that same segment would have 25% of translated sentence A's words and 30% of translated sentence B's words. An alternative to this is time proportion (i.e. trying to match 25% of sentence A's total audio time)

- Whisper is better at recognizing speech than (even paid) pyannote is at diarizing it. Thus sometimes there were speech segments with no speaker attached to it. I 

- note to install TTS, do pip install coqui-tts