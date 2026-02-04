# dub-lite

Free/BYOM (bring your own model) dubbing.

**Motivation:** While learning languages, I restricted myself to only watching content in my target language. But I love YouTube and felt limited by what I wanted to but could not watch. Not state of the art, but allows language learners to continue listening in their target language. For best translations, use Gemini models (although this is significantly slower than Groq models).

Note, this is still a work in progress. Translation, diarization, etc. models are only so good at the moment, so I'm working on engineering around that, while making the processing quicker.

---

## Examples

### Dubbing Example

| Original Video | Dubbed Video |
| -------------- | ------------ |
| <video src="https://github.com/user-attachments/assets/3575c600-285c-4e70-abfe-b4830b08440a"></video> | <video src="https://github.com/user-attachments/assets/f66f66f2-8210-42a7-a8a0-703d2cd1cbd3"></video> |

### Demo Process

The following video demonstrates the complete dubbing workflow:

<video src="https://github.com/user-attachments/assets/0ece70f1-72ff-463c-a668-19b831ec2e8b" width="100"></video>

---

## Roadmap

- [ ] Launch at dub-lite.alishali.info
- [ ] Allow video saves (so that users can dump a bunch of videos and come back later to view)
- [ ] Connect to GPUs for TTS, diarization, etc. speedups
- [ ] Add orig caption option

## Notes and Design Decisions

- Translated sentences are mapped back to segments via word proportion. That is, if a segment had 25% of sentence A's words and 30% of sentence B's words, after the translation map back, that same segment would have 25% of translated sentence A's words and 30% of translated sentence B's words. An alternative to this is time proportion (i.e. trying to match 25% of sentence A's total audio time)
- Whisper is better at recognizing speech than (even paid) pyannote is at diarizing it.
- To install TTS → `pip install coqui-tts`
