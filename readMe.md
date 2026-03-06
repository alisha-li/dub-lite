# dub-lite

Free/BYOM (bring your own model) dubbing.

**Motivation:** Designed for language learners who want to watch anything while staying immersed. Could also be used by content creators for quick distribution to global audiences. For best translations, use Gemini models, although this is significantly slower than Groq or free TranslateGemma.

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

## Recommendations: 
Works better the less overlapping speakers and background noise there is.

For YouTube downloads I recommend [yt-dlp](https://github.com/yt-dlp/yt-dlp?tab=readme-ov-file#installation).

---

## To Do:

- [ ] Preserve lyrics in music - need some music extraction preprocessing
- [ ] Add cleanup logics for old videos
- [ ] Set up hub where you can view or download batches of video

## Notes and Design Decisions
- In multilingual videos (ones with multiple languages), has a tendency to repeat lines many times. Would only recommend videos with one language.
- Translated sentences are mapped back to segments via word proportion. That is, if a segment had 25% of sentence A's words and 30% of sentence B's words, after the translation map back, that same segment would have 25% of translated sentence A's words and 30% of translated sentence B's words. An alternative to this is time proportion (i.e. trying to match 25% of sentence A's total audio time)
- Whisper is better at recognizing speech than (even paid) pyannote is at diarizing it.
- To install TTS → `pip install coqui-tts`
- Chose Modal since pricing is based on amount of time GPU is actually in use.