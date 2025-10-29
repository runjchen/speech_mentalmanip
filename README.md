# Detecting Mental Manipulation in Speech via Synthetic Multi-Speaker Dialogue

Status: Under review

Anonymous artifact link: [https://anonymous.4open.science/r/speech_mentalmanip-E798/](https://anonymous.4open.science/r/speech_mentalmanip-E798)



## TL;DR
We extend the text-only mental manipulation benchmark into speech by rendering each dialogue as multi-speaker, voice-consistent TTS. 
This enables 1:1 comparisons between text and audio. Models and humans both struggle more on audio, 
highlighting modality-specific ambiguity and the subjectivity of mental manipulation.

## What's in this repo?
```
speech_mentalmanip/
├─ README.md                                       # You are here
├─ multi_speaker_TTS_audios_generation_scripts/    # TTS by turn
├─ prediction/                                     # Model Evaluations
├─ composed_audios_dataset/                        # Composed audio dataset
│  ├─ text_conversations_mental_manipulative_composed_audios_batch_01.zip
│  ├─ text_conversations_mental_manipulative_composed_audios_batch_02.zip
│  └─ text_conversations_non_mental_manipulative_composed_audios.zip
└─ human_annotations/                              # Human annotation data and results
   ├─ annotations_audio_compiled.csv              # Compiled audio annotations
   ├─ annotations_text_compiled.csv               # Compiled text annotations
   ├─ audio_files/                                 # Audio files used for annotation
   └─ template/                                    # Annotation templates
```
