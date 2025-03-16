import torch
from transformers import pipeline

audio = "sample.mp3"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-tamil-medium", chunk_length_s=30, device=device)
transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")

print('Transcription: ', transcribe(audio)["text"])