import requests

ELEVENLAB = "https://elevenlabs.io/app/home"
ELEVENLAB_API = "USE YOUR API KEY"
VOICE_ID = {"Xb7hH8MSUJpSbSDYk0k2"}


import io
import wave
import numpy as np
from openai import OpenAI
import os
api_key = "USE YOUR API KEY"

client = OpenAI(api_key=api_key)

def create_wav_bytes_from_pcm(audio_data, sample_rate=16000):
    """Create WAV bytes from raw PCM 16-bit mono audio data (numpy array or list)."""
    audio_pcm = np.array(audio_data, dtype=np.int16)

    with io.BytesIO() as wav_buffer:

        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_pcm.tobytes())
        wav_buffer.seek(0)
        return wav_buffer.read()

def stt_streaming(audio_data):
    """
    Convert raw PCM 16-bit mono audio data bytes (or numpy array) to
    WAV and transcribe using OpenAI Whisper API.

    Returns transcription text.
    """

    
    wav_bytes = create_wav_bytes_from_pcm(audio_data, sample_rate=16000)

    
    transcription = client.audio.transcriptions.create(
        file=io.BytesIO(wav_bytes),
        model="whisper-1",
        response_format="text"
    )

    return transcription

def tts_stream(text: str) -> bytes:
    r = requests.post(
        f"{OPENAI_API}/text-to-speech/{VOICE_ID}", 
        headers={"xi-api-key": OPENAI_KEY},
        json={"text": text, "voice_settings": {"stability": 0.5}}
    )
    r.raise_for_status()
    return r.content
