import os
import re
import hashlib
import edge_tts

AUDIO_DIR = "media/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

def sanitize_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", text.strip())[:50]

def get_audio_path(text: str) -> str:
    filename = sanitize_filename(text) + ".mp3"
    return os.path.join(AUDIO_DIR, filename)

async def generate_tts_audio(text: str) -> str:
    path = get_audio_path(text)
    if not os.path.exists(path):
        communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
        await communicate.save(path)
    return path
