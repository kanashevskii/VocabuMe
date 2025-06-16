import os
import hashlib
import re
import edge_tts

AUDIO_DIR = "media/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

def sanitize_filename(text: str) -> str:
    """Sanitize a string so it's safe for filenames."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", text.strip())[:50]

def get_audio_path(text: str) -> str:
    """Return path for a TTS file using a hashed name to hide the word."""
    hashed = hashlib.sha1(text.encode("utf-8")).hexdigest()
    filename = f"{hashed}.mp3"
    hashed_path = os.path.join(AUDIO_DIR, filename)
    # support previously generated files named after the text
    legacy_path = os.path.join(AUDIO_DIR, sanitize_filename(text) + ".mp3")
    if not os.path.exists(hashed_path) and os.path.exists(legacy_path):
        os.rename(legacy_path, hashed_path)
    return hashed_path

async def generate_tts_audio(text: str) -> str:
    path = get_audio_path(text)
    if not os.path.exists(path):
        communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
        await communicate.save(path)
    return path
