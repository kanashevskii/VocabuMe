import os
import hashlib
import re
import edge_tts

AUDIO_DIR = "media/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

def sanitize_filename(text: str) -> str:
    """Sanitize a string so it's safe for filenames."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", text.strip())[:80]

def get_audio_path(text: str) -> str:
    """
    Return path for a TTS file.
    - Primary: sanitized word-based filename to enable reuse.
    - Fallback: hashed filename for legacy files; we check both.
    """
    word_filename = f"{sanitize_filename(text)}.mp3"
    word_path = os.path.join(AUDIO_DIR, word_filename)

    hashed = hashlib.sha1(text.encode("utf-8")).hexdigest()
    hashed_path = os.path.join(AUDIO_DIR, f"{hashed}.mp3")

    # If legacy hashed exists, prefer it, else use word-based; if both exist, keep hashed to avoid duplicates.
    if os.path.exists(hashed_path):
        return hashed_path
    return word_path

async def generate_tts_audio(text: str) -> str:
    path = get_audio_path(text)
    if not os.path.exists(path):
        communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
        await communicate.save(path)
    return path
