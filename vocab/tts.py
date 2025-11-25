import os
import hashlib
import re
import shutil
import edge_tts

AUDIO_DIR = "media/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

def sanitize_filename(text: str) -> str:
    """Sanitize a string so it's safe for filenames."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", text.strip())[:80]

def get_plain_path(text: str) -> str:
    """Return path for a TTS file named after the text."""
    filename = sanitize_filename(text) + ".mp3"
    return os.path.join(AUDIO_DIR, filename)


def get_hashed_path(text: str) -> str:
    """Return hashed path for a temporary TTS file."""
    hashed = hashlib.sha1(text.encode("utf-8")).hexdigest()
    filename = f"{hashed}.mp3"
    return os.path.join(AUDIO_DIR, filename)

def get_audio_path(text: str) -> str:
    """
    Return path for a TTS file.
    - Primary: sanitized word-based filename to enable reuse.
    - Fallback: hashed filename for legacy files; we check both.
    """
    plain_path = get_plain_path(text)
    hashed_path = get_hashed_path(text)
    if os.path.exists(plain_path):
        return plain_path
    if os.path.exists(hashed_path):
        return hashed_path
    return plain_path

async def generate_tts_audio(text: str) -> str:
    """Generate or return the path of a TTS file named after the text."""
    plain_path = get_plain_path(text)
    hashed_path = get_hashed_path(text)
    if not os.path.exists(plain_path) and os.path.exists(hashed_path):
        os.rename(hashed_path, plain_path)
    if not os.path.exists(plain_path):
        communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
        await communicate.save(plain_path)
    return plain_path


async def generate_temp_audio(text: str) -> str:
    """Return a temporary hashed copy of the TTS file for sending."""
    base_path = await generate_tts_audio(text)
    hashed_path = get_hashed_path(text)
    shutil.copy(base_path, hashed_path)
    return hashed_path
