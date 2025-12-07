import os
import hashlib
import re
import shutil
import edge_tts
import logging
from gtts import gTTS
from edge_tts.exceptions import NoAudioReceived

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

def _is_valid_audio(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        return False


async def generate_tts_audio(text: str) -> str:
    """
    Generate or return the path of a TTS file named after the text.
    gTTS — основной, edge-tts — резерв.
    """
    if not text or not text.strip():
        raise ValueError("Empty text for TTS")

    plain_path = get_plain_path(text)
    hashed_path = get_hashed_path(text)
    if not os.path.exists(plain_path) and os.path.exists(hashed_path):
        os.rename(hashed_path, plain_path)

    # if we already have a valid file — reuse
    if _is_valid_audio(plain_path):
        return plain_path

    # Primary: gTTS (быстрее/стабильнее в нашей среде)
    last_err = None
    try:
        tts = gTTS(text)
        tts.save(plain_path)
        if _is_valid_audio(plain_path):
            logging.info("gTTS generated audio for %s", text)
            return plain_path
    except Exception as exc:  # noqa: BLE001
        last_err = exc
        logging.exception("gTTS failed for %s: %s", text, exc)

    # Fallback: edge-tts с несколькими голосами
    attempts = 2
    voices = ["en-US-AriaNeural", "en-US-JennyNeural", "en-GB-RyanNeural"]
    for voice in voices:
        for _ in range(attempts):
            try:
                communicate = edge_tts.Communicate(text, voice=voice)
                await communicate.save(plain_path)
                if _is_valid_audio(plain_path):
                    logging.info("edge-tts generated audio for %s with %s", text, voice)
                    return plain_path
                logging.warning("Generated empty TTS for %s with %s, retrying", text, voice)
            except NoAudioReceived as exc:
                last_err = exc
                logging.warning("No audio received for %s with %s", text, voice)
            except Exception as exc:  # noqa: BLE001 logging to avoid silent failures
                last_err = exc
                logging.exception("TTS generation failed for %s with %s: %s", text, voice, exc)

    # cleanup empty file so we don't keep sending empty mp3
    if os.path.exists(plain_path) and os.path.getsize(plain_path) == 0:
        try:
            os.remove(plain_path)
        except Exception:
            logging.warning("Failed to remove empty TTS file %s", plain_path)

    if last_err:
        raise last_err
    raise ValueError(f"Failed to generate audio for {text}")


async def generate_temp_audio(text: str) -> str:
    """Return a temporary hashed copy of the TTS file for sending."""
    base_path = await generate_tts_audio(text)
    hashed_path = get_hashed_path(text)
    shutil.copy(base_path, hashed_path)
    return hashed_path
