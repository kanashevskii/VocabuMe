import os
import hashlib
import re
import shutil
import edge_tts
import logging
from edge_tts.exceptions import NoAudioReceived

try:
    from gtts import gTTS
except ImportError:  # pragma: no cover - depends on runtime env
    gTTS = None

AUDIO_DIR = "media/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

VOICE_BY_LANGUAGE = {
    "en": ["en-US-AriaNeural", "en-US-JennyNeural", "en-GB-RyanNeural"],
    "ka": ["ka-GE-EkaNeural", "ka-GE-GiorgiNeural"],
}


def normalize_tts_language(text: str, language_code: str | None = None) -> str:
    value = (language_code or "").strip().lower()
    if value in VOICE_BY_LANGUAGE:
        return value
    if re.search(r"[\u10A0-\u10FF]", text):
        return "ka"
    return "en"


def sanitize_filename(text: str) -> str:
    """Sanitize a string so it's safe for filenames."""
    normalized = re.sub(r"\s+", "_", text.strip(), flags=re.UNICODE)
    normalized = re.sub(r"[^\w]", "_", normalized, flags=re.UNICODE)
    normalized = normalized.strip("_")[:40] or "audio"
    return normalized

def get_plain_path(text: str, language_code: str | None = None) -> str:
    """Return path for a TTS file named after the text."""
    lang = normalize_tts_language(text, language_code)
    suffix = hashlib.sha1(f"{lang}:{text}".encode("utf-8")).hexdigest()[:10]
    filename = f"{lang}_{sanitize_filename(text)}_{suffix}.mp3"
    return os.path.join(AUDIO_DIR, filename)


def get_hashed_path(text: str, language_code: str | None = None) -> str:
    """Return hashed path for a temporary TTS file."""
    lang = normalize_tts_language(text, language_code)
    hashed = hashlib.sha1(f"{lang}:{text}".encode("utf-8")).hexdigest()
    filename = f"{hashed}.mp3"
    return os.path.join(AUDIO_DIR, filename)

def get_audio_path(text: str, language_code: str | None = None) -> str:
    """
    Return path for a TTS file.
    - Primary: sanitized word-based filename to enable reuse.
    - Fallback: hashed filename for legacy files; we check both.
    """
    plain_path = get_plain_path(text, language_code=language_code)
    hashed_path = get_hashed_path(text, language_code=language_code)
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


async def generate_tts_audio(text: str, language_code: str | None = None) -> str:
    """
    Generate or return the path of a TTS file named after the text.
    gTTS — основной, edge-tts — резерв.
    """
    if not text or not text.strip():
        raise ValueError("Empty text for TTS")

    lang = normalize_tts_language(text, language_code)
    plain_path = get_plain_path(text, language_code=lang)
    hashed_path = get_hashed_path(text, language_code=lang)
    if not os.path.exists(plain_path) and os.path.exists(hashed_path):
        os.rename(hashed_path, plain_path)

    # if we already have a valid file — reuse
    if _is_valid_audio(plain_path):
        return plain_path

    # Primary: gTTS when installed.
    last_err = None
    if gTTS is not None:
        try:
            tts = gTTS(text, lang=lang)
            tts.save(plain_path)
            if _is_valid_audio(plain_path):
                logging.info("gTTS generated audio for %s", text)
                return plain_path
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logging.exception("gTTS failed for %s: %s", text, exc)
    else:
        logging.warning("gTTS is not installed, falling back to edge-tts for %s", text)

    # Fallback: edge-tts с несколькими голосами
    attempts = 2
    voices = VOICE_BY_LANGUAGE.get(lang, VOICE_BY_LANGUAGE["en"])
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


async def generate_temp_audio(text: str, language_code: str | None = None) -> str:
    """Return a temporary hashed copy of the TTS file for sending."""
    base_path = await generate_tts_audio(text, language_code=language_code)
    hashed_path = get_hashed_path(text, language_code=language_code)
    shutil.copy(base_path, hashed_path)
    return hashed_path
