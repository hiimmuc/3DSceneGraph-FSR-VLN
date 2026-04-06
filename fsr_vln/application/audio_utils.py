"""Audio utilities for HoloAgent — STT via faster-whisper, TTS via piper-tts.

All heavy imports are guarded so the Streamlit app loads even if these
libraries are not installed (features are gracefully disabled).
"""

import io
import logging
import os
import tempfile
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# STT — faster-whisper
# ---------------------------------------------------------------------------


def is_whisper_available() -> bool:
    """Return True if faster-whisper is installed."""
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _get_whisper_model(model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
    """Load and cache the WhisperModel singleton."""
    from faster_whisper import WhisperModel

    logger.info("Loading faster-whisper model '%s' on %s (%s)...", model_size, device, compute_type)
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_audio(
    audio_bytes: bytes,
    language: Optional[str] = None,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
) -> str:
    """Transcribe raw audio bytes to text using faster-whisper.

    Args:
        audio_bytes: Raw audio data (WAV, MP3, OGG, or any ffmpeg-supported format).
        language: Optional language hint (e.g. ``"en"``). Auto-detected if None.
        model_size: Whisper model size (tiny/base/small/medium/large-v2).
        device: Inference device (``"cpu"`` or ``"cuda"``).
        compute_type: Quantization type (``"int8"``, ``"float16"``, ``"float32"``).

    Returns:
        Transcribed text string, or empty string on failure.
    """
    if not is_whisper_available():
        logger.warning("faster-whisper not installed — STT unavailable.")
        return ""

    try:
        model = _get_whisper_model(model_size, device, compute_type)

        suffix = ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        kwargs = {"beam_size": 5}
        if language:
            kwargs["language"] = language

        segments, _ = model.transcribe(tmp_path, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text
    except Exception as e:
        logger.error("STT transcription error: %s", e)
        return ""
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# TTS — piper-tts
# ---------------------------------------------------------------------------


def is_piper_available() -> bool:
    """Return True if piper-tts is installed."""
    try:
        from piper import PiperVoice  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _get_piper_voice(voice_model: str, use_cuda: bool = False):
    """Load and cache the PiperVoice singleton."""
    from piper import PiperVoice

    logger.info("Loading piper voice from '%s' (CUDA=%s)...", voice_model, use_cuda)
    return PiperVoice.load(voice_model, use_cuda=use_cuda)


def synthesize_speech(text: str, voice_model: str = "", use_cuda: bool = False) -> Optional[bytes]:
    """Synthesize text to speech and return raw WAV bytes.

    Args:
        text: Text to synthesize.
        voice_model: Path to the piper ``.onnx`` voice model file.
        use_cuda: Whether to use GPU for inference.

    Returns:
        WAV audio bytes, or None if TTS is unavailable or synthesis fails.
    """
    if not is_piper_available():
        logger.warning("piper-tts not installed — TTS unavailable.")
        return None

    if not voice_model:
        logger.warning("No piper voice model specified — TTS unavailable.")
        return None

    if not text.strip():
        return None

    try:
        import wave

        voice = _get_piper_voice(voice_model, use_cuda)
        buf = io.BytesIO()

        with wave.open(buf, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

        return buf.getvalue()
    except Exception as e:
        logger.error("TTS synthesis error: %s", e)
        return None
