try:
    from .model import XHMossTranscribeDiarizeModel
    from .encoder_model import XHMossTranscribeDiarizeEncoder
except ModuleNotFoundError as error:
    if error.name != "xhquant":
        raise
    XHMossTranscribeDiarizeModel = None
    XHMossTranscribeDiarizeEncoder = None
from .host_fusion import (
    AUDIO_WINDOW_SAMPLES,
    AUDIO_WINDOW_SECONDS,
    SAMPLING_RATE,
    SupportedUtteranceExceededError,
    fuse_embeddings,
    load_supported_utterance,
    transcribe_supported_utterance,
)

__all__ = [
    "XHMossTranscribeDiarizeEncoder",
    "XHMossTranscribeDiarizeModel",
    "AUDIO_WINDOW_SAMPLES",
    "AUDIO_WINDOW_SECONDS",
    "SAMPLING_RATE",
    "SupportedUtteranceExceededError",
    "fuse_embeddings",
    "load_supported_utterance",
    "transcribe_supported_utterance",
]
