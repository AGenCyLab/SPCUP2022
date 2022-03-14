import soundfile as sf
import numpy as np


def read_audio_file(audio_path: str, duration: float = 6.0) -> np.ndarray:
    """
        Reads a given audio file. Slices it up or trims it down if the length 
        is not exactly equal to self.duration. By default, max duration is 6 
        seconds according to the paper
        """
    audio, sample_rate = sf.read(audio_path)

    # padding
    if len(audio) < duration * sample_rate:
        audio = np.tile(audio, int((duration * sample_rate) // len(audio)) + 1)

    # trim
    audio = audio[0 : (int(duration * sample_rate))]
    audio = np.expand_dims(audio, axis=0)

    return np.asarray(audio, dtype=np.float32)

