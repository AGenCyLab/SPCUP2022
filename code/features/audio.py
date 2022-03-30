import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
sys.path.append(str(ROOT))

import torch
import librosa


class MFCC(object):
    """MFCC.
    Args:
        sr: [int]
            sampling rate

        hop_length: [int]
            number of samples between successive frames

        n_mfcc: [int]
            number of MFCCs to return
    """

    def __init__(self, sr=16000, hop_length=256, n_mfcc=20):
        self.sr = sr
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc

    def __call__(self, sample):
        y, label = sample
        sr, hop_length, n_mfcc = self.sr, self.hop_length, self.n_mfcc
        # Compute MFCC features from the raw signal
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc
        )

        return mfcc, label


class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
    Not sure if you need this.
    """

    def __call__(self, sample):
        y, label = sample
        y = torch.from_numpy(y)
        return y, label
