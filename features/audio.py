import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
sys.path.append(str(ROOT))

CQCC_PATH = ROOT.joinpath("features", "cqcc")
sys.path.append(str(CQCC_PATH))

import torch
import librosa
from features.cqcc.CQCC.cqcc import cqcc


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


class CQCC(object):
    def __init__(
        self,
        num_coeffs_to_keep: int = 256,
        fs: int = 16000,
        B: int = 96,
        d: int = 16,
        cf: int = 19,
        ZsdD: str = "ZsdD",
    ):
        self.num_coeffs_to_keep = num_coeffs_to_keep
        self.fs = fs
        self.B = B
        self.fmax = self.fs / 2
        self.fmin = self.fmax / 2 ** 9
        self.d = d
        self.cf = cf
        self.ZsdD = ZsdD

    def __call__(self, sample):
        y, label = sample
        y = y.reshape(-1, 1)

        cqcc_feature, _, _, _, _, _, _ = cqcc(
            y,
            self.fs,
            self.B,
            self.fmax,
            self.fmin,
            self.d,
            self.cf,
            self.ZsdD,
        )

        if self.num_coeffs_to_keep == -1:
            return cqcc_feature, label

        return cqcc_feature[: self.num_coeffs_to_keep, :], label


class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
    Not sure if you need this.
    """

    def __call__(self, sample):
        y, label = sample
        y = torch.from_numpy(y)
        return y, label


# Call to your training function
# transform = transforms.Compose([
#     MFCC(params1),
#     MelSpectogram(params2),
#     ToTensor()
# ])
# # somewhere in the train.py file
# data_module = SPCUP22DataModule(
#         training_config["training"]["batch_size"],
#         dataset_root=pathlib.Path("./data/spcup22").absolute(),
#         config_file_path=args.dataset_config_file_path,
#         should_include_unseen_in_training_data=args.include_unseen_in_training_data,
#         should_load_eval_data=args.load_eval_data,
#         transform=transform,
#    )
