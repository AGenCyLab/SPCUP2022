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
