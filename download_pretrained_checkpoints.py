"""
Downloads the pretrained checkpoints. The directory structure is as follows:

checkpoints/
├── gmm
│   ├── cqcc_with_aug
│   │   ├── attributes.json
│   │   ├── model
│   │   │   ├── config.json
│   │   │   └── parameters.pt
│   │   └── params.json
│   ├── cqcc_without_aug
│   │   ├── attributes.json
│   │   ├── model
│   │   │   ├── config.json
│   │   │   └── parameters.pt
│   │   └── params.json
│   ├── mfcc_wihtout_aug
│   │   ├── attributes.json
│   │   ├── model
│   │   │   ├── config.json
│   │   │   └── parameters.pt
│   │   └── params.json
│   └── mfcc_with_aug
│       ├── attributes.json
│       ├── model
│       │   ├── config.json
│       │   └── parameters.pt
│       └── params.json
├── resnet18
│   ├── resnet18_on_train_dataset
│   │   └── last.ckpt
│   └── resnet18_on_train_dataset_augmented
│       └── last.ckpt
├── resnet34
│   ├── resnet34_on_train_dataset
│   │   └── last.ckpt
│   └── resnet34_on_train_dataset_augmented
│       └── last.ckpt
├── svm
│   ├── without_unseen
│   │   └── svm-03-12-2022-20-46-01-0.05.pkl
│   ├── with_unseen
│   │   └── last.pkl
│   └── with_unseen_aug
│       └── last.pkl
├── tssdnet
│   ├── inc_tssdnet_with_unseen
│   │   └── last.ckpt
│   ├── inc_tssdnet_with_unseen_aug
│   │   └── last.ckpt
│   ├── res_tssdnet_with_unseen
│   │   └── last.ckpt
│   └── res_tssdnet_with_unseen_aug
│       └── last.ckpt
└── vgg16
    ├── vgg16_on_train_dataset
    │   └── last.ckpt
    └── vgg16_on_train_dataset_augmented
        └── last.ckpt
"""

import sys
import pathlib

ROOT = pathlib.Path(__file__).absolute().parent
sys.path.append(str(ROOT))

import zipfile
import os
from utils.dataset import download_file


CHECKPOINT_DOWNLOAD_PATH = ROOT.joinpath("checkpoints")
CHECKPOINTS_URL = "https://onedrive.live.com/download?cid=7AC29D06407D53E6&resid=7AC29D06407D53E6%21108&authkey=AAzzpqUBTLhdXQc"
ZIP_FILENAME = "checkpoints.zip"


if __name__ == "__main__":
    zip_file_path = str(CHECKPOINT_DOWNLOAD_PATH.joinpath(ZIP_FILENAME))
    os.makedirs(CHECKPOINT_DOWNLOAD_PATH, exist_ok=True)

    download_file(CHECKPOINTS_URL, CHECKPOINT_DOWNLOAD_PATH, ZIP_FILENAME)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(str(CHECKPOINT_DOWNLOAD_PATH))

    os.remove(zip_file_path)
