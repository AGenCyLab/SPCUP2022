"""
Downloads the pretrained checkpoints. The directory structure is as follows:

checkpoints/
├── tssdnet
│   ├── inc_tssdnet_with_unseen
│   │   └── last.ckpt
│   ├── inc_tssdnet_with_unseen_aug
│   │   └── last.ckpt
│   ├── res_tssdnet_with_unseen
│   │   └── last.ckpt
│   └── res_tssdnet_with_unseen_aug
│       └── last.ckpt
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
