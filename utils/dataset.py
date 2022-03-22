import yaml
import os
import zipfile
import requests
import pathlib
import math
from tqdm import tqdm


def download_file(
    link: str, download_folder_root: pathlib.Path, filename: str
):
    """
    A function for downloading files from links with progress bar
    """
    data_file_path = download_folder_root.joinpath(filename)

    response = requests.get(link, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024
    wrote = 0

    data_file = open(data_file_path, "wb")

    for chunk in tqdm(
        response.iter_content(chunk_size=chunk_size),
        total=math.ceil(total_size // chunk_size),
        unit="KB",
        unit_scale=True,
    ):
        wrote = wrote + len(chunk)
        data_file.write(chunk)

    data_file.close()


class SPCUP22DatasetDownloader:
    """
    Class for downloading the SPCUP22 dataset. Depends on the existence of a
    config yaml file with the following structure:

    spcup22:
        raw_audio:
            training:
                part1:
                    link: https://www.dropbox.com/s/36yqmymkva2bwdi/spcup_2022_training_part1.zip?dl=1
                    filename: spcup_2022_training_part1.zip
                    default_path: data/spcup22/training/part1/spcup_2022_training_part1 
                part2:
                    link: https://www.dropbox.com/s/wsmlthhri29fb79/spcup_2022_unseen.zip?dl=1
                    filename: spcup_2022_unseen.zip
                    default_path: data/spcup22/training/part2/spcup_2022_unseen
            evaluation:
                part1:
                    link: https://www.dropbox.com/s/ftkyvwxgr9wl7jf/spcup_2022_eval_part1.zip?dl=1
                    filename: spcup_2022_eval_part1.zip
                    default_path: data/spcup22/evaluation/spcup_2022_eval_part1

        mel_features:
            training:
                ...
            evaluation:
                ...
    """

    def __init__(
        self,
        config_file_path: str,
        dataset_name: str = "spcup22",
        unzip_after_download: bool = True,
        data_type: str = "raw_audio",
        remove_zip_after_extraction: bool = False,
    ) -> None:
        """
        config_file_path: str
            the path to the dataset.yaml file

        dataset_name: str
            by default it's "spcup22", however, if other datasets are introduced
            we can change the name according to the entry in the dataset.yaml
            file

        data_type: str 
            one of ("raw_audio", "mel_features", ...) (add other types as needed)
            the dataset.yaml file should be updated accordingly
        """
        self.root = pathlib.Path(__file__).parent.parent
        self.config_file_path = config_file_path
        self.dataset_name = dataset_name
        self.remove_zip_after_extraction = remove_zip_after_extraction

        with open(self.config_file_path, mode="r") as yaml_file_object:
            self.config = yaml.load(yaml_file_object, Loader=yaml.FullLoader)[
                dataset_name
            ][data_type]

        self.download_folder_root = pathlib.Path(self.root).joinpath(
            "data", data_type, self.dataset_name
        )

        if not self.download_folder_root.exists():
            os.makedirs(str(self.download_folder_root), exist_ok=True)

        self.unzip_after_download = unzip_after_download

    def download_datasets(self):
        """
        Downloads all the datasets as defined in the dataset.yaml config file
        """
        for dataset_type, dataset_link_data in self.config.items():
            for part_name, part_values in dataset_link_data.items():
                link = part_values["link"]
                filename = part_values["filename"]
                default_path = part_values["default_path"]

                zip_file_path = self.download_folder_root.joinpath(filename)
                extraction_dir = self.download_folder_root.joinpath(
                    default_path
                )

                if zip_file_path.exists():
                    if not extraction_dir.exists():
                        print("Extracting [{}]...".format(zip_file_path))
                        self.unzip_file(zip_file_path, extraction_dir)
                    else:
                        print(
                            "Skipping downloading [{}]...".format(
                                zip_file_path
                            )
                        )
                else:
                    if not extraction_dir.exists():
                        print("Downloading [{}]...".format(link))
                        download_file(
                            link, self.download_folder_root, filename
                        )
                        print("Extracting [{}]...".format(zip_file_path))
                        self.unzip_file(zip_file_path, extraction_dir)

                    print("Skipping downloading [{}]...".format(zip_file_path))

    def unzip_file(self, zip_file_path: str, extraction_dir: str):
        """
        Uzips a zip file given a zip file path and an extraction directory
        """
        print("Unzipping [{}] ...".format(zip_file_path))
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(extraction_dir)

        if self.remove_zip_after_extraction:
            os.remove(zip_file_path)
