import yaml


def load_config_file(config_file_path: str):
    with open(config_file_path, mode="r") as config_file_object:
        config = yaml.load(config_file_object, Loader=yaml.FullLoader)
        return config

