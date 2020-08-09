import yaml


def load_config(config_path):
    r"""Loads config file and returns a dictionary."""
    with open(config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config
