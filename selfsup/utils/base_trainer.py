import os
from selfsup.utils.path_manager import PathManager


class BaseTrainer():
    r"""Base class Trainer. All trainers should inherit from this class."""
    def __init__(self, config):
        output_path = os.path.join(config["outputs_path"], config["method"], config["experiment_name"])
        self.path_manager = PathManager(output_path)
        
    def train(self):
        pass