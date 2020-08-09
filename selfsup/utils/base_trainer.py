import torch
import os
from selfsup.utils.path_manager import PathManager
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class BaseTrainer():
    r"""Base class Trainer. All trainers should inherit from this class."""
    def __init__(self, config):
        self.config = config
        
        # Create output folders
        output_path = os.path.join(config["outputs_path"], config["method"], config["experiment_name"])
        self.path_manager = PathManager(output_path)
        
        # Summary writer
        now = datetime.now()
        dt_string = now.strftime("%d_%m-%H_%M_%S")
        self.writer = SummaryWriter(log_dir=os.path.join(self.path_manager.logs_path, dt_string))
        
        # Set device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")
        
    def train(self):
        pass