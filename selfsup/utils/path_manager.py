import os


class PathManager():
    r"""
    Manages output folder by initializing them and saving paths
    to each output sub-folder.
    """
    def __init__(self, output_path):
        # Set paths
        self.output_path = output_path
        self.checkpoints_path = os.path.join(self.output_path, "checkpoints")
        self.logs_path = os.path.join(self.output_path, "logs")
        self._creat_folders()

    def _creat_folders(self):
        r"""Creates sub-folders in the output folder."""
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.checkpoints_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
