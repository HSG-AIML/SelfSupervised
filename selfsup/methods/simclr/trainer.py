from selfsup.utils.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, config):
        super(Trainer, self).__init__(config)
        print("Hello I'm SimCLR trainer.")

    def train(self):
        print("SimCLR is training now.")