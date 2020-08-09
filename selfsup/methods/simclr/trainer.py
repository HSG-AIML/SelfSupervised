import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from selfsup.utils.base_trainer import BaseTrainer
from selfsup.methods.simclr.models.resnet import ResNet
from selfsup.methods.simclr.loss import NTXentLoss
from selfsup.methods.simclr.datasets.coco import get_coco_dataloaders


class Trainer(BaseTrainer):
    def __init__(self, config):
        super(Trainer, self).__init__(config)
        # Criterion
        self.criterion = NTXentLoss(self.device, self.config["batch_size"], **self.config["loss"])
        # Dataloaders
        self._get_dataloaders()
        # Model and optimizer
        self.model = ResNet(**self.config["model"]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), 3e-4, 
                                    weight_decay=eval(self.config['weight_decay']))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                             T_max=len(self.train_loader), 
                                                             eta_min=0,
                                                             last_epoch=-1)

    def _get_dataloaders(self):
        r"""Gets dataloaders for train and validation."""
        if self.config["dataset"]["dataset_name"] == "coco2014":
            self.train_loader, self.valid_loader = get_coco_dataloaders(self.config)
        else:
            raise RuntimeError("Dataset is not defined.")

    def _simclr_step(self, xis, xjs):
        r"""Applies model to an input pair and returns the loss."""
        # Get the representations and the projections
        ris, zis = self.model(xis)  # [N,C]
        rjs, zjs = self.model(xjs)  # [N,C]

        # Normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        
        # Compute loss
        loss = self.criterion(zis, zjs)
        return loss

    def train(self):
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        for epoch_counter in range(self.config['epochs']):
            for (xis, xjs) in self.train_loader:
                self.optimizer.zero_grad()
                # Transfer data to self.device
                xis, xjs = xis.to(self.device), xjs.to(self.device)

                loss = self._simclr_step(xis, xjs )
                
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(f"Step {n_iter}, loss: {loss.item()}")
                    
                loss.backward()

                self.optimizer.step()
                n_iter += 1

            # Validate the model
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate_epoch()
                if valid_loss < best_valid_loss:
                    # Save the best model weights
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), os.path.join(self.path_manager.checkpoints_path, 'best_model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=n_iter)

            # Warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()

            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)
    
    def _validate_epoch(self):
        # Validation steps
        with torch.no_grad():
            self.model.eval()
            valid_loss = 0.0
            for (xis, xjs) in self.valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._simclr_step(xis, xjs)
                valid_loss += loss.item()
            valid_loss /= len(self.valid_loader)
        self.model.train()
        return valid_loss