import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import datetime as dt
from tqdm import tqdm
from selfsup.utils.base_trainer import BaseTrainer
from selfsup.methods.simclr.models.resnet import ResNet
from selfsup.methods.simclr.loss import NTXentLoss
from selfsup.methods.simclr.datasets.coco import get_coco_dataloaders


class Trainer(BaseTrainer):
    r"""Trainer for SimCLR method."""
    def __init__(self, config):
        # Call super class constructor
        super(Trainer, self).__init__(config)

        # Init optimization criterion
        self.criterion = NTXentLoss(self.device, self.config["batch_size"], **self.config["loss"])

        # Init data loaders
        self._get_dataloaders()

        # Init the backbone model
        if self.config["model"]["base_model"].startswith("resnet"):
            self.model = ResNet(**self.config["model"]).to(self.device)
        else:
            raise RuntimeError(f'Base model {self.config["model"]["base_model"]} not defined.')
        
        # Init the optimizer
        self.optimizer = optim.Adam(params=self.model.parameters(), 
                                    lr=self.config["lr"], 
                                    weight_decay=eval(self.config['weight_decay']))

        # Init the learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, 
                                                              T_max=len(self.train_loader), 
                                                              eta_min=self.config['scheduler_eta_min'], 
                                                              last_epoch=self.config['scheduler_last_epoch'])

    def _get_dataloaders(self):
        r"""Returns data loaders for train and validation."""
        # Case: COCO-2014 dataset
        if self.config["dataset"]["dataset_name"] == "coco2014":
            self.train_loader, self.valid_loader = get_coco_dataloaders(self.config)
        else:
            # Raise error if dataset is defined
            raise RuntimeError("Dataset is not defined.")

    def _simclr_step(self, xis, xjs):
        r"""Sigle SimCLR step: feeds an input pair to the model and returns the loss."""
        # Obtain representations and projection feature vectors
        ris, zis = self.model(xis)  # [N,C]
        rjs, zjs = self.model(xjs)  # [N,C]

        # Normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        
        # Compute contrastive loss
        loss = self.criterion(zis, zjs)

        return loss

    # define simclr training process TODO: Maybe rename to "training_epoch"?
    def train(self):
        r"""Start training of the model."""
        # Init global iteration count
        self.global_itr = 0

        # Init best validation loss with +infinity
        best_valid_loss = np.inf

        # Iterate over training epochs
        for epoch_counter in range(self.config['epochs']):
            # Train the model for one epoch
            self._train_epoch(epoch=epoch_counter)

            # Case: validation epoch reached
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                # Run model validation
                valid_loss = self._validate_epoch(epoch=epoch_counter)
                # Case: improved model found (lower loss the the previous valid loss)
                if valid_loss < best_valid_loss:
                    # Update best validation loss
                    best_valid_loss = valid_loss
                    # Save model parameters
                    torch.save(self.model.state_dict(), os.path.join(self.path_manager.checkpoints_path, 'best_model.pth'))
                    # Save optimizer parameters
                    torch.save(self.optimizer.state_dict(), os.path.join(self.path_manager.checkpoints_path, 'best_optim.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=self.global_itr)

            # Warmup for the first n epochs
            if epoch_counter >= self.config["warmup_epochs"]:
                # Update learning rate 
                self.scheduler.step()

            # Record the actual learning rate
            self.writer.add_scalar('cosine_lr_decay', self.scheduler.get_lr()[0], global_step=self.global_itr)

    def _train_epoch(self, epoch):
        r"""Trains the model for one epoch."""
        # wrap training loader into progress bar
        self.train_loader = tqdm(self.train_loader)

        # Iterate over training mini batches
        for (xis, xjs) in self.train_loader:
            # Reset optimizer gradients
            self.optimizer.zero_grad()

            # Push mini batch data to compute device
            xis = xis.to(self.device)
            xjs = xjs.to(self.device)

            # Run single simclr training iteration
            loss = self._simclr_step(xis, xjs)

            # Log iteration and corresponding loss
            self.train_loader.set_description(
                (f"[INFO {dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')}] SimCLR Model Train :: "
                 f"epoch: {epoch}/{self.config['epochs']}, train-iteration loss: {np.round(loss.item(), 5)}.")
            )

            # Case: logging step reached
            if self.global_itr % self.config['log_every_n_steps'] == 0:
                # Record training loss
                self.writer.add_scalar('train_loss', loss, global_step=self.global_itr)

            # Run backward pass
            loss.backward()

            # Optimize model parameter
            self.optimizer.step()

            # Increase global iteration count
            self.global_itr += 1

    def _validate_epoch(self, epoch):
        r"""Runs model vaidation."""
        print("Validating model ...")

        # wrap validation loader into progress bar
        self.valid_loader = tqdm(self.valid_loader)

        # Ignore gradients
        with torch.no_grad():
            # Set model mode to validation
            self.model.eval()

            # Init validation loss
            valid_loss = 0.0

            # Iterate over training mini batches
            for (xis, xjs) in self.valid_loader:
                # Push mini batch data to compute device
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                # Run single simclr validation iteration
                loss = self._simclr_step(xis, xjs)

                # Log iteration and corresponding loss
                self.train_loader.set_description(
                    (f"[INFO {dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')}] SimCLR Model Valid :: "
                     f"epoch: {epoch}/{self.config['epochs']}, valid-iteration loss: {np.round(loss.item(), 5)}.")
                )

                # Collect and accumulate validation loss
                valid_loss += loss.item()

            # Normalize validation loss by number of mini batches
            valid_loss /= len(self.valid_loader)

        # Set model into training mode
        self.model.train()

        return valid_loss