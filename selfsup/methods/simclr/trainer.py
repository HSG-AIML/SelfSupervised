# import python libraries
import numpy as np

# import torch libraries
import torch
import torch.optim as optim
import torch.nn.functional as F

# import project libraries
from selfsup.utils.base_trainer import BaseTrainer
from selfsup.methods.simclr.models.resnet import ResNet
from selfsup.methods.simclr.loss import NTXentLoss
from selfsup.methods.simclr.datasets.coco import get_coco_dataloaders

# define trainer class
class Trainer(BaseTrainer):

    # define class constructor
    def __init__(self, config):

        # call super class constructor
        super(Trainer, self).__init__(config)

        # init optimization criterion
        self.criterion = NTXentLoss(self.device, self.config["batch_size"], **self.config["loss"])

        # init data loaders
        self._get_dataloaders()

        # init the backbone model
        self.model = ResNet(**self.config["model"]).to(self.device)

        # init the optimizer TODO: Fixed init learning rate? Maybe flexible hyperparameter?
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=3e-4, weight_decay=eval(self.config['weight_decay']))

        # init the learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=len(self.train_loader), eta_min=0, last_epoch=-1)

    # define data loader initializations
    def _get_dataloaders(self):

        r"""Gets dataloaders for train and validation."""

        # case: coco2014 dataset selected
        if self.config["dataset"]["dataset_name"] == "coco2014":

            # init the coco2014 data loaders
            self.train_loader, self.valid_loader = get_coco_dataloaders(self.config)

        # case: unknown dataset selected
        else:

            # raise runtime error
            raise RuntimeError("Dataset is not defined.")

    # define single simclr training step
    def _simclr_step(self, xis, xjs):

        r"""Applies model to an input pair and returns the loss."""

        # obtain representations and projection feature vectors
        ris, zis = self.model(xis)  # [N,C]
        rjs, zjs = self.model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        
        # compute contrastive loss
        loss = self.criterion(zis, zjs)

        # return contrastive loss
        return loss

    # define simclr training process TODO: Maybe rename to "training_epoch"?
    def train(self):

        # init iteration count
        n_iter = 0

        # TODO: Remove, since not used?
        valid_n_iter = 0

        # init best validation loss
        best_valid_loss = np.inf

        # iterate over training epochs
        for epoch_counter in range(self.config['epochs']):

            # iterate over training mini batches
            for (xis, xjs) in self.train_loader:

                # reset optimizer gradients
                self.optimizer.zero_grad()

                # push mini batch data to compute device
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                # run single simclr training iteration
                loss = self._simclr_step(xis, xjs)

                # case: logging step reached
                if n_iter % self.config['log_every_n_steps'] == 0:

                    # record training loss
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                    # log training iteration and corresponding loss
                    print(f"Step {n_iter}, loss: {loss.item()}")

                # run backward pass
                loss.backward()

                # optimize model parameter
                self.optimizer.step()

                # increase iteration count
                n_iter += 1

            # case: validation epoch reached
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:

                # run model validation
                valid_loss = self._validate_epoch()

                # case: improved model found
                if valid_loss < best_valid_loss:

                    # reset best validation loss
                    best_valid_loss = valid_loss

                    # save model parameters TODO: import os library?, What's about saving also the optimizer and learning rate?
                    torch.save(self.model.state_dict(), os.path.join(self.path_manager.checkpoints_path, 'best_model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=n_iter)

            # warmup for the first n epochs TODO: Maybe flexible hyperparameter?
            if epoch_counter >= 10:

                # update learning rate TODO: Should be self.scheduler.step()?
                scheduler.step()

            # record the actual learning rate TODO: Should be self.scheduler.get_lr()?
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    # define single simclr validation epoch
    def _validate_epoch(self):

        # ignore gradients
        with torch.no_grad():

            # set model into validation mode
            self.model.eval()

            # init validation loss
            valid_loss = 0.0

            # iterate over training mini batches
            for (xis, xjs) in self.valid_loader:

                # push mini batch data to compute device
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                # run single simclr validation iteration
                loss = self._simclr_step(xis, xjs)

                # collect and accumulate validation loss
                valid_loss += loss.item()

            # normalize validation loss by number of mini batches
            valid_loss /= len(self.valid_loader)

        # set model into training mode
        self.model.train()

        # return validation loss
        return valid_loss