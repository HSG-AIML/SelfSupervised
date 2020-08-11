import numpy as np
import torch


class NTXentLoss(torch.nn.Module):
    r"""Calculates SimCLR Loss and returns the loss as a scalar."""
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        # Call super class constructor
        super(NTXentLoss, self).__init__()

        # Set compute device
        self.device = device

        # Init parameters and functions for computing loss
        self.batch_size = batch_size
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        r"""Returns similarity function."""
        # Case: cosine similarity enabled
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        # Case: non cosine similarity enabled
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        r"""Returns correlation mask."""
        # Define diagonal mask
        diag = np.eye(2 * self.batch_size)

        # Define lower diagonal mask
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)

        # Define upper diagonal mask
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)

        # Merge diagonal, lower diagonal, and upper diagonal mask
        mask = torch.from_numpy((diag + l1 + l2))

        # Invert mask and convert to boolean
        mask = (1 - mask).type(torch.bool)

        # Push mask to compute device
        mask = mask.to(self.device)

        return mask

    def _dot_simililarity(self, x, y):
        r"""Computes dot similarity between two tensors.
        x shape: (N, 1, C)
        y shape: (1, C, 2N)
        v shape: (N, 2N)
        """
        # Compute dot similarity
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)

        return v

    def _cosine_simililarity(self, x, y):
        r"""Computes consine similarity between two tensors.
        x shape: (N, 1, C)
        y shape: (1, 2N, C)
        v shape: (N, 2N)
        """
        # Compute cosine similarity
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))

        return v

    def forward(self, zis, zjs):
        r"""Computes SimCLR loss.
        Inputs: two batches of tensors where each batch is embeddings
                extracted from random transformations of the same input.
        """
        # Stack representations row-wise
        representations = torch.cat([zjs, zis], dim=0)

        # Compute similarity matrix based on either dot or cosine similarity
        similarity_matrix = self.similarity_function(representations, representations)

        # Determine positive sample scores
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)

        # Concatenate positive samples scores
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        # Compute masked similarity matrix of negative scores based on either dot or cosine similarity
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        # Concatenate positive and negative sample scores
        logits = torch.cat((positives, negatives), dim=1)

        # Scale logits by temperature
        logits /= self.temperature

        # Init zero labels and push to compute device
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()

        # Determine cross entropy between logits and labels TODO: add dimensionality
        loss = self.criterion(logits, labels)

        # Normalize loss by batch size (mutliplied by 2 since we have two batch of inputs)
        loss = loss / (2 * self.batch_size)

        return loss
