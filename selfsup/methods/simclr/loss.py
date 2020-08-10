# import python libraries
import numpy as np

# import torch libraries
import torch


# define contrastive loss clas
class NTXentLoss(torch.nn.Module):

    # define class constructor
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):

        # call super class constructor
        super(NTXentLoss, self).__init__()

        # set compute device
        self.device = device

        # set mini batch size
        self.batch_size = batch_size

        # set loss temperature
        self.temperature = temperature

        # set loss softmax
        self.softmax = torch.nn.Softmax(dim=-1)

        # set same sample mask
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)

        # set similarity function
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)

        # set loss criterion
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    # define get similarity function
    def _get_similarity_function(self, use_cosine_similarity):

        # case: cosine similarity enabled
        if use_cosine_similarity:

            # init cosine similarity
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

            # return cosine similarity
            return self._cosine_simililarity

        # case: non cosine similarity enabled
        else:

            # return dot similarity
            return self._dot_simililarity

    # define get correlation mask
    def _get_correlated_mask(self):

        # define diagonal mask
        diag = np.eye(2 * self.batch_size)

        # define lower diagonal mask
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)

        # define upper diagonal mask
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)

        # merge diagonal, lower diagonal, and upper diagonal mask
        mask = torch.from_numpy((diag + l1 + l2))

        # invert mask and convert to boolean
        mask = (1 - mask).type(torch.bool)

        # push mask to compute device
        mask = mask.to(self.device)

        # return correlated mask
        return mask

    # define dot similarity
    def _dot_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)

        # compute dot similarity
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)

        # return dot similarity
        return v

    # define cosine similarity
    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)

        # compute cosine similarity
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))

        # return cosine similarity
        return v

    # define loss forward pass
    def forward(self, zis, zjs):

        # stack representations row-wise
        representations = torch.cat([zjs, zis], dim=0)

        # compute similarity matrix based on either dot or cosine similarity
        similarity_matrix = self.similarity_function(representations, representations)

        # determine positive sample scores
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)

        # concatenate positive samples scores
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        # compute masked similarity matrix of negative scores based on either dot or cosine similarity
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        # concatenate positive and negative sample scores
        logits = torch.cat((positives, negatives), dim=1)

        # scale logits by temperature
        logits /= self.temperature

        # init zero labels and push to compute device
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()

        # determine cross entropy between logits and labels TODO: add dimensionality
        loss = self.criterion(logits, labels)

        # normalize loss by batch size
        loss = loss / (2 * self.batch_size)

        # return contrastive loss
        return loss
