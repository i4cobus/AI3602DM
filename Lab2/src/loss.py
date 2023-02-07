import torch
import torch.nn as nn
from torch import Tensor


class NegativeSamplingLoss():
    def __init__(self, tot_num: int, n_negs: int, EPS: float = 1e-7):
        """
        Negatvie Sampling Loss computer.
        Create an instance of this object and directly call it.

        Args:
            tot_num (int): Number of classes
            n_negs (int): Number of samples per negative sampling process
            EPS (float, optional): For numerical stability.
                Defaults to 1e-7.
        """
        self.EPS = EPS
        self.n_samples = tot_num
        self.n_negs = n_negs

    def __call__(
            self,
            model: nn.Module,
            preds: Tensor,
            batch_sz: int) -> Tensor:
        """
        Given a model and the prediction of current batch,
        randomly samples negative examples and compute approximate loss

        Args:
            model (Module): The inference model
            preds (Tensor): Prediction for current batch.
                Expects the shape to be BND,
                where N is the length of a neighbouring sequence,
                D is the dimension of embedding
            batch_sz (int): Size of current batch

        Returns:
            Negative sampling loss.
        """
        DEVICE = preds.device
        window_sz = preds.shape[1] // 2

        cur = preds[:, window_sz, :]
        ctx = torch.cat(
            (preds[:, :window_sz, :], preds[:, window_sz + 1:, :]),
            axis=1)

        # generate negative samples
        negs = torch.randint(
            self.n_samples,
            (batch_sz, 2*window_sz, self.n_negs))
        negs = negs.reshape(batch_sz, -1).to(DEVICE)

        neg_pred = model.forward(negs)

        neg_pred = neg_pred.reshape(
            batch_sz, 2*window_sz, self.n_negs, -1)

        # negative sampling loss
        pos_prob = torch.einsum(
            'ij, ikj -> ik', cur, ctx)
        neg_prob = torch.einsum(
            'ij, iklj -> ikl', cur, neg_pred)
        pos_prob = torch.log(torch.sigmoid(pos_prob) + self.EPS)
        neg_prob = torch.mean(
            torch.log(torch.sigmoid(neg_prob) + self.EPS), axis=-1)

        loss = - torch.sum((pos_prob - neg_prob))
        loss /= batch_sz

        return loss
