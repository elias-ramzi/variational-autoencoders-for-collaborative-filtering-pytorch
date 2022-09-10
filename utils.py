from __future__ import division

import torch
# import numpy as np
import pickle
# import bottleneck as bn


def count_parameters(model):
    """
    this functions returns the learnable parameters of a model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_weights_pkl(fname, weights):
    with open(fname, 'wb') as f:
        pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)


def load_weights_pkl(fname):
    with open(fname, 'rb') as f:
        weights = pickle.load(f)
    return weights


def get_parameters(model, bias=False):
    for k, m in model.named_parameters():
        if bias:
            if k.endswith('.bias'):
                yield m
        else:
            if k.endswith('.weight'):
                yield m


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# borrowed from https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb
# def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
#     '''
#     normalized discounted cumulative gain@k for binary relevance
#     ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
#     '''
#     batch_users = X_pred.shape[0]
#     idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
#     topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
#     idx_part = np.argsort(-topk_part, axis=1)
#     idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
#     # build the discount template
#     tp = 1. / np.log2(np.arange(2, k + 2))
#
#     DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk] * tp).sum(axis=1)
#     IDCG = np.array([(tp[:min(int(n), k)]).sum() for n in np.minimum(k, heldout_batch.sum(axis=1))])
#     return DCG / IDCG


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    # X_pred = torch.from_numpy(X_pred)
    # heldout_batch = torch.from_numpy(heldout_batch)

    idx = X_pred.topk(k, -1).indices
    target = heldout_batch.gather(1, idx)
    positives = heldout_batch.sum(-1)
    recall = target.sum(-1) / torch.where(positives < k, positives, torch.tensor(k, device=X_pred.device).float())
    return recall.cpu().numpy()


def average_precision(X_pred, heldout_batch, k=None,):
    # X_pred = torch.from_numpy(X_pred)
    # heldout_batch = torch.from_numpy(heldout_batch)
    if k is None:
        k = heldout_batch.size(-1)
    idx = X_pred.topk(k, -1).indices
    sorted_target = heldout_batch.gather(1, idx)

    relevance_mask = torch.ones_like(sorted_target)
    if k:
        relevance_mask[:, k:] = 0

    normalizing_factor = (sorted_target * relevance_mask).sum(-1)

    ranks = torch.arange(1, 1 + sorted_target.size(1), device=sorted_target.device)
    pos_ranks = torch.cumsum(sorted_target, -1) * sorted_target * relevance_mask

    ap = (pos_ranks / ranks).sum(-1) / normalizing_factor
    return ap.cpu().numpy()


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=None,):
    # X_pred = torch.from_numpy(X_pred)
    # heldout_batch = torch.from_numpy(heldout_batch)
    if k is None:
        k = heldout_batch.size(-1)
    idx = X_pred.topk(k, -1).indices
    sorted_target = heldout_batch.gather(1, idx)
    gt = heldout_batch.topk(k, -1).values

    ranks = torch.arange(1, sorted_target.size(1) + 1, device=sorted_target.device)

    dcg = (sorted_target / torch.log2(1 + ranks)).sum(-1)
    idcg = (gt / torch.log2(1 + ranks)).sum(-1)

    ndcg = dcg / idcg
    return ndcg.cpu().numpy()
