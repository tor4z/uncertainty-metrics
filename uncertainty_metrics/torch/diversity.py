# coding=utf-8
# Copyright 2020 The Uncertainty Metrics Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metrics for model diversity."""

import itertools
import torch


def disagreement(logits_1, logits_2):
    """Disagreement between the predictions of two classifiers."""
    preds_1 = torch.argmax(logits_1, dim=-1).type(torch.int32)
    preds_2 = torch.argmax(logits_2, dim=-1).type(torch.int32)
    return torch.mean((preds_1 != preds_2).type(torch.float32))


def double_fault(logits_1, logits_2, labels):
    """Double fault [1] is the number of examples both classifiers predict wrong.

    Args:
      logits_1: tf.Tensor.
      logits_2: tf.Tensor.
      labels: tf.Tensor.

    Returns:
      Scalar double-fault diversity metric.

    ## References

    [1] Kuncheva, Ludmila I., and Christopher J. Whitaker. "Measures of diversity
        in classifier ensembles and their relationship with the ensemble
        accuracy." Machine learning 51.2 (2003): 181-207.
    """
    preds_1 = torch.argmax(logits_1, dim=-1).type(labels.dtype)
    preds_2 = torch.argmax(logits_2, dim=-1).type(labels.dtype)

    res = torch.where(preds_1 != labels)
    res = torch.stack(res).t()
    fault_1_idx = torch.squeeze(res)
    fault_1_idx = fault_1_idx.type(torch.int32)

    preds_2_at_idx = torch.gather(preds_2, fault_1_idx)
    labels_at_idx = torch.gather(labels, fault_1_idx)

    double_faults = preds_2_at_idx != labels_at_idx
    double_faults = double_faults.type(torch.float32)
    return torch.mean(double_faults)


def logit_kl_divergence(logits_1, logits_2):
    """Average KL divergence between logit distributions of two classifiers."""
    probs_1 = torch.softmax(logits_1, dim=-1)
    probs_2 = torch.softmax(logits_2, dim=-1)
    vals = kl_divergence(probs_1, probs_2)
    return torch.mean(vals)


def kl_divergence(p, q, clip=False):
    """Generalized KL divergence [1] for unnormalized distributions.

    Args:
      p: tf.Tensor.
      q: tf.Tensor.
      clip: bool.

    Returns:
      tf.Tensor of the Kullback-Leibler divergences per example.

    ## References

    [1] Lee, Daniel D., and H. Sebastian Seung. "Algorithms for non-negative
    matrix factorization." Advances in neural information processing systems.
    2001.
    """
    if clip:
        p = torch.clamp(p, torch.finfo(torch.float32).eps, 1)
        q = torch.clamp(q, torch.finfo(torch.float32).eps, 1)
    return torch.sum(p * torch.log(p / q), dim=-1)


def lp_distance(x, y, p=1):
    """l_p distance."""
    diffs_abs = torch.abs(x - y)
    summation = torch.sum(torch.pow(diffs_abs, p), dim=-1)
    return torch.mean(torch.pow(summation, 1. / p), dim=-1)


def cosine_distance(x, y):
    """Cosine distance between vectors x and y."""
    x_norm = torch.sqrt(torch.sum(torch.pow(x, 2), dim=-1))
    x_norm = x_norm.view(-1, 1)
    y_norm = torch.sqrt(torch.sum(torch.pow(y, 2), dim=-1))
    y_norm = y_norm.view((-1, 1))
    normalized_x = x / x_norm
    normalized_y = y / y_norm
    return torch.mean(torch.sum(normalized_x * normalized_y, dim=-1))


# TODO(ghassen): we could extend this to take an arbitrary list of metric fns.
def average_pairwise_diversity(probs, num_models, error=None):
    """Average pairwise distance computation across models."""
    if probs.shape[0] != num_models:
        raise ValueError('The number of models {0} does not match '
                         'the probs length {1}'.format(num_models, probs.shape[0]))

    pairwise_disagreement = []
    pairwise_kl_divergence = []
    pairwise_cosine_distance = []
    for pair in list(itertools.combinations(range(num_models), 2)):
        probs_1 = probs[pair[0]]
        probs_2 = probs[pair[1]]
        pairwise_disagreement.append(disagreement(probs_1, probs_2))
        pairwise_kl_divergence.append(
            torch.mean(kl_divergence(probs_1, probs_2)))
        pairwise_cosine_distance.append(cosine_distance(probs_1, probs_2))

    # TODO(ghassen): we could also return max and min pairwise metrics.
    average_disagreement = torch.mean(torch.stack(pairwise_disagreement))
    if error is not None:
        average_disagreement /= (error + torch.finfo(torch.float32).eps)
    average_kl_divergence = torch.mean(torch.stack(pairwise_kl_divergence))
    average_cosine_distance = torch.mean(torch.stack(pairwise_cosine_distance))

    return {
        'disagreement': average_disagreement,
        'average_kl': average_kl_divergence,
        'cosine_similarity': average_cosine_distance
    }


def variance_bound(probs, labels, num_models):
    """Empirical upper bound on the variance for an ensemble model.

    This term was introduced in arxiv.org/abs/1912.08335 to obtain a tighter
    PAC-Bayes upper bound; we use the empirical variance of Theorem 4.

    Args:
      probs: tensor of shape `[num_models, batch_size, n_classes]`.
      labels: tensor of sparse labels, of shape `[batch_size]`.
      num_models: number of models in the ensemble.

    Returns:
      A (float) upper bound on the empirical ensemble variance.
    """
    if probs.shape[0] != num_models:
        raise ValueError('The number of models {0} does not match '
                         'the probs length {1}'.format(num_models, probs.shape[0]))
    batch_size = probs.shape[1]
    labels = labels.type(torch.int32)

    # batch_indices maps point `i` to its associated label `l_i`.
    batch_indices = torch.stack([torch.arange(0, batch_size), labels], dim=1)
    # Shape: [num_models, batch_size, batch_size].
    batch_indices = batch_indices * torch.ones([num_models, 1, 1])

    # Replicate batch_indices across the `num_models` index.
    ensemble_indices = torch.arange(0, num_models).view([num_models, 1, 1])
    ensemble_indices = ensemble_indices * torch.ones([1, batch_size, 1])
    # Shape: [num_models, batch_size, n_classes].
    indices = torch.cat([ensemble_indices, batch_indices], dim=-1)

    # Shape: [n_models, n_points].
    # per_model_probs[n, i] contains the probability according to model `n` that
    # point `i` in the batch has its true label.
    per_model_probs = gather_nd(probs, indices)

    max_probs, _ = torch.max(per_model_probs, dim=0)  # Shape: [n_points]
    avg_probs = torch.mean(per_model_probs, dim=0)  # Shape: [n_points]

    return .5 * torch.mean(
        torch.square((per_model_probs - avg_probs) / max_probs))


def gather_nd(params, indices):
    """params is of "n" dimensions and has size [x1, x2, x3, ..., xn],
    indices is of 2 dimensions  and has size [num_samples, m] (m <= n)
    """
    assert type(indices) == torch.Tensor
    return params[indices.transpose(0, 1).long().numpy().tolist()]
