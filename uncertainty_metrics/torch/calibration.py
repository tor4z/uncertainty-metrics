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

# Lint as: python3
"""Calibration metrics for probabilistic predictions.

Calibration is a property of probabilistic prediction models: a model is said to
be well-calibrated if its predicted probabilities over a class of events match
long-term frequencies over the sampling distribution.
"""
import numpy as np

import torch
from torch import nn
import torch.distributions as td
from scipy.stats import truncnorm


class ExpectedCalibrationError:
    """Expected Calibration Error.

    Expected calibration error (Guo et al., 2017, Naeini et al., 2015) is a scalar
    measure of calibration for probabilistic models. Calibration is defined as the
    level to which the accuracy over a set of predicted decisions and true
    outcomes associated with a given predicted probability level matches the
    predicted probability. A perfectly calibrated model would be correct `p`% of
    the time for all examples for which the predicted probability was `p`%, over
    all values of `p`.

    This metric can be computed as follows. First, cut up the probability space
    interval [0, 1] into some number of bins. Then, for each example, store the
    predicted class (based on a threshold of 0.5 in the binary case and the max
    probability in the multiclass case), the predicted probability corresponding
    to the predicted class, and the true label into the corresponding bin based on
    the predicted probability. Then, for each bin, compute the average predicted
    probability ("confidence"), the accuracy of the predicted classes, and the
    absolute difference between the confidence and the accuracy ("calibration
    error"). Expected calibration error can then be computed as a weighted average
    calibration error over all bins, weighted based on the number of examples per
    bin.

    Perfect calibration under this setup is when, for all bins, the average
    predicted probability matches the accuracy, and thus the expected calibration
    error equals zero. In the limit as the number of bins goes to infinity, the
    predicted probability would be equal to the accuracy for all possible
    probabilities.

    References:
      1. Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. On Calibration of Modern
         Neural Networks. in International Conference on Machine Learning (ICML)
         cs.LG, (Cornell University Library, 2017).
      2. Naeini, M. P., Cooper, G. F. & Hauskrecht, M. Obtaining Well Calibrated
         Probabilities Using Bayesian Binning. Proc Conf AAAI Artif Intell 2015,
         2901-2907 (2015).
    """

    _setattr_tracking = False  # Automatic tracking breaks some unit tests

    def __init__(self, num_bins=15, name=None, dtype=None):
        """Constructs an expected calibration error metric.

        Args:
          num_bins: Number of bins to maintain over the interval [0, 1].
          name: Name of this metric.
          dtype: Data type.
        """
        self.dtype = dtype
        self.name = name
        self.num_bins = num_bins

        self.correct_sums = nn.Parameter(torch.zeros((num_bins,)), requires_grad=True)
        self.prob_sums = nn.Parameter(torch.zeros((num_bins,)), requires_grad=True)
        self.counts = nn.Parameter(torch.zeros((num_bins,)), requires_grad=True)

    def update_state(self, labels, probabilities, **kwargs):
        """Updates this metric.

        This will flatten the labels and probabilities, and then compute the ECE
        over all predictions.

        Args:
          labels: Tensor of shape [..., ] of class labels in [0, k-1].
          probabilities: Tensor of shape [..., ], [..., 1] or [..., k] of normalized
            probabilities associated with the True class in the binary case, or with
            each of k classes in the multiclass case.
          **kwargs: Other potential keywords, which will be ignored by this method.
        """
        del kwargs  # unused
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        if not isinstance(probabilities, torch.Tensor):
            probabilities = torch.tensor(probabilities)

        probabilities = probabilities.type(self.dtype)

        # Flatten labels to [N, ] and probabilities to [N, 1] or [N, k].
        if labels.dim() != 1:
            labels = labels.view(-1)
        if probabilities.dim() != 2 or (probabilities.shape[0] !=
                                        labels.shape[0]):
            probabilities = probabilities.view((labels.shape[0], -1))
        # Extend any probabilities of shape [N, 1] to shape [N, 2].
        # NOTE: XLA does not allow for different shapes in the branches of a
        # conditional statement. Therefore, explicit indexing is used.
        given_k = probabilities.shape[-1]
        k = np.maximum(2, given_k)
        if given_k < 2:
            probabilities = torch.cat([1. - probabilities, probabilities], -1)[:, -k:]

        pred_labels = torch.argmax(probabilities, dim=-1)
        pred_probs, _ = torch.max(probabilities, dim=-1)
        correct_preds = (pred_labels == labels.type(pred_labels.dtype)).type(self.dtype)

        bin_indices = torch.histc(pred_probs, bins=self.num_bins, min=0.0, max=1.0)
        batch_correct_sums = torch.zeros(self.num_bins).scatter_add(
            dim=0,
            index=bin_indices,
            src=correct_preds
        )

        batch_prob_sums = torch.zeros(self.num_bins).scatter_add(
            dim=0,
            index=bin_indices,
            src=pred_probs
        )
        batch_counts = torch.zeros(self.num_bins).scatter_add(
            dim=0,
            index=bin_indices,
            src=torch.ones_like(bin_indices),
        )
        batch_counts = batch_counts.type(self.dtype)
        self.correct_sums += batch_correct_sums
        self.prob_sums += batch_prob_sums
        self.counts += batch_counts

    def result(self):
        """Computes the expected calibration error."""
        non_empty = self.counts != 0
        correct_sums = self.correct_sums[non_empty]
        prob_sums = self.prob_sums[non_empty]
        counts = self.counts[non_empty]
        accs = correct_sums / counts
        confs = prob_sums / counts
        total_count = torch.sum(counts)
        return torch.sum(counts / total_count * torch.abs(accs - confs))

    def reset_states(self):
        """Resets all of the metric state variables.

        This function is called between epochs/steps,
        when a metric is evaluated during training.
        """
        pass
        # tf.keras.backend.batch_set_value([(v, [0., ] * self.num_bins) for v in
        #                                   self.variables])


def _bin_centers(num_bins):
    """Split the unit interval into `num_bins` equal parts and return centers."""
    return torch.linspace(0.0, 1.0 - 1.0 / num_bins, num_bins) + 0.5 / num_bins


def _prob_posterior(num_bins, bin_n, bin_pmean):
    """Compute posterior mean and standard deviation for unknown per-bin means.

    Args:
      num_bins: int, positive, the number of equal-sized bins to divide the unit
        interval into.
      bin_n: Tensor, shape (num_bins,), containing the absolute counts of
        observations falling into each bin.
      bin_pmean: Tensor, shape (num_bins,), containing the average predicted
        probability value that falls into this bin.

    Returns:
      posterior_means: Tensor, (num_bins,), the per-bin posterior mean.
      posterior_stddev: Tensor, (num_bins,), the per-bin posterior standard
        deviation.
    """
    bin_centers = _bin_centers(num_bins)
    sigma0_sq = (1.0 / num_bins) / 12.0  # matching moment of Uniform(Bin)
    sigma_sq = (1.0 / num_bins) / 12.0

    sigman_sq = 1.0 / (bin_n / sigma_sq + 1.0 / sigma0_sq)
    posterior_means = sigman_sq * (
            bin_centers / sigma0_sq + bin_n * (bin_pmean / sigma_sq))
    posterior_stddev = torch.sqrt(sigman_sq)

    return posterior_means, posterior_stddev


def _compute_ece(prob, bin_mean_prob):
    """Compute the expected calibration error (ECE).

    Args:
      prob: Tensor, shape (2,num_bins), containing the probabilities over the
        {incorrect,correct}x{0,1,..,num_bins-1} events.
      bin_mean_prob: Tensor, shape (1,num_bins), containing the average
        probability within each bin.

    Returns:
      ece: Tensor, scalar, the expected calibration error.
    """
    pz_given_b = prob / torch.unsqueeze(torch.sum(prob, dim=0), 0)
    prob_correct = prob[1, :] / torch.sum(prob[1, :])
    ece = torch.sum(prob_correct * torch.abs(pz_given_b[1, :] - bin_mean_prob))

    return ece


def _compute_calibration_bin_statistics(
        num_bins, logits=None, probabilities=None,
        labels_true=None, labels_predicted=None):
    """Compute binning statistics required for calibration measures.

    Args:
      num_bins: int, number of probability bins, e.g. 10.
      logits: Tensor, (n,nlabels), with logits for n instances and nlabels.
      probabilities: Tensor, (n,nlabels), with probs for n instances and nlabels.
      labels_true: Tensor, (n,), with tf.int32 or tf.int64 elements containing
        ground truth class labels in the range [0,nlabels].
      labels_predicted: Tensor, (n,), with tf.int32 or tf.int64 elements
        containing decisions of the predictive system.  If `None`, we will use
        the argmax decision using the `logits`.

    Returns:
      bz: Tensor, shape (2,num_bins), tf.int32, counts of incorrect (row 0) and
        correct (row 1) predictions in each of the `num_bins` probability bins.
      pmean_observed: Tensor, shape (num_bins,), tf.float32, the mean predictive
        probabilities in each probability bin.
    """
    if (logits is None) == (probabilities is None):
        raise ValueError(
            "_compute_calibration_bin_statistics expects exactly one of logits or "
            "probabilities.")
    elif probabilities is None:
        if logits.get_shape().as_list()[-1] == 1:  # pytype: disable=attribute-error
            raise ValueError(
                "_compute_calibration_bin_statistics expects logits for binary"
                " classification of shape (n, 2) for nlabels=2 but got ",
                logits.get_shape())  # pytype: disable=attribute-error
        probabilities = torch.softmax(logits, dim=1)
    elif (probabilities.get_shape().as_list()[-1] == 1 or
          len(probabilities.get_shape().as_list()) == 1):
        raise ValueError(
            "_compute_calibration_bin_statistics expects probabilities for binary"
            " classification of shape (n, 2) for nlabels=2 but got ",
            probabilities.get_shape())

    if labels_predicted is None:
        # If no labels are provided, we take the label with the maximum probability
        # decision.  This corresponds to the optimal expected minimum loss decision
        # under 0/1 loss.
        pred_y = torch.argmax(probabilities, dim=1).type(torch.int32)
    else:
        pred_y = labels_predicted

    correct = (pred_y == labels_true).type(torch.int32)

    # Collect predicted probabilities of decisions
    # prob_y = tf.compat.v1.batch_gather(probabilities,
    #                                    torch.unsqueeze(pred_y, dim=1))  # p(pred_y | x)
    prob_y = torch.gather(probabilities,
                          index=pred_y.unsqueeze(dim=1),
                          dim=1)
    prob_y = prob_y.view(-1)

    # Compute b/z histogram statistics:
    # bz[0,bin] contains counts of incorrect predictions in the probability bin.
    # bz[1,bin] contains counts of correct predictions in the probability bin.
    bins = torch.histc(prob_y, min=0.0, max=1.0, bins=num_bins)
    event_bin_counts = torch.bincount(
        correct * num_bins + bins,
        minlength=2 * num_bins)
    event_bin_counts = event_bin_counts.view(2, num_bins)

    # Compute mean predicted probability value in each of the `num_bins` bins
    pmean_observed = torch.zeros(num_bins).scatter_add(
        dim=0,
        index=bins,
        src=prob_y
    )
    tiny = np.finfo(np.float32).tiny
    pmean_observed = pmean_observed / (
            torch.sum(event_bin_counts, dim=0).type(torch.float32) + tiny)

    return event_bin_counts, pmean_observed


def bayesian_expected_calibration_error(num_bins, logits=None,
                                        probabilities=None,
                                        labels_true=None,
                                        labels_predicted=None,
                                        num_ece_samples=500):
    """Sample from the posterior of the Expected Calibration Error (ECE).

    The Bayesian ECE is defined via a posterior over ECEs given the observed data.
    With a large number of observations it will closely match the ordinary ECE but
    with few observations it will provide uncertainty estimates about the ECE.

    This method produces iid samples from the posterior over ECE values and for
    practical use you can summarize these samples, for example by computing the
    mean or quantiles.  For example, the following code will compute a 10%-90%
    most probable region as well as the median ECE estimate.

    ```
    ece_samples = bayesian_expected_calibration_error(10, logits=logits,
                                                      labels=labels)
    tfp.stats.percentile(ece_samples, [10.0, 50.0, 90.0])
    ```

    Args:
      num_bins: int, number of probability bins, e.g. 10.
      logits: Tensor, (n,nlabels), with logits for n instances and nlabels.
      probabilities: Tensor, (n,nlabels), with probs for n instances and nlabels.
      labels_true: Tensor, (n,), with tf.int32 or tf.int64 elements containing
        ground truth class labels in the range [0,nlabels].
      labels_predicted: Tensor, (n,), with tf.int32 or tf.int64 elements
        containing decisions of the predictive system.  If `None`, we will use
        the argmax decision using the `logits`.
      num_ece_samples: int, number of posterior samples of the ECE to create.

    Returns:
      ece_samples: Tensor, (ece_samples,), tf.float32 elements, ECE samples.
    """
    if (logits is None) == (probabilities is None):
        raise ValueError(
            "bayesian_expected_calibration_error expects exactly one of logits or "
            "probabilities.")
    elif probabilities is None:
        if logits.get_shape().as_list()[-1] == 1:  # pytype: disable=attribute-error
            raise ValueError(
                "bayesian_expected_calibration_error expects logits for binary"
                " classification of shape (n, 2) for nlabels=2 but got ",
                logits.get_shape())  # pytype: disable=attribute-error
        probabilities = torch.softmax(logits, dim=1)
    elif (probabilities.get_shape().as_list()[-1] == 1 or
          len(probabilities.get_shape().as_list()) == 1):
        raise ValueError(
            "bayesian_expected_calibration_error expects probabilities for binary"
            " classification of shape (n, 2) for nlabels=2 but got ",
            probabilities.get_shape())

    event_bin_counts, pmean_observed = _compute_calibration_bin_statistics(
        num_bins, probabilities=probabilities,
        labels_true=labels_true, labels_predicted=labels_predicted)
    event_bin_counts = event_bin_counts.type(torch.float32)

    # Compute posterior over probability value in each bin
    bin_n = torch.sum(event_bin_counts, dim=0)
    post_ploc, post_pscale = _prob_posterior(num_bins, bin_n, pmean_observed)
    bin_centers = _bin_centers(num_bins)
    half_bin = 0.5 * (1.0 / num_bins)
    # post_pmean = tfd.TruncatedNormal(
    #     loc=post_ploc, scale=post_pscale,
    #     low=bin_centers - half_bin, high=bin_centers + half_bin)

    # Compute the Dirichlet posterior over b/z probabilities
    prior_alpha = 1.0 / num_bins  # Perk's Dirichlet prior
    post_alpha = (event_bin_counts + prior_alpha).view(2 * num_bins,)
    posterior_event = td.Dirichlet(post_alpha)

    # Sample ECEs from the analytic and posteriors, which factorize
    ece_samples = torch.stack(
        [_compute_ece(posterior_event.sample().view((2, num_bins)),
                      truncated_normal_sample(loc=post_ploc, scale=post_pscale,
                                              low=(bin_centers - half_bin), high=bin_centers))
         for _ in range(num_ece_samples)])

    return ece_samples


def truncated_normal_sample(loc, scale, low, high, size=None):
    values = truncnorm.rvs(low, high, loc=loc,
                           scale=scale, size=size)
    return torch.tensor(values)
