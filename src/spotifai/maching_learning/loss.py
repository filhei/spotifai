import torch

from torch.nn import Module, ModuleList, Conv1d, TransformerEncoderLayer, Linear


class ContrastiveLoss(Module):
    """Contrastive loss function from Barlow Twins (https://arxiv.org/abs/2103.03230).
    The method has been extended to allow multiple positive examples per batch by averaging the correlation over all possible permutations of positive pairs (using a closed form solution).
    z_a and z_b are the normalized representations/projections (model outputs), each containing the same number of samples.
    The observations with same batch index i should be similiar (positive examples), such that z_a[i] has a positive example in z_b[i].
    If multiple positive examples are provided, 'groups' must be provided to define which indices belong together/to the same contrastive class (see example).

    Args:
        z_a (torch.tensor): First set of model projections/feature representations with shape (n_observations, projection_dim). Assuned to be normalized to mean: 0, std: 1 along batch dimension.
        z_b (torch.tensor): Second set of model projections/feature representations with same shape. Each observation must match the contrastive class of the corresponding index of z_a.
        groups (list[list[int]]): Subdivision of observations over contrastive classes. For each group of positive examples,
                                  one list of the corresponding indices should exist in the main list.
                                  [[1, 2], [3, 4], [5], [6]] indicates that the first two observations belong together, as do the third and fourth.

    Returns:
        torch.float: loss value.

    Example:
        z_a and z_b each have shape (16, feature_dim). Each contrastive class has 4 corresponding observations in each projection:
        indices [0, 1, 2, 3] in both z_a and z_b should have similar representations. To inform the loss funciton about this, set

        >>> groups = [
        ...     [ 0,  1,  2,  3],
        ...     [ 4,  5,  6,  7],
        ...     [ 8,  9, 10, 11],
        ...     [12, 13, 14, 15],
        ... ]

    """

    def __init__(self, coefficient=0.01):
        """Initialization.

        Args:
            coefficient (float): Coefficient that determines the contribution from off diagonal elelements (negative examples).
        """
        super().__init__()
        self.coefficient = coefficient

    def forward(self, z_a, z_b, groups=None):

        # shape: (batch, feature)
        assert z_a.dim() == 2
        assert z_b.dim() == 2
        assert z_a.shape == z_b.shape

        n = len(z_a)

        if groups is not None:
            # TODO: document derivation of barlow twins loss for multiple positive examples
            assert (
                sum([len(g) for g in groups]) == n
            ), "every observation must be accounted for"

            aggregation_a = torch.zeros(len(groups), z_a.shape[1], device=z_a.device)
            aggregation_b = torch.zeros_like(aggregation_a)

            for i, group in enumerate(groups):
                # average corr over all permutations: take the correlation between the reduced vectors of (scaled) sums of the positive examples
                scale = len(group)

                aggregation_a[i] = z_a[group].sum(dim=0)
                aggregation_b[i] = z_b[group].sum(dim=0) / scale

            z_a = aggregation_a
            z_b = aggregation_b

        # correlations
        c = z_a.T @ z_b / n

        # diagonal target: 1
        diagonal_diff = torch.diagonal(c) - 1
        diagonal_loss = torch.dot(diagonal_diff, diagonal_diff)

        # off-diagonal target: 0
        diagonal_mask = 1.0 - torch.eye(len(c), device=c.device)
        off_diagonal_diff = (c * diagonal_mask).flatten()
        off_diagonal_loss = torch.dot(off_diagonal_diff, off_diagonal_diff)

        loss = diagonal_loss + self.coefficient * off_diagonal_loss

        # TODO: is scaling by n relevant? (since we always calculate same-shape cross-correlation matrix) => rather scale by squared(feature_size)?
        loss /= n

        return loss


def similarity_groups(half_batch_size, buffer_size):
    """Utility to create the 'groups' argument for the forward pass of the ContrastiveLoss.
    Each (half) batch is assumed to be sampled from the same contrastive class, and the buffer size is the number of (half) batches.
    The groups are then (with m = half_batch_size):
    [
        [0, 1, ..., m-1],
        [m, m+1, ..., 2*m-1],
        ...
    ]

    Args:
        half_batch_size (int): Size of half batch (which corresponds to the number of positive examples).
        buffer_size (int): Number of batches that will be buffered.

    Returns:
        groups (list[list[int]]): List of list of indices of observations from the corresponding contrastive class.
    """
    groups = [
        torch.arange(half_batch_size) + half_batch_size * i for i in range(buffer_size)
    ]
    return groups


# TODO: Implement CLIP-inspired contrastive loss
# class ContrastiveLoss (Normlized Temperature-scaled CrossEntropy NT-Xent)
# def forward(self, input: Tensor, target: Tensor) -> Tensor:
#     return F.cross_entropy(
#         input,
#         target,
#         weight=self.weight,
#         ignore_index=self.ignore_index,
#         reduction=self.reduction,
#         label_smoothing=self.label_smoothing,
#     )
