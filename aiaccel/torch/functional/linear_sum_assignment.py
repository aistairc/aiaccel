import numpy as np
from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment

import torch


def linear_sum_assignment(cost_matrix: torch.Tensor, maximize: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the linear sum assignment problem for a batch of cost matrices.
    Args:
        cost_matrix (torch.Tensor): A tensor of shape (..., m, n)
            representing the cost matrix for each assignment problem.
        maximize (bool): If True, the problem is treated as a maximization problem.
            If False, it is treated as a minimization problem. Defaults to False.
    Returns:
        tuple: A tuple containing two tensors:
            - row_indices: Indices of the rows assigned to each column.
            - col_indices: Indices of the columns assigned to each row.
    """

    assert cost_matrix.ndim >= 2, "cost_matrix must have at least 2 dimensions"

    *batch_shape, m, n = cost_matrix.shape

    row_ind_list, col_ind_list = [], []
    for cm in cost_matrix.reshape(-1, m, n).cpu().numpy():
        row_ind, col_ind = scipy_linear_sum_assignment(cm, maximize=maximize)

        row_ind_list.append(row_ind)
        col_ind_list.append(col_ind)

    row_indices = torch.from_numpy(np.stack(row_ind_list).reshape(*batch_shape, -1)).to(cost_matrix.device)
    col_indices = torch.from_numpy(np.stack(col_ind_list).reshape(*batch_shape, -1)).to(cost_matrix.device)

    return row_indices, col_indices
