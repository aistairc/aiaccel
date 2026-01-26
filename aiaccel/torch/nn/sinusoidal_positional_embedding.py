# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    """Generates sinusoidal positional embeddings for input tensors.

    Args:
        d_emb (int, optional): Embedding dimension. Defaults to 256.
        tau (float, optional): Base for exponential decay of frequencies. Defaults to 10000.
        dims (int | list[int], optional): Dimensions along which to apply positional encoding. Defaults to 1.
        normalize (bool, optional): Whether to normalize coordinates to [0, 2*pi]. Defaults to False.

    Returns:
        torch.Tensor: Positional embeddings with the same leading shape as input ``x`` and last dimension ``d_emb``.
    """

    div_term: torch.Tensor

    def __init__(self, d_emb: int = 256, tau: float = 10000, dims: int | list[int] = 1, normalize: bool = False):
        super().__init__()

        self.dims = [dims] if isinstance(dims, int) else dims

        assert d_emb % (2 * len(self.dims)) == 0

        self.normalize = normalize

        d_emb_ = d_emb // len(self.dims)
        self.register_buffer("div_term", 1 / tau ** (torch.arange(0, d_emb_, 2) / d_emb_))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pos_list = []
        for dim in self.dims:
            if mask is not None:
                assert mask.shape == x.shape[:-1] and mask.dtype == torch.bool

                coordinates = (~mask).cumsum(dim).sub_(1)
            else:
                shape_ = [1] * (x.dim() - 1)
                shape_[dim] = x.shape[dim]

                coordinates = torch.arange(x.shape[dim], device=x.device).view(*shape_).expand(x.shape[:-1])

            if self.normalize:
                coordinates = 1 + coordinates
                coordinates = 2 * math.pi * coordinates / coordinates.select(dim, -1).unsqueeze(dim).add(1e-6)

            coordinates = coordinates.unsqueeze(-1) * self.div_term
            pos_ = torch.stack((coordinates.sin(), coordinates.cos()), dim=-1).flatten(-2)
            pos_list.append(pos_)

        pos = torch.cat(pos_list, dim=-1)

        if mask is not None:
            pos.masked_fill_(mask.unsqueeze(-1), 0)

        return pos
