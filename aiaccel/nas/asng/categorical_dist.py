from __future__ import annotations

from typing import Union

import numpy as np


class Categorical:
    """
    Represents a categorical distribution for categorical variables.

    Args:
        categories (List[int]): The numbers of categories for each dimension.

    Attributes:
        d (int): The number of dimensions.
        C (List[int]): Specified categories for each dimension.
        Cmax (int): The maximum number of categories across all dimensions.
        theta (np.ndarray): Parameter matrix of the distribution.
        valid_param_num (int): Number of valid parameters.
        valid_d (int): Number of valid dimensions.
    """

    def __init__(self, categories: list[int]) -> None:
        self.d = len(categories)
        self.C = categories
        self.Cmax = max(categories)
        self.theta = np.zeros((self.d, self.Cmax))
        # Initialize theta by 1/C for each dimension
        for i, ci in enumerate(self.C):
            self.theta[i, :ci] = 1.0 / ci
        # Number of valid parameters
        self.valid_param_num = sum(c - 1 for c in self.C)
        # Number of valid dimensions
        self.valid_d = sum(1 for c in self.C if c > 1)

    def sampling_lam(self, lam: int) -> np.ndarray:
        """
        Draw samples from the categorical distribution.

        Args:
            lam (int): The number of samples to draw.

        Returns:
            np.ndarray: Sampled variables (one-hot representation) with shape (lam, d, Cmax).
        """
        rand = np.random.rand(lam, self.d, 1)
        cum_theta = self.theta.cumsum(axis=1)
        X = (cum_theta - self.theta <= rand) & (rand < cum_theta)
        return X

    def sampling(self) -> np.ndarray:
        """
        Draw a single sample from the categorical distribution.

        Returns:
            np.ndarray: A sampled variable (one-hot representation) with shape (d, Cmax).
        """
        rand = np.random.rand(self.d, 1)
        cum_theta = self.theta.cumsum(axis=1)
        return (cum_theta - self.theta <= rand) & (rand < cum_theta)

    def mle(self) -> np.ndarray:
        """
        Return the most likely categories.

        Returns:
            np.ndarray: Array of the most likely categories (one-hot representation) with shape (d, Cmax).
        """
        m = self.theta.argmax(axis=1)
        return np.eye(self.Cmax)[m]

    def loglikelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the log likelihood for given samples.

        Args:
            X (np.ndarray): Samples (one-hot representation) with shape (lam, d, maxK).

        Returns:
            np.ndarray: Array of log likelihoods for each sample.
        """
        # return np.sum(X * np.log(self.theta), axis=(1, 2))
        log_theta = np.log(self.theta, out=np.zeros_like(self.theta), where=self.theta != 0)
        return np.sum(X * log_theta, axis=(1, 2))

    def log_header(self) -> list[str]:
        """
        Generate the header for logging purposes.

        Returns:
            List[str]: A list of header strings.
        """
        return [f"theta{i}_{j}" for i in range(self.d) for j in range(self.C[i])]

    def log(self) -> list[str]:
        """
        Generate a log of the current theta values.

        Returns:
            List[str]: A list of theta values in string format for logging.
        """
        return [f"{self.theta[i, j]:.6f}" for i in range(self.d) for j in range(self.C[i])]

    def load_theta_from_log(self, theta: Union[list[float], np.ndarray]) -> None:
        """
        Load theta values from a log.

        Args:
            theta (Union[List[float], np.ndarray]): The theta values to load.
        """
        self.theta = np.zeros((self.d, self.Cmax))
        k = 0
        for i in range(self.d):
            self.theta[i, : self.C[i]] = theta[k : k + self.C[i]]
            k += self.C[i]

    def _load_theta_from_log(self, theta: Union[list[float], np.ndarray]) -> None:
        """
        Load theta values from a log.

        Args:
            theta (Union[List[float], np.ndarray]): The theta values to load.
        """
        self.theta = np.zeros((self.d, self.Cmax))
        k = 0
        for i in range(self.d):
            self.theta[i, : self.C[i]] = theta[k : k + self.C[i]]
            k += self.C[i]
