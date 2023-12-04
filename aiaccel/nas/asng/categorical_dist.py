import numpy as np


class Categorical(object):
    """Categorical distribution for categorical variables parametrized by :math:`\\{ \\theta \\}_{i=1}^{(d \\times K)}`.

    Args:
        categories (array_like[int]): The numbers of categories.

    Attributes:
        d (int): The number of categories.
        C (array_like): Specified categories.
        Cmax (int): The maximum number of categories.
        theta (np.ndarray[float]):
        valid_param_num (int):
        valid_d (int):
    """

    def __init__(self, categories):
        self.d = len(categories)
        self.C = categories
        self.Cmax = np.max(categories)
        self.theta = np.zeros((self.d, self.Cmax))
        # initialize theta by 1/C for each dimensions
        for i in range(self.d):
            self.theta[i, : self.C[i]] = 1.0 / self.C[i]
        # pad zeros to unused elements
        for i in range(self.d):
            self.theta[i, self.C[i] :] = 0.0
        # number of valid parameters
        self.valid_param_num = int(np.sum(self.C - 1))
        # valid dimension size
        self.valid_d = len(self.C[self.C > 1])

    def sampling_lam(self, lam):
        """Draw :math:`\\lambda` samples from the categorical distribution.

        Args:
            lam (int): sample size :math:`\\lambda`

        Returns:
            array_like[bool]: An array of sampled variables from the
                categorical distribution (one-hot representation) with shape of
                (lam, d, Cmax).
        """
        rand = np.random.rand(lam, self.d, 1)  # range of random number is [0, 1)
        cum_theta = self.theta.cumsum(axis=1)  # (d, Cmax)
        X = (cum_theta - self.theta <= rand) & (rand < cum_theta)
        return X

    def sampling(self):
        """Draw a sample from the categorical distribution.

        Returns:
            array_like[bool]: An array of sampled variables from the
                categorical distribution (one-hot representation) with shape of
                (d, Cmax).
        """
        rand = np.random.rand(self.d, 1)  # range of random number is [0, 1)
        cum_theta = self.theta.cumsum(axis=1)  # (d, Cmax)

        # x[i, j] becomes 1 iff cum_theta[i, j] - theta[i, j] <= rand[i] < cum_theta[i, j]
        x = (cum_theta - self.theta <= rand) & (rand < cum_theta)
        return x

    def mle(self):
        """Return the most likely categories.

        Returns:
            array_like[bool]: An array of categorical variables
            (one-hot representation) with shape of (d, Cmax).
        """
        m = self.theta.argmax(axis=1)
        x = np.zeros((self.d, self.Cmax))
        for i, c in enumerate(m):
            x[i, c] = 1
        return x

    def loglikelihood(self, X):
        """Calculate log likelihood.

        Args:
            X (array_like[bool]): An array of samples (one-hot representation)
            with shape of (lam, d, maxK).

            Returns:
                array_like[float]: An array of log likelihoods with shape of
                (lam, ).
        """
        return (X * np.log(self.theta)).sum(axis=2).sum(axis=1)

    def log_header(self):
        header_list = []
        for i in range(self.d):
            header_list += ["theta%d_%d" % (i, j) for j in range(self.C[i])]
        return header_list

    def log(self):
        theta_list = []
        for i in range(self.d):
            theta_list += ["%f" % self.theta[i, j] for j in range(self.C[i])]
        return theta_list

    def load_theta_from_log(self, theta):
        self.theta = np.zeros((self.d, self.Cmax))
        k = 0
        for i in range(self.d):
            for j in range(self.C[i]):
                self.theta[i, j] = theta[k]
                k += 1
