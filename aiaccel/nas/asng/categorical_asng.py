import numpy as np
from nas.asng.categorical_dist import Categorical


class CategoricalASNG:
    """Adaptive stochastic natural gradient method on multivariate categorical distribution.

    Args:
        categories (numpy.ndarray): Array containing the numbers of categories of each dimension.
        alpha (float): Threshold of SNR in ASNG algorithm.
        init_delta (float): Initial value of delta.
        delta_max (float): Maximum value of Delta.
        init_theta (numpy.ndarray): Initial parameter of theta. Its shape must be (len(categories), max(categories)).
    """

    def __init__(self, categories, params=None, alpha=1.5, eps=0, init_delta=1.0, delta_max=np.inf, init_theta=None):
        self.p_model = Categorical(categories)

        if init_theta is not None:
            self.p_model.theta = init_theta

        self.N = np.sum(categories - 1)
        self.delta = init_delta
        self.Delta = 1.0
        self.delta_max = delta_max
        self.alpha = alpha
        self.gamma = 0.0
        self.s = np.zeros(self.N)

        self.eps = eps

        if params is not None:
            self.params = params
            self.params_max = np.zeros([len(self.params), 1])
        else:
            self.params = np.zeros([len(categories), max(categories)])
            self.params_max = np.zeros([len(self.params), 1])

        for i in range(len(self.params)):
            self.params_max[i] = np.max(self.params[i]) + 1e-10

    def get_delta(self):
        return self.delta / self.Delta

    def sampling(self):
        return self.p_model.sampling()

    def update(self, ms, losses, range_restriction=True):
        delta = self.get_delta()
        _ = delta * self.N**-0.5
        params_sum = np.sum(self.params * self.p_model.theta, axis=1).reshape([self.params.shape[0], 1])

        u, idx = self.utility(losses)
        mu_W, var_W = u.mean(), u.var()
        if var_W == 0:
            return

        ngrad = np.mean((u - mu_W)[:, np.newaxis, np.newaxis] * (ms[idx] - self.p_model.theta), axis=0)
        nreg = (self.params - params_sum) * self.p_model.theta

        # Too small natural gradient leads ngnorm to 0.
        if (np.abs(ngrad) < 1e-18).all():
            print("skip update")
            return

        self.p_model.theta += delta * (ngrad - self.eps * nreg / np.max(self.params))

        for i in range(self.p_model.d):
            ci = self.p_model.C[i]

            # Constraint for theta (minimum value of theta and sum of theta = 1.0)
            theta_min = 1.0 / (self.p_model.valid_d * (ci - 1)) if range_restriction and ci > 1 else 0.0
            self.p_model.theta[i, :ci] = np.maximum(self.p_model.theta[i, :ci], theta_min)
            theta_sum = self.p_model.theta[i, :ci].sum()
            tmp = theta_sum - theta_min * ci
            self.p_model.theta[i, :ci] -= (theta_sum - 1.0) * (self.p_model.theta[i, :ci] - theta_min) / tmp

            # Ensure the summation to 1
            self.p_model.theta[i, :ci] /= self.p_model.theta[i, :ci].sum()

    @staticmethod
    def utility(f, rho=0.25, negative=True):
        """
        Ranking Based Utility Transformation

        w(f(x)) / lambda =
            1/mu  if rank(x) <= mu
            0     if mu < rank(x) < lambda - mu
            -1/mu if lambda - mu <= rank(x)

        where rank(x) is the number of at least equally good
        points, including it self.

        The number of good and bad points, mu, is ceil(lambda/4).
        That is,
            mu = 1 if lambda = 2
            mu = 1 if lambda = 4
            mu = 2 if lambda = 6, etc.

        If there exist tie points, the utility values are
        equally distributed for these points.
        """
        eps = 1e-14
        idx = np.argsort(f)
        lam = len(f)
        mu = int(np.ceil(lam * rho))
        _w = np.zeros(lam)
        _w[:mu] = 1 / mu
        _w[lam - mu :] = -1 / mu if negative else 0
        w = np.zeros(lam)
        istart = 0
        for i in range(f.shape[0] - 1):
            if f[idx[i + 1]] - f[idx[i]] < eps * f[idx[i]]:
                pass
            elif istart < i:
                w[istart : i + 1] = np.mean(_w[istart : i + 1])
                istart = i + 1
            else:
                w[i] = _w[i]
                istart = i + 1
        w[istart:] = np.mean(_w[istart:])
        return w, idx
