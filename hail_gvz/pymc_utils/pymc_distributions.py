import aesara.tensor as at
import numpy as np
import pymc as mc

from pykelihood.distributions import GPD, Beta

tol = 1e-3
threshold = 8.06
exp_threshold = np.exp(threshold) - 1
link_pot = lambda x: np.log1p(x) - threshold
inverse_link_pot = lambda x: np.exp(x + threshold) - 1
link_beta = lambda x: x / exp_threshold
inverse_link_beta = lambda x: x * exp_threshold


def sigmoid(x):
    return 1 / (1 + (-x).exp())


def custom(x, alpha=0.02):
    return at.switch(x <= 0, (alpha * x).exp(), 1 + alpha * x)


class Matern32Chordal(mc.gp.cov.Stationary):
    def __init__(self, input_dims, ls, r=6378.137, active_dims=None):
        if input_dims != 2:
            raise ValueError("Chordal distance is only defined on 2 dimensions")
        super().__init__(input_dims, ls=ls, active_dims=active_dims)
        self.r = r

    def lonlat2xyz(self, lonlat):
        lonlat = np.deg2rad(lonlat)
        return self.r * at.stack(
            [
                at.cos(lonlat[..., 0]) * at.cos(lonlat[..., 1]),
                at.sin(lonlat[..., 0]) * at.cos(lonlat[..., 1]),
                at.sin(lonlat[..., 1]),
            ],
            axis=-1,
        )

    def chordal_dist(self, X, Xs=None):
        if Xs is None:
            Xs = X
        X, Xs = at.broadcast_arrays(
            self.lonlat2xyz(X[..., :, None, :]), self.lonlat2xyz(Xs[..., None, :, :])
        )
        return at.sqrt(at.sum(((X - Xs) / self.ls) ** 2, axis=-1) + 1e-12)

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        r = self.chordal_dist(X, Xs)
        return (1.0 + np.sqrt(3.0) * r) * at.exp(-np.sqrt(3.0) * r)


class RatQuadChordal(mc.gp.cov.Stationary):
    def __init__(self, input_dims, ls, alpha=15, r=6378.137, active_dims=None):
        if input_dims != 2:
            raise ValueError("Chordal distance is only defined on 2 dimensions")
        super().__init__(input_dims, ls=ls, active_dims=active_dims)
        self.r = r
        self.alpha = alpha

    def lonlat2xyz(self, lonlat):
        lonlat = np.deg2rad(lonlat)
        return self.r * at.stack(
            [
                at.cos(lonlat[..., 0]) * at.cos(lonlat[..., 1]),
                at.sin(lonlat[..., 0]) * at.cos(lonlat[..., 1]),
                at.sin(lonlat[..., 1]),
            ],
            axis=-1,
        )

    def chordal_dist(self, X, Xs=None):
        if Xs is None:
            Xs = X
        X, Xs = at.broadcast_arrays(
            self.lonlat2xyz(X[..., :, None, :]), self.lonlat2xyz(Xs[..., None, :, :])
        )
        return at.sqrt(at.sum(((X - Xs) / self.ls) ** 2, axis=-1) + 1e-12)

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        r = self.chordal_dist(X, Xs)
        return (1.0 + r ** 2 / (2 * self.alpha)) ** (-1 / self.alpha)


class Matern52Chordal(mc.gp.cov.Stationary):
    def __init__(self, input_dims, ls, r=6378.137, active_dims=None):
        if input_dims != 2:
            raise ValueError("Chordal distance is only defined on 2 dimensions")
        super().__init__(input_dims, ls=ls, active_dims=active_dims)
        self.r = r

    def lonlat2xyz(self, lonlat):
        lonlat = np.deg2rad(lonlat)
        return self.r * at.stack(
            [
                at.cos(lonlat[..., 0]) * at.cos(lonlat[..., 1]),
                at.sin(lonlat[..., 0]) * at.cos(lonlat[..., 1]),
                at.sin(lonlat[..., 1]),
            ],
            axis=-1,
        )

    def chordal_dist(self, X, Xs=None):
        if Xs is None:
            Xs = X
        X, Xs = at.broadcast_arrays(
            self.lonlat2xyz(X[..., :, None, :]), self.lonlat2xyz(Xs[..., None, :, :])
        )
        return at.sqrt(at.sum(((X - Xs) / self.ls) ** 2, axis=-1) + 1e-12)

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        r = self.chordal_dist(X, Xs)
        return (1.0 + np.sqrt(5.0) * r + 5.0 * (r ** 2) / 3) * at.exp(-np.sqrt(5.0) * r)


def gpd_without_loc(scale, shape, value):
    def random(scale, shape, rng, size):
        distro = GPD(scale=scale, shape=shape)
        return distro.rvs(size=size, random_state=rng)

    def logp(value, scale, shape):
        def key_term_nonzero_shape(x):
            arr = (1 + shape * x / scale)
            arr = arr ** ((-1 / shape) - 1)
            return arr

        def key_term_zero_shape(x):
            arr = (-x / scale).exp()
            return arr

        def key_term(x, shape):
            return at.switch(at.abs(shape) >= tol, key_term_nonzero_shape(x), key_term_zero_shape(x))

        gpd_term = (1 / scale) * key_term(value, shape)
        log_pdf_gpd = gpd_term.log()
        valid = log_pdf_gpd[value >= 0.]
        valid = valid[~at.isnan(valid)]
        valid = valid[~at.isinf(valid)]
        return valid.sum()

    return mc.DensityDist('pot', scale, shape, logp=logp,
                          random=random, observed=value)


def beta_distri(mu, sigma, value):
    def random(mu, sigma, rng, size):
        mu = at.switch(mu < 1, at.switch(mu > 0, mu, tol), 1 - tol)
        sigma = at.switch(sigma < 1, at.switch(sigma > 0, sigma, tol), 1 - tol)
        try:
            dist = mc.Beta.dist(mu=mu, sigma=sigma)
            return mc.draw(dist)
        except:
            kappa = mu * (1 - mu) / sigma ** 2 - 1
            alpha = kappa * mu
            beta = kappa * (1 - mu)
            dist = Beta(alpha=alpha, beta=beta)
            return dist.rvs(size, rng)

    def logp(value, mu, sigma):
        value = at.switch(value < 1, at.switch(value > 0, value, tol), 1 - tol)
        mu = at.switch(mu < 1, at.switch(mu > 0, mu, tol), 1 - tol)
        sigma = at.switch(sigma < 1, at.switch(sigma > 0, sigma, tol), 1 - tol)
        dist = mc.Beta.dist(mu=mu, sigma=sigma)
        logp = mc.logp(dist, value)
        logp = at.switch(at.isinf(logp), -10 ** 5, logp)
        return logp.sum()

    return mc.DensityDist('beta_sum', mu, sigma, logp=logp,
                          random=random, observed=value)


def beta_distri_unobserved(mu, sigma):
    def random(mu, sigma, rng, size):
        mu = at.switch(mu < 1, at.switch(mu > 0, mu, tol), 1 - tol)
        sigma = at.switch(sigma < 1, at.switch(sigma > 0, sigma, tol), 1 - tol)
        try:
            dist = mc.Beta.dist(mu=mu, sigma=sigma)
            return mc.draw(dist)
        except:
            pass

    def logp(value, mu, sigma):
        mu = at.switch(mu < 1, at.switch(mu > 0, mu, tol), 1 - tol)
        sigma = at.switch(sigma < 1, at.switch(sigma > 0, sigma, tol), 1 - tol)
        dist = mc.Beta.dist(mu=mu, sigma=sigma)
        logp = mc.logp(dist, value)
        logp = at.switch(at.isinf(logp), -10 ** 3, logp)
        return logp

    return mc.DensityDist.dist(mu, sigma, class_name='beta', logp=logp,
                               random=random)


def gpd_without_loc_unobserved(scale, shape):
    def random(scale, shape, rng, size):
        distro = GPD(scale=scale, shape=shape)
        try:
            return distro.rvs(size=size, random_state=rng)
        except:
            pass

    def logp(value, scale, shape):
        def key_term_nonzero_shape(x):
            arr = (1 + shape * x / scale)
            arr = arr ** ((-1 / shape) - 1)
            return arr

        def key_term_zero_shape(x):
            arr = (-x / scale).exp()
            return arr

        def key_term(x, shape):
            return at.switch(at.abs(shape) >= tol, key_term_nonzero_shape(x), key_term_zero_shape(x))

        gpd_term = (1 / scale) * key_term(value, shape)
        log_pdf_gpd = gpd_term.log()
        valid = log_pdf_gpd
        return valid

    return mc.DensityDist.dist(scale, shape, class_name='pot', logp=logp,
                               random=random)


def combined_beta_gpd(beta, gpd, value):
    def random(beta, gpd, rng, size):
        return np.where(value > exp_threshold, inverse_link_pot(gpd), inverse_link_beta(beta))

    def logp(value, beta, gpd):
        total_ll = at.switch(value > exp_threshold, mc.logp(gpd, link_pot(value)), mc.logp(beta, link_beta(value)))
        valid = total_ll[~at.isnan(total_ll)]
        return valid.sum()

    return mc.DensityDist('cond_damage', beta, gpd, logp=logp,
                          random=random, observed=value, dims='point')


def combined_bern_beta_gpd(bern, beta, gpd, value):
    def random(bern, beta, gpd, rng, size):
        return np.where(bern == 1, inverse_link_pot(gpd), inverse_link_beta(beta))

    def logp(value, bern, beta, gpd):
        total_ll = at.switch(value > exp_threshold, mc.logp(gpd, link_pot(value)), mc.logp(beta, link_beta(value)))
        valid = total_ll[~at.isnan(total_ll)]
        return valid.sum()

    return mc.DensityDist('cond_damage', bern, beta, gpd, logp=logp,
                          random=random, observed=value, dims='point')


def poisson_counts(psi, lambd, value):
    def random(psi, lambd, rng, size):
        dist = mc.ZeroInflatedPoisson.dist(psi=psi, mu=lambd)
        try:
            sample = mc.draw(dist, random_seed=rng)
            return sample
        except:
            pass

    def logp(value, psi, lambd):
        dist = mc.ZeroInflatedPoisson.dist(psi=psi, mu=lambd)
        logp = mc.logp(dist, value)
        valid = at.switch(at.isnan(logp), -1e5, logp)
        valid = at.switch(at.isinf(valid), -1e5, valid)
        return valid.sum()

    return mc.DensityDist('counts', psi, lambd, logp=logp, observed=value,
                          random=random, dims='point')


def poisson_counts_unobserved(psi, lambd):
    def random(psi, lambd, rng, size):
        dist = mc.ZeroInflatedPoisson.dist(psi=psi, mu=lambd)
        try:
            sample = mc.draw(dist, random_seed=rng)
            return sample
        except:
            pass

    def logp(value, psi, lambd):
        dist = mc.ZeroInflatedPoisson.dist(psi=psi, mu=lambd)
        logp = mc.logp(dist, value)
        return logp.sum()

    return mc.DensityDist.dist(psi, lambd, class_name='poisson_counts', logp=logp,
                               random=random)


def binom_counts_alphamu(psi, mu, alpha, value):
    def random(psi, mu, alpha, rng, size):
        dist = mc.ZeroInflatedNegativeBinomial.dist(psi=psi, mu=mu, alpha=alpha)
        try:
            sample = mc.draw(dist, random_seed=rng)
            return sample
        except:
            pass

    def logp(value, psi, mu, alpha, ):
        dist = mc.ZeroInflatedNegativeBinomial.dist(psi=psi, mu=mu, alpha=alpha)
        logp = mc.logp(dist, value)
        return logp.sum()

    return mc.DensityDist('counts', psi, mu, alpha, logp=logp, observed=value,
                          random=random, dims='point')


def binom_counts_alphamu_unobserved(psi, mu, alpha):
    def random(psi, mu, alpha, rng, size):
        dist = mc.ZeroInflatedNegativeBinomial.dist(psi=psi, mu=mu, alpha=alpha)
        try:
            sample = mc.draw(dist, random_seed=rng)
            return sample
        except:
            pass

    def logp(value, psi, mu, alpha, ):
        dist = mc.ZeroInflatedNegativeBinomial.dist(psi=psi, mu=mu, alpha=alpha)
        logp = mc.logp(dist, value)
        return logp.sum()

    return mc.DensityDist.dist(psi, mu, alpha, class_name='binom_counts', logp=logp,
                               random=random)


def binom_counts(psi, n, p, value):
    def random(psi, n, p, rng, size):
        dist = mc.ZeroInflatedBinomial.dist(psi=psi, p=p, n=n)
        try:
            sample = mc.draw(dist, random_seed=rng)
            return sample
        except:
            pass

    def logp(value, psi, n, p):
        dist = mc.ZeroInflatedBinomial.dist(psi=psi, p=p, n=n)
        logp = mc.logp(dist, value)
        return logp.sum()

    return mc.DensityDist('counts', psi, n, p, logp=logp, observed=value,
                          random=random, dims='point')


def binom_counts_unobserved(psi, n, p):
    def random(psi, n, p, rng, size):
        dist = mc.ZeroInflatedBinomial.dist(psi=psi, p=p, n=n)
        try:
            sample = mc.draw(dist, random_seed=rng)
            return sample
        except:
            pass

    def logp(value, psi, n, p):
        dist = mc.ZeroInflatedBinomial.dist(psi=psi, p=p, n=n)
        logp = mc.logp(dist, value)
        return logp.sum()

    return mc.DensityDist.dist(psi, n, p, class_name='binom_counts', logp=logp,
                               random=random)


def bernoulli_distri(p, value):
    def random(p, rng, size):
        p = at.switch(p < 1, at.switch(p > 0, p, tol), 1 - tol)
        dist = mc.Bernoulli.dist(p=p)
        return mc.draw(dist, random_seed=rng)

    def logp(value, p):
        p = at.switch(p < 1, at.switch(p > 0, p, tol), 1 - tol)
        dist = mc.Bernoulli.dist(p=p)
        logp = mc.logp(dist, value)
        valid = at.switch(at.isnan(logp), -1e5, logp)
        valid = at.switch(at.isinf(valid), -1e5, valid)
        return valid.sum()

    return mc.DensityDist('counts', p, observed=value, logp=logp, dims='point',
                          random=random)


def bernoulli_distri_unobserved(p):
    def random(p, rng, size):
        p = at.switch(p < 1, at.switch(p > 0, p, tol), 1 - tol)
        dist = mc.Bernoulli.dist(p=p)
        return mc.draw(dist, random_seed=rng)

    def logp(value, p):
        dist = mc.Bernoulli.dist(p=p)
        logp = mc.logp(dist, value)
        return logp.sum()

    return mc.DensityDist.dist(p, class_name='over_threshold', logp=logp,
                               random=random)


def combined_dirac_beta(bern, beta, value):
    def random(bern, beta, rng, size):
        try:
            return np.where(bern[0] == 1, beta, 0)
        except:
            pass

    def logp(value, bern, beta):
        total_ll = at.switch(value > 0, mc.logp(beta, value), mc.logp(mc.DiracDelta.dist(c=0), value))
        return total_ll.sum()

    return mc.DensityDist('counts', bern, beta, logp=logp,
                          random=random, observed=value, dims='point')
