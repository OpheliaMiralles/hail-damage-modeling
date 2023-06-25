import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as mc
from pymc import Uniform

from pymc_utils.pymc_distributions import beta_distri, RatQuadChordal

plt.rcParams["figure.constrained_layout.use"] = True
DATA_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/data/GVZ_Datenlieferung_an_ETH/')
FITS_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/fits/')
scaling_factor = 100
tol = 1e-5
threshold = 8.06
link_beta = lambda x: x / (np.exp(threshold) - 1)


def sigmoid(x):
    return 1 / (1 + (-x).exp())


def logit(x):
    return np.log(x / (1 - x))


def beta_nonextreme(model, data, _below):
    _grid = model.grid_idx[_below]
    _exp = model.exposure[_below]
    _threshold = model.threshold
    _season = model.season_idx[_below]
    _X = model.lonlat_mapping
    _poh = model.poh[_below]
    _meshs = model.meshs[_below]
    _exp = model.exposure[_below]
    _time = model.date_idx[_below]
    dates = model.coords['time']
    with model:
        # alpha
        constant_mu = Uniform(f"constant_mu", lower=0., upper=10)
        sigma_seasonal = Uniform(f"sigma_seasonal_mu", lower=0, upper=2, dims='season')
        cov_annual = mc.gp.cov.Cosine(1, 365)
        latent_annual = mc.gp.Latent(cov_func=cov_annual)
        _X_annual_cycle = mc.ConstantData('days_since_start',
                                          np.array((pd.to_datetime(dates) - np.min(dates)).days)[:, None])
        latent_cyclic = latent_annual.prior("eps_mu", _X_annual_cycle, dims="time", jitter=1e-7)  # corr inter-cells
        cyclic_mu = sigma_seasonal[_season] * latent_cyclic[_time]
        sigma_grid = Uniform("sigma_grid", lower=0., upper=2, initval=1.)
        ls = mc.Gamma("length_scale_sigma", mu=5, sigma=2., initval=15)
        latent = mc.gp.Latent(cov_func=RatQuadChordal(2, ls), )
        eps = latent.prior("eps_grid", _X, dims="grid", jitter=1e-6)  # corr inter-cells
        gridwise_mu = Uniform(f"grid_scale", lower=-10, upper=10, dims='grid')
        grid_mu = gridwise_mu[_grid] + eps[_grid] * sigma_grid
        coef_meshs = Uniform("coef_meshs", lower=-5., upper=5, initval=0.)
        coef_poh = Uniform("coef_poh", lower=-5., upper=5, initval=0.)
        coef_exposure = Uniform("coef_exposure", lower=-5., upper=5, initval=0.)
        alpha = constant_mu + grid_mu + coef_poh * _poh + coef_meshs * _meshs + coef_exposure * _exp + cyclic_mu
        mu = sigmoid(alpha)
        # sigma
        kappa = mc.HalfNormal('precision', sigma=1000)  # precision parameter
        sigma = (mu * (1 - mu) / (kappa + 1)).sqrt()
        observed = link_beta(data.pos_error.loc[_below.eval({})])
        beta = beta_distri(mu, sigma=sigma, value=observed)
    return beta


def build_beta_model(data):
    data = data.reset_index()
    below_idx = data.index#np.array(data[np.log1p(data.pos_error) < 8.06].index)
    above_idx = data.index#np.array(data[np.log1p(data.pos_error) >= 8.06].index)
    grid_idx, gridcells = pd.factorize(data["gridcell"], sort=True)
    dates_continuous = pd.date_range(data.claim_date.min(), data.claim_date.max(), freq='1d')
    date_idx, dates = np.array(data.time), dates_continuous
    location_idx, locations = pd.factorize(data['location'], sort=True)
    season_idx, seasons = pd.factorize(data.season, sort=True)
    X = data.groupby('gridcell').mean()[['longitude_grid', 'latitude_grid']].values
    mapping_time_season, _ = pd.factorize(data.groupby('time', sort=False).season.mean(), sort=True)
    mapping_location_grid, _ = pd.factorize(data.groupby('location', sort=False).gridcell.mean(), sort=True)
    coords = {"grid": gridcells,
              'location': locations,
              'above': above_idx,
              'below': below_idx,
              'point': data.index,
              'time': dates,
              'unique_timesteps_data': np.array(data.time.unique()),
              'season': seasons,
              "feature": ["longitude", "latitude"]}
    model = mc.Model(coords=coords)
    with model:
        _threshold = mc.ConstantData('threshold', threshold)
        _below = mc.ConstantData('below_idx', below_idx, dims='below')
        _above = mc.ConstantData('above_idx', above_idx, dims='above')
        _grid = mc.ConstantData("grid_idx", grid_idx, dims="point")
        _time = mc.ConstantData("date_idx", date_idx, dims="point")
        _season = mc.ConstantData('season_idx', season_idx, dims='point')
        _locations = mc.ConstantData("location_idx", location_idx, dims="point")
        _dates = mc.ConstantData('date', dates, dims='time')
        _lonlat = mc.ConstantData('lonlat_mapping_location', data.reset_index().groupby('location').mean()[['longitude', 'latitude']].values, dim='location')
        _X = mc.ConstantData("lonlat_mapping", X, dims=("grid", "feature"))
        _mapping_time_season = mc.ConstantData('mapping_time_season', mapping_time_season, dims='unique_timesteps_data')
        _mapping_location_grid = mc.ConstantData('mapping_location_grid', mapping_location_grid, dims='location')
        # covariates
        _exp = mc.ConstantData('exposure', np.log(data.exposure), dims='point')
        _meshs = mc.ConstantData('meshs', data.mean_meshs / scaling_factor, dims='point')
        _poh = mc.ConstantData('poh', data.mean_poh / scaling_factor, dims='point')
    beta_nonextreme(model, data, _below)
    return model


def fit_marginal_model_for_claim_values(data):
    model = build_beta_model(data)
    with model:
        # fit model
        print('Fitting beta model for non extreme claims...')
        trace = mc.sample(4000, init="jitter+adapt_diag_grad", chains=1,
                          cores=1, progressbar=True, step=mc.DEMetropolisZ())
    return trace
