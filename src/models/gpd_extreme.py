import aesara.tensor as at
import numpy as np
import pandas as pd
import pymc as mc
from pymc import Uniform

from constants import DATA_ROOT, scaling_factor, threshold
from pykelihood.distributions import GPD
from pymc_utils.pymc_distributions import Matern32Chordal, gpd_without_loc
from threshold_selection import threshold_selection_GoF

link_pot = lambda x: np.log1p(x) - threshold


def get_gpd_params(data):
    gpd_params = GPD.fit(data, loc=data.min()).flattened_params
    return [g.value for g in gpd_params]


def get_threshold(hail_data):
    ext_data = hail_data.groupby('claim_date').error.agg(risk_metric)
    return threshold_selection_GoF(ext_data, min_threshold=2, max_threshold=10)[0]


def risk_metric(x):
    return np.log1p(np.sum(x / scaling_factor))


def custom(x):
    return at.switch(x <= 0, x.exp(), 1 + x)


def gpd_extreme(model, _above):
    _grid = model.grid_idx[_above]
    _X = model.lonlat_mapping
    _season = model.season_idx[_above]
    _poh = model.poh[_above]
    _meshs = model.meshs[_above]
    _exp = model.exposure[_above]
    with model:
        # shape parameter
        init_shape = [-0.12960985886908538, -4.743355860126307e-05]
        shape = mc.Normal("shape", mu=init_shape, sigma=[0.1, 0.05], dims='season', initval=[0] * 2)[_season]
        # scale parameter
        constant_scale = Uniform(f"constant_scale", lower=-40, upper=40)
        coef_meshs = Uniform("coef_meshs", lower=-5., upper=5)
        coef_exposure = Uniform("coef_exposure", lower=-5., upper=5)
        coef_crossed = Uniform("coef_crossed", lower=-5., upper=5)
        sigma = Uniform("sigma_scale", lower=0, upper=15)
        ls = mc.Gamma("length_scale_scale", mu=5, sigma=1)
        latent = mc.gp.Latent(cov_func=Matern32Chordal(2, ls), )
        eps = latent.prior("eps_scale", _X, dims="grid", jitter=1e-7)  # corr inter-cells
        glm = constant_scale + coef_meshs * _meshs + coef_exposure * _exp + coef_crossed * _poh * _meshs + eps[
            _grid] * sigma
        scale = (glm / scaling_factor).exp()
    return scale, shape


def build_gpd_model(data):
    data = data.reset_index()
    below_idx = np.array(data[np.log1p(data.pos_error) < 8.06].index)
    above_idx = np.array(data[np.log1p(data.pos_error) >= 8.06].index)
    claim_values = pd.read_csv(DATA_ROOT / 'processed.csv', parse_dates=['claim_date'])
    grid_idx, _ = pd.factorize(data["gridcell"], sort=True)
    _, gridcells = pd.factorize(claim_values["gridcell"], sort=True)
    season_idx, seasons = pd.factorize(data.season, sort=True)[0], [0, 1]
    dates_continuous = pd.date_range(data.claim_date.min(), data.claim_date.max(), freq='1d')
    date_idx, dates = np.array(data.time), dates_continuous
    X = claim_values.groupby('gridcell').mean()[['longitude_grid', 'latitude_grid']].values
    coords = {"grid": gridcells,
              'above': above_idx,
              'below': below_idx,
              'point': data.index,
              'time': dates,
              'unique_timesteps_data': np.array(data.time.unique()),
              'season': seasons,
              "feature": ["longitude", "latitude"]}
    model = mc.Model(coords=coords)
    with model:
        _below = mc.ConstantData('below_idx', below_idx, dims='below')
        _above = mc.ConstantData('above_idx', above_idx, dims='above')
        mc.ConstantData('longitude', data.longitude)
        mc.ConstantData('latitude', data.latitude)
        mc.ConstantData('climada_dmg', data.climada_dmg)
        mc.ConstantData('threshold', threshold)
        mc.ConstantData("grid_idx", grid_idx, dims="point")
        mc.ConstantData('season_idx', season_idx, dims='point')
        mc.ConstantData("lonlat_mapping", X, dims=("grid", "feature"))
        # covariates
        mc.ConstantData('unscaled_exposure', data.exposure, dims='point')
        mc.ConstantData('exposure', (np.log(data.exposure) - np.log(data.exposure).min()) / (
                np.log(data.exposure).max() - np.log(data.exposure).min()), dims='point')
        mc.ConstantData('meshs', data.MESHS / scaling_factor, dims='point')
        mc.ConstantData('poh', data.POH / scaling_factor, dims='point')
    scale, shape = gpd_extreme(model, _above)
    observed = link_pot(data.pos_error.loc[_above.eval({})])
    with model:
        gpd_without_loc(scale, shape, observed)
    return model


def fit_marginal_model_for_claim_values(data):
    model = build_gpd_model(data)
    with model:
        # fit model
        print('Fitting GPD model for extreme claims...')
        trace = mc.sample(500, tune=200, init="jitter+adapt_diag_grad", chains=2,
                          cores=1, progressbar=True)
    return trace
