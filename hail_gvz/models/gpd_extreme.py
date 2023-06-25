import datetime
import pathlib

import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as mc
from pymc import Uniform

from data.data_processing import get_train_data
from pykelihood.distributions import GPD
from pymc_utils.pymc_distributions import Matern32Chordal, gpd_without_loc, bernouilli_over_threshold
from threshold_selection import threshold_selection_GoF

DATA_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/data/GVZ_Datenlieferung_an_ETH/')
FITS_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/fits/')
scaling_factor = 100
tol = 1e-5
threshold = 8.06
exp_threshold = np.exp(threshold) - 1
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


def gpd_extreme(model, data):
    _above = model.above_idx
    _grid = model.grid_idx
    _location = model.location_idx
    _time = model.date_idx
    _threshold = model.threshold
    _X = model.lonlat_mapping
    _season = model.season_idx
    _poh = model.poh
    _meshs = model.meshs
    _exp = model.exposure
    init_loc, init_scale, init_shape = GPD.fit(link_pot(data.pos_error.loc[_above.eval({})])).flattened_params
    with model:
        # shape parameter
        shape = Uniform("shape", lower=-1, upper=1., dims='season', initval=[init_shape.value] * 3)[_season]
        coef_meshs = Uniform("coef_meshs", lower=-50., upper=50, initval=0.)
        coef_poh = Uniform("coef_poh", lower=-50., upper=50, initval=0.)
        coef_crossed = Uniform("coef_crossed", lower=-50., upper=50, initval=0.)
        coef_exposure = Uniform("coef_exposure", lower=-50., upper=50, initval=0.)
        # scale parameter
        constant_scale = Uniform(f"constant_scale", lower=-100, upper=100, initval=init_scale.value)
        sigma = Uniform("sigma_scale", lower=0, upper=100)
        ls = mc.Gamma("length_scale_scale", mu=200, sigma=50)
        latent = mc.gp.Latent(cov_func=Matern32Chordal(2, ls), )
        eps = latent.prior("eps_scale", _X, dims="grid", jitter=1e-7)  # corr inter-cells
        scale = custom(constant_scale + eps[_grid] * sigma + coef_poh * _poh + coef_meshs * _meshs \
                       + coef_crossed * _meshs * _poh + coef_exposure * _exp)
        # observed
        observed = link_pot(data.pos_error.loc[_above.eval({})])
        gpd_without_loc(scale[_above], shape[_above], observed)
        above_bool = (data.pos_error - exp_threshold > 0).astype(int)

        def key_term_nonzero_shape(x):
            arr = (1 + shape * x / scale)
            arr = at.switch(arr < 0, 0, arr)
            arr = arr ** (-1 / shape)
            return arr

        def key_term_zero_shape(x):
            arr = (-x / scale).exp()
            return arr

        def key_term(x, shape):
            return at.switch(at.abs(shape) >= tol, key_term_nonzero_shape(x), key_term_zero_shape(x))

        mc.Deterministic('intensity', key_term(np.repeat(threshold, len(data)), shape))
        bernouilli_over_threshold(scale, shape, above_bool)


def build_gpd_model(data):
    data = data.reset_index()
    below_idx = np.array(data[np.log1p(data.pos_error) < 8.06].index)
    above_idx = np.array(data[np.log1p(data.pos_error) >= 8.06].index)
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
    gpd_extreme(model, data)
    return model


def fit_marginal_model_for_claim_values(data):
    model = build_gpd_model(data)
    with model:
        # fit model
        print('Fitting GPD model for extreme claims...')
        trace = mc.sample(2000, tune=100, init="adapt_diag", chains=3,
                          cores=1, progressbar=True)
    return trace


if __name__ == '__main__':
    train_data = get_train_data()
    trace = fit_marginal_model_for_claim_values(train_data)
    name = datetime.datetime.now().strftime('%Y%m%d_%H:%M')
    path = str(pathlib.Path(FITS_ROOT / 'claim_values_poisson' / name).with_suffix('.nc'))
    ndf = trace.to_netcdf(path)

    az.plot_trace(trace, ['shape', 'constant_scale', 'coef_poh', 'coef_meshs', 'coef_exposure', 'coef_crossed'])
    plt.show()
    az.plot_trace(trace, ['sigma_scale', 'length_scale_scale'])
    plt.show()
