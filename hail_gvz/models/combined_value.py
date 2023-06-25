import pathlib

import aesara.tensor as at
import arviz as az
import numpy as np
import pandas as pd
import pymc as mc
import xarray as xr
from pymc import Uniform

from pykelihood.distributions import GPD
from pymc_utils.pymc_distributions import Matern32Chordal, RatQuadChordal, sigmoid, beta_distri_unobserved, combined_beta_gpd, gpd_without_loc_unobserved, bernoulli_distri, combined_bern_beta_gpd, \
    bernoulli_distri_unobserved
from threshold_selection import threshold_selection_GoF

DATA_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/data/GVZ_Datenlieferung_an_ETH/')
FITS_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/fits/')
scaling_factor = 100
tol = 1e-3
threshold = 8.06
exp_threshold = np.exp(threshold) - 1
link_pot = lambda x: np.log1p(x) - threshold
link_beta = lambda x: x / (np.exp(threshold) - 1)
name_pot = '20230228_05:00'
name_beta = '20230220_12:41'
name_bern = '20230221_12:58'
trace_beta_solo = az.from_netcdf(str(pathlib.Path(FITS_ROOT / 'claim_values' / name_beta).with_suffix('.nc')))
trace_gpd_solo = az.from_netcdf(str(pathlib.Path(FITS_ROOT / 'claim_values' / name_pot).with_suffix('.nc')))
trace_bern_solo = az.from_netcdf(str(pathlib.Path(FITS_ROOT / 'claim_values' / name_bern).with_suffix('.nc')))


def get_gpd_params(data):
    gpd_params = GPD.fit(data, loc=data.min()).flattened_params
    return [g.value for g in gpd_params]


def get_threshold(hail_data):
    ext_data = hail_data.groupby('claim_date').error.agg(risk_metric)
    return threshold_selection_GoF(ext_data, min_threshold=2, max_threshold=10)[0]


def risk_metric(x):
    return np.log1p(np.sum(x / scaling_factor))


def initialize_model(data):
    claim_values = pd.read_csv(DATA_ROOT / 'processed.csv', parse_dates=['claim_date'])
    grid_idx, _ = pd.factorize(data["gridcell"], sort=True)
    _, gridcells = pd.factorize(claim_values["gridcell"], sort=True)
    season_idx, seasons = pd.factorize(data.season, sort=True)[0], [0, 1]
    X = claim_values.groupby('gridcell').mean()[['longitude_grid', 'latitude_grid']].values
    coords = {"grid": gridcells,
              'point': data.index,
              'season': seasons,
              "feature": ["longitude", "latitude"]}
    model = mc.Model(coords=coords)
    with model:
        mc.ConstantData('longitude', data.longitude)
        mc.ConstantData('latitude', data.latitude)
        mc.ConstantData('climada_dmg', data.climada_dmg)
        mc.ConstantData('threshold', threshold)
        mc.ConstantData("grid_idx", grid_idx, dims="point")
        mc.ConstantData('season_idx', season_idx, dims='point')
        mc.ConstantData("lonlat_mapping", X, dims=("grid", "feature"))
        # covariates
        mc.ConstantData('unscaled_exposure', data.exposure, dims='point')
        mc.ConstantData('exposure', (np.log(data.exposure) - np.log(data.exposure).min()) / (np.log(data.exposure).max() - np.log(data.exposure).min()), dims='point')
        mc.ConstantData('meshs', data.MESHS / scaling_factor, dims='point')
        mc.ConstantData('poh', data.POH / scaling_factor, dims='point')
    return model


def beta_model(model):
    _grid = model.grid_idx
    _X = model.lonlat_mapping
    _poh = model.poh
    _meshs = model.meshs
    _exp = model.exposure
    with model:
        # alpha
        constant_mu = Uniform(f"constant_mu", lower=0., upper=10, initval=trace_beta_solo.posterior.constant_mu.mean(['chain', 'draw']))
        sigma_grid = Uniform("sigma_grid", lower=0., upper=2, initval=trace_beta_solo.posterior.sigma_grid.mean(['chain', 'draw']))
        ls = mc.Gamma("length_scale_sigma", mu=5, sigma=2., initval=trace_beta_solo.posterior.length_scale_sigma.mean(['chain', 'draw']))
        latent = mc.gp.Latent(cov_func=RatQuadChordal(2, ls), )
        eps = latent.prior("eps_grid", _X, dims="grid", jitter=1e-6, initval=trace_beta_solo.posterior.eps_grid.mean(['chain', 'draw']))  # corr inter-cells
        gridwise_mu = Uniform(f"grid_scale", lower=-10, upper=10, dims='grid', initval=trace_beta_solo.posterior.grid_scale.mean(['chain', 'draw']))
        grid_mu = gridwise_mu[_grid] + eps[_grid] * sigma_grid
        coef_meshs = Uniform("coef_meshs_b", lower=-5., upper=5, initval=trace_beta_solo.posterior.coef_meshs_b.mean(['chain', 'draw']))
        coef_poh = Uniform("coef_poh_b", lower=-5., upper=5, initval=trace_beta_solo.posterior.coef_poh_b.mean(['chain', 'draw']))
        coef_exposure = Uniform("coef_exposure_b", lower=-5., upper=5, initval=trace_beta_solo.posterior.coef_exposure.mean(['chain', 'draw']))
        alpha = constant_mu + grid_mu + coef_poh * _poh + coef_meshs * _meshs + coef_exposure * _exp
        mu = sigmoid(alpha)
        # sigma
        kappa = mc.HalfNormal('precision', sigma=1000, initval=trace_beta_solo.posterior.precision.mean(['chain', 'draw']))  # precision parameter
        sigma = (mu * (1 - mu) / (kappa + 1)).sqrt()
    return mu, sigma


def gpd_model(model):
    _grid = model.grid_idx
    _X = model.lonlat_mapping
    _season = model.season_idx
    _poh = model.poh
    _meshs = model.meshs
    _exp = model.exposure
    with model:
        # shape parameter
        init_shape = [-0.12960985886908538, -4.743355860126307e-05]
        shape = mc.Normal("shape", mu=init_shape, sigma=[0.1, 0.05], dims='season', initval=init_shape)[_season]
        # scale parameter
        constant_scale = Uniform(f"constant_scale", lower=-40, upper=40, initval=-2.0202707317519466)
        coef_meshs = Uniform("coef_meshs", lower=-5., upper=5, initval=trace_gpd_solo.posterior.coef_meshs.mean(['chain', 'draw']))
        coef_exposure = Uniform("coef_exposure", lower=-5., upper=5, initval=trace_gpd_solo.posterior.coef_exposure.mean(['chain', 'draw']))
        coef_crossed = Uniform("coef_crossed", lower=-5., upper=5, initval=trace_gpd_solo.posterior.coef_crossed.mean(['chain', 'draw']))
        sigma = Uniform("sigma_scale", lower=0, upper=15, initval=1.)
        ls = 90  # mc.Gamma("length_scale_scale", mu=5, sigma=1, initval=0.)
        latent = mc.gp.Latent(cov_func=Matern32Chordal(2, ls), )
        eps = latent.prior("eps_scale", _X, dims="grid", jitter=1e-7, initval=trace_gpd_solo.posterior.eps_scale.mean(['chain', 'draw']))  # corr inter-cells
        glm = constant_scale + coef_meshs * _meshs + coef_exposure * _exp + coef_crossed * _poh * _meshs + eps[_grid] * sigma
        scale = (glm / scaling_factor).exp()
        mc.Potential('bound', -at.switch(shape < -tol, at.abs(-scale / shape - 7.603124653521049), 0).sum() / scaling_factor)
    return scale, shape


def bernoulli_over_threshold(model):
    _season = model.season_idx
    _grid = model.grid_idx
    _poh = model.poh
    _meshs = model.meshs
    _exp = model.exposure
    with model:
        coef_meshs = Uniform("coef_meshs_bern", lower=-100., upper=100, initval=trace_bern_solo.posterior.coef_meshs_bern.mean(['chain', 'draw']))
        coef_crossed_bern = Uniform("coef_crossed_bern", lower=-100., upper=100, initval=trace_bern_solo.posterior.coef_crossed_bern.mean(['chain', 'draw']))
        seasonal = Uniform("seasonal_bern", lower=-50., upper=50, dims='season', initval=trace_bern_solo.posterior.seasonal_bern.mean(['chain', 'draw']))
        gridcell = Uniform("gridcell_bern", lower=-100., upper=100, dims='grid', initval=trace_bern_solo.posterior.gridcell_bern.mean(['chain', 'draw']))
        coef_exposure = Uniform("coef_exposure_bern", lower=-100., upper=100, initval=trace_bern_solo.posterior.coef_exposure_bern.mean(['chain', 'draw']))
        coef_poh = Uniform("coef_poh_bern", lower=-100., upper=100, initval=trace_bern_solo.posterior.coef_poh_bern.mean(['chain', 'draw']))
        constant_alpha = Uniform(f"constant_alpha", lower=0, upper=400, initval=trace_bern_solo.posterior.constant_alpha.mean(['chain', 'draw']))
        alpha = constant_alpha + coef_exposure * _exp + gridcell[_grid] + seasonal[_season] \
                + coef_meshs * _meshs + coef_poh * _poh + coef_crossed_bern * _poh * _meshs
        p = sigmoid(alpha)
    return p


def define_model_for_days_with_claims(model, data, fit=False):
    observed = data.pos_error
    above_bool = (observed - exp_threshold > 0).astype(int)
    mu, sigma = beta_model(model)
    scale, shape = gpd_model(model)
    p = bernoulli_over_threshold(model)
    with model:
        gpd = gpd_without_loc_unobserved(scale, shape)
        beta = beta_distri_unobserved(mu, sigma)
        if fit:
            bernoulli_distri(p, above_bool)
            combined_beta_gpd(beta, gpd, observed)
        else:
            bern = bernoulli_distri_unobserved(p)
            combined_bern_beta_gpd(bern, beta, gpd, observed)


def build_model(data, fit=False):
    data = data.sort_values('claim_date').reset_index()
    model = initialize_model(data)
    define_model_for_days_with_claims(model, data, fit)
    return model


def get_chosen_variables_for_model(model, nb_draws=None):
    vars = [v.name for v in model.free_RVs]
    bern_vars = [v for v in vars if 'bern' in v or v == 'constant_alpha']
    beta_vars = [v for v in vars[:9]]
    pot_vars = [v for v in vars if v not in bern_vars and v not in beta_vars and v in [v for v in trace_gpd_solo.posterior.variables]]
    trace_bern = trace_bern_solo.posterior[bern_vars]
    trace_beta = trace_beta_solo.posterior[beta_vars]
    trace_pot = trace_gpd_solo.posterior[pot_vars]
    traces = [trace_bern, trace_beta, trace_pot]
    common_draws = nb_draws or np.min([trace.draw.shape[0] for trace in traces])
    trace_bern = trace_bern.isel(draw=slice(0, common_draws))
    trace_beta = trace_beta.isel(draw=slice(0, common_draws))
    trace_pot = trace_pot.isel(draw=slice(0, common_draws))
    traces = [trace_bern, trace_beta, trace_pot]
    return xr.merge(traces)


def fit_marginal_model_for_claim_values(data):
    model = build_model(data, fit=True)
    with model:
        # fit model
        print('Fitting model...')
        trace = mc.sample(2000, init="adapt_diag", chains=1,
                          cores=1, progressbar=True)
    return trace

if __name__ == '__main__':
    import arviz as az
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.pylab as pylab
    matplotlib.rcParams["text.usetex"] = False
    params = {'legend.fontsize': 'x-large',
              'axes.facecolor': '#eeeeee',
              'axes.labelsize': 'xx-large', 'axes.titlesize': 20, 'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    pylab.rcParams.update(params)
    az.plot_trace(trace_beta_solo)
    plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
    tt = trace_gpd_solo.posterior.rename({'shape': r'xi', 'coef_meshs': r'sigma_1', 'constant_scale': r'sigma_0', 'coef_crossed': r'sigma_2', 'coef_exposure': r'sigma_3'})
    az.plot_autocorr(tt, var_names=[r'xi', r'sigma_1',  r'sigma_0',  r'sigma_2',  r'sigma_3'], ax=axes.flatten())
    fig.suptitle('Autocorrelation through time for parameters of the GPD model', fontsize=20)
    fig.show()
