import datetime
import pathlib

import aesara.tensor as at
import arviz as az
import geopandas
import geoplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as mc
from pymc import Uniform
from shapely.geometry import Point

from constants import FITS_ROOT, PLOT_ROOT, pow_climada, origin, scaling_factor
from data.hailcount_data_processing import get_train_data, get_grid_mapping, get_exposure
from pymc_utils.pymc_distributions import sigmoid, Matern32Chordal, binom_counts_alphamu


def distance_from_coordinates(z1, z2):
    """

    :param z1: tuple of lon, lat for the first place
    :param z2: tuple of lon, lat for the second place
    :return: distance between the 2 places in km
    """
    lon1, lat1 = z1[..., 0], z1[..., 1]
    lon2, lat2 = z2[..., 0], z2[..., 1]
    # Harvestine formula
    r = 6371  # radius of Earth (KM)
    p = np.pi / 180
    try:
        a = 0.5 - ((lat2 - lat1) * p).cos() / 2 + (lat1 * p).cos() * (lat2 * p).cos() * (
                1 - ((lon2 - lon1) * p).cos()) / 2
        d = 2 * r * a.sqrt().arcsin()
    except:
        a = 0.5 - (np.cos((lat2 - lat1) * p)) / 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (
                1 - np.cos((lon2 - lon1) * p)) / 2
        d = 2 * r * np.arcsin((np.sqrt(a)))
    return d


def hauteur(z1, z2, z3):
    a = distance_from_coordinates(z1, z2)
    b = distance_from_coordinates(z2, z3)
    c = distance_from_coordinates(z1, z3)
    p = (a + b + c) / 2
    try:
        # should not happen but when point on the line, small calculation errors make the S nan
        pa = at.switch(p - a < 0, 0, p - a)
        pb = at.switch(p - b < 0, 0, p - b)
        pc = at.switch(p - c < 0, 0, p - c)
        S = (p * pa * pb * pc).sqrt()
    except:
        pa = np.where(p - a < 0, 0, p - a)
        pb = np.where(p - b < 0, 0, p - b)
        pc = np.where(p - c < 0, 0, p - c)
        S = np.sqrt((p * pa * pb * pc))
    d = 2 * S / a
    return d


def hauteur_degrees(z1, z2, z3):
    z1_series = geopandas.GeoSeries([Point(x, y) for (x, y) in z1], crs='epsg:4326')
    z2_series = geopandas.GeoSeries([Point(x, y) for (x, y) in z2], crs='epsg:4326')
    z3_series = geopandas.GeoSeries([Point(x, y) for (x, y) in z3], crs='epsg:4326')
    a = z1_series.distance(z2_series)
    b = z2_series.distance(z3_series)
    c = z1_series.distance(z3_series)
    p = (a + b + c) / 2
    S = np.sqrt((p * (p - a) * (p - b) * (p - c)))
    d = 2 * S / a
    return d


def poisson_counts_model(model):
    _cnt_grid = model.cnt_gridcell
    _grid = model.grid_idx
    _X = model.lonlat_mapping
    _season = model.season_idx
    _wind_dir = model.wind_dir
    _wind_speed = model.wind_speed
    _poh = model.poh
    _meshs = model.meshs
    _climadacnt = model.climada_cnt
    _obscnt = model.obs_cnt
    with model:
        lambd0 = Uniform('constant_intensity', lower=0., upper=100)
        coef_meshs_cnt = Uniform("coef_meshs_cnt", lower=-50., upper=50)
        coef_ws_cnt = Uniform("coef_ws_cnt", lower=-50., upper=50)
        sigma_dist = mc.Gamma("sigma_dist", mu=10, sigma=2)
        coef_climada = Uniform("coef_climada", lower=[-10.] * pow_climada, upper=[10] * pow_climada, initval=[0.] * pow_climada)
        coef_cross_climada_dist = Uniform("coef_cross_climada_dist", lower=-50., upper=50)
        sigma = Uniform("sigma_poisson", lower=0, upper=10)
        sigma1 = mc.Uniform("sigma1", lower=0, upper=10)
        ls = mc.Gamma("ls", mu=200, sigma=50)
        deviation_from_origin = mc.Uniform('deviation_from_origin', lower=[-1] * 3, upper=[1] * 3, dims='season')[_season]
        new_origin = (origin[0], origin[1] + deviation_from_origin)
        slope = _wind_dir
        new_lats = slope * (_X[..., 0][_grid] - origin[0]) + new_origin[1]
        new_origin_array = at.stack([np.tile(new_origin[0], new_origin[1].shape.eval({})), new_origin[1]], axis=1)
        dist_from_line = hauteur(new_origin_array, at.stack([_X[..., 0][_grid], new_lats], axis=1), _X[_grid])
        dist_term = sigma_dist / (1 + dist_from_line) - 1
        latent = mc.gp.Latent(cov_func=Matern32Chordal(2, ls), )
        eps = latent.prior("eps_scale", _X, dims="grid", jitter=1e-7)
        seasonal = mc.Uniform('seasonal_comp', lower=[-50] * 3, upper=[50] * 3, dims='season')[_season]
        climada_vec = np.array([_climadacnt.eval({}) ** p for p in range(1, pow_climada + 1)])
        pointwise_block = lambd0 + at.math.dot(coef_climada, climada_vec) + coef_ws_cnt * _wind_speed * dist_term + coef_cross_climada_dist * _climadacnt * dist_term \
                          + coef_meshs_cnt * _meshs * dist_term + seasonal
        climada_block = pointwise_block + eps[
            _grid] * sigma
        # null_climada_block = pointwise_block + eps[_grid] * sigma1
        logmu = climada_block  # at.switch(_climadacnt >= 1, climada_block, null_climada_block)
        glm_mu = logmu / scaling_factor
        # psi
        psi0 = mc.Gamma('psi0', alpha=2, beta=2)
        c = mc.Uniform('c', -50, 50)
        d = mc.Uniform('d', -50, 50)
        glm_psi = psi0 + c * (_climadacnt >= 1).astype(int) + d * dist_term * _meshs * _poh + seasonal
        psi = sigmoid(glm_psi / scaling_factor ** 2)
        alpha = mc.Gamma('alpha', 2, 2)
        # observed
        observed = _obscnt
        mu = at.where(glm_mu.exp() > _cnt_grid, _cnt_grid, glm_mu.exp())
        binom_counts_alphamu(psi=psi, mu=mu, alpha=alpha, value=observed)


def build_poisson_model(data, m, exposure):
    mapping = m
    data = data.reset_index()
    mean_wind_dir = data.groupby(['claim_date'], as_index=False).agg({'wind_dir': 'mean'}).rename(columns={'wind_dir': 'mean_wind_dir'})
    data = data.merge(mean_wind_dir, how='left', on=['claim_date'])
    count_gridcell = mapping.groupby('gridcell').latitude.count().rename('cnt_gridcell')
    data = data.merge(count_gridcell.reset_index(), on='gridcell', how='left')
    exp_gridcell = exposure.merge(mapping.set_index(['latitude', 'longitude']), left_index=True, right_index=True, how='right').fillna(0.).groupby('gridcell').agg({'volume': 'sum', 'value': 'sum'})
    grid_idx, gridcells = pd.factorize(data.gridcell, sort=True)
    time_idx, time = pd.factorize(data.claim_date, sort=True)
    season_idx, seasons = pd.factorize(data.season, sort=True)[0], [0, 1, 2]
    X_grid = mapping[['lon_grid', 'lat_grid', 'gridcell']].drop_duplicates().set_index('gridcell')
    mapping_time_season, _ = pd.factorize(data.groupby('time', sort=False).season.mean(), sort=True)
    coords = {"grid": gridcells,
              'point': data.index,
              'location': mapping.index,
              'time': time,
              'season': seasons,
              "feature": ["longitude", "latitude"]}
    model = mc.Model(coords=coords)
    with model:
        mc.ConstantData("gridcell", data.gridcell, dims="point")
        mc.ConstantData("grid_idx", grid_idx, dims="point")
        mc.ConstantData("time_idx", time_idx, dims="point")
        mc.ConstantData('season_idx', season_idx, dims='point')  # N
        mc.ConstantData("lonlat_mapping", X_grid, dims=("grid", "feature"))  # p x 2
        # covariates
        mc.ConstantData('exposure', exp_gridcell.value, dims='grid')
        mc.ConstantData('volume', (exp_gridcell.volume - exp_gridcell.volume.min()) / (exp_gridcell.volume.max() - exp_gridcell.volume.min()), dims='grid')
        mc.ConstantData('meshs', data.meshs / scaling_factor, dims='point')
        mc.ConstantData('poh', data.poh / scaling_factor, dims='point')
        mc.ConstantData('wind_dir', data.mean_wind_dir / 360, dims='point')
        mc.ConstantData('cnt_gridcell', data.cnt_gridcell, dims='point')
        mc.ConstantData('wind_speed', (data.wind_speed - data.wind_speed.min()) / (data.wind_speed.max() - data.wind_speed.min()), dims='point')
        mc.ConstantData('climada_cnt', data.climadacnt, dims='point')
        # observed
        mc.ConstantData('obs_cnt', data.obscnt, dims='point')
    poisson_counts_model(model)
    return model


def fit_marginal_model_for_claim_count(data, mapping, exposure):
    model = build_poisson_model(data, mapping, exposure)
    with model:
        # fit model
        print('Fitting model for claim count...')
        trace = mc.sample(200, tune=150, chains=1, init='jitter+adapt_diag_grad',
                          progressbar=True)
    return trace


if __name__ == '__main__':
    train_data = get_train_data(suffix='GVZ_emanuel')
    name_mod = datetime.datetime.now().strftime('%Y%m%d_%H:%M')
    mapping = get_grid_mapping(suffix='GVZ_emanuel')
    exposure = get_exposure()
    trace = fit_marginal_model_for_claim_count(train_data, mapping, exposure)
    f = trace.posterior.eps_scale.mean('draw').to_dataframe().reset_index()
    f = f.rename(columns={'grid': 'gridcell'}).merge(mapping.drop_duplicates(subset='gridcell')[['geometry', 'gridcell']], on='gridcell', how='left')
    f = f.set_geometry('geometry')
    geoplot.choropleth(f, hue='eps_scale', legend=True)
    path_plots = PLOT_ROOT / name_mod
    path_plots.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_plots / 'eps.png', DPI=200)
    plt.show()
    plt.clf()

    trace.log_likelihood.counts.plot()
    plt.savefig(path_plots / 'll.png', dpi=200)
    plt.show()
    path = str(pathlib.Path(FITS_ROOT / 'claim_counts' / name_mod).with_suffix('.nc'))
    ndf = trace.to_netcdf(path)

    fig, axes = plt.subplots(ncols=4, nrows=2, constrained_layout=True, figsize=(15, 10))
    tt = trace.posterior.isel(draw=slice(55, None), season=[0, 2]).rename({'constant_intensity': r'$\mu_0$',
                                                                           'coef_climada': r'$\mu_1$',
                                                                           'coef_cross_climada_dist': r'$\mu_2$',
                                                                           'coef_meshs_cnt': '$\mu_3$',
                                                                           'coef_ws_cnt': "$\mu_4$",
                                                                           'sigma_dist': "$\sigma_m$",
                                                                           'seasonal_comp': r'$\epsilon$'})
    vars = ['$\mu_0$', '$\mu_1$', '$\mu_2$', '$\mu_4$', r'$\epsilon$']
    az.plot_autocorr(tt,
                     var_names=vars,
                     ax=axes, max_lag=100)
    var_names = ['$\mu_0$', '$\mu_{11}$', '$\mu_{12}$', '$\mu_{13}$', '$\mu_2$', '$\mu_3$',
                 r'$\epsilon_1$', r'$\epsilon_2$']

    for ax, var in zip(axes.flatten(), var_names):
        ax.set_title(r"{}".format(var))
    fig.suptitle('Autocorrelation through time for parameters of the Negative Binomial model', fontsize=20)
    fig.show()
    fig.savefig(path_plots / 'autocorr_counts.png', dpi=200)

    fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(15, 5))
    tt = trace.posterior.isel(draw=slice(55, None)).rename(
        {'alpha': r'$\alpha$'})
    az.plot_posterior(tt, var_names=[r'$\alpha$'], filter_vars="like",
                      ax=axes[0])
    axes[1].plot(tt[r'$\alpha$'].sel(chain=0))
    axes[0].set_title(r'Autocorrelation through time for $\alpha$')
    axes[1].set_title(r'Evolution of $\alpha$ after initial burn-in sample')
    fig.suptitle('Diagnostic plots for the shape parameter of the Negative Binomial model', fontsize=20)
    fig.show()
    fig.savefig(path_plots / 'diag_alpha_poisson.png', dpi=200)
