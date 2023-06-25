import datetime
import pathlib
from glob import glob

import aesara.tensor as at
import arviz as az
import cartopy
import geopandas
import geoplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as mc
from shapely.geometry import Point
import xarray as xr
from data.hailcount_data_processing import get_train_data, get_grid_mapping, get_exposure, delta, timestep, get_validation_data, get_test_data
from pymc_utils.pymc_distributions import sigmoid, Matern32Chordal, beta_distri, bernoulli_distri, bernoulli_distri_unobserved, combined_dirac_beta, beta_distri_unobserved

PRED_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/prediction/')
DATA_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/data/GVZ_Datenlieferung_an_ETH/Ophelia/')
FITS_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/fits/')
tol = 1e-5
scaling_factor = 1e6
pow_climada = 3
origin = (8.36278492095831, 47.163852336888695)
timestep_days = pd.to_timedelta(timestep).days
area = delta ** 2
previous = '20230313_08:43'
PLOT_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/plots/')
trace_previous = az.from_netcdf(str(pathlib.Path(FITS_ROOT / 'claim_counts' / previous).with_suffix('.nc')))
threshold = 0.3


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


def beta_prop_model(model, fit=True):
    _cnt_grid = model.cnt_gridcell
    _grid = model.grid_idx
    _X = model.lonlat_mapping
    _season = model.season_idx
    _wind_dir = model.wind_dir
    _wind_speed = model.wind_speed
    _poh = model.poh
    _meshs = model.meshs
    _exp = model.exposure
    _vol = model.volume
    _climadacnt = model.climada_cnt / scaling_factor
    _obscnt = model.obs_cnt
    with model:
        lambd0 = mc.Uniform('constant_intensity', lower=0., upper=100)
        # coef_poh_cnt = Uniform("coef_poh_cnt", lower=-50., upper=50, initval=0)
        coef_meshs_cnt = mc.Uniform("coef_meshs_cnt", lower=-50., upper=50, initval=0)
        # coef_ws_cnt = Uniform("coef_ws_cnt", lower=-50., upper=50, initval=0)
        sigma_dist = mc.Uniform("sigma_dist", 20, 500)
        coef_climada = mc.Uniform("coef_climada", lower=[-10.] * pow_climada, upper=[10] * pow_climada)
        coef_cross_climada_dist = mc.Uniform("coef_cross_climada_dist", lower=-50., upper=50)
        sigma = mc.Uniform("sigma_poisson", lower=0, upper=2)
        sigma1 = mc.Uniform("sigma1", lower=0, upper=2)
        ls = float(trace_previous.posterior.ls.mean())  # mc.Gamma("ls", mu=200, sigma=50, initval=trace_previous.posterior.ls.mean())
        deviation_from_origin = mc.Uniform('deviation_from_origin', lower=[-1] * 3, upper=[1] * 3, dims='season')[
            _season]
        new_origin = (origin[0], origin[1] + deviation_from_origin)
        slope = _wind_dir
        new_lats = slope * (_X[..., 0][_grid] - origin[0]) + new_origin[1]
        new_origin_array = at.stack([np.tile(new_origin[0], new_origin[1].shape.eval({})), new_origin[1]], axis=1)
        dist_from_line = hauteur(new_origin_array, at.stack([_X[..., 0][_grid], new_lats], axis=1), _X[_grid])
        dist_term = sigma_dist / (1 + dist_from_line) - 1
        latent = mc.gp.Latent(cov_func=Matern32Chordal(2, ls), )
        eps = latent.prior("eps_scale", _X, dims="grid", jitter=1e-8, initval=trace_previous.posterior.eps_scale.mean(['draw', 'chain']))  # corr inter-cells
        seasonal = mc.Uniform('seasonal_comp', lower=[-50] * 3, upper=[50] * 3, dims='season', initval=trace_previous.posterior.seasonal_comp.mean(['chain', 'draw']))[_season]
        climada_vec = np.array([np.log1p(_climadacnt.eval({})) ** p for p in range(1, pow_climada + 1)])
        pointwise_block = lambd0 + at.math.dot(coef_climada, climada_vec) + coef_cross_climada_dist * _climadacnt * dist_term \
                          + coef_meshs_cnt * _meshs * dist_term + seasonal  # coef_ws_cnt * _wind_speed + coef_poh_cnt * _poh dist_term
        climada_block = pointwise_block + eps[
            _grid] * sigma
        null_climada_block = pointwise_block + eps[_grid] * sigma1
        logmu = at.switch(_climadacnt > 0, climada_block, null_climada_block)
        glm_mu = logmu / scaling_factor
        mu = sigmoid(glm_mu)
        kappa = mc.HalfNormal('precision', sigma=1000)  # precision parameter
        sigma = (mu * (1 - mu) / (kappa + 1)).sqrt()
        # psi
        p1 = len(_obscnt[_obscnt <= 0].eval({})) / len(_obscnt.eval({}))
        psi0 = mc.Uniform('psi0', initval=p1)
        c = mc.Uniform('c', -50, 50)
        glm_psi = psi0 + c.dot([glm_mu])
        psi = sigmoid(glm_psi / scaling_factor)
        observed = _obscnt[_obscnt > 0].eval({})
        cat = (_obscnt > 0).astype(int).eval({})
        if fit:
            bernoulli_distri(p=psi, value=cat)
            beta_distri(mu=glm_mu[_obscnt > 0], sigma=sigma[_obscnt > 0], value=observed)
        else:
            bern = bernoulli_distri_unobserved(psi)
            beta = beta_distri_unobserved(mu=glm_mu, sigma=sigma)
            combined_dirac_beta(bern, beta, _obscnt)

        # extreme counts
        # p2 = len(_obscnt[_obscnt > threshold].eval({})) / len(_obscnt.eval({}))
        # p3 = 1 - (p1 + p2)
        # shape = mc.Normal('shape_poisson', mu=0.24, sigma=0.25, initval=0.24)
        # scale = (glm_mu + sigma2 * eps[_grid]).exp()
        # # weights
        # weights = mc.Dirichlet('weights', psi.T)
        # sum_w_supzero = weights[..., 1:].sum(axis=-1)[_obscnt > 0]
        # sum_w_supzero = at.switch(sum_w_supzero <= 0, tol, sum_w_supzero)
        # w2 = (weights[_obscnt > 0, 1:].T / sum_w_supzero).T
        # valid_weights = at.isclose(at.sum(w2, axis=-1), 1)
        # # observed
        # observed = _obscnt[_obscnt > 0].eval({})
        # cat = at.switch(_obscnt > threshold, 2, at.switch(_obscnt > 0, 1, 0)).eval({})
        # # extremes are values above threshold
        # beta = beta_distri_unobserved(mu=glm_mu[_obscnt > 0][valid_weights], sigma=sigma[_obscnt > 0][valid_weights])
        # gpd = gpd_without_loc_unobserved(scale[_obscnt > 0][valid_weights], shape)
        # mc.Categorical('zero_below_above', p=weights, observed=cat)
        # mc.Mixture('nonzero', w2[valid_weights], [beta, gpd], observed=observed)


def build_beta_model(data, m, exposure, fit=False):
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
        mc.ConstantData("lonlat_mapping", X_grid, dims=("gridcell", "feature"))  # p x 2
        # covariates
        mc.ConstantData('exposure', exp_gridcell.value, dims='gridcell')
        mc.ConstantData('volume', (exp_gridcell.volume - exp_gridcell.volume.min()) / (exp_gridcell.volume.max() - exp_gridcell.volume.min()), dims='grid')
        mc.ConstantData('meshs', data.meshs / scaling_factor, dims='point')
        mc.ConstantData('poh', data.poh / scaling_factor, dims='point')
        mc.ConstantData('wind_dir', data.mean_wind_dir / 360, dims='point')
        mc.ConstantData('cnt_gridcell', data.cnt_gridcell, dims='point')
        mc.ConstantData('wind_speed', (data.wind_speed - data.wind_speed.min()) / (data.wind_speed.max() - data.wind_speed.min()), dims='point')
        mc.ConstantData('climada_cnt', data.climadacnt, dims='point')
        # observed
        mc.ConstantData('obs_cnt', data.obscnt / data.cnt_gridcell, dims='point')
    beta_prop_model(model, fit)
    return model


def fit_marginal_model_for_claim_count(data, mapping, exposure):
    model = build_beta_model(data, mapping, exposure, fit=True)
    with model:
        # fit model
        print('Fitting model for claim count...')
        trace = mc.sample(100, tune=100, target_accept=0.95, chains=1,
                          cores=1, progressbar=True)
    return trace


if __name__ == '__main__':
    train_data = get_train_data()
    valid_data = get_validation_data()
    test_data = get_test_data()

    mapping = get_grid_mapping()
    exposure = get_exposure()
    trace = az.from_netcdf(str(pathlib.Path(FITS_ROOT / 'claim_counts' / '20230312_15:16').with_suffix('.nc')))#fit_marginal_model_for_claim_count(train_data, mapping, exposure)
    name_mod = datetime.datetime.now().strftime('%Y%m%d_%H:%M')
    path = str(pathlib.Path(FITS_ROOT / 'claim_counts' / name_mod).with_suffix('.nc'))
    ndf = trace.to_netcdf(path)
    f = trace.posterior.eps_scale.mean('draw').to_dataframe().reset_index()
    f = f.rename(columns={'grid': 'gridcell'}).merge(mapping.drop_duplicates(subset='gridcell')[['geometry', 'gridcell']], on='gridcell', how='left')
    f = f.set_geometry('geometry')
    geoplot.choropleth(f, hue='eps_scale', legend=True)
    plt.show()
    trace.log_likelihood.beta_sum.mean('chain').plot()
    plt.show()
    trace.log_likelihood.over_threshold.mean('chain').plot()
    plt.show()

    for data, name in zip([train_data], ['train']):#, valid_data, test_data , 'val', 'test'
        p = mc.sample_posterior_predictive(trace, model=build_beta_model(data, mapping, exposure))
        d = xr.merge([p.constant_data.meshs,
                      p.constant_data.grid_idx,
                      p.constant_data.time_idx,
                      p.constant_data.poh,
                      p.constant_data.volume,
                      p.constant_data.cnt_gridcell,
                      p.constant_data.climada_cnt,
                      (p.observed_data.counts*p.constant_data.cnt_gridcell).rename('obscnt'),
                      (p.posterior_predictive.counts * p.constant_data.cnt_gridcell).quantile(0.05, ['chain', 'draw']).rename('lb_counts').drop('quantile'),
                      (p.posterior_predictive.counts * p.constant_data.cnt_gridcell).quantile(1 - 0.05, ['chain', 'draw']).rename('ub_counts').drop('quantile'),
                      (p.posterior_predictive.counts*p.constant_data.cnt_gridcell).mean(['chain', 'draw']).rename('pred_cnt')]).drop_dims('grid')
        counts_df = d.to_dataframe().assign(claim_date=lambda x: data.reset_index().claim_date.unique()[x.time_idx]) \
            .assign(gridcell=lambda x: mapping.gridcell.unique()[x.grid_idx]).merge(mapping.drop_duplicates(subset='gridcell')[['gridcell', 'geometry']], on='gridcell',
                                                                                    how='left')
        m = mapping
        j = counts_df.set_geometry('geometry')
        j = j.assign(mean_pred=lambda x: x.pred_cnt / x.cnt_gridcell)
        path = PRED_ROOT / name_mod
        path.mkdir(parents=True, exist_ok=True)
        j.set_index(['claim_date', 'gridcell']).to_csv(path / f'mean_pred_{name}_{name_mod}.csv')
        p.to_netcdf(str(pathlib.Path(path / f'pred_{name}_{name_mod}').with_suffix('.nc')))

        geom_roots = glob(str(pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/data/GVZ_Datenlieferung_an_ETH/') / 'SHAPEFILE/swissTLMRegio_Boundaries_LV95/swissTLMRegio_KANTONSGEBIET_LV95.shp'))
        df_polygon = pd.concat([geopandas.read_file(geom_root) for geom_root in geom_roots]).to_crs(epsg='4326')
        df_polygon = df_polygon[df_polygon.NAME == 'Zürich']
        lakes = glob(str(pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/data/GVZ_Datenlieferung_an_ETH/') / 'SHAPEFILE/swissTLMRegio_Product_LV95/Hydrography/swissTLMRegio_Lake.shp'))
        df_lakes = geopandas.read_file(lakes[0]).to_crs(epsg='4326')

        for date in j.claim_date.unique():
            spec_date = j[j.claim_date == date]
            poscounts = spec_date[spec_date.obscnt >= 1]
            pospred = spec_date[spec_date.pred_cnt >= 1]
            if len(poscounts) > 10:
                fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 10), constrained_layout=True, subplot_kw={'projection': cartopy.crs.PlateCarree()})
                geoplot.choropleth(poscounts, hue='obscnt', cmap='OrRd', legend=True, ax=ax1, legend_kwargs={'shrink': 0.7})
                geoplot.choropleth(spec_date, hue='climada_cnt', cmap='OrRd', legend=True, ax=ax2, legend_kwargs={'shrink': 0.7})
                geoplot.choropleth(pospred, hue='ub_counts', cmap='OrRd', legend=True, ax=ax3, legend_kwargs={'shrink': 0.7})
                for ax in [ax1, ax2, ax3]:
                    df_polygon.plot(ax=ax, facecolor='oldlace', edgecolor='grey')
                    df_lakes.plot(ax=ax, facecolor='royalblue', alpha=0.5)
                    ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
                    ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), color='black')
                    ax.set_extent([m.longitude.min(), m.longitude.max(), m.latitude.min(), m.latitude.max()])
                fig.suptitle(date)
                fig.show()
                path = PLOT_ROOT / name_mod / f'pred_poisson_{name}_{date}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, DPI=200)
