import pathlib
from glob import glob

import cartopy
import geopandas
import geoplot
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import arviz as az
from data.hailcount_data_processing import get_exposure, get_grid_mapping, get_train_data, get_validation_data, \
    get_test_data
from diagnostic.metrics import log_spectral_distance_from_xarray, spatially_convolved_ks_stat
from models.counts import hauteur

matplotlib.rcParams["text.usetex"] = True
params = {'legend.fontsize': 'xx-large',
          'axes.facecolor': '#eeeeee',
          'axes.labelsize': 'xx-large', 'axes.titlesize': 20, 'xtick.labelsize': 'xx-large',
          'ytick.labelsize': 'xx-large'}
pylab.rcParams.update(params)
DATA_ROOT = pathlib.Path('/Volumes/ExtremeSSD/hail_gvz/data/GVZ_Datenlieferung_an_ETH/')
FITS_ROOT = pathlib.Path('/Volumes/ExtremeSSD/hail_gvz/fits')
PLOT_ROOT = pathlib.Path('/Volumes/ExtremeSSD/hail_gvz/plots/')
PRED_ROOT = pathlib.Path('/Volumes/ExtremeSSD/hail_gvz/prediction/')
threshold = 8.06
CRS = 'EPSG:2056'
exp_threshold = np.exp(threshold) - 1
exposure = get_exposure()
mapping = get_grid_mapping(suffix='GVZ_emanuel')
conf = 0.05
tol = 1e-2
quants = np.linspace(tol, 1, 200)  # franchise claims
claim_values = pd.read_csv(DATA_ROOT / 'processed.csv', parse_dates=['claim_date'])
# Climada damages downscaled per building
climada_damages = pd.read_csv(DATA_ROOT / 'Ophelia' / 'GVZ_emanuel' / 'climada_dmg.csv', index_col=['latitude', 'longitude'],
                              usecols=lambda x: ('2' in x) or x in ['latitude', 'longitude'])
climada_damages_date = [climada_damages[c].rename('climadadmg').to_frame().assign(claim_date=pd.to_datetime(c)) for c in climada_damages.columns]
climada_pred_pos = pd.concat([d[d.climadadmg > 0] for d in climada_damages_date])
climada_damages = climada_pred_pos.reset_index().assign(geom=lambda x: geopandas.points_from_xy(x.longitude, x.latitude, crs=CRS).to_crs('epsg:4326')) \
    .set_geometry('geom').assign(longitude=lambda x: x.geom.x).assign(latitude=lambda x: x.geom.y).merge(mapping[['longitude', 'latitude', 'gridcell', 'geometry']],
                                                                                                         on=['latitude', 'longitude'], how='left').set_geometry('geometry')
climada_damages = climada_damages.set_index(['longitude', 'latitude', 'claim_date'])
# Climada counts per building using PAA
climada_counts = pd.read_csv(DATA_ROOT / 'Ophelia' / 'GVZ_emanuel' / 'climada_cnt.csv', index_col=['latitude', 'longitude'],
                             usecols=lambda x: ('2' in x) or x in ['latitude', 'longitude'])
climada_counts_date = [climada_counts[c].rename('climadadmg').to_frame().assign(claim_date=pd.to_datetime(c)) for c in climada_counts.columns]
climada_count_pos = pd.concat([d[d.climadadmg > 0] for d in climada_counts_date])
climada_counts = climada_count_pos.reset_index().assign(geom=lambda x: geopandas.points_from_xy(x.longitude, x.latitude, crs=CRS).to_crs('epsg:4326')) \
    .set_geometry('geom').assign(longitude=lambda x: x.geom.x).assign(latitude=lambda x: x.geom.y).merge(mapping[['longitude', 'latitude', 'gridcell', 'geometry']],
                                                                                                         on=['latitude', 'longitude'], how='left').set_geometry('geometry')
climada_counts = climada_counts.set_index(['longitude', 'latitude', 'claim_date'])
# Observed damages
obs_damages = claim_values[['claim_date', 'longitude', 'latitude', 'claim_value', 'MESHS']].merge(mapping[['latitude', 'longitude', 'gridcell']], on=['latitude', 'longitude'], how='left')
obs_dmg_gridcell = obs_damages.drop(columns=['longitude', 'latitude']).groupby(['gridcell', 'claim_date']).agg({'claim_value': 'sum', 'MESHS': 'mean'})
poh = pd.read_csv(DATA_ROOT / 'Ophelia' / 'poh.csv', index_col=['latitude', 'longitude'],
                  usecols=lambda x: ('2' in x) or x in ['latitude', 'longitude'])
meshs = pd.read_csv(DATA_ROOT / 'Ophelia' / 'meshs.csv', index_col=['latitude', 'longitude'],
                    usecols=lambda x: ('2' in x) or x in ['latitude', 'longitude'])
geom_roots = glob(str(DATA_ROOT/'SHAPEFILE/swissTLMRegio_Boundaries_LV95/swissTLMRegio_KANTONSGEBIET_LV95.shp'))
df_polygon = pd.concat([geopandas.read_file(geom_root) for geom_root in geom_roots]).to_crs(epsg='4326')
df_polygon = df_polygon[df_polygon.NAME == 'Zürich']
lakes = glob(str(DATA_ROOT / 'SHAPEFILE/swissTLMRegio_Product_LV95/Hydrography/swissTLMRegio_Lake.shp'))
df_lakes = geopandas.read_file(lakes[0]).to_crs(epsg='4326')

def plot_mc_diagnostics(name_pot, name_beta, name_counts):
    trace_beta_solo = az.from_netcdf(str(pathlib.Path(FITS_ROOT / 'claim_values' / name_beta).with_suffix('.nc')))
    trace_gpd_solo = az.from_netcdf(str(pathlib.Path(FITS_ROOT / 'claim_values' / name_pot).with_suffix('.nc')))
    trace_counts_solo = az.from_netcdf(str(pathlib.Path(FITS_ROOT / 'claim_counts' / name_counts).with_suffix('.nc')))
    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(15, 10))
    tt = trace_gpd_solo.posterior.rename(
        {'shape': r'$\xi$', 'coef_meshs': r'$\sigma_1$', 'constant_scale': r'$\sigma_0$', 'coef_crossed': r'$\sigma_2$',
         'coef_exposure': r'$\sigma_3$'})
    az.plot_autocorr(tt.isel(chain=0), var_names=[r'$\sigma_0$', r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$'],
                     ax=axes.flatten(), textsize=16)
    fig.suptitle('Autocorrelation through time for parameters of the GPD model', fontsize=20)
    fig.show()
    fig.savefig(f'{PLOT_ROOT}/autocorr_gpd.png', dpi=200)

    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(15, 10))
    tt = trace_gpd_solo.posterior.rename(
        {'shape': r'$\xi$'})
    az.plot_posterior(tt, var_names=[r'$\xi$'], filter_vars="like",
                      ax=axes[:, 0], hdi_prob=0.95)
    axes[0, 1].plot(tt[r'$\xi$'].sel(season=0, chain=0))
    axes[1, 1].plot(tt[r'$\xi$'].sel(season=1, chain=0))
    for i in range(2):
        axes[i, 0].set_title(r'KDE plot for the posterior of $\xi_{}$'.format(i + 1))
    for i in range(2):
        axes[i, 1].set_title(r'Evolution of $\xi_{}$ after initial burn-in sample'.format(i + 1))
    fig.suptitle('Diagnostic plots for the shape parameter of the GPD model', fontsize=20)
    fig.show()
    fig.savefig(f'{PLOT_ROOT}/diag_shape_gpd.png', dpi=200)

    fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(15, 5))
    tt = trace_counts_solo.posterior.isel(draw=slice(55, None)).rename(
        {'alpha': r'$\alpha$'})
    az.plot_posterior(tt, var_names=[r'$\alpha$'], filter_vars="like",
                      ax=axes[0], hdi_prob=0.95)
    axes[1].plot(tt[r'$\alpha$'].sel(chain=0))
    axes[0].set_title(r'KDE plot for the posterior of $\alpha$')
    axes[1].set_title(r'Evolution of $\alpha$ after initial burn-in sample')
    fig.suptitle('Diagnostic plots for the shape parameter of the Negative Binomial model', fontsize=20)
    fig.show()
    fig.savefig(f'{PLOT_ROOT}/diag_alpha_poisson.png', dpi=200)

    fig, axes = plt.subplots(ncols=4, nrows=2, constrained_layout=True, figsize=(15, 10))
    tt = trace_counts_solo.posterior.isel(draw=slice(55, None), season=[0, 2]).rename({'constant_intensity': r'$\mu_0$',
                                                                                       'coef_climada': r'$\mu_1$',
                                                                                       'coef_cross_climada_dist': r'$\mu_2$',
                                                                                       'coef_meshs_cnt': '$\mu_3$',
                                                                                       'coef_ws_cnt': "$\mu_4$",
                                                                                       'sigma_dist': "$\sigma_m$",
                                                                                       'seasonal_comp': r'$\epsilon$'})
    vars = ['$\mu_0$', '$\mu_1$', '$\mu_2$', '$\mu_4$', r'$\epsilon$']
    az.plot_autocorr(tt,
                     var_names=vars,
                     ax=axes)
    var_names = ['$\mu_0$', '$\mu_{11}$', '$\mu_{12}$', '$\mu_{13}$', '$\mu_2$', '$\mu_3$',
                 r'$\epsilon_1$', r'$\epsilon_2$']

    for ax, var in zip(axes.flatten(), var_names):
        ax.set_title(r"{}".format(var))
    fig.suptitle('Autocorrelation through time for parameters of the Negative Binomial model', fontsize=20)
    fig.show()
    fig.savefig(f'{PLOT_ROOT}/autocorr_poisson.png', dpi=200)

    beta_latex_var = r"\nu"
    beta_varname_dic = {"precision":r"$\kappa$",
                         "constant_mu":r"${}_0$".format(beta_latex_var),
                         "coef_meshs_b": r"${}_1$".format(beta_latex_var),
                         "coef_poh_b": r"${}_2$".format(beta_latex_var)}
    fig, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True, figsize=(15, 10))
    tt = trace_beta_solo.posterior.rename(beta_varname_dic).isel(draw=slice(2000, None))
    az.plot_forest(tt, var_names=[v for v in beta_varname_dic.values()], filter_vars="like",
                      ax=axes.flatten(), hdi_prob=0.95, combined=True, ess=True)
    axes[0].set_title(r'Posterior estimates and $95\%$ CI')
    axes[1].set_title(r'Effective sample size')
    fig.suptitle('Diagnostic plots for parameters of the Beta model', fontsize=20)
    fig.show()
    fig.savefig(f'{PLOT_ROOT}/diag_forest_beta.png', dpi=200)


def plot_climada_count_to_buildings_ratio():
    i = climada_damages.groupby(['gridcell', 'claim_date']).climadadmg.count().to_frame().merge(obs_damages.groupby(['gridcell', 'claim_date']).claim_value.count().to_frame(), how='outer',
                                                                                                left_index=True, right_index=True).fillna(
        0.)
    cnt_gridcell = mapping.groupby('gridcell').latitude.count().rename('cnt_gridcell')
    i = i.merge(cnt_gridcell, left_on='gridcell', right_index=True, how='left').reset_index()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 6), constrained_layout=True)
    ax1.scatter(i.claim_date, 100 * i.climadadmg / i.cnt_gridcell, marker='x', color='slategrey', s=6)
    ax2.scatter(i.claim_date, 100 * i.claim_value / i.cnt_gridcell, marker='x', color='slategrey', s=6)
    ax1.set_title('a) CLIMADA per-building')
    ax2.set_title('b) Observed')
    for ax in [ax1, ax2]:
        ax.hlines(100, i.claim_date.min(), i.claim_date.max(), color='black')
        ax.set_xlabel('time')
        ax.set_ylabel('ratio (\%)')
    fig.suptitle('Impacted buildings/Total number of buildings per cell', fontsize=20)
    fig.show()
    fig.savefig(PLOT_ROOT / 'claim_building_ratio.png', DPI=200)


def plot_climada_compensation():
    g = climada_damages.groupby('claim_date').climadadmg.sum().to_frame().merge(obs_damages.groupby('claim_date').claim_value.sum().to_frame(), how='outer', left_index=True, right_index=True).fillna(
        0.)
    h = climada_damages.climadadmg.to_frame().merge(obs_damages.set_index(['claim_date', 'longitude', 'latitude']).claim_value.to_frame(), how='outer', left_index=True, right_index=True).fillna(
        0.)
    i = climada_damages.groupby(['gridcell', 'claim_date']).climadadmg.count().to_frame().merge(obs_damages.groupby(['gridcell', 'claim_date']).claim_value.count().to_frame(), how='outer',
                                                                                                left_index=True, right_index=True).fillna(
        0.)
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(22, 7), constrained_layout=True)
    for data, ax in zip([i, h, g], [ax1, ax2, ax3]):
        ax.scatter(data.claim_value, data.climadadmg, marker='x', color='slategrey')
        ax.plot(data.claim_value,
                data.claim_value, label='$x=y$', color='slategrey')
        ax.legend(loc='upper left')
        ax.set_xlabel('observed')
        ax.set_ylabel('predicted')
        ax.set_yscale(matplotlib.scale.FuncScaleLog(axis=1, functions=[lambda x: 1 + x, lambda x: x - 1]))
        ax.set_xscale(matplotlib.scale.FuncScaleLog(axis=0, functions=[lambda x: 1 + x, lambda x: x - 1]))
    ax1.set_title('a) Number of buildings with positive damages per 2km gridcell per day')
    ax2.set_title('b) Claim value for single buildings per day')
    ax3.set_title('c) Total damages (CHF) over the canton per day')
    fig.suptitle('Compensation mechanism for the per-building CLIMADA damage prediction', fontsize=20)
    fig.show()
    fig.savefig(PLOT_ROOT / 'compensation_climada.png', DPI=200)


def plot_example_day_and_line():
    dates = [pd.to_datetime('2004-07-08'), pd.to_datetime('2009-05-12'), pd.to_datetime('2012-06-30')]
    devs = [0.14, 0.02, 0.18]
    coef = [0.5, 1, 1]
    train_cond = lambda x: (int(pd.to_datetime(x).strftime('%Y')) >= 2008)
    test_cond = lambda x: (int(pd.to_datetime(x).strftime('%Y')) < 2005)
    dates_str = [d.strftime('%Y-%m-%d') for d in dates]
    origin = (8.36278492095831, 47.163852336888695)
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(15, 10), constrained_layout=True,
                             subplot_kw={'projection': cartopy.crs.PlateCarree()})
    for ax in axes.flatten():
        df_polygon.plot(ax=ax, facecolor='oldlace', edgecolor='grey')
        df_lakes.plot(ax=ax, facecolor='royalblue', alpha=0.5)
        ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
        ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), color='black')
    for d, d_str, ax, co, dev in zip(dates, dates_str, axes.T, coef, devs):
        ax1 = ax[0]
        ax2 = ax[1]
        if train_cond(d):
            tt = get_train_data().reset_index()
        elif test_cond(d):
            tt = get_test_data().reset_index()
        else:
            tt = get_validation_data().reset_index()
        tt_day = tt[tt.claim_date == d]
        deviation_from_origin = dev
        new_origin = (origin[0], origin[1] + deviation_from_origin)
        slope = tt_day.wind_dir.mean() / 360
        tt_day = tt_day.merge(mapping[['gridcell', 'lon_grid', 'lat_grid', 'geometry']].drop_duplicates('gridcell'), how='left', on='gridcell').set_geometry('geometry')
        _X = np.array(tt_day[['lon_grid', 'lat_grid']])
        new_lats = co * slope * (_X[..., 0] - origin[0]) + new_origin[1]
        a = np.array([new_origin])
        new_origin_array = np.tile(a, (len(_X), 1))
        dist_from_line = hauteur(new_origin_array, np.stack([_X[..., 0], new_lats], axis=1), _X).eval({})
        tt_day = tt_day.assign(dist_from_line=dist_from_line) \
            .assign(dist_term=lambda x: np.exp(-(x.dist_from_line / 80) ** 2 / 2))
        ax1.plot(_X[..., 0], new_lats, color='red', ls='--')
        geoplot.choropleth(tt_day[tt_day.obscnt > 0], ax=ax1, hue='obscnt', cmap='OrRd', legend=True, legend_kwargs={'label': 'count', 'shrink': 0.8})
        geoplot.choropleth(tt_day, ax=ax2, hue='dist_from_line', cmap=matplotlib.cm.get_cmap('OrRd_r'), legend=True, legend_kwargs={'label': 'km', 'shrink': 0.8})
        ax1.set_title(f'Observed claim count {d_str}')
        ax2.set_title(f'Distance from line {d_str}')
    for ax in axes.flatten():
        ax.set_extent([mapping.longitude.min() - 0.01, mapping.longitude.max() + 0.01,
                       mapping.latitude.min() - 0.01, mapping.latitude.max() + 0.01])
    fig.show()
    fig.savefig(PLOT_ROOT / f'examples_line.png')


def plot_poh_meshs_winds(date):
    date_str = date.strftime('%Y-%m-%d')
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 7), constrained_layout=True, subplot_kw={'projection': cartopy.crs.PlateCarree()})
    loc_meshs = meshs[date_str].rename("MESHS").reset_index().assign(geometry=lambda x: geopandas.GeoSeries.from_xy(x.longitude, x.latitude, crs=CRS).to_crs('epsg:4326')).set_geometry('geometry')
    loc_poh = poh[date_str].rename('POH').reset_index().assign(geometry=lambda x: geopandas.GeoSeries.from_xy(x.longitude, x.latitude, crs=CRS).to_crs('epsg:4326')).set_geometry('geometry')
    wd = get_train_data().wind_dir.reset_index()
    loc_wd = wd[wd.claim_date == date].merge(mapping.drop_duplicates('gridcell')[['gridcell', 'geometry']], on='gridcell', how='left').set_geometry('geometry')
    for ax in [ax1, ax2, ax3]:
        df_polygon.plot(ax=ax, facecolor='oldlace', edgecolor='grey')
        df_lakes.plot(ax=ax, facecolor='royalblue', alpha=0.5)
        ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
        ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), color='black')
    geoplot.pointplot(loc_poh, ax=ax1, hue='POH', cmap='YlOrRd', legend=True, s=3, legend_kwargs={'label': '\%', 'shrink': 0.8})
    geoplot.pointplot(loc_meshs, ax=ax2, hue='MESHS', cmap='YlGnBu', legend=True, s=3, legend_kwargs={'label': 'mm', 'shrink': 0.8})
    geoplot.choropleth(loc_wd, ax=ax3, hue='wind_dir', cmap='YlGn', legend=True, legend_kwargs={'label': 'degrees', 'shrink': 0.8})
    ax1.set_title('POH')
    ax2.set_title('MESHS')
    ax3.set_title('Wind direction')
    for ax in [ax1, ax2, ax3]:
        ax.set_extent([mapping.longitude.min() - 0.01, mapping.longitude.max() + 0.01,
                       mapping.latitude.min() - 0.01, mapping.latitude.max() + 0.01])
    fig.suptitle(f'Hail risk variables and wind direction on the {date_str}', fontsize=20)
    fig.show()
    fig.savefig(PLOT_ROOT / f'variables_{date_str}.png')


def plot_exposure():
    exposure = get_exposure()
    exposure = exposure \
        .reset_index().assign(chf_cubem=lambda x: np.where(x.volume > 0, x.value / x.volume, 0)).assign(geometry=lambda x: geopandas.GeoSeries.from_xy(x.longitude, x.latitude)).set_geometry(
        'geometry')
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 7), constrained_layout=True, subplot_kw={'projection': cartopy.crs.PlateCarree()})
    for ax in [ax1, ax2]:
        df_polygon.plot(ax=ax, facecolor='oldlace', edgecolor='grey')
        df_lakes.plot(ax=ax, facecolor='royalblue', alpha=0.5)
        ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
        ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), color='black')
    geoplot.pointplot(exposure, ax=ax1, hue='value', cmap='Purples', legend=True, s=3, norm=matplotlib.colors.LogNorm(vmin=exposure.value.min(),
                                                                                                                      vmax=exposure.value.max()),
                      legend_kwargs={'label': 'CHF',
                                     'shrink': 0.8})
    geoplot.pointplot(exposure, ax=ax2, hue='chf_cubem', cmap='Purples', norm=matplotlib.colors.LogNorm(vmin=1, vmax=exposure.chf_cubem.max()),
                      legend=True, s=3, legend_kwargs={'label': r'CHF.$m^{-3}$', 'shrink': 0.8})
    for ax in [ax1, ax2]:
        ax.set_extent([mapping.longitude.min() - 0.01, mapping.longitude.max() + 0.01,
                       mapping.latitude.min() - 0.01, mapping.latitude.max() + 0.01])
    ax1.set_title('Insured value')
    ax2.set_title('Insured value per cube meter')
    fig.suptitle(r'Exposure for individual buildings in the canton of Z\"urich', fontsize=20)
    fig.show()
    fig.savefig(PLOT_ROOT / f'exposure.png')


def get_paa(counts):
    cnt_gridcell = mapping.groupby('gridcell').latitude.count().rename('cnt_grid')
    counts = counts.merge(cnt_gridcell.to_frame(), on='gridcell', how='left')
    counts = counts.assign(obs_paa=100 * counts.obscnt / counts.cnt_grid) \
        .assign(ub_paa=100 * counts.ub_counts / counts.cnt_grid) \
        .assign(pred_paa=100 * counts.pred_cnt / counts.cnt_grid) \
        .assign(lb_paa=100 * counts.lb_counts / counts.cnt_grid).assign(climada_paa=100 * counts.climada_cnt / counts.cnt_grid)
    mean_by_meshs = counts.groupby('meshs').mean()
    obs_paa = mean_by_meshs.obs_paa
    pred_paa = mean_by_meshs.pred_paa
    lb_paa = mean_by_meshs.lb_paa
    ub_paa = mean_by_meshs.ub_paa
    climada_paa = mean_by_meshs.climada_paa
    return obs_paa, pred_paa, lb_paa, ub_paa, climada_paa


def plot_paa(counts, name):
    # smoothing = 2
    obs_paa, pred_paa, lb_paa, ub_paa, climada_paa = get_paa(counts)
    cuts = pd.DataFrame(np.array(pd.cut(obs_paa.reset_index().meshs, bins=50)), index=obs_paa.index, columns=['interval'])
    sclimada_paa = climada_paa.to_frame().merge(cuts, left_index=True, right_index=True, how='left').groupby('interval').mean()['climada_paa']
    sobs_paa = obs_paa.to_frame().replace(0, np.nan).interpolate().merge(cuts, left_index=True, right_index=True, how='left').groupby('interval').mean()['obs_paa']
    spred_paa = pred_paa.to_frame().replace(0, np.nan).interpolate().merge(cuts, left_index=True, right_index=True, how='left').groupby('interval').mean()['pred_paa']
    slb_paa = lb_paa.to_frame().replace(0, np.nan).interpolate().merge(cuts, left_index=True, right_index=True, how='left').groupby('interval').mean()['lb_paa']
    sub_paa = ub_paa.to_frame().replace(0, np.nan).interpolate().merge(cuts, left_index=True, right_index=True, how='left').groupby('interval').mean()['ub_paa']
    fig, ax = plt.subplots(ncols=1, figsize=(10, 6), constrained_layout=True)
    sobs_paa.rename('Observed').plot(ax=ax, color='salmon', legend=True)
    sclimada_paa.rename('CLIMADA').plot(ax=ax, color='goldenrod', legend=True)
    spred_paa.rename('Predicted mean').plot(ax=ax, color='navy', legend=True)
    slb_paa.rename(f'{int(round(100 * (1 - conf), 0))}\% predicted range').plot(ax=ax, color='navy', ls='--', legend=True)
    sub_paa.plot(ax=ax, color='navy', ls='--', legend=False)
    ax.set_xlabel('MESHS (mm)')
    ax.set_ylabel('PAA (\%)')
    ax.set_yscale('log')
    inter = [pd.Interval(float(x._text.split(',')[0].split('(')[-1]), float(x._text.split(',')[-1].split(']')[0])) for x in ax.get_xticklabels()[:-1]]
    ax.set_xticklabels([int(i.mid) for i in inter] + [''])
    fig.suptitle(f'a) Mean percentage of affected assets (PAA) computed on test set', fontsize=20)
    fig.show()
    path = PLOT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / 'paa.png', DPI=200)


def get_series_predicted_damages(counts, sizes_df):
    pred_dmg = counts[counts.pred_cnt >= 1][['mean_pred', 'mean_pred_lb', 'mean_pred_ub', 'claim_date', 'gridcell']].merge(mapping[['gridcell', 'latitude', 'longitude']], on='gridcell') \
        .assign(
        claim_date=lambda x: pd.to_datetime(x.claim_date)) \
        .merge(sizes_df.reset_index(), on=['latitude', 'longitude', 'claim_date']).assign(pred_dmg=lambda x: x.pred_dmg * x.mean_pred).assign(pred_dmg_lb=lambda x: x.pred_dmg_lb * x.mean_pred_lb) \
        .assign(pred_dmg_ub=lambda x: x.pred_dmg_ub * x.mean_pred_ub)
    pred_dmg_gridcell = pred_dmg.groupby(['gridcell', 'claim_date']).agg({'pred_dmg': 'sum', 'meshs': 'mean'})
    lb_dmg_gridcell = pred_dmg.groupby(['gridcell', 'claim_date']).agg({'pred_dmg_lb': 'sum', 'meshs': 'mean'})
    ub_dmg_gridcell = pred_dmg.groupby(['gridcell', 'claim_date']).agg({'pred_dmg_ub': 'sum', 'meshs': 'mean'})
    return pred_dmg_gridcell, lb_dmg_gridcell, ub_dmg_gridcell


def qq_total_per_gridcell(series_predicted_damages, name):
    fig, ax = plt.subplots(ncols=1, figsize=(10, 6), constrained_layout=True)
    pred_dmg_gridcell, lb_dmg_gridcell, ub_dmg_gridcell = series_predicted_damages
    func = lambda x: x
    obs = func(obs_dmg_gridcell.claim_value)
    pred = func(pred_dmg_gridcell.groupby(['gridcell', 'claim_date']).sum())
    lb = func(lb_dmg_gridcell.groupby(['gridcell', 'claim_date']).sum())
    ub = func(ub_dmg_gridcell.groupby(['gridcell', 'claim_date']).sum())
    climada = func(climada_damages.groupby(['gridcell', 'claim_date']).climadadmg.sum())
    ax.scatter(obs.quantile(quants), pred.quantile(quants), marker='x', color='navy', label='Predicted mean')
    ax.scatter(obs.quantile(quants), climada.quantile(quants), marker='x', color='goldenrod', label='CLIMADA')
    ax.plot(obs.quantile(quants), lb.quantile(quants).rolling(50).mean(),
            ls='--', color='navy',
            label=f'{int(round(100 * (1 - conf), 0))}\% predicted range')
    ax.plot(obs.quantile(quants), ub.quantile(quants),
            ls='--', color='navy')
    ax.plot(obs.quantile(quants), obs.quantile(quants), color='navy', label=r'$x=y$')
    fig.suptitle(f'b) QQ plot of the hail damages (CHF) per 2km gridcell', fontsize=20)
    ax.set_xlabel('observed')
    ax.set_ylabel('predicted')
    ax.legend(loc='upper left')
    ax.set_xscale(matplotlib.scale.FuncScaleLog(axis=0, functions=[lambda x: 1 + x, lambda x: x - 1]))
    ax.set_yscale(matplotlib.scale.FuncScaleLog(axis=1, functions=[lambda x: 1 + x, lambda x: x - 1]))
    fig.show()
    path = PLOT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / 'qq_gridcell.png', DPI=200)


def qq_total_per_location(series_predicted_damages, name):
    fig, ax = plt.subplots(ncols=1, figsize=(10, 6), constrained_layout=True)
    pred_dmg_gridcell, lb_dmg_gridcell, ub_dmg_gridcell = series_predicted_damages
    func = lambda x: x
    quants = np.linspace(tol, 1 - tol, 200)  # franchise claims
    obs = func(obs_damages.claim_value)
    pred = func(pred_dmg_gridcell)
    lb = func(lb_dmg_gridcell)
    ub = func(ub_dmg_gridcell)
    climada = func(climada_damages.climadadmg)
    ax.scatter(obs.quantile(quants), pred.quantile(quants), marker='x', color='navy', label='Predicted mean')
    q = lb.quantile(quants).rolling(100).mean()
    q.iloc[0] = 100
    ax.scatter(obs.quantile(quants), climada.quantile(quants), marker='x', color='goldenrod', label='CLIMADA')
    ax.plot(obs.quantile(quants), q.interpolate(),
            ls='--', color='navy',
            label=f'{int(round(100 * (1 - conf), 0))}\% predicted range')
    ax.plot(obs.quantile(quants), ub.quantile(quants),
            ls='--', color='navy')
    ax.plot(obs.quantile(quants), obs.quantile(quants), color='navy', label=r'$x=y$')
    fig.suptitle(f'a) QQ plot of the hail damages (CHF) per location', fontsize=20)
    ax.legend(loc='upper left')
    ax.set_xlabel('observed')
    ax.set_ylabel('predicted')
    ax.set_xscale(matplotlib.scale.FuncScaleLog(axis=0, functions=[lambda x: 1 + x, lambda x: x - 1]))
    ax.set_yscale(matplotlib.scale.FuncScaleLog(axis=1, functions=[lambda x: 1 + x, lambda x: x - 1]))
    fig.show()
    path = PLOT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / 'qq_location.png', DPI=200)


def qq_total_gricell_day(series_predicted_damages, name):
    pred_dmg_gridcell, lb_dmg_gridcell, ub_dmg_gridcell = series_predicted_damages
    fig, ax = plt.subplots(ncols=1, figsize=(10, 6), constrained_layout=True)
    func = lambda x: x
    aove_threshold = obs_dmg_gridcell
    ax.scatter(func(aove_threshold.claim_value.groupby('claim_date').sum()).quantile(quants), func(pred_dmg_gridcell.groupby('claim_date').sum()).quantile(quants), marker='x', color='navy',
               label='Predicted mean')
    ax.scatter(func(aove_threshold.claim_value.groupby('claim_date').sum()).quantile(quants), func(climada_damages.climadadmg.groupby('claim_date').sum()).quantile(quants), marker='x',
               color='goldenrod',
               label='CLIMADA')
    ax.plot(func(aove_threshold.claim_value.groupby('claim_date').sum()).quantile(quants),
            func(lb_dmg_gridcell.groupby('claim_date').sum()).quantile(quants).rolling(10).mean(),
            ls='--', color='navy',
            label=f'{int(round(100 * (1 - conf), 0))}\% predicted range')
    ax.plot(func(aove_threshold.claim_value.groupby('claim_date').sum()).quantile(quants),
            func(ub_dmg_gridcell.groupby('claim_date').sum()).quantile(quants).rolling(10).mean(),
            ls='--', color='navy')
    ax.plot(func(aove_threshold.claim_value.groupby('claim_date').sum()).quantile(quants),
            func(aove_threshold.claim_value.groupby('claim_date').sum()).quantile(quants), color='navy', label='$x=y$')
    ax.legend(loc='upper left')
    ax.set_xlabel('observed')
    ax.set_ylabel('predicted')
    ax.set_xscale(matplotlib.scale.FuncScaleLog(axis=0, functions=[lambda x: 1 + x, lambda x: x - 1]))
    ax.set_yscale(matplotlib.scale.FuncScaleLog(axis=1, functions=[lambda x: 1 + x, lambda x: x - 1]))
    fig.suptitle(f'QQ plot of the total damages (CHF) over the canton per day', fontsize=14)
    fig.show()
    path = PLOT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / 'qq_gridcell_day.png', DPI=200)


def pointplot_daily_claims(sizes_df, name):
    fig, ax = plt.subplots(ncols=1, figsize=(10, 6), constrained_layout=True)
    g = claim_values.groupby('claim_date').claim_value.sum().to_frame() \
        .merge(sizes_df.groupby('claim_date').mean_pred_size.sum().to_frame(), left_index=True, right_index=True, how='outer') \
        .merge(climada_damages.groupby('claim_date').climadadmg.sum().to_frame(), left_index=True, right_index=True, how='outer')
    g = g.fillna(0.)
    func = lambda x: np.log1p(x)
    ax.scatter(func(g.claim_value), func(g.mean_pred_size.rolling(2).mean()), color='navy', marker='x', label='Predicted mean')
    ax.scatter(func(g.claim_value), func(g.climadadmg), color='goldenrod', marker='x', label='CLIMADA')
    ax.plot(func(g.claim_value), func(g.claim_value), color='navy', label='$x=y$')
    ax.set_xlabel('observed')
    ax.set_ylabel('predicted')
    fig.suptitle(f'Point plot of the total log-damages (CHF) over the canton per day', fontsize=20)
    fig.show()
    path = PLOT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / 'pointplot_gridcell_day.png', DPI=200)


def qq_counts(counts, name):
    fig, ax = plt.subplots(ncols=1, figsize=(10, 6), constrained_layout=True)
    func = lambda x: x
    pred_cnt = func(counts[counts.pred_cnt >= 1].dropna().pred_cnt).quantile(quants)
    lb_cnt = func(counts[counts.pred_cnt >= 1].dropna().lb_counts).quantile(quants)
    ub_cnt = func(counts[counts.pred_cnt >= 1].dropna().ub_counts).quantile(quants)
    obs_cnt = func(claim_values.drop(columns=['gridcell']).merge(mapping[['gridcell', 'latitude', 'longitude']], on=['latitude', 'longitude'], how='left')
                   .groupby(['claim_date', 'gridcell']).claim_value.count()).quantile(quants)
    climada_cnt = climada_counts.groupby(['claim_date', 'gridcell']).climadadmg.sum().rename('climadacnt')
    climada_cnt = func(climada_cnt[climada_cnt >= 1]).quantile(quants)
    ax.scatter(obs_cnt, pred_cnt, marker='x', color='navy',
               label='Predicted mean')
    ax.scatter(obs_cnt, climada_cnt, marker='x',
               color='goldenrod',
               label='CLIMADA')
    ax.plot(obs_cnt, lb_cnt, ls='--', color='navy', label=f'{int(round(100 * (1 - conf), 0))}\% predicted range')
    ax.plot(obs_cnt, ub_cnt, ls='--', color='navy')
    ax.plot(obs_cnt, obs_cnt, color='navy', label=r'$x=y$')
    ax.legend(loc='upper left')
    ax.set_xlabel('observed')
    ax.set_ylabel('predicted')
    ax.set_xscale(matplotlib.scale.FuncScaleLog(axis=0, functions=[lambda x: 1 + x, lambda x: x - 1]))
    ax.set_yscale(matplotlib.scale.FuncScaleLog(axis=1, functions=[lambda x: 1 + x, lambda x: x - 1]))
    fig.suptitle(f'b) QQ plot of the number of claims per 2km gridcell per day', fontsize=20)
    fig.show()
    path = PLOT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / 'qq_counts.png', DPI=200)


def plot_mdr(series_predicted_damages_with_meshs, name):
    fig, ax = plt.subplots(ncols=1, figsize=(10, 6), constrained_layout=True)
    pred_dmg_gridcell, lb_dmg_gridcell, ub_dmg_gridcell = series_predicted_damages_with_meshs
    q05 = lb_dmg_gridcell.copy()
    q95 = ub_dmg_gridcell.copy()
    mean = pred_dmg_gridcell.copy()
    cd = climada_damages.copy()
    sum = exposure.value.sum()
    mean['meshs'] = 100 * mean.meshs
    q05['meshs'] = 100 * q05.meshs
    q05['meshs'] = 100 * q95.meshs
    mean = mean.assign(pred_mdr=100 * pred_dmg_gridcell.pred_dmg / sum)
    q05 = q05.assign(lb_mdr=100 * q05.pred_dmg_lb / sum)
    q95 = q95.assign(ub_mdr=100 * q95.pred_dmg_ub / sum)
    cd = cd.assign(climada_mdr=100 * climada_damages.climadadmg / sum)
    og = obs_dmg_gridcell.assign(obs_mdr=100 * obs_dmg_gridcell.claim_value / sum)
    obs_mdr = og.groupby('mean_meshs').obs_mdr.mean().rolling(2).mean()
    pred_mdr = mean.groupby('meshs').pred_mdr.mean().rolling(2).mean()
    lb_mdr = q05.groupby('meshs').lb_mdr.mean().rolling(2).mean()
    ub_mdr = q95.groupby('meshs').ub_mdr.mean().rolling(2).mean()
    climada_mdr = cd.groupby('meshs').climada_mdr.mean().rolling(2).mean()
    ax.plot(obs_mdr, color='salmon', label='Observed')
    ax.plot(climada_mdr, color='goldenrod', label='CLIMADA')
    ax.plot(pred_mdr, color='navy', label='Predicted mean')
    ax.plot(lb_mdr, color='navy', ls='--', label=f'{int(round(100 * (1 - conf), 0))}\% CI')
    ax.plot(ub_mdr, color='navy', ls='--')
    ax.set_xlabel('MESHS (mm)')
    ax.set_ylabel('MDR (\%)')
    ax.legend()
    fig.suptitle(f'Mean damage ratio')
    fig.show()
    path = PLOT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / 'mdr.png', DPI=200)


def plot_pct_above_threshold(series_predicted_damages, name):
    fig, ax = plt.subplots(ncols=1, figsize=(10, 6), constrained_layout=True)
    log_pow = np.linspace(2, 15, 100)
    mean, lb, ub = series_predicted_damages
    ax.scatter([np.exp(t) - 1 for t in log_pow], [len(mean[np.log1p(mean) > t]) / len(mean) for t in log_pow], marker='x', color='navy',
               label='Predicted mean')
    ax.scatter([np.exp(t) - 1 for t in log_pow], [len(obs_damages[np.log1p(obs_damages.claim_value) > t]) / len(obs_damages) for t in log_pow],
               marker='x', color='salmon', label='Observed')
    ax.scatter([np.exp(t) - 1 for t in log_pow], [len(climada_damages[np.log1p(climada_damages.climadadmg) > t]) / len(climada_damages) for t in log_pow], marker='x',
               color='goldenrod', label='CLIMADA')
    ax.plot([np.exp(t) - 1 for t in log_pow], [len(lb[np.log1p(lb) > t].rolling(50).mean()) / len(lb) for t in log_pow],
            ls='--', color='navy', label=f'{int(round(100 * (1 - conf), 0))}\% predicted range')
    ax.plot([np.exp(t) - 1 for t in log_pow], [len(ub[np.log1p(ub) > t]) / len(ub) for t in log_pow], ls='--', color='navy')
    ax.set_xscale(matplotlib.scale.FuncScaleLog(axis=0, functions=[lambda x: 1 + x, lambda x: x - 1]))
    ax.legend()
    ax.set_xlabel('threshold')
    ax.set_ylabel('proportion of claim values above threshold')
    fig.suptitle('Proportion of claims above a given threshold for a single building', fontsize=14)
    fig.show()
    path = PLOT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / 'pct_above_thresh.png', DPI=200)


def lsd_input_vs_predicted(inputs, targets, predicted, name):
    metric = log_spectral_distance_from_xarray
    fig, ax = plt.subplots(ncols=1, figsize=(10, 6), constrained_layout=True)
    metric_input = xr.DataArray(metric(targets, inputs),
                                coords=targets.mean('claim_date').coords, name='CLIMADA')
    metric_predicted = xr.DataArray(metric(targets, predicted),
                                    coords=targets.mean('claim_date').coords, name='predicted')
    ds = xr.merge([metric_input, metric_predicted])
    clipped_df = np.sqrt(ds).to_dataframe()
    ax.scatter(clipped_df.predicted, clipped_df.CLIMADA, marker='x', s=4, color='slategrey')
    ax.plot(clipped_df.predicted, clipped_df.predicted, color='slategrey')
    ax.set_xlabel(f'Predicted')
    ax.set_ylabel(f'CLIMADA')
    fig.suptitle('a) Log-spectral distance', fontsize=20)
    fig.show()
    path = PLOT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / 'lsd.png', DPI=200)


def skss_input_vs_predicted(inputs, targets, predicted, name):
    dims = [all.dims[v] for v in all.dims]
    metric = spatially_convolved_ks_stat
    fig, ax = plt.subplots(ncols=1, figsize=(10, 6), constrained_layout=True)
    m_in = metric(targets.to_numpy().reshape(dims + [1]), inputs.to_numpy().reshape(dims + [1]))
    m_pred = metric(targets.to_numpy().reshape(dims + [1]), predicted.to_numpy().reshape(dims + [1]))
    ax.scatter(m_pred, m_in, marker='x', s=4, color='slategrey')
    ax.plot(m_pred, m_pred, color='slategrey')
    ax.set_xlabel(f'Predicted')
    ax.set_ylabel(f'CLIMADA')
    fig.suptitle('b) Spatially convolved KSS', fontsize=20)
    fig.show()
    path = PLOT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / 'skss.png', DPI=200)


def plot_errors_counts(counts, name):
    fig, ax = plt.subplots(ncols=1, figsize=(10, 6), constrained_layout=True)
    c = counts.groupby('claim_date').sum()
    zero_counts = c[c.obscnt == 0]
    non_zero_counts = c[c.obscnt > 0]
    fa_pred = len(zero_counts[zero_counts.pred_cnt >= 1])
    fa_climada = len(zero_counts[zero_counts.climada_cnt > 0])
    tp_pred = len(non_zero_counts[non_zero_counts.pred_cnt >= 1])
    tp_climada = len(non_zero_counts[non_zero_counts.climada_cnt > 0])
    fn_pred = len(non_zero_counts[non_zero_counts.pred_cnt == 0])
    fn_climada = len(non_zero_counts[non_zero_counts.climada_cnt == 0])
    tn_pred = len(zero_counts[zero_counts.pred_cnt == 0])
    tn_climada = len(zero_counts[zero_counts.climada_cnt == 0])
    fa_rate_pred = fa_pred / (fa_pred + tn_pred)
    sensitivity_pred = tp_pred / (tp_pred + fn_pred)
    specificity_pred = tn_pred / (fa_pred + tn_pred)
    pos_pred_value_pred = tp_pred / (tp_pred + fa_pred)
    fa_rate_clim = fa_climada / (fa_climada + tn_climada)
    sensitivity_clim = tp_climada / (tp_climada + fn_climada)
    specificity_clim = tn_climada / (fa_climada + tn_climada)
    pos_pred_value_clim = tp_climada / (tp_climada + fa_climada)
    df = pd.DataFrame([[fa_rate_pred, fa_rate_clim, (fa_rate_pred - fa_rate_clim) / fa_rate_clim],
                       [sensitivity_pred, sensitivity_clim, (sensitivity_pred - sensitivity_clim) / sensitivity_clim],
                       [specificity_pred, specificity_clim, (specificity_pred - specificity_clim)],
                       [pos_pred_value_pred, pos_pred_value_clim, (pos_pred_value_pred - pos_pred_value_clim) / pos_pred_value_clim]],
                      index=['False Alarm', 'Sensitivity', 'Specificity', 'Positive Predictive Value'],
                      columns=['Predicted mean', 'CLIMADA', 'Variation'])
    print(df)
    (100 * df).plot.bar(ax=ax, color=['navy', 'goldenrod'])
    ax.set_ylabel('rate (\%)')
    fig.suptitle('Count of claims per day', fontsize=14)
    fig.show()
    path = PLOT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / 'rates.png', DPI=200)


def plot_errors_individual_counts(counts, name):
    fig, ax = plt.subplots(ncols=1, figsize=(10, 6), constrained_layout=True)
    c = counts
    zero_counts = c[c.obscnt == 0]
    non_zero_counts = c[c.obscnt > 0]
    fa_pred = len(zero_counts[zero_counts.pred_cnt >= 1])
    fa_climada = len(zero_counts[zero_counts.climada_cnt > 0])
    tp_pred = len(non_zero_counts[non_zero_counts.pred_cnt >= 1])
    tp_climada = len(non_zero_counts[non_zero_counts.climada_cnt > 0])
    fn_pred = len(non_zero_counts[non_zero_counts.pred_cnt == 0])
    fn_climada = len(non_zero_counts[non_zero_counts.climada_cnt == 0])
    tn_pred = len(zero_counts[zero_counts.pred_cnt == 0])
    tn_climada = len(zero_counts[zero_counts.climada_cnt == 0])
    fa_rate_pred = fa_pred / (fa_pred + tn_pred)
    sensitivity_pred = tp_pred / (tp_pred + fn_pred)
    specificity_pred = tn_pred / (fa_pred + tn_pred)
    pos_pred_value_pred = tp_pred / (tp_pred + fa_pred)
    fa_rate_clim = fa_climada / (fa_climada + tn_climada)
    sensitivity_clim = tp_climada / (tp_climada + fn_climada)
    specificity_clim = tn_climada / (fa_climada + tn_climada)
    pos_pred_value_clim = tp_climada / (tp_climada + fa_climada)
    df = pd.DataFrame([[fa_rate_pred, fa_rate_clim, (fa_rate_pred - fa_rate_clim) / fa_rate_clim],
                       [sensitivity_pred, sensitivity_clim, (sensitivity_pred - sensitivity_clim) / sensitivity_clim],
                       [specificity_pred, specificity_clim, (specificity_pred - specificity_clim)],
                       [pos_pred_value_pred, pos_pred_value_clim, (pos_pred_value_pred - pos_pred_value_clim) / pos_pred_value_clim]],
                      index=['False Alarm', 'Sensitivity', 'Specificity', 'Positive Predictive Value'],
                      columns=['Predicted mean', 'CLIMADA', 'Variation'])
    print(df)
    (100 * df).plot.bar(ax=ax, color=['navy', 'goldenrod'])
    ax.set_ylabel('rate (\%)')
    fig.suptitle('Count of claims per gridcell per day', fontsize=14)
    fig.show()
    path = PLOT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / 'rates_individual.png', DPI=200)


def plot_all():
    plot_climada_compensation()
    plot_climada_count_to_buildings_ratio()
    plot_example_day_and_line()
    plot_poh_meshs_winds(pd.to_datetime('2021-06-28'))
    plot_exposure()
    counts = {}
    for name_counts, scaling in zip(['20230424_11:06_GVZ_emanuel'], [1e2]):
        name_sizes = f'combined_20230221_12:58_20230228_05:00_20230220_12:41_{name_counts}'
        name = name_sizes
        path_to_counts = glob(str(pathlib.Path(PRED_ROOT / name_counts / '*').with_suffix('.csv')))
        path_to_sizes = glob(str(pathlib.Path(PRED_ROOT / name_sizes / '*').with_suffix('.csv')))
        az_counts = {p.split('/')[-1].split('.')[0]: pd.read_csv(p) for p in path_to_counts}
        counts[name_counts] = pd.concat(az_counts.values())
        counts[name_counts] = counts[name_counts].assign(meshs=lambda x: scaling * x.meshs)
        qq_counts(counts[name_counts], name)
        if 'emanuel' not in name_counts:
            cc = climada_counts.groupby(['gridcell', 'claim_date']).climadadmg.sum().rename('climada_cnt').reset_index()
            counts[name_counts].claim_date = pd.to_datetime(counts[name_counts].claim_date)
            counts[name_counts] = counts[name_counts].drop(columns=['climada_cnt']).merge(cc, on=['gridcell', 'claim_date'], how='left').fillna(0.)
        plot_paa(counts[name_counts], name)
        sizes_df = pd.concat([pd.read_csv(d).assign(claim_date=pd.to_datetime(d.split('/')[-1].split('.')[0])).set_index(['gridcell', 'claim_date']) for d in path_to_sizes])
        mean, lb, ub = sizes_df.mean_pred_size.rename('pred_dmg'), sizes_df.lb_pred.rename('pred_dmg_lb'), sizes_df.ub_pred.rename('pred_dmg_ub')
        qq_total_per_gridcell([mean, lb, ub], name)
        qq_total_per_location([mean, lb, ub], name)
        climada_xarray = climada_damages.reset_index().groupby(['gridcell', 'claim_date'], as_index=False).climadadmg.sum() \
            .merge(mapping.drop_duplicates('gridcell')[['gridcell', 'lat_grid', 'lon_grid']], on='gridcell', how='left') \
            .set_index(['claim_date', 'lon_grid', 'lat_grid']).climadadmg.to_xarray().fillna(0.)
        pred_xarray = sizes_df.reset_index().groupby(['gridcell', 'claim_date'], as_index=False).mean_pred_size.sum() \
            .merge(mapping.drop_duplicates('gridcell')[['gridcell', 'lat_grid', 'lon_grid']], on='gridcell', how='left') \
            .set_index(['claim_date', 'lon_grid', 'lat_grid']).mean_pred_size.to_xarray().fillna(0.)
        obs_xarray = claim_values.drop(columns=['gridcell']).merge(mapping, on=['latitude', 'longitude'], how='left') \
            .groupby(['claim_date', 'lon_grid', 'lat_grid']).claim_value.sum().to_xarray().fillna(0.)
        all = xr.merge([climada_xarray, pred_xarray, obs_xarray]).fillna(0.)
        lsd_input_vs_predicted(all.climadadmg,
                               all.claim_value,
                               all.mean_pred_size, name)
        skss_input_vs_predicted(all.climadadmg,
                                all.claim_value,
                                all.mean_pred_size, name)


if __name__ == '__main__':
    plot_all()
