from glob import glob

import cartopy
import cv2
import geopandas
import geoplot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from constants import DATA_ROOT, PLOT_ROOT, claim_values, df_polygon, df_lakes, CRS, suffix
from data.climada_processing import process_climada_perbuilding_positive_damages
from data.hailcount_data_processing import get_train_data, get_test_data, get_validation_data, get_exposure, get_grid_mapping
from extreme_values_visualization import extremogram_plot
from models.counts import hauteur
from threshold_selection import threshold_selection_GoF
from utils import grid_from_geopandas_pointcloud, aggregate_claim_data, compute_extremal_correlation_over_grid, get_extremal_corr, get_spearman_corr

plot_path = PLOT_ROOT / 'exploratory'
exposure = get_exposure()
mapping = get_grid_mapping(suffix=suffix)
poh = pd.read_csv(DATA_ROOT / 'poh.csv', index_col=['latitude', 'longitude'],
                  usecols=lambda x: ('2' in x) or x in ['latitude', 'longitude'])
meshs = pd.read_csv(DATA_ROOT / 'meshs.csv', index_col=['latitude', 'longitude'],
                    usecols=lambda x: ('2' in x) or x in ['latitude', 'longitude'])
climada_damages = process_climada_perbuilding_positive_damages()
obs_damages = claim_values[['claim_date', 'longitude', 'latitude', 'claim_value', 'MESHS']].merge(mapping[['latitude', 'longitude', 'gridcell']], on=['latitude', 'longitude'], how='left')
spatial_extent = [mapping.longitude.min() - 0.01, mapping.longitude.max() + 0.01,
                  mapping.latitude.min() - 0.01, mapping.latitude.max() + 0.01]


def plot_average_claim_values_time(df):
    gdf = df.reset_index().assign(month=lambda x: x.claim_date.dt.month) \
        .assign(month_name=lambda x: x.claim_date.dt.strftime('%B')).assign(year=lambda x: x.claim_date.dt.year)
    data_boxplot = gdf[['claim_value', 'claim_date', 'month_name', 'latitude', 'longitude', 'month']].set_index(['claim_date', 'latitude', 'longitude', 'month_name', 'month'])
    fig, ax = plt.subplots(ncols=1, figsize=(8, 7), constrained_layout=True)
    sns.boxenplot(data_boxplot.reset_index().sort_values('month'), x='month_name', y='claim_value', ax=ax)
    ax.set_ylabel('CHF')
    ax.set_xlabel('')
    ax.set_yscale('log')
    ax.set_title('a) Hail-related damages distribution per month')
    fig.show()
    fig.savefig(str(plot_path / 'average_total_claims.png'), DPI=200)


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
    fig.savefig(plot_path / 'claim_building_ratio.png', DPI=200)


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
    fig.savefig(plot_path / 'compensation_climada.png', DPI=200)


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
            tt = get_train_data(suffix=suffix).reset_index()
        elif test_cond(d):
            tt = get_test_data(suffix=suffix).reset_index()
        else:
            tt = get_validation_data(suffix=suffix).reset_index()
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
        ax.set_extent(spatial_extent)
    fig.show()
    fig.savefig(plot_path / f'examples_line.png')


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
        ax.set_extent(spatial_extent)
    fig.suptitle(f'Hail risk variables and wind direction on the {date_str}', fontsize=20)
    fig.show()
    fig.savefig(plot_path / f'variables_{date_str}.png')


def plot_exposure():
    exp = exposure \
        .reset_index().assign(chf_cubem=lambda x: np.where(x.volume > 0, x.value / x.volume, 0)).assign(geometry=lambda x: geopandas.GeoSeries.from_xy(x.longitude, x.latitude)).set_geometry(
        'geometry')
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 7), constrained_layout=True, subplot_kw={'projection': cartopy.crs.PlateCarree()})
    for ax in [ax1, ax2]:
        df_polygon.plot(ax=ax, facecolor='oldlace', edgecolor='grey')
        df_lakes.plot(ax=ax, facecolor='royalblue', alpha=0.5)
        ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
        ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), color='black')
    geoplot.pointplot(exp, ax=ax1, hue='value', cmap='Purples', legend=True, s=3, norm=matplotlib.colors.LogNorm(vmin=exp.value.min(),
                                                                                                                 vmax=exp.value.max()),
                      legend_kwargs={'label': 'CHF',
                                     'shrink': 0.8})
    geoplot.pointplot(exp, ax=ax2, hue='chf_cubem', cmap='Purples', norm=matplotlib.colors.LogNorm(vmin=1, vmax=exp.chf_cubem.max()),
                      legend=True, s=3, legend_kwargs={'label': r'CHF.$m^{-3}$', 'shrink': 0.8})
    for ax in [ax1, ax2]:
        ax.set_extent(spatial_extent)
    ax1.set_title('Insured value')
    ax2.set_title('Insured value per cube meter')
    fig.suptitle(r'Exposure for individual buildings in the canton of Z\"urich', fontsize=20)
    fig.show()
    fig.savefig(plot_path / f'exposure.png')


def plot_diagnostic_logsum_claimvalues(df):
    ext_data = np.log1p(df.groupby('claim_date').claim_value.sum())
    ts, fig = threshold_selection_GoF(ext_data, min_threshold=2, max_threshold=10, plot=True)
    fig.savefig(str(plot_path / 'threshold_selection.png'), DPI=200)
    extremogram_plot(ext_data.to_frame(name='data')
                     .assign(days_since_start=lambda x: (x.index - x.index.min()).days)
                     .assign(threshold=ts[0]), h_range=np.linspace(1, 100, 15),
                     path_to_figure=str(plot_path))


def plot_grid(df):
    fig, ax = plt.subplots(ncols=1, figsize=(6, 5), constrained_layout=True)
    g = geopandas.GeoSeries(df.geometry.unique())
    g.plot(edgecolor='salmon', facecolor='salmon', linewidth=2, alpha=0.2, ax=ax)
    df.assign(geom_point=lambda x: geopandas.points_from_xy(x['longitude'], x['latitude'])).set_geometry('geom_point').plot(markersize=1, ax=ax, color='navy')
    ax.set_xlabel(r'longitude ($^\circ$)')
    ax.set_ylabel(r'latitude ($^\circ$)')
    ax.set_title(r'a) Convex hull of the grid used for the modelling of spatial effects')
    fig.show()
    fig.savefig(str(plot_path / 'grid.png'), DPI=200)


def plot_grid_extremal_corr_space(df):
    grid = grid_from_geopandas_pointcloud(df)
    agg_df = aggregate_claim_data(df, grid)
    agg = compute_extremal_correlation_over_grid(agg_df)
    fig, ax = plt.subplots(ncols=1, figsize=(6, 5), constrained_layout=True)
    agg['corr'].plot(kind='bar', x='dist', width=0.8, color='slategrey', ax=ax)
    ax.set_title(r'Extremal correlation $\pi$ over a grid')
    ax.set_xlabel('distance (km)')
    ax.set_ylabel(r'$\pi$')
    fig.savefig(str(plot_path / 'extremal_corr_over_grid.png'), DPI=200)


def plot_values_over_grid(df, aggfunc=lambda x: x.mean()):
    df.exposure = np.log(df.exposure)
    cnt = df.groupby('gridcell').claim_value.count()
    valid = cnt[cnt > 100].index
    all = df[df.gridcell.isin(valid)]
    fig1, ax1 = plt.subplots(figsize=(8, 6), constrained_layout=True)
    geoplot.choropleth(all.dissolve(['latitude_grid', 'longitude_grid'], aggfunc=aggfunc)[['geometry', 'exposure']],
                       hue='exposure', cmap='Reds', legend=True, edgecolor='white', linewidth=1, ax=ax1)
    ax1.set_title('Average exposure value over the grid (CHF)')
    fig1.savefig(str(PLOT_ROOT / 'av_exposure_value_grid.png'), DPI=200)
    fig2, ax2 = plt.subplots(figsize=(8, 6), constrained_layout=True)
    geoplot.choropleth(all.dissolve(['latitude_grid', 'longitude_grid'], aggfunc=aggfunc)[['geometry', 'claim_value']],
                       hue='claim_value', cmap='Reds',
                       legend=True, edgecolor='white', linewidth=1, ax=ax2)
    ax2.set_title('b) Average claim value over the grid')
    fig2.savefig(str(PLOT_ROOT / 'av_claim_value_grid.png'), DPI=200)
    fig3, ax3 = plt.subplots(figsize=(8, 6), constrained_layout=True)
    count = all.dissolve(['latitude_grid', 'longitude_grid'], aggfunc=lambda x: x.count())[['geometry', 'exposure']]
    geoplot.choropleth(count, hue='exposure', cmap='Reds', legend=True, edgecolor='white', linewidth=1, ax=ax3)
    ax3.set_title('c) Count of claims over the grid')
    fig3.savefig(str(plot_path / 'count_claims_grid.png'), DPI=200)


def plot_correlation(df_hail, dim='space'):
    grid = grid_from_geopandas_pointcloud(df_hail)
    agg = aggregate_claim_data(df_hail, grid)
    if dim == 'time':
        letter = ''
    else:
        letter = 'b'
    column = 'value'
    spearman = get_spearman_corr(agg, dim, column)
    extremal = get_extremal_corr(agg, dim, column)
    fig, ax = plt.subplots(ncols=1, figsize=(8, 7), constrained_layout=True)
    spearman['corr'].rolling(3).mean().loc[:80].plot(marker='x', x='h', color='slategrey', ax=ax, label=r'Spearman coefficient $\rho$')
    ax.fill_between(spearman.loc[:80].index, spearman.q05.rolling(3).mean().loc[:80], spearman.q95.rolling(3).mean().loc[:80], color='slategrey', alpha=0.3)
    xlabel = 'lag (days)' if dim == 'time' else 'distance (km)'
    extremal['corr'].rolling(3).mean().loc[:80].plot(marker='x', x='h', color='royalblue', ax=ax, label=r'Extremal correlation $\pi$')
    ax.fill_between(extremal.loc[:80].index, extremal.q05.rolling(3).mean().loc[:80], extremal.q95.rolling(3).mean().loc[:80], color='royalblue', alpha=0.3)
    ax.legend()
    ax.set_title(f'{letter}) Evolution of the autocorrelation over {dim}')
    ax.set_xlabel(xlabel)
    fig.show()
    fig.savefig(str(plot_path / f'correlation_{dim}_summer_{column}.png'), DPI=200)


def plot_examples(data):
    data = data.reset_index().assign(geom_point=lambda x: geopandas.points_from_xy(x['longitude'], x['latitude'])).set_geometry('geom_point')

    geom_roots = glob(str(DATA_ROOT / 'SHAPEFILE/swissTLMRegio_Boundaries_LV95/swissTLMRegio_*_LV95.shp'))
    df_polygon = pd.concat([geopandas.read_file(geom_root) for geom_root in geom_roots]).to_crs(epsg='4326').clip_by_rect(xmin=data.longitude.min(), xmax=data.longitude.max(),
                                                                                                                          ymin=data.latitude.min(), ymax=data.latitude.max()).to_crs(epsg='2056')
    dates = []
    for d in np.unique(data.claim_date):
        df = data[data.claim_date == d]
        if len(df) > 30:
            dates.append(d)
    era5 = '/Volumes/ExtremeSSD/wind_downscaling_gan/data/ERA5'

    def plot_wind_components_from_array(u: np.ndarray, v: np.ndarray, longitudes: np.ndarray, latitudes: np.ndarray,
                                        ax=None, proj=None,
                                        title=''):
        if ax is None:
            proj = cartopy.crs.PlateCarree()
            ax = plt.axes(projection=proj)
        proj = proj or cartopy.crs.PlateCarree()
        ax.quiver(longitudes, latitudes,
                  u, v, transform=proj, scale_units='height', scale=10)
        ax.set_title(title)
        return ax

    for d in dates:
        p = glob(era5 + f"/{str(d)[:10].replace('-', '')}*surface*daily*")
        if len(p):
            df = data[data.claim_date == d]
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, constrained_layout=True, figsize=(15, 10), subplot_kw={'projection': cartopy.crs.epsg(2056)})
            df_polygon.plot(ax=ax1, facecolor='goldenrod', edgecolor='grey', alpha=0.2)
            df_polygon.plot(ax=ax2, facecolor='goldenrod', edgecolor='grey', alpha=0.2)
            vmin = float(data.claim_value.quantile(0.05))
            vmax = float(data.claim_value.quantile(0.95))
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            sm = matplotlib.cm.ScalarMappable(norm=norm, cmap='Reds')
            cmap = sm.cmap
            geoplot.pointplot(df, norm=norm, cmap=cmap, hue='climada_dmg', ax=ax2)
            geoplot.pointplot(df, norm=norm, cmap=cmap, hue='claim_value', ax=ax3, legend=True, legend_kwargs={'shrink': 0.7})
            ax2.set_title('CLIMADA predicted damages (CHF)', fontsize=16)
            ax3.set_title('Observed damages (CHF)', fontsize=16)
            ax1.set_title('Wind direction', fontsize=16)
            era5_date = xr.open_mfdataset(p)
            df_polygon.plot(ax=ax3, facecolor='goldenrod', edgecolor='grey', alpha=0.2)
            plot_wind_components_from_array(era5_date.u10.max('time').to_numpy(), era5_date.v10.max('time').to_numpy(),
                                            longitudes=era5_date.longitude.to_numpy(), latitudes=era5_date.latitude.to_numpy(), ax=ax1, proj=cartopy.crs.PlateCarree())
            for ax in [ax1, ax2, ax3]:
                ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
                ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), color='black')
                ax.set_extent(spatial_extent)
            fig.suptitle(str(d)[:10], fontsize=20)
            fig.show()
            fig.savefig(str(plot_path / 'example_days' / f'{str(d)[:10]}.png'), DPI=200)


def plot_exploratory():
    plot_climada_compensation()
    plot_climada_count_to_buildings_ratio()
    plot_example_day_and_line()
    plot_poh_meshs_winds(pd.to_datetime('2021-06-28'))
    plot_exposure()
    plot_average_claim_values_time(claim_values)
    plot_correlation(claim_values, dim='space')
    im1 = cv2.imread(str(PLOT_ROOT / 'average_total_claims.png'))
    im2 = cv2.imread(str(PLOT_ROOT / f'correlation_space_summer_value.png'))
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(ncols=2, constrained_layout=True, figsize=(16, 7))
    ax1.imshow(im1)
    ax2.imshow(im2)
    ax1.axis('off')
    ax2.axis('off')
    fig.savefig(str(plot_path / 'distri_claim_values.png'), dpi=200)
