import pathlib
from glob import glob

import cartopy
import cartopy.crs as ccrs
import contextily as ctx
import cv2
import geopandas
import geoplot
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from data.hailcount_data_processing import get_grid_mapping
from extreme_values_visualization import extremogram_plot
from threshold_selection import threshold_selection_GoF
from utils import grid_from_geopandas_pointcloud, aggregate_claim_data, compute_extremal_correlation_over_grid, get_extremal_corr, get_spearman_corr

mapping = get_grid_mapping(suffix='GVZ_emanuel')
mapping = mapping.assign(geom_point=geopandas.GeoSeries.from_wkt(mapping.geom_point))
params = {'legend.fontsize': 'xx-large',
          'axes.facecolor': '#eeeeee',
          'axes.labelsize': 'xx-large', 'axes.titlesize': 20, 'xtick.labelsize': 'xx-large',
          'ytick.labelsize': 'xx-large'}
pylab.rcParams.update(params)
matplotlib.rcParams["text.usetex"] = True
PLOT_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/plots/')
DATA_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/data/GVZ_Datenlieferung_an_ETH/')
EXPOSURES_ROOT = pathlib.Path(DATA_ROOT / 'GVZ_Exposure_202201').with_suffix('.csv')
HAILSTORM_ROOT = pathlib.Path(DATA_ROOT / 'GVZ_Hail_Loss_200001_to_202203').with_suffix('.csv')
CRS = 'EPSG:2056'
CCRS = ccrs.epsg(2056)


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
    fig.savefig(str(PLOT_ROOT / 'average_total_claims.png'), DPI=200)


def plot_climada_exposure(exp):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5), subplot_kw={'projection': ccrs.epsg(3857)})
    exp.plot_basemap(axis=ax1)
    ax1.set_title('Pointwise log exposure')
    exp.plot_hexbin(axis=ax2)
    ax2.set_title('Total log exposure over spatial bins')
    ctx.add_basemap(ax=ax2, origin='upper', url=ctx.providers.Stamen.Terrain)
    fig.show()
    fig.savefig(str(PLOT_ROOT / 'climada_exposure.png'), DPI=200)


def plot_diagnostic_logsum_claimvalues(df):
    ext_data = np.log1p(df.groupby('claim_date').claim_value.sum())
    ts, fig = threshold_selection_GoF(ext_data, min_threshold=2, max_threshold=10, plot=True)
    fig.savefig(str(PLOT_ROOT / 'threshold_selection.png'), DPI=200)
    extremogram_plot(ext_data.to_frame(name='data')
                     .assign(days_since_start=lambda x: (x.index - x.index.min()).days)
                     .assign(threshold=ts[0]), h_range=np.linspace(1, 100, 15),
                     path_to_figure=str(PLOT_ROOT))


def plot_grid(df):
    fig, ax = plt.subplots(ncols=1, figsize=(6, 5), constrained_layout=True)
    g = geopandas.GeoSeries(df.geometry.unique())
    g.plot(edgecolor='salmon', facecolor='salmon', linewidth=2, alpha=0.2, ax=ax)
    df.assign(geom_point=lambda x: geopandas.points_from_xy(x['longitude'], x['latitude'])).set_geometry('geom_point').plot(markersize=1, ax=ax, color='navy')
    ax.set_xlabel(r'longitude ($^\circ$)')
    ax.set_ylabel(r'latitude ($^\circ$)')
    ax.set_title(r'a) Convex hull of the grid used for the modelling of spatial effects')
    fig.show()
    fig.savefig(str(PLOT_ROOT / 'grid.png'), DPI=200)


def plot_grid_extremal_corr_space(df):
    grid = grid_from_geopandas_pointcloud(df)
    agg_df = aggregate_claim_data(df, grid)
    agg = compute_extremal_correlation_over_grid(agg_df)
    fig, ax = plt.subplots(ncols=1, figsize=(6, 5), constrained_layout=True)
    agg['corr'].plot(kind='bar', x='dist', width=0.8, color='slategrey', ax=ax)
    ax.set_title(r'Extremal correlation $\pi$ over a grid')
    ax.set_xlabel('distance (km)')
    ax.set_ylabel(r'$\pi$')
    fig.savefig(str(PLOT_ROOT / 'extremal_corr_over_grid.png'), DPI=200)


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
    fig3.savefig(str(PLOT_ROOT / 'count_claims_grid.png'), DPI=200)


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
    fig.savefig(str(PLOT_ROOT / f'correlation_{dim}_summer_{column}.png'), DPI=200)


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
                ax.set_extent([mapping.longitude.min(), mapping.longitude.max(),
                               mapping.latitude.min(), mapping.latitude.max()])
            fig.suptitle(str(d)[:10], fontsize=20)
            fig.show()
            fig.savefig(str(PLOT_ROOT / 'example_days' / f'{str(d)[:10]}.png'), DPI=200)


def plot_all(df):
    plot_average_claim_values_time(df)
    plot_correlation(df, dim='space')
    im1 = cv2.imread(str(PLOT_ROOT / 'average_total_claims.png'))
    im2 = cv2.imread(str(PLOT_ROOT / f'correlation_space_summer_value.png'))
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(ncols=2, constrained_layout=True, figsize=(16, 7))
    ax1.imshow(im1)
    ax2.imshow(im2)
    ax1.axis('off')
    ax2.axis('off')
    fig.savefig(str(PLOT_ROOT / 'figure_6.png'), dpi=200)


if __name__ == '__main__':
    print("Loading corrected data")
    data = pd.read_csv(str(DATA_ROOT / 'processed.csv'), index_col=[0, 1, 2, 3], parse_dates=[3]).drop(columns=['time']).reset_index()
    data['geometry'] = geopandas.GeoSeries.from_wkt(data['geometry'])
    data = data.set_geometry('geometry')
    print("Plotting")
    PLOT_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/plots/exploratory/')
    plot_all(data)
