import math
import pathlib

import cartopy
import geopandas
import geoplot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import PLOT_ROOT, DATA_ROOT, PRED_ROOT, claim_values, CRS, confidence, nb_draws, df_polygon, df_lakes, suffix
from data.hailcount_data_processing import get_train_data, get_test_data, get_validation_data, get_grid_mapping
from diagnostic.prediction import generate_prediction_for_date, get_pred_counts_for_day

mapping = get_grid_mapping(suffix=suffix)
spatial_extent = [mapping.longitude.min() - 0.01, mapping.longitude.max() + 0.01,
                  mapping.latitude.min() - 0.01, mapping.latitude.max() + 0.01]


def plot_counts(counts_csv, name_counts):
    path = PLOT_ROOT / name_counts
    path.mkdir(parents=True, exist_ok=True)
    climada_counts = pd.read_csv(DATA_ROOT / 'GVZ_emanuel' / 'climada_cnt.csv', index_col=['latitude', 'longitude'],
                                 usecols=lambda x: ('2' in x) or x in ['latitude', 'longitude'])
    climada_counts_date = [climada_counts[c].rename('climadadmg').to_frame().assign(claim_date=pd.to_datetime(c)) for c in climada_counts.columns]
    climada_count_pos = pd.concat([d[d.climadadmg > 0] for d in climada_counts_date])
    climada_counts = climada_count_pos.reset_index().assign(geom=lambda x: geopandas.points_from_xy(x.longitude, x.latitude, crs='EPSG:2056').to_crs('epsg:4326')) \
        .set_geometry('geom').assign(longitude=lambda x: x.geom.x).assign(latitude=lambda x: x.geom.y).merge(mapping[['longitude', 'latitude', 'gridcell']],
                                                                                                             on=['latitude', 'longitude'], how='left')
    climada_counts = climada_counts.set_index(['longitude', 'latitude', 'claim_date'])
    cc = climada_counts.dissolve(['gridcell', 'claim_date'], 'sum').climadadmg.rename('climada_cnt').reset_index().merge(mapping.drop_duplicates('gridcell')[['gridcell', 'geometry']],
                                                                                                                         on='gridcell', how='left').set_geometry('geometry')
    for date in counts_csv.claim_date.unique():
        spec_date = counts_csv[counts_csv.claim_date == date]
        poscounts = spec_date[spec_date.obscnt >= 1]
        pospred = spec_date[spec_date.pred_cnt >= 1]
        if len(poscounts) > 10:
            vmin = 0
            vmax = poscounts.obscnt.max()
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            sm = matplotlib.cm.ScalarMappable(norm=norm, cmap='OrRd')
            cmap = sm.cmap
            fig, axes = plt.subplots(ncols=3, figsize=(15, 7),
                                     constrained_layout=True, subplot_kw={'projection': cartopy.crs.PlateCarree()})
            ax1, ax2, ax3 = axes.flatten()
            geoplot.choropleth(poscounts, hue='obscnt', cmap=cmap, norm=norm, legend=False, ax=ax1)
            ax1.set_title(f'Observed: {round(poscounts.obscnt.sum(), 1)} claims')

            if pd.to_datetime(date) in cc.claim_date.unique():
                pos_clim = cc[cc.claim_date == date]
                geoplot.choropleth(pos_clim[pos_clim.climada_cnt >= 1], hue='climada_cnt', cmap=cmap, norm=norm, legend=False, ax=ax2)
                nb = pos_clim[pos_clim.climada_cnt >= 1].climada_cnt.sum()
            else:
                nb = 0.0
            ax2.set_title(f'CLIMADA: {round(nb, 1)} claims')
            geoplot.choropleth(pospred, hue='pred_cnt', cmap=cmap, norm=norm, legend=False, ax=ax3)
            ax3.set_title(f'Predicted: {round(pospred.pred_cnt.sum(), 1)} claims')
            fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02, extend='both')
            for ax in axes:
                df_polygon.plot(ax=ax, facecolor='oldlace', edgecolor='grey')
                df_lakes.plot(ax=ax, facecolor='royalblue', alpha=0.5)
                ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
                ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), color='black')
                ax.set_extent(spatial_extent)
            fig.suptitle(date, fontsize=20)
            # fig.show()
            path_plot = path / f'pred_poisson_{name_counts}_{date}.png'
            fig.savefig(path_plot, DPI=200)


def select_random_date(name_counts):
    random = np.random.choice([0], 1)[0]
    suffix = {0: 'train', 1: 'val', 2: 'test'}
    predicted_counts = pd.read_csv(str(PRED_ROOT / f'{name_counts}' / f'mean_pred_{suffix[random]}_{name_counts}.csv'), parse_dates=['claim_date'])
    dc_pos = predicted_counts[predicted_counts.pred_cnt >= 1.]
    random_date = pd.to_datetime(np.random.choice(dc_pos.claim_date.unique(), 1)[0])
    dc_day = dc_pos[dc_pos.claim_date == random_date].set_index(['gridcell', 'claim_date'])
    return (dc_day, random_date)


def generate_map_for_date(random_date, dc_day, smooth_spatially=False, bin=False, name_sizes=None):
    random_date = pd.to_datetime(random_date)
    date_str = random_date.strftime('%Y%m%d')
    if name_sizes:
        path_to_predicted_damages = pathlib.Path(PRED_ROOT / name_sizes / date_str).with_suffix('.csv')
        # Combining counts and sizes
        if not path_to_predicted_damages.is_file():
            pred_pos = generate_prediction_for_date(random_date, dc_day, mapping=mapping, nb_draws=nb_draws, smooth_spatially=smooth_spatially, bin=bin, path_to_sizes=path_to_predicted_damages.parent)
        else:
            pred_pos = pd.read_csv(str(path_to_predicted_damages)).assign(geometry=lambda x: geopandas.GeoSeries.from_wkt(x['geometry'])).set_geometry('geometry')
        generate_map_from_predicted_sizes(random_date, pred_pos, path=PLOT_ROOT / name_sizes)


def generate_map_from_predicted_sizes(random_date, predicted_sizes, path=None, name_sizes=''):
    random_date = pd.to_datetime(random_date)
    date_str = random_date.strftime('%Y%m%d')
    # Combining counts and sizes
    obs_pos = claim_values[claim_values.claim_date == random_date].drop(columns=['gridcell', 'geometry']) \
        .merge(mapping[['latitude', 'longitude', 'gridcell', 'geom_point']], on=['longitude', 'latitude'], how='left').set_geometry('geom_point')
    pred_pos = predicted_sizes.assign(geom_point=lambda x: geopandas.GeoSeries.from_xy(x.longitude, x.latitude)).set_geometry('geom_point')
    # Combining counts and sizes
    climada_damages = pd.read_csv(DATA_ROOT / 'GVZ_emanuel' / 'climada_dmg.csv', index_col=['latitude', 'longitude'],
                                  usecols=lambda x: ('2' in x) or x in ['latitude', 'longitude'])
    if random_date.strftime('%Y-%m-%d') in climada_damages.columns:
        climada_pred_pos = climada_damages[random_date.strftime('%Y-%m-%d')].rename('climadadmg')
        climada_pred_pos = climada_pred_pos[climada_pred_pos > 0].reset_index().assign(geom=lambda x: geopandas.points_from_xy(x.longitude, x.latitude, crs=CRS).to_crs('epsg:4326')) \
            .set_geometry('geom').assign(longitude=lambda x: x.geom.x).assign(latitude=lambda x: x.geom.y).merge(mapping[['longitude', 'latitude', 'gridcell']],
                                                                                                                 on=['latitude', 'longitude'], how='left')
        climada_pred_pos = climada_pred_pos.groupby('gridcell').climadadmg.sum().reset_index().merge(mapping.drop_duplicates('gridcell')[['gridcell', 'geometry']], on='gridcell',
                                                                                                     how='left').set_geometry('geometry')
        # Plots
    fig, axes = plt.subplots(ncols=3, figsize=(19, 7), constrained_layout=True, subplot_kw={'projection': cartopy.crs.PlateCarree()})
    ax1, ax2, ax3 = axes.flatten()
    vmin = min(obs_pos['claim_value'].min(), pred_pos['mean_pred_size'].min()) if len(obs_pos) else pred_pos['mean_pred_size'].min()
    vmax = max(obs_pos['claim_value'].max(), pred_pos['mean_pred_size'].max()) if len(obs_pos) else pred_pos['mean_pred_size'].max()
    if random_date.strftime('%Y-%m-%d') in climada_damages.columns:
        cvmin = min(min(climada_pred_pos['climadadmg'].min(), obs_pos['claim_value'].min()), pred_pos['mean_pred_size'].min())
        cvmax = max(max(climada_pred_pos['climadadmg'].max(), obs_pos['claim_value'].max()), pred_pos['mean_pred_size'].max())
    else:
        cvmin = vmin
        cvmax = vmax
    norm = matplotlib.colors.LogNorm(vmin=max(vmin, 1), vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap='OrRd')
    cnorm = matplotlib.colors.FuncNorm([lambda x: x, lambda x: x], vmin=cvmin, vmax=cvmax)
    csm = matplotlib.cm.ScalarMappable(norm=cnorm, cmap='OrRd')
    cmap = sm.cmap
    ccmap = csm.cmap
    for ax in [ax1, ax2, ax3]:
        df_polygon.plot(ax=ax, facecolor='oldlace', edgecolor='grey')
        df_lakes.plot(ax=ax, facecolor='royalblue', alpha=0.5)
        ax.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
        ax.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), color='black')
    geoplot.pointplot(obs_pos, hue='claim_value', norm=norm, cmap=cmap, legend=False, ax=ax1, s=2)
    if random_date.strftime('%Y-%m-%d') in climada_damages.columns:
        geoplot.choropleth(climada_pred_pos, hue='climadadmg', norm=cnorm, cmap=ccmap, legend=False, ax=ax2)
        ax2.set_title(f'CLIMADA predicted damages: CHF{number_to_scientific(climada_pred_pos["climadadmg"].sum())}', fontsize=18)
    else:
        ax2.set_title(f'CLIMADA predicted damages: CHF0', fontsize=18)
    geoplot.pointplot(pred_pos, hue='mean_pred_size', cmap=cmap, norm=norm, legend=False, ax=ax3, s=3)
    for ax in [ax1, ax2, ax3]:
        ax.set_extent(spatial_extent)
    ax1.set_title(f'Observed damages: CHF{number_to_scientific(obs_pos["claim_value"].sum())}', fontsize=18)
    ax3.set_title(
        f'Predicted damages: CHF{number_to_scientific(pred_pos["mean_pred_size"].sum())} / {int(100 * (1 - confidence))}\%CI[{number_to_scientific(pred_pos["lb_pred"].sum())},'
        f' {number_to_scientific(pred_pos["ub_pred"].sum())}]', fontsize=18)
    fig.colorbar(sm, ax=axes, fraction=0.6, label='per-building damage (CHF)', pad=0.02, extend='both', aspect=70)
    fig.colorbar(csm, ax=axes, fraction=0.6, label='per-cell damage (CHF)', pad=0.02, extend='both', aspect=70)
    fig.suptitle(random_date.strftime('%Y-%m-%d'), fontsize=20)
    # fig.show()
    p = path or PLOT_ROOT / name_sizes
    path_to_plots = pathlib.Path(p / date_str, with_suffix='png')
    path_to_plots.parent.mkdir(exist_ok=True, parents=True)
    if not path_to_plots.is_file():
        fig.savefig(str(path_to_plots), DPI=200)


def number_to_scientific(number):
    if number == 0:
        return number
    power_of_ten = math.floor(math.log10(np.abs(number)))
    if power_of_ten < 3:
        return round(number, 1)
    elif power_of_ten < 6:
        return f'{round(number / 1e3, 1)}K'
    elif power_of_ten < 9:
        return f'{round(number / 1e6, 1)}M'
    else:
        return f'{round(number / 1e9, 1)}B'
