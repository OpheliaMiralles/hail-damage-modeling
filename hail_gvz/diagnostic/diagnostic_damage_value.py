import pathlib

import arviz as az
import geopandas
import geoplot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as mc
import seaborn as sns
import xarray as xr

from data.haildamage_data_processing import get_validation_data, get_test_data
from exploratory_da.utils import process_exposure_data
from models.combined_value import build_model, get_chosen_variables_for_model

matplotlib.rcParams["text.usetex"] = True
az.style.use("arviz-darkgrid")
DATA_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/data/GVZ_Datenlieferung_an_ETH/')
PLOT_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/plots/')
FITS_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/fits/')
threshold = 8.06
exp_threshold = np.exp(threshold) - 1
tol = 1e-2
confidence = 0.15

validation_data = get_validation_data()
test_data = get_test_data()
model = build_model(validation_data, fit=True)
trace = get_chosen_variables_for_model(model)
predicted = mc.sample_posterior_predictive(trace, model)


def rmse(df1, df2):
    return np.sqrt(((1 - df2 / df1) ** 2).mean('draw'))


def relbias(df1, df2):
    return (1 - df2 / df1).mean('draw')


def confidence_coverage(df1, df2):
    q05 = df2.quantile(0.05, dim='draw')
    q95 = df2.quantile(0.95, dim='draw')
    return ((q05 <= df1) & (df1 <= q95)).astype(int)


def distance_from_closest_bound(df1, df2):
    q05 = df2.quantile(0.05, dim='draw')
    q95 = df2.quantile(0.95, dim='draw')
    dist_from_up = q95 - df1
    dist_from_low = df1 - q05
    dist_from_closest = xr.where(dist_from_up < dist_from_low, dist_from_up, dist_from_low)
    return xr.where(dist_from_closest > 0, 0, dist_from_closest)


def get_observed_paa(data):
    exposure = process_exposure_data()
    nb_assets_total = len(exposure)
    nb_claims = data.groupby('MESHS').claim_value.count()
    return 100 * nb_claims / nb_assets_total


def make_merged_dataset(predicted, data, aggfunc=lambda x: x.mean(['chain', 'draw'])):
    data = data.sort_values('claim_date').reset_index()
    merged = xr.merge([aggfunc(predicted.posterior_predictive.cond_damage).rename('scaled_damages'),
                       predicted.observed_data.over_threshold.rename('true_bool'),
                       predicted.posterior_predictive.over_threshold.median(['chain', 'draw']).rename('pred_bool')]).to_dataframe()
    merged = merged.merge(data, left_index=True, right_index=True).assign(geom_point=lambda x: geopandas.points_from_xy(x['longitude'], x['latitude'])).set_geometry('geom_point')
    merged['geometry'] = geopandas.GeoSeries.from_wkt(merged['geometry'])
    merged = merged.assign(predicted=lambda x: x.scaled_damages + x.climada_dmg).rename(columns={'claim_value': 'observed'})
    return merged


def plot_predicted_vs_observed(data, predicted, level='grid'):
    merged = make_merged_dataset(predicted, data)
    if level == 'grid':
        merged = merged.set_geometry('geometry')
    vmin = float(merged.observed.quantile(0.1))
    vmax = float(merged.observed.quantile(0.9))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap='Reds')
    cmap = sm.cmap
    if level == 'time':
        toplot = merged.assign(year=lambda x: pd.to_datetime(x['time']).dt.year).assign(month=lambda x: pd.to_datetime(x['time']).dt.month).set_index('time')
    else:
        toplot = merged
    fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(15, 7))
    ax3, ax4 = axes.flatten()
    if level == 'location':
        geoplot.pointplot(toplot, ax=ax4, hue='observed', cmap=cmap, norm=norm, edgecolor='white', linewidth=1, legend=False)
        geoplot.pointplot(toplot, ax=ax3, hue='predicted', cmap=cmap, norm=norm, edgecolor='white', linewidth=1, legend=False)
        fig.colorbar(sm, ax=axes)
        ax3.set_title(f'Mean prediction over space at the {level} level')
        ax4.set_title(f'Observed claims over space at the {level} level')
    elif level == 'grid':
        geoplot.choropleth(toplot, ax=ax4, hue='observed', cmap=cmap, norm=norm, edgecolor='white', linewidth=1, legend=False)
        geoplot.choropleth(toplot, ax=ax3, hue='predicted', cmap=cmap, norm=norm, edgecolor='white', linewidth=1, legend=False)
        fig.colorbar(sm, ax=axes)
        ax3.set_title(f'Predicted claims over space at the {level} level')
        ax4.set_title(f'Observed claims over space at the {level} level')
    elif level == 'time':
        for col, color, alpha in zip(['observed', 'predicted'], ['salmon', 'royalblue'], [0.8, 0.4]):
            toplot.groupby('year')[col].mean().plot(ax=ax3, color=color, label=col.capitalize(), kind='bar', alpha=alpha)
            toplot.groupby('month')[col].mean().plot(ax=ax4, color=color, label=col.capitalize(), kind='bar', alpha=alpha)
        ax3.legend()
        ax3.set_title(f'Yearly averaged claim value')
        ax4.legend()
        ax4.set_title(f'Monthly averaged claim value')
    fig.show()
    return fig


def plot_qq(predicted):
    quants = np.linspace(tol, 1 - tol, 1000)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    quants_x = predicted.posterior_predictive.cond_damage.quantile(quants, dim='point').rename({'quantile': 'quant'})
    q05_quants = quants_x.quantile(confidence, dim=['chain', 'draw'])
    q95_quants = quants_x.quantile(1 - confidence, dim=['chain', 'draw'])
    mean_quants = quants_x.mean(dim=['chain', 'draw'])
    observed_quantiles = predicted.observed_data.cond_damage.quantile(quants)
    ax.scatter(observed_quantiles, mean_quants, color='navy', marker='o', s=6)
    ax.fill_between(observed_quantiles, q05_quants, q95_quants, color='navy', alpha=0.2)
    #ax.plot(observed_quantiles, observed_quantiles, color='navy')
    ax.set_ylabel('Predicted damages (CHF)')
    ax.set_xlabel('Observed damages (CHF)')
    fig.suptitle(f'QQ Plot of predicted vs observed claim values')
    fig.show()
    fig.savefig(str(PLOT_ROOT / 'qq_all.png'), DPI=200)


def plot_mdr(predicted, data):
    paa = get_observed_paa(data)
    mean = make_merged_dataset(predicted, data)
    q05 = make_merged_dataset(predicted, data, aggfunc=lambda x: x.quantile(confidence, ['chain', 'draw']))
    q95 = make_merged_dataset(predicted, data, aggfunc=lambda x: x.quantile(1 - confidence, ['chain', 'draw']))
    mean_mdr = mean.groupby('MESHS').sum()
    mean_mdr = mean_mdr.assign(mdr_pred=100 * mean_mdr.predicted / mean_mdr['exposure']) \
        .assign(mdr_obs=100 * mean_mdr.observed / mean_mdr['exposure']).assign(mdr_climada=100 * mean_mdr.climada_dmg / mean_mdr['exposure'])
    q05_mdr = q05.groupby('MESHS').sum()
    q05_mdr = q05_mdr.assign(mdr_pred=100 * q05_mdr.predicted / q05_mdr['exposure']).assign(mdr_obs=100 * q05_mdr.observed / q05_mdr['exposure'])
    q95_mdr = q95.groupby('MESHS').sum()
    q95_mdr = q95_mdr.assign(mdr_pred=100 * q95_mdr.predicted / q95_mdr['exposure']).assign(mdr_obs=100 * q95_mdr.observed / q95_mdr['exposure'])
    dic = {}
    for q, d in zip(['q05', 'mean', 'q95'], [q05_mdr, mean_mdr, q95_mdr]):
        p = d.mdr_pred.rolling(10).mean()
        dic[q] = p
    dic['obs'] = mean_mdr.mdr_obs.rolling(10).mean()
    dic['climada'] = mean_mdr.mdr_climada.rolling(10).mean()
    toplot = pd.DataFrame.from_dict(dic)
    fig, ax = plt.subplots(ncols=1, figsize=(10, 5), tight_layout=True)
    ax.plot(toplot.obs * paa, color='salmon', label='Observed')
    ax.plot(toplot.climada, color='goldenrod', label='CLIMADA')
    ax.plot(toplot['mean'] * paa, color='navy', label='Predicted mean')
    ax.plot(toplot.q05 * paa, color='navy', ls='--', label=f'{int(round(100 * (1 - confidence), 0))}\% CI')
    ax.plot(toplot.q95 * paa, color='navy', ls='--')
    ax.set_xlabel('MESHS (mm)')
    ax.set_ylabel('MDR (\%)')
    ax.legend()
    fig.suptitle(f'Mean damage ratio computed on test set')
    fig.show()
    fig.savefig(str(PLOT_ROOT / 'mdr.png'), DPI=200)


def boxplot(merged):
    data_boxplot = merged[['observed', 'predicted', 'climada_dmg', 'month', 'location', 'claim_date']].set_index(['claim_date', 'location', 'month'])
    data_boxplot = pd.concat([data_boxplot[['observed']].assign(label='Observed').rename(columns={'observed': 'data'}), data_boxplot[['predicted']]
                             .assign(label='Predicted').rename(columns={'predicted': 'data'}), data_boxplot[['climada_dmg']].assign(label='CLIMADA').rename(columns={'climada_dmg': 'data'})])
    lim = data_boxplot.data.quantile(0.995)
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    sns.boxplot(data_boxplot[data_boxplot.label!='CLIMADA'].reset_index(), x='month', y='data', hue='label', ax=ax)
    ax.set_ylim((0, lim))
    ax.set_title('Boxplot of the predicted and observed damage distribution per month.')
    fig.show()
    fig.savefig(str(PLOT_ROOT / 'boxplot_time.png'), DPI=200)


def diagnostic(data):
    merged = make_merged_dataset(predicted, data)
    plot_mdr(predicted, data)
    boxplot(merged)
    plot_qq(predicted)


if __name__ == '__main__':
    diagnostic(test_data)
