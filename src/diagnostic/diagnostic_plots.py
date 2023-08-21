import pathlib
from glob import glob

import arviz as az
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from constants import FITS_ROOT, PRED_ROOT, PLOT_ROOT, claim_values, quants, confidence, suffix, name_pot, name_beta, name_bern, name_counts, scaling_factor, tol
from data.climada_processing import process_climada_counts, process_climada_perbuilding_positive_damages
from data.hailcount_data_processing import get_train_data, get_test_data, get_validation_data, get_exposure, get_grid_mapping
from diagnostic.map_generation import generate_map_for_date
from diagnostic.metrics import log_spectral_distance_from_xarray, spatially_convolved_ks_stat
from diagnostic.prediction import get_pred_counts_for_day

exposure = get_exposure()
mapping = get_grid_mapping(suffix=suffix)
obs_damages = claim_values[['claim_date', 'longitude', 'latitude', 'claim_value', 'MESHS']].merge(mapping[['latitude', 'longitude', 'gridcell']], on=['latitude', 'longitude'], how='left')
obs_dmg_gridcell = obs_damages.drop(columns=['longitude', 'latitude']).groupby(['gridcell', 'claim_date']).agg({'claim_value': 'sum', 'MESHS': 'mean'})
climada_counts = process_climada_counts()
climada_damages = process_climada_perbuilding_positive_damages()


def plot_mc_diagnostics(name_pot, name_beta, name_counts):
    name_dir = f'combined_20230221_12:58_{name_beta}_{name_pot}_{name_counts}'
    trace_beta_solo = az.from_netcdf(str(pathlib.Path(FITS_ROOT / 'claim_values' / name_beta).with_suffix('.nc')))
    trace_gpd_solo = az.from_netcdf(str(pathlib.Path(FITS_ROOT / 'claim_values' / name_pot).with_suffix('.nc')))
    trace_counts_solo = az.from_netcdf(str(pathlib.Path(FITS_ROOT / 'claim_counts' / name_counts).with_suffix('.nc')))
    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(15, 10))
    tt = trace_gpd_solo.posterior.rename(
        {'shape': r'$\xi$', 'coef_meshs': r'$\sigma_1$', 'constant_scale': r'$\sigma_0$', 'coef_crossed': r'$\sigma_2$',
         'coef_exposure': r'$\sigma_3$'})
    az.plot_autocorr(tt.isel(chain=0), var_names=[r'$\sigma_0$', r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$'],
                     ax=axes.flatten(), textsize=16, max_lag=100)
    fig.suptitle('Autocorrelation through time for parameters of the GPD model', fontsize=20)
    fig.show()
    fig.savefig(f'{PLOT_ROOT}/{name_dir}/autocorr_gpd.png', dpi=200)

    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(15, 10))
    tt = trace_gpd_solo.posterior.rename(
        {'shape': r'$\xi$'})
    az.plot_autocorr(tt, var_names=[r'$\xi$'], filter_vars="like",
                     ax=axes[:, 0])
    axes[0, 1].plot(tt[r'$\xi$'].sel(season=0, chain=0))
    axes[1, 1].plot(tt[r'$\xi$'].sel(season=1, chain=0))
    for i in range(2):
        axes[i, 0].set_title(r'Autocorrelation through time for $\xi_{}$'.format(i + 1))
    for i in range(2):
        axes[i, 1].set_title(r'Evolution of $\xi_{}$ after initial burn-in sample'.format(i + 1))
    fig.suptitle('Diagnostic plots for the shape parameter of the GPD model', fontsize=20)
    fig.show()
    fig.savefig(f'{PLOT_ROOT}/{name_dir}/diag_shape_gpd.png', dpi=200)

    fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(15, 5))
    tt = trace_counts_solo.posterior.isel(draw=slice(55, None)).rename(
        {'alpha': r'$\alpha$'})
    az.plot_autocorr(tt, var_names=[r'$\alpha$'], filter_vars="like",
                     ax=axes[0])
    axes[1].plot(tt[r'$\alpha$'].sel(chain=0))
    axes[0].set_title(r'Autocorrelation through time for $\alpha$')
    axes[1].set_title(r'Evolution of $\alpha$ after initial burn-in sample')
    fig.suptitle('Diagnostic plots for the shape parameter of the Negative Binomial model', fontsize=20)
    fig.show()
    fig.savefig(f'{PLOT_ROOT}/{name_dir}/diag_alpha_poisson.png', dpi=200)

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
                     ax=axes, max_lag=100)
    var_names = ['$\mu_0$', '$\mu_{11}$', '$\mu_{12}$', '$\mu_{13}$', '$\mu_2$', '$\mu_3$',
                 r'$\epsilon_1$', r'$\epsilon_2$']

    for ax, var in zip(axes.flatten(), var_names):
        ax.set_title(r"{}".format(var))
    fig.suptitle('Autocorrelation through time for parameters of the Negative Binomial model', fontsize=20)
    fig.show()
    fig.savefig(f'{PLOT_ROOT}/{name_dir}/autocorr_poisson.png', dpi=200)

    beta_latex_var = r"\nu"
    beta_varname_dic = {"precision": r"$\kappa$",
                        "coef_meshs_b": r"${}_1$".format(beta_latex_var),
                        "coef_poh_b": r"${}_2$".format(beta_latex_var)}
    fig, axes = plt.subplots(ncols=2, nrows=3, constrained_layout=True, figsize=(15, 10))
    tt = trace_beta_solo.posterior.rename(beta_varname_dic)
    az.plot_trace(tt, var_names=[v for v in beta_varname_dic.values()],
                  axes=axes)
    fig.suptitle('Diagnostic plots for parameters of the Beta model', fontsize=20)
    fig.show()
    fig.savefig(f'{PLOT_ROOT}/{name_dir}/diag_trace_beta.png', dpi=200)


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
    obs_paa, pred_paa, lb_paa, ub_paa, climada_paa = get_paa(counts)
    bounds = [20, 80]
    cuts = pd.DataFrame(np.array(pd.cut(obs_paa.reset_index().meshs, bins=50)), index=obs_paa.index, columns=['interval'])
    sclimada_paa = climada_paa.to_frame().merge(cuts, left_index=True, right_index=True, how='left').groupby('interval').mean()['climada_paa']
    sobs_paa = obs_paa.to_frame().replace(0, np.nan).interpolate().merge(cuts, left_index=True, right_index=True, how='left').groupby('interval').mean()['obs_paa']
    spred_paa = pred_paa.to_frame().replace(0, np.nan).interpolate().merge(cuts, left_index=True, right_index=True, how='left').groupby('interval').mean()['pred_paa']
    slb_paa = lb_paa.to_frame().replace(0, np.nan).interpolate().merge(cuts, left_index=True, right_index=True, how='left').groupby('interval').mean()['lb_paa']
    sub_paa = ub_paa.to_frame().replace(0, np.nan).interpolate().merge(cuts, left_index=True, right_index=True, how='left').groupby('interval').mean()['ub_paa']
    fig, ax = plt.subplots(ncols=1, figsize=(10, 6), constrained_layout=True)
    sobs_paa.rename('Observed').loc[bounds[0]:bounds[-1]].plot(ax=ax, color='salmon', legend=True)
    sclimada_paa.rename('CLIMADA').loc[bounds[0]:bounds[-1]].plot(ax=ax, color='goldenrod', legend=True)
    spred_paa.rename('Predicted mean').loc[bounds[0]:bounds[-1]].plot(ax=ax, color='navy', legend=True)
    slb_paa.rename(f'{int(round(100 * (1 - confidence), 0))}\% predicted range').loc[bounds[0]:bounds[-1]].plot(ax=ax, color='navy', ls='--', legend=True)
    sub_paa.loc[bounds[0]:bounds[-1]].plot(ax=ax, color='navy', ls='--', legend=False)
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
    fig, ax = plt.subplots(ncols=1, figsize=(8, 7), constrained_layout=True)
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
            label=f'{int(round(100 * (1 - confidence), 0))}\% predicted range')
    ax.plot(obs.quantile(quants), ub.quantile(quants),
            ls='--', color='navy')
    ax.plot(obs.quantile(quants), obs.quantile(quants), color='navy', label=r'$x=y$')
    fig.suptitle(f'c) QQ plot of the hail damage (CHF) per 2km gridcell', fontsize=20)
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
    fig, ax = plt.subplots(ncols=1, figsize=(8, 7), constrained_layout=True)
    pred_dmg_gridcell, lb_dmg_gridcell, ub_dmg_gridcell = series_predicted_damages
    func = lambda x: x
    quants = np.linspace(tol, 1 - 5e-3, 200)  # franchise claims + outlier
    obs = func(obs_damages.claim_value)
    pred = func(pred_dmg_gridcell)
    lb = func(lb_dmg_gridcell)
    ub = func(ub_dmg_gridcell)
    climada = func(climada_damages.climadadmg)
    ax.scatter(obs.quantile(quants), pred.quantile(quants), marker='x', color='navy', label='Predicted mean')
    q = lb.quantile(quants).rolling(100).mean()
    q.iloc[0] = 100
    ax.scatter(obs.quantile(quants), climada.quantile(quants), marker='x', color='goldenrod', label='Downscaled CLIMADA')
    ax.plot(obs.quantile(quants), q.interpolate(),
            ls='--', color='navy',
            label=f'{int(round(100 * (1 - confidence), 0))}\% predicted range')
    ax.plot(obs.quantile(quants), ub.quantile(quants),
            ls='--', color='navy')
    ax.plot(obs.quantile(quants), obs.quantile(quants), color='navy', label=r'$x=y$')
    fig.suptitle(f'b) QQ plot of the hail damage (CHF) per location', fontsize=20)
    ax.legend(loc='lower right')
    ax.set_xlabel('observed')
    ax.set_ylabel('predicted')
    ax.set_xscale(matplotlib.scale.FuncScaleLog(axis=0, functions=[lambda x: 1 + x, lambda x: x - 1]))
    ax.set_yscale(matplotlib.scale.FuncScaleLog(axis=1, functions=[lambda x: 1 + x, lambda x: x - 1]))
    fig.show()
    path = PLOT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / 'qq_location.png', DPI=200)

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
    ax.plot(obs_cnt, lb_cnt, ls='--', color='navy', label=f'{int(round(100 * (1 - confidence), 0))}\% predicted range')
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
    ax.plot(lb_mdr, color='navy', ls='--', label=f'{int(round(100 * (1 - confidence), 0))}\% CI')
    ax.plot(ub_mdr, color='navy', ls='--')
    ax.set_xlabel('MESHS (mm)')
    ax.set_ylabel('MDR (\%)')
    ax.legend()
    fig.suptitle(f'Mean damage ratio')
    fig.show()
    path = PLOT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / 'mdr.png', DPI=200)


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
    return clipped_df


def skss_input_vs_predicted(all, name):
    inputs = all.climadadmg
    targets = all.claim_value
    predicted = all.mean_pred_size
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
    return m_in, m_pred


def plot_errors_counts(counts, name):
    fig, ax = plt.subplots(ncols=1, figsize=(10, 6), constrained_layout=True)
    c = counts.groupby('claim_date').sum()
    neg_nb_cnt = 0
    zero_counts = c[c.obscnt <= neg_nb_cnt]
    non_zero_counts = c[c.obscnt > neg_nb_cnt]
    fa_pred = len(zero_counts[zero_counts.pred_cnt >= 1])
    fa_climada = len(zero_counts[zero_counts.climada_cnt > 0])
    tp_pred = len(non_zero_counts[non_zero_counts.pred_cnt >= 1])
    tp_climada = len(non_zero_counts[non_zero_counts.climada_cnt > 0])
    fn_pred = len(non_zero_counts[non_zero_counts.pred_cnt == 0])
    print(non_zero_counts[non_zero_counts.pred_cnt == 0].obscnt.quantile(np.linspace(0, 1, 10)))
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
    fig.suptitle('Count of claims per day', fontsize=20)
    fig.show()
    path = PLOT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / 'rates.png', DPI=200)


def plot_all():
    name_sizes = f'combined_{name_bern}_{name_beta}_{name_pot}_{name_counts}'
    plot_mc_diagnostics(name_pot, name_beta, name_counts)
    name = name_sizes
    path_to_counts = glob(str(pathlib.Path(PRED_ROOT / name_counts / '*').with_suffix('.csv')))
    path_to_sizes = glob(str(pathlib.Path(PRED_ROOT / name_sizes / '*').with_suffix('.csv')))
    az_counts = {p.split('/')[-1].split('.')[0]: pd.read_csv(p) for p in path_to_counts}
    counts = pd.concat(az_counts.values())
    counts = counts.assign(meshs=lambda x: scaling_factor * x.meshs)
    plot_errors_counts(counts, name)
    qq_counts(counts, name)
    plot_paa(counts, name)
    sizes_df = pd.concat([pd.read_csv(d).assign(claim_date=pd.to_datetime(d.split('/')[-1].split('.')[0])).set_index(['gridcell', 'claim_date']) for d in path_to_sizes])
    mean, lb, ub = sizes_df.mean_pred_size.rename('pred_dmg'), sizes_df.lb_pred.rename('pred_dmg_lb'), sizes_df.ub_pred.rename('pred_dmg_ub')
    qq_total_per_gridcell([mean, lb, ub], name_sizes)
    qq_total_per_location([mean, lb, ub], name_sizes)
    climada_xarray = climada_damages.reset_index().groupby(['gridcell', 'claim_date'], as_index=False).climadadmg.sum() \
        .merge(mapping.drop_duplicates('gridcell')[['gridcell', 'lat_grid', 'lon_grid']], on='gridcell', how='left') \
        .set_index(['claim_date', 'lon_grid', 'lat_grid']).climadadmg.to_xarray().fillna(0.)
    pred_xarray = sizes_df.reset_index().groupby(['gridcell', 'claim_date'], as_index=False).mean_pred_size.sum() \
        .merge(mapping.drop_duplicates('gridcell')[['gridcell', 'lat_grid', 'lon_grid']], on='gridcell', how='left') \
        .set_index(['claim_date', 'lon_grid', 'lat_grid']).mean_pred_size.to_xarray().fillna(0.)
    obs_xarray = claim_values.drop(columns=['gridcell']).merge(mapping, on=['latitude', 'longitude'], how='left') \
        .groupby(['claim_date', 'lon_grid', 'lat_grid']).claim_value.sum().to_xarray().fillna(0.)
    all = xr.merge([climada_xarray, pred_xarray, obs_xarray]).fillna(0.)
    lsd_df = lsd_input_vs_predicted(all.climadadmg,
                                    all.claim_value,
                                    all.mean_pred_size, name_sizes)
    skss_in, skss_pred = skss_input_vs_predicted(all, name_sizes)
    scaled_lsd = ((-lsd_df.CLIMADA + lsd_df.predicted) / lsd_df.CLIMADA).rename('LSD')
    scaled_skss = pd.Series(((-skss_in + skss_pred) / skss_in).numpy().flatten()).replace(np.inf, np.nan).dropna().rename('SKSS')
    fig, ax = plt.subplots(ncols=1, figsize=(8, 7), constrained_layout=True)
    props = dict(widths=0.7, patch_artist=True, medianprops=dict(color="lightgrey"))
    bp1 = ax.boxplot(scaled_skss.dropna(), labels=['SKSS'], positions=[0], **props)
    ax2 = ax.twinx()
    bp2 = ax2.boxplot(scaled_lsd, labels=['LSD'], positions=[1], **props)
    colors = ['navy', 'indianred']
    for bplot, axx, color in zip((bp1, bp2), [ax, ax2], colors):
        for patch in bplot['boxes']:
            patch.set_facecolor(color)
            axx.tick_params(axis='y', colors=color)
    ax.set_xlabel('metric')
    ax.set_ylabel('variation from CLIMADA')
    fig.show()
    fig.suptitle(f'a) Scaled variation of metric values', fontsize=20)
    path = PLOT_ROOT / name_sizes
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / 'metric_values.png', DPI=200)
    im1 = cv2.imread(str(PLOT_ROOT / name_sizes / 'metric_values.png'))
    im2 = cv2.imread(str(PLOT_ROOT / name_sizes / f'qq_location.png'))
    im3 = cv2.imread(str(PLOT_ROOT / name_sizes / f'qq_gridcell.png'))
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, constrained_layout=True, figsize=(20, 6))
    ax1.imshow(im1)
    ax2.imshow(im2)
    ax3.imshow(im3)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    fig.show()
    fig.savefig(str(PLOT_ROOT / name_sizes / 'metrics_qq_claim_values.png'), dpi=200)
    for data, n in zip([get_train_data(suffix=suffix), get_test_data(suffix=suffix), get_validation_data(suffix=suffix)], ['train', 'test', 'val']):
        dates = [pd.to_datetime('2004-07-08'), pd.to_datetime('2012-06-30'), pd.to_datetime('2021-06-28')]
        for d in dates:
            if d in data.reset_index().claim_date.unique():
                dc_day = get_pred_counts_for_day(data, d, name_counts)
                if len(dc_day[dc_day.pred_cnt >= 1]):
                    print(f'Selecting date {d}')
                    generate_map_for_date(d, dc_day, name_sizes=name_sizes)
                else:
                    print(f'No count on day {d}')