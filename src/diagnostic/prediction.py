import pathlib

import arviz as az
import geopandas
import numpy as np
import pandas as pd
import pymc as mc
import xarray as xr

from constants import FITS_ROOT, PRED_ROOT, claim_values, nb_draws, confidence, train_cond, test_cond, valid_cond, suffix, quantile_prediction_counts
from data.hailcount_data_processing import get_exposure, get_grid_mapping
from exploratory_da.utils import associate_data_with_grid, grid_from_geopandas_pointcloud
from models.combined_value import build_model, get_chosen_variables_for_model
from models.counts import build_poisson_model

exposure = get_exposure()
mapping = get_grid_mapping(suffix=suffix)
weights = claim_values[claim_values.claim_date >= pd.to_datetime('2008-01-01')].groupby(
    ['latitude', 'longitude']).gridcell.count().rename('weight').reset_index()
weights = exposure.merge(weights, on=['latitude', 'longitude'], how='left').fillna(0.).assign(
    weight=lambda x: x.weight * x.value / x.volume)[['weight', 'latitude', 'longitude']]
grid = grid_from_geopandas_pointcloud(
    claim_values.drop_duplicates(['latitude', 'longitude']).assign(
        geom_point=lambda x: geopandas.points_from_xy(x['longitude'], x['latitude'])).set_geometry('geom_point'))


def generate_and_save_counts_prediction(data, name_set, name_counts):
    trace_counts = az.from_netcdf(str(pathlib.Path(FITS_ROOT / 'claim_counts' / name_counts).with_suffix('.nc')))
    path = PRED_ROOT / f'{name_counts}'
    path.mkdir(parents=True, exist_ok=True)
    p = mc.sample_posterior_predictive(trace_counts, model=build_poisson_model(data, mapping, exposure))
    d = xr.merge([p.constant_data.meshs,
                  p.constant_data.grid_idx,
                  p.constant_data.time_idx,
                  p.constant_data.poh,
                  p.constant_data.climada_cnt,
                  p.observed_data.counts.rename('obscnt'),
                  p.posterior_predictive.counts.quantile(quantile_prediction_counts, ['chain', 'draw']).drop('quantile').rename('pred_cnt'),
                  p.posterior_predictive.counts.where(p.posterior_predictive.counts > 0).mean(['chain', 'draw']).rename(
                      'mean_pos'),
                  p.posterior_predictive.counts.quantile(quantile_prediction_counts - confidence / 2, ['chain', 'draw']).drop('quantile').rename('lb_counts'),
                  p.posterior_predictive.counts.quantile(quantile_prediction_counts + confidence / 2, ['chain', 'draw']).drop('quantile').rename('ub_counts'),
                  p.posterior_predictive.counts.mean(['chain', 'draw']).rename('mean_cnt')])
    counts_df = d.to_dataframe().assign(claim_date=lambda x: data.reset_index().claim_date.unique()[x.time_idx]) \
        .assign(gridcell=lambda x: mapping.gridcell.unique()[x.grid_idx]).merge(
        mapping.drop_duplicates(subset='gridcell')[['gridcell', 'geometry']], on='gridcell',
        how='left')
    j = counts_df.set_geometry('geometry')
    count_gridcell = mapping.groupby('gridcell').latitude.count().rename('cnt_gridcell').reset_index()
    j = j.merge(count_gridcell, on='gridcell', how='left')
    j = j.assign(mean_pred=lambda x: x.pred_cnt / x.cnt_gridcell)
    j.set_index(['claim_date', 'gridcell']).to_csv(path / f'mean_pred_{name_set}_{name_counts}.csv')
    p.to_netcdf(str(pathlib.Path(path / f'pred_{name_set}_{name_counts}').with_suffix('.nc')))


def get_pred_counts_for_day(data_counts, date, name_counts):
    random = [train_cond(date), valid_cond(date), test_cond(date)].index(True)
    suffix = {0: 'train', 1: 'val', 2: 'test'}
    predicted_counts = pd.read_csv(str(PRED_ROOT / name_counts / f'mean_pred_{suffix[random]}_{name_counts}.csv'),
                                   parse_dates=['claim_date'])
    dc_day = predicted_counts[predicted_counts.claim_date == date].set_index(['gridcell', 'claim_date'])
    dc_day = dc_day.merge(data_counts[['climadadmg', 'climadacnt']], left_index=True,
                          right_on=['gridcell', 'claim_date'], how='left')
    return dc_day


def generate_prediction_for_date(date, dc_day, mapping=mapping, nb_draws=nb_draws, smooth_spatially=False, bin=False,
                                 path_to_sizes=None):
    date_str = pd.to_datetime(date).strftime('%Y%m%d')
    dc_day = dc_day.reset_index().rename(
        columns={'gridcell': 'gridcell_poisson', 'poh': 'mean_poh_hd', 'meshs': 'mean_meshs_hd',
                 'volume': 'mean_vol_hd'}).assign(
        geometry=lambda x: geopandas.GeoSeries.from_wkt(x.geometry)).set_geometry('geometry')
    pos_counts = dc_day[dc_day.pred_cnt >= 1] if not bin else dc_day[dc_day.pred_cnt >= 0.5]

    def build_sizes_test_set_from_counts(pos_counts):
        cells_of_interest = mapping[mapping.gridcell.isin(pos_counts.gridcell_poisson)].set_geometry(
            'geometry')  # restrict on gridcell with positive counts for prediction of sizes
        counts_refactored = pos_counts.merge(
            cells_of_interest[['gridcell', 'latitude', 'longitude']].rename(columns={'gridcell': 'gridcell_poisson'}),
            on='gridcell_poisson')  # expand variable values for each gridcell to match location level
        counts_refactored.claim_date = pd.to_datetime(counts_refactored.claim_date)
        cv_with_gridcells = claim_values.drop(columns=['gridcell', 'geometry']).merge(mapping,
                                                                                      on=['longitude', 'latitude'],
                                                                                      how='left').rename(
            columns={'gridcell': 'gridcell_poisson'})
        new_set = counts_refactored.merge(cv_with_gridcells[
                                              ['gridcell_poisson', 'claim_date', 'longitude', 'latitude', 'POH',
                                               'MESHS', 'climada_dmg', 'pos_error']],
                                          on=['gridcell_poisson', 'longitude', 'latitude', 'claim_date'], how='left')
        new_set.POH = np.where(np.isnan(new_set.POH), new_set.mean_poh_hd, new_set.POH)
        new_set.MESHS = np.where(np.isnan(new_set.MESHS), new_set.mean_meshs_hd, new_set.MESHS)
        new_set.pos_error = new_set.pos_error.fillna(0.)
        new_set.climada_dmg = new_set.climada_dmg.fillna(0.)
        new_set = new_set.assign(month=lambda x: x.claim_date.dt.month.astype(int)).assign(
            year=lambda x: x.claim_date.dt.year.astype(int)).assign(
            season=lambda x: np.where((x.month >= 6) & (x.month <= 8), 0, 1)).assign(
            geom_point=lambda x: geopandas.points_from_xy(x['longitude'], x['latitude'])).set_geometry('geom_point')
        data_vars = ['geom_point', 'gridcell_poisson',
                     'MESHS', 'POH', 'pos_error', 'climada_dmg',
                     'year', 'month', 'season']
        vars = data_vars + ['longitude', 'latitude', 'claim_date', 'geometry']
        agg = associate_data_with_grid(new_set[vars], grid, vars=data_vars).assign(
            latitude_grid=lambda x: x.geometry.centroid.y).assign(longitude_grid=lambda x: x.geometry.centroid.x)
        agg = agg.merge(exposure.reset_index().rename(columns={'value': 'exposure'}), on=['latitude', 'longitude'],
                        how='left')
        # additional cells with no predictive value - need to se how to expand prediction to new gridcells
        agg = agg[agg.gridcell.isin(claim_values["gridcell"].unique())].dropna(subset='exposure')
        return agg

    # Prediction of sizes
    size_set = build_sizes_test_set_from_counts(pos_counts)
    model_sizes = build_model(size_set)
    trace_sizes = get_chosen_variables_for_model(model_sizes, nb_draws=nb_draws)
    predicted_sizes = mc.sample_posterior_predictive(trace_sizes, model_sizes)
    g = xr.merge([predicted_sizes.posterior_predictive.cond_damage, predicted_sizes.constant_data.unscaled_exposure])
    predicted_sizes.posterior_predictive = predicted_sizes.posterior_predictive.assign(
        scaled=(predicted_sizes.posterior_predictive.dims,
                np.where(g.cond_damage > g.unscaled_exposure, g.unscaled_exposure, g.cond_damage)))
    predicted_sizes.posterior_predictive = predicted_sizes.posterior_predictive.assign(
        cond_damage=predicted_sizes.posterior_predictive.scaled)
    if path_to_sizes:
        d = xr.merge([(
                              predicted_sizes.posterior_predictive.cond_damage + predicted_sizes.constant_data.climada_dmg.rename(
                          {'climada_dmg_dim_0': 'point'})).rename('pred_size'),
                      predicted_sizes.constant_data.longitude.rename({'longitude_dim_0': 'point'}),
                      predicted_sizes.constant_data.latitude.rename({'latitude_dim_0': 'point'})]).expand_dims(
            {'claim_date': [date]})
        path_to_nc = pathlib.Path(path_to_sizes / date_str).with_suffix('.nc')
        path_to_nc.parent.mkdir(exist_ok=True, parents=True)
        d.to_netcdf(str(path_to_nc))
    # Combining counts and sizes
    predicted_size = (
            predicted_sizes.posterior_predictive.cond_damage + predicted_sizes.constant_data.climada_dmg.rename(
        {'climada_dmg_dim_0': 'point'})).mean(['chain', 'draw'])
    q05 = (predicted_sizes.posterior_predictive.cond_damage + predicted_sizes.constant_data.climada_dmg.rename(
        {'climada_dmg_dim_0': 'point'})).quantile(confidence, dim=['chain', 'draw'])
    q95 = (predicted_sizes.posterior_predictive.cond_damage + predicted_sizes.constant_data.climada_dmg.rename(
        {'climada_dmg_dim_0': 'point'})).quantile(1 - confidence, dim=['chain', 'draw'])
    ps_dataframe = size_set.sort_values('claim_date').reset_index().assign(mean_pred_size=predicted_size).assign(
        lb_pred=q05).assign(ub_pred=q95)
    if not smooth_spatially:
        ps_dataframe = ps_dataframe.merge(weights, on=['latitude', 'longitude'], how='left').sort_values('exposure')
        mean = pd.concat([d[['gridcell_poisson', 'longitude', 'latitude', 'claim_date', 'mean_pred_size',
                             'weight']].rename(columns={'gridcell_poisson': 'gridcell'})
                         .set_index(['longitude', 'latitude', 'gridcell', 'claim_date'])
                         .sort_values(by='weight', ascending=False).head(
            int(round(pos_counts.set_index('gridcell_poisson')['pred_cnt'].loc[g], 0))).mean_pred_size for g, d in
                          ps_dataframe.groupby('gridcell_poisson')])
        lb = pd.concat([d[['gridcell_poisson', 'longitude', 'latitude', 'claim_date', 'lb_pred', 'weight']].rename(
            columns={'gridcell_poisson': 'gridcell'})
                       .set_index(['longitude', 'latitude', 'gridcell', 'claim_date'])
                       .sort_values(by='weight', ascending=False).head(
            int(round(pos_counts.set_index('gridcell_poisson')['pred_cnt'].loc[g], 0))).lb_pred for g, d in
                        ps_dataframe.groupby('gridcell_poisson')])
        ub = pd.concat([d[['gridcell_poisson', 'longitude', 'latitude', 'claim_date', 'ub_pred', 'weight']].rename(
            columns={'gridcell_poisson': 'gridcell'})
                       .set_index(['longitude', 'latitude', 'gridcell', 'claim_date'])
                       .sort_values(by='weight', ascending=False).head(
            int(round(pos_counts.set_index('gridcell_poisson')['pred_cnt'].loc[g], 0))).ub_pred for g, d in
                        ps_dataframe.groupby('gridcell_poisson')])
    else:
        complete_df = ps_dataframe.merge(pos_counts, on=['gridcell_poisson', 'claim_date'], how='left').rename(
            {'gridcell_poisson': 'gridcell'}).set_index(['gridcell', 'claim_date'])
        mean = (complete_df.mean_pred_size * complete_df.mean_pred).rename('mean_pred_size')
        lb = (complete_df.lb_pred * complete_df.mean_pred).rename('lb_pred')
        ub = (complete_df.ub_pred * complete_df.mean_pred).rename('ub_pred')
    pred_pos = pd.concat([mean, lb, ub], axis=1).merge(mapping[['longitude', 'latitude', 'gridcell', 'geometry']],
                                                       on=['gridcell', 'latitude', 'longitude'],
                                                       how='left').set_geometry('geometry')
    if path_to_sizes:
        pred_pos.to_csv(str(pathlib.Path(path_to_sizes / date_str).with_suffix('.csv')))
    return pred_pos
