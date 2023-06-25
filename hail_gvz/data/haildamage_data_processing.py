import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from exploratory_da.utils import grid_from_geopandas_pointcloud, associate_claim_data_with_grid, process_haildamage_data, \
    process_exposure_data, associate_exposure_data_with_grid

plt.rcParams["figure.constrained_layout.use"] = True
DATA_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/data/GVZ_Datenlieferung_an_ETH/')
FITS_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/fits/')
scaling_factor = 1
tol = 1e-5


# HAIL DAMAGES
def prepare_data():
    df_hail = process_haildamage_data()
    grid = grid_from_geopandas_pointcloud(df_hail)
    agg = associate_claim_data_with_grid(df_hail, grid)
    exp = process_exposure_data()
    agg_exp = associate_exposure_data_with_grid(exp, grid).drop(columns=['gridcell', 'geometry']).groupby(['longitude', 'latitude'], as_index=False).agg({'value': 'sum'})
    all = agg.merge(agg_exp, how='inner', on=['longitude', 'latitude']).assign(latitude_grid=lambda x: x.geometry.centroid.y).assign(longitude_grid=lambda x: x.geometry.centroid.x)
    unique_locs = all.groupby(['longitude', 'latitude']).claim_value.mean().index
    location_mapping = [n for n, l in enumerate(unique_locs)]
    location_mapping_df = pd.DataFrame(location_mapping, columns=['location'], index=unique_locs)
    all = all.set_index(['gridcell', 'longitude', 'latitude', 'claim_date']).merge(location_mapping_df, left_on=['longitude', 'latitude'], right_index=True)
    all = all.reset_index().assign(year=lambda x: pd.to_datetime(x['claim_date']).dt.year).assign(month=lambda x: pd.to_datetime(x['claim_date']).dt.month) \
        .assign(season=lambda x: np.where((x['month'] >= 10) | (x['month'] <= 3), 0, x['month'] - 3)).set_index(['gridcell', 'longitude', 'latitude', 'claim_date']).rename(
        columns={'value': 'exposure'})
    all['error'] = all.claim_value - all.climada_dmg
    all = all.assign(pos_error=lambda x: np.where(x['error'] < 0, 0, x['error']))
    mean_covariates_gridcell = all.groupby(['gridcell', 'claim_date']).agg({'MESHS': 'mean', 'POH': 'mean'}).rename(columns={'MESHS': 'mean_meshs', 'POH': 'mean_poh'})
    all = all.merge(mean_covariates_gridcell, left_on=['gridcell', 'claim_date'], right_index=True, how='left')
    return all


def get_train_test_validation_split(data=None):
    if data is None:
        data_to_split = prepare_data()
    else:
        data_to_split = data.copy()
    train, not_for_training = sklearn.model_selection.train_test_split(data_to_split, train_size=2 / 3, stratify=data.gridcell)
    test, validation = sklearn.model_selection.train_test_split(not_for_training, train_size=1 / 2)
    return train, test, validation


def get_train_test_validation_split_monotonic_years(data=None):
    if data is None:
        data_to_split = prepare_data()
    else:
        data_to_split = data.copy()
    train = data_to_split[data_to_split.year < 2016]
    test = data_to_split[data_to_split.year >= 2019]
    validation = data_to_split[(data_to_split.year >= 2016) & (data_to_split.year < 2019)]
    return train, test, validation


def save_train_test_validation_data(data=None, method='random', freq='1d'):
    trainset, testset, validationset = get_train_test_validation_split(data) if method != 'monotonic' else get_train_test_validation_split_monotonic_years(data)
    for set, name in zip([trainset, testset, validationset], ['train', 'test', 'validation']):
        if 'time' in set.columns:
            set = set.drop(columns='time')
        set.claim_date = pd.to_datetime(set.claim_date)
        dates_continuous = pd.date_range(set.reset_index().claim_date.min(), set.reset_index().claim_date.max(), freq=freq)
        dates_mapping = [n for n, l in enumerate(dates_continuous)]
        time_mapping_df = pd.DataFrame(dates_mapping, columns=['time'], index=dates_continuous)
        set = set.merge(time_mapping_df, left_on=['claim_date'], right_index=True)
        set = set.assign(season=lambda x: np.where((x.month >= 6) & (x.month<=8), 0, 1))
        set.to_csv(str(DATA_ROOT / f'{name}.csv'))


def get_train_data():
    trainset = pd.read_csv(str(DATA_ROOT / 'train.csv'), index_col=[0, 1, 2, 3], parse_dates=[3])
    return trainset


def get_test_data():
    trainset = pd.read_csv(str(DATA_ROOT / 'test.csv'), index_col=[0, 1, 2, 3], parse_dates=[3])
    return trainset


def get_validation_data():
    trainset = pd.read_csv(str(DATA_ROOT / 'validation.csv'), index_col=[0, 1, 2, 3], parse_dates=[3])
    return trainset
