import pathlib

import geopandas
import numpy as np
import pandas as pd
import xarray as xr

from exploratory_da.utils import process_data_with_modelled_formatting, grid_from_geopandas_pointcloud, process_exposure_data

DATA_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/data/GVZ_Datenlieferung_an_ETH/Ophelia')

FITS_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/fits/')
scaling_factor = 1
tol = 1e-5
CRS = 'EPSG:2056'
delta = 0.015
timestep = '1d'


def sparsify_data(suffix=None):
    root_to_counts = DATA_ROOT / suffix if suffix else DATA_ROOT
    suffix_str = f'_{suffix}' if suffix else ''
    for src, name in (
            # (f'df_poh{suffix_str}.csv', 'poh'),
            # (f'df_meshs{suffix_str}.csv', 'meshs'),
            (f'df_imp_modelled{suffix_str}_PAA.csv', 'climada_cnt'),
            (f'df_imp_modelled{suffix_str}.csv', 'climada_dmg'),
            # (f'df_imp_observed{suffix_str}.csv', 'obs_cnt')
    ):
        print(f"Reading {name}")
        r = process_data_with_modelled_formatting(root_to_counts / src)
        r.to_csv(root_to_counts / f'{name}.csv')


def read_and_process_counts(dl=False, suffix=None):
    root_to_counts = DATA_ROOT / suffix if suffix else DATA_ROOT
    root_to_winds = '/Volumes/ExtremeSSD/wind_downscaling_gan/data/ERA5/'
    train_cond = lambda x: (int(pd.to_datetime(x).strftime('%Y')) >= 2008)
    test_cond = lambda x: (int(pd.to_datetime(x).strftime('%Y')) < 2005)
    valid_cond = lambda x: (int(pd.to_datetime(x).strftime('%Y')) < 2008) & (int(pd.to_datetime(x).strftime('%Y')) >= 2005)
    print("Creating train, test and validation sets")
    for cond, name in zip([train_cond, test_cond, valid_cond], ['train', 'test', 'validation']):
        print(f"Creating {name} set")
        poh = pd.read_csv(root_to_counts / f'poh.csv', index_col=['latitude', 'longitude'], usecols=lambda x: ('2' in x and cond(x)) or x in ['latitude', 'longitude'])
        poh = poh.groupby(level=['latitude', 'longitude']).max()
        meshs = pd.read_csv(root_to_counts / f'meshs.csv', index_col=['latitude', 'longitude'], usecols=lambda x: ('2' in x and cond(x)) or x in ['latitude', 'longitude'])
        meshs = meshs.groupby(level=['latitude', 'longitude']).max()
        obscnt = pd.read_csv(root_to_counts / f'obs_cnt.csv', index_col=['latitude', 'longitude'], usecols=lambda x: ('2' in x and cond(x)) or x in ['latitude', 'longitude'])
        obscnt = (obscnt > 0).astype(int)
        obscnt = obscnt.groupby(level=['latitude', 'longitude']).sum()
        climadacnt = pd.read_csv(root_to_counts / f'climada_cnt.csv', index_col=['latitude', 'longitude'], usecols=lambda x: ('2' in x and cond(x)) or x in ['latitude', 'longitude'])
        climada_dmg = pd.read_csv(root_to_counts / f'climada_dmg.csv', index_col=['latitude', 'longitude'], usecols=lambda x: ('2' in x and cond(x)) or x in ['latitude', 'longitude'])
        climada_dmg = climada_dmg.groupby(level=['latitude', 'longitude']).sum()
        # climadacnt = (climadacnt > 0).astype(int)
        climadacnt = climadacnt.groupby(level=['latitude', 'longitude']).sum()
        valid_cols = poh.columns
        unique_locs = poh.reset_index()[['latitude', 'longitude']].drop_duplicates(['latitude', 'longitude'])
        unique_locs = geopandas.GeoDataFrame(unique_locs, geometry=geopandas.points_from_xy(unique_locs.longitude, unique_locs.latitude), crs=CRS).to_crs('epsg:4326') \
            .assign(longitude=lambda g: g.geometry.x, latitude=lambda g: g.geometry.y)
        print("Creating grid")
        grid = grid_from_geopandas_pointcloud(unique_locs, delta=delta)
        dates = [pd.to_datetime(d) for d in valid_cols]
        dates_period = pd.date_range(np.min(dates), np.max(dates), freq=timestep)
        dates_continuous = pd.date_range(np.min(dates), np.max(dates), freq='1d')
        dates_mapping = np.array(range(len(dates_period)))
        time_mapping_df = pd.DataFrame(dates_mapping, index=pd.Series(dates_period, name='claim_date'), columns=['time'])
        continuous_time = pd.DataFrame(dates_continuous, index=pd.Series(dates_continuous, name='claim_date'))
        continuous_time = continuous_time.merge(time_mapping_df, left_index=True, right_index=True, how='left').drop(columns=[0]).ffill()
        suffix = 'DL' if dl else suffix
        target = DATA_ROOT / suffix if suffix else DATA_ROOT

        def get_agg_var(time_df, grid_df, var_df, var_name, aggfunc):
            vars = var_df.columns
            var_df_geo = geopandas.GeoDataFrame(var_df.reset_index(), geometry=geopandas.points_from_xy(var_df.reset_index().longitude, var_df.reset_index().latitude), crs=CRS).to_crs('epsg:4326') \
                .rename(columns={'geometry': 'geom_point'}).set_geometry('geom_point')
            vars_gridcell = var_df_geo.sjoin(grid_df).rename(columns={'index_right': 'gridcell'}) \
                .merge(grid_df.reset_index().rename(columns={'index': 'gridcell'}), on='gridcell', how='left').set_geometry('geometry') \
                .assign(lon_grid=lambda x: x.geometry.centroid.x).assign(lat_grid=lambda x: x.geometry.centroid.y)
            if var_name == 'obscnt' and name == 'train':
                # to save grid once only
                grid_array = vars_gridcell.drop(columns=vars).set_geometry('geom_point').assign(longitude=lambda x: x.geometry.x).assign(latitude=lambda x: x.geometry.y)
                grid_array.to_csv(target / 'location_grid_mapping.csv')
            var_gridcell = vars_gridcell.dissolve('gridcell', aggfunc).drop(columns=['geometry', 'latitude', 'longitude', 'lat_grid', 'lon_grid']).reset_index().set_index(['gridcell']).stack().rename(
                var_name).reset_index().rename(
                columns={'level_1': 'claim_date'})
            var_gridcell.claim_date = pd.to_datetime(var_gridcell.claim_date)
            var_gridcell = var_gridcell.set_index(['gridcell', 'claim_date'])
            var_gridcell_agg = var_gridcell.merge(time_df, left_on='claim_date', right_index=True).groupby(['gridcell', 'time', 'claim_date']).agg(aggfunc)
            return var_gridcell_agg

        obscnt_gridcell_agg = get_agg_var(continuous_time, grid, obscnt, 'obscnt', 'sum')
        climadacnt_gridcell_agg = get_agg_var(continuous_time, grid, climadacnt, 'climadacnt', 'sum')
        climadadmg_gridcell_agg = get_agg_var(continuous_time, grid, climada_dmg, 'climadadmg', 'sum')
        poh_gridcell_agg = get_agg_var(continuous_time, grid, poh, 'poh', 'max')
        meshs_gridcell_agg = get_agg_var(continuous_time, grid, meshs, 'meshs', 'max')
        grid_comp = grid.assign(latitude=lambda x: x.geometry.centroid.y).assign(longitude=lambda x: x.geometry.centroid.x)
        lats_grid, lons_grid = grid_comp['latitude'].unique(), grid_comp['longitude'].unique()
        winds = xr.open_mfdataset([root_to_winds + f'{d}_era5_surface_daily.nc' for d in [d.strftime('%Y%m%d') for d in dates_period]]).sel(latitude=lats_grid, longitude=lons_grid, method='nearest') \
            .assign_coords({'latitude': lats_grid, 'longitude': lons_grid, 'time': dates_period}).rename({'time': 'claim_date'})
        daily_winds = winds.assign(wind_speed=np.sqrt(winds.u10 ** 2 + winds.v10 ** 2)).assign(wind_dir=180 * (np.arctan2(winds.u10, winds.v10) / np.pi + 1))
        wind_speed = daily_winds['wind_speed'].to_dataframe().merge(time_mapping_df, left_on='claim_date', right_index=True, how='left') \
            .merge(grid_comp.reset_index().rename(columns={'index': 'gridcell'}).set_index(['latitude', 'longitude'])[['gridcell']], left_on=['latitude', 'longitude'], right_index=True) \
            .reset_index().drop(columns=['latitude', 'longitude']).set_index(['gridcell', 'time', 'claim_date'])
        wind_dir = daily_winds['wind_dir'].to_dataframe().merge(time_mapping_df, left_on='claim_date', right_index=True, how='left') \
            .merge(grid_comp.reset_index().rename(columns={'index': 'gridcell'}).set_index(['latitude', 'longitude'])[['gridcell']], left_on=['latitude', 'longitude'], right_index=True) \
            .reset_index().drop(columns=['latitude', 'longitude']).set_index(['gridcell', 'time', 'claim_date'])
        season = [np.where((x.month < 6), 0, np.where(x.month >= 9, 2, 1)) for x in dates]
        season_df = pd.DataFrame([season] * len(poh), index=poh.index, columns=dates)
        seasons = get_agg_var(continuous_time, grid, season_df, 'season', 'max')
        dataset = pd.concat([poh_gridcell_agg, meshs_gridcell_agg, obscnt_gridcell_agg, climadacnt_gridcell_agg, climadadmg_gridcell_agg, seasons], axis=1).fillna(0.)
        with_winds = pd.concat([dataset, wind_dir.loc[dataset.index], wind_speed.loc[dataset.index]], axis=1)
        with_winds.to_csv(target / f'{name}.csv')


def get_train_data(dl=False, suffix=None):
    suffix = 'DL' if dl else suffix
    target = DATA_ROOT / suffix if suffix else DATA_ROOT
    trainset = pd.read_csv(str(target / 'train.csv'), index_col=[0, 1, 2], parse_dates=['claim_date'])
    return trainset


def get_test_data(dl=False, suffix=None):
    suffix = 'DL' if dl else suffix
    target = DATA_ROOT / suffix if suffix else DATA_ROOT
    trainset = pd.read_csv(str(target / 'test.csv'), index_col=[0, 1, 2], parse_dates=['claim_date'])
    return trainset


def get_validation_data(dl=False, suffix=None):
    suffix = 'DL' if dl else suffix
    target = DATA_ROOT / suffix if suffix else DATA_ROOT
    trainset = pd.read_csv(str(target / 'validation.csv'), index_col=[0, 1, 2], parse_dates=['claim_date'])
    return trainset


def get_grid_mapping(dl=False, suffix=None):
    suffix = 'DL' if dl else suffix
    target = DATA_ROOT / suffix if suffix else DATA_ROOT
    m = pd.read_csv(str(target / 'location_grid_mapping.csv'))
    mapping = m.assign(geometry=lambda x: geopandas.GeoSeries.from_wkt(x.geometry)).set_geometry('geometry').rename(columns={'index': 'location'}).sort_values('gridcell')
    return mapping


def get_exposure():
    exp = process_exposure_data()
    exp = exp.groupby(['latitude', 'longitude']).agg({'value': 'sum', 'volume': 'sum'})
    return exp
