import geopandas
import pandas as pd

from constants import DATA_ROOT, suffix, CRS
from data.hailcount_data_processing import get_grid_mapping

mapping = get_grid_mapping(suffix=suffix)


def process_climada_perbuilding_positive_damages():
    # Climada damages downscaled per building
    climada_damages = pd.read_csv(DATA_ROOT / suffix / 'climada_dmg.csv', index_col=['latitude', 'longitude'],
                                  usecols=lambda x: ('2' in x) or x in ['latitude', 'longitude'])
    climada_damages_date = [climada_damages[c].rename('climadadmg').to_frame().assign(claim_date=pd.to_datetime(c)) for c in climada_damages.columns]
    climada_pred_pos = pd.concat([d[d.climadadmg > 0] for d in climada_damages_date])
    climada_damages = climada_pred_pos.reset_index().assign(geom=lambda x: geopandas.points_from_xy(x.longitude, x.latitude, crs=CRS).to_crs('epsg:4326')) \
        .set_geometry('geom').assign(longitude=lambda x: x.geom.x).assign(latitude=lambda x: x.geom.y).merge(mapping[['longitude', 'latitude', 'gridcell', 'geometry']],
                                                                                                             on=['latitude', 'longitude'], how='left').set_geometry('geometry')
    climada_damages = climada_damages.set_index(['longitude', 'latitude', 'claim_date'])
    return climada_damages


def process_climada_counts():
    # Climada counts per building using PAA
    climada_counts = pd.read_csv(DATA_ROOT / suffix / 'climada_cnt.csv', index_col=['latitude', 'longitude'],
                                 usecols=lambda x: ('2' in x) or x in ['latitude', 'longitude'])
    climada_counts_date = [climada_counts[c].rename('climadadmg').to_frame().assign(claim_date=pd.to_datetime(c)) for c in climada_counts.columns]
    climada_count_pos = pd.concat([d[d.climadadmg > 0] for d in climada_counts_date])
    climada_counts = climada_count_pos.reset_index().assign(geom=lambda x: geopandas.points_from_xy(x.longitude, x.latitude, crs=CRS).to_crs('epsg:4326')) \
        .set_geometry('geom').assign(longitude=lambda x: x.geom.x).assign(latitude=lambda x: x.geom.y).merge(mapping[['longitude', 'latitude', 'gridcell', 'geometry']],
                                                                                                             on=['latitude', 'longitude'], how='left').set_geometry('geometry')
    climada_counts = climada_counts.set_index(['longitude', 'latitude', 'claim_date'])
    return climada_counts
