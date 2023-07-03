import glob
import pathlib
import warnings
from itertools import combinations

import matplotlib

from extreme_values_visualization import extremogram_plot

matplotlib.rcParams["text.usetex"] = True
import cartopy.crs as ccrs
import geopandas
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, MultiPolygon
from shapely.geometry import Polygon
from shapely.ops import triangulate
import xarray as xr
import os

warnings.filterwarnings('ignore')
DATA_ROOT = pathlib.Path(os.getenv('DATA_ROOT', ''))
PLOT_ROOT = pathlib.Path(os.getenv('PLOT_ROOT', ''))
MESHS = xr.open_mfdataset(glob.glob('/Volumes/ExtremeSSD/hail_gvz/data/MZC/*.nc'), coords='minimal')
POH = xr.open_mfdataset(glob.glob('/Volumes/ExtremeSSD/hail_gvz/data/BZC/*.nc'), coords='minimal')
EXPOSURES_ROOT = pathlib.Path(DATA_ROOT / 'GVZ_Exposure_202201').with_suffix('.csv')
HAILSTORM_ROOT_UNPROCESSED = pathlib.Path(DATA_ROOT / 'GVZ_Hail_Loss_200001_to_202203').with_suffix('.csv')
HAILSTORM_ROOT_PROCESSED = pathlib.Path(DATA_ROOT / 'GVZ_Hail_Loss_date_corrected7').with_suffix('.csv')
MODELLED_ROOT = pathlib.Path(DATA_ROOT / 'GVZ_imp_modelled').with_suffix('.csv')
CRS = 'EPSG:2056'
CCRS = ccrs.epsg(2056)
TRANSLATION_EXPOSURE_COLUMNS = {'KoordinateNord': 'latitude', 'KoordinateOst': 'longitude', 'Versicherungssumme': 'value',
                                'Volumen': 'volume', 'VersicherungsID': 'id', 'Baujahr': 'construction_year',
                                'Nutzung': 'building_type', 'Nutzungscode': 'building_type_id',
                                'Adresse': 'address'}
TRANSLATION_HAILDATA_COLUMNS = {'KoordinateNord': 'latitude', 'KoordinateOst': 'longitude', 'Versicherungssumme': 'value',
                                'Volumen': 'volume', 'VersicherungsID': 'id', 'Baujahr': 'construction_year',
                                'Nutzung': 'building_type', 'Nutzungscode': 'building_type_id', 'Adresse': 'address',
                                'Schadennummer': 'claim_id', 'Schadendatum': 'claim_date', 'Schadensumme': 'claim_value'}


def process_exposure_data():
    df_exp = pd.read_csv(EXPOSURES_ROOT, sep=';').rename(columns=TRANSLATION_EXPOSURE_COLUMNS)
    gdf = geopandas.GeoDataFrame(
        df_exp, geometry=geopandas.points_from_xy(df_exp.longitude, df_exp.latitude), crs=CRS)
    gdf.to_crs(epsg=4326, inplace=True)
    gdf = gdf.assign(longitude=lambda x: x.geometry.x).assign(latitude=lambda x: x.geometry.y)
    return gdf


def process_modelled_data():
    df_mod = pd.read_csv(MODELLED_ROOT, sep=';')
    df_mod = df_mod.drop(columns=['latitude', 'longitude']).rename(columns={'KoordinateNord': 'latitude', 'KoordinateOst': 'longitude'})
    df_mod = df_mod.drop(columns=list(np.intersect1d(list(TRANSLATION_HAILDATA_COLUMNS.keys()), df_mod.columns)) + ['Unnamed: 0', 'impf_', 'centr_HL']) \
        .set_index(['latitude', 'longitude', 'value']).stack().reset_index().rename(columns={'level_3': 'claim_date', 0: 'climada_dmg'})
    df_mod.claim_date = pd.to_datetime(df_mod.claim_date)
    df_mod = df_mod.set_index(['latitude', 'longitude', 'claim_date'])
    return df_mod


def process_data_with_modelled_formatting(root):
    df_mod = pd.read_csv(root, sep=';', index_col=['KoordinateNord', 'KoordinateOst'], usecols=lambda x: '2' in x or x == 'KoordinateNord' or x == 'KoordinateOst')
    print(df_mod.shape)
    df_mod = df_mod.reset_index().rename(columns={'KoordinateNord': 'latitude', 'KoordinateOst': 'longitude'})
    df_mod = df_mod.set_index(['latitude', 'longitude'])
    df_mod = df_mod.rename(columns={d: pd.to_datetime(d) for d in df_mod.columns}).sort_index(axis=1)
    df_mod.dtype = pd.SparseDtype(float, 0)
    return df_mod


def pre_process_haildamage_data(path: pathlib.Path):
    d = pd.read_csv(path, sep=';').rename(columns=TRANSLATION_HAILDATA_COLUMNS)
    d['claim_date'] = d['claim_date'].astype(str)
    d = d.assign(claim_date=lambda x: np.where(x['claim_date'].str.len() < 8, '0' + x['claim_date'], x['claim_date']))
    d['claim_date'] = pd.to_datetime(d['claim_date'], format='%d%m%Y')
    dic_agg = {'claim_value': 'sum'}
    if 'POH' in d.columns:
        dic_agg['POH'] = 'max'
        dic_agg['MESHS'] = 'max'
    d = d.groupby(['latitude', 'longitude', 'claim_date']).agg(dic_agg).reset_index()
    return d


def process_haildamage_data(concat_unprocessed=False):
    dfp = pre_process_haildamage_data(HAILSTORM_ROOT_PROCESSED)
    if concat_unprocessed:
        df = pre_process_haildamage_data(HAILSTORM_ROOT_UNPROCESSED)
        missing_lons = set(df.longitude) - set(np.intersect1d(dfp.longitude, df.longitude))
        missing_lats = set(df.latitude) - set(np.intersect1d(dfp.latitude, df.latitude))
        missing_dates = set(df.claim_date) - set(pd.to_datetime(np.intersect1d(dfp.claim_date, df.claim_date)))
        missing_dates_out_hail_season = [d for d in missing_dates if (d.month <= 3 or d.month >= 10)]  # aggregation per dates during hail season is preserved
        unprocessed_deleted = df[(df.claim_date.isin(missing_dates_out_hail_season)) | (df.longitude.isin(missing_lons)) | (df.latitude.isin(missing_lats))]
        df_hail = pd.concat([dfp, unprocessed_deleted]).fillna(
            0.)  # the deleted data added here is the one automatically deleted during processing because out of hail season or POH too low in a 5km radius
    else:
        df_hail = dfp
    df_modelled = process_modelled_data().groupby(['latitude', 'longitude', 'claim_date']).agg({'climada_dmg': 'sum'})
    df_hail = df_hail.merge(df_modelled, on=['latitude', 'longitude', 'claim_date'], how='left')
    df_hail.climada_dmg = df_hail.climada_dmg.fillna(0.)
    gdf = geopandas.GeoDataFrame(
        df_hail, geometry=geopandas.points_from_xy(df_hail.longitude, df_hail.latitude), crs=CRS)
    gdf.to_crs(epsg=4326, inplace=True)
    gdf = gdf.assign(longitude=lambda x: x.geometry.x).assign(latitude=lambda x: x.geometry.y)
    return gdf


def distance_from_lonlat(lon1, lat1, lon2, lat2):
    r = 6371  # radius of Earth (KM)
    p = np.pi / 180
    a = 0.5 - np.cos((lat2 - lat1) * p) / 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    d = 2 * r * np.arcsin(np.sqrt(a))  # 2*R*asin
    return d


def triangulate_haildamage_data():
    df_hail_event, df_hail = process_haildamage_data()
    geo = df_hail.reset_index()
    gdf = geopandas.GeoDataFrame(
        geo, geometry=geopandas.points_from_xy(geo.longitude, geo.latitude), crs=CRS)
    gdf.to_crs(epsg=4326, inplace=True)
    gdf = gdf.assign(longitude=lambda x: x.geometry.x).assign(latitude=lambda x: x.geometry.y)
    x = gdf.geometry.x
    y = gdf.geometry.y
    stack = np.vstack((x, y)).T
    mp = MultiPoint(points=stack)
    tri = triangulate(mp)
    mesh = geopandas.GeoDataFrame(geometry=tri)
    aggregated = gdf.sjoin(mesh, predicate='within')
    return aggregated


def get_polygon_df_from_kde_plot(kde):
    level_polygons = []
    i = 0
    for col in kde.collections:
        paths = []
        # Loop through all polygons that have the same intensity level
        for contour in col.get_paths():
            # Create a polygon for the countour
            # First polygon is the main countour, the rest are holes
            for ncp, cp in enumerate(contour.to_polygons()):
                x = cp[:, 0]
                y = cp[:, 1]
                new_shape = Polygon([(i[0], i[1]) for i in zip(x, y)])
                if ncp == 0:
                    poly = new_shape
                else:
                    # Remove holes, if any
                    poly = poly.difference(new_shape)

            # Append polygon to list
            paths.append(poly)
        # Create a MultiPolygon for the contour
        multi = MultiPolygon(paths)
        # Append MultiPolygon and level as tuple to list
        level_polygons.append(multi)
        i += 1
    polygons = geopandas.GeoDataFrame({'geometry': level_polygons[:-1]}, crs='epsg:4326')
    disjoint_pols = []
    for i in range(len(polygons) - 1):
        dis = geopandas.GeoSeries(polygons.iloc[i], crs='epsg:4326') \
            .difference(geopandas.GeoSeries(polygons.iloc[i + 1], crs='epsg:4326'))
        disjoint_pols.append(dis.values[0])
    disjoint_pols.append(geopandas.GeoSeries(polygons.iloc[-1], crs='epsg:4326').values[0])
    to_return = geopandas.GeoDataFrame({'geometry': disjoint_pols})
    to_return = to_return.set_crs(epsg=4326)
    return to_return


def grid_from_geopandas_pointcloud(geopandas_pointcloud, delta=0.1):
    xmin, ymin, xmax, ymax = geopandas_pointcloud.total_bounds
    width = delta
    height = delta
    rows = int(np.ceil((ymax - ymin) / height))
    cols = int(np.ceil((xmax - xmin) / width))
    XleftOrigin = xmin
    XrightOrigin = xmin + width
    YtopOrigin = ymax
    YbottomOrigin = ymax - height
    polygons = []
    for i in range(cols):
        Ytop = YtopOrigin
        Ybottom = YbottomOrigin
        for j in range(rows):
            polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)]))
            Ytop = Ytop - height
            Ybottom = Ybottom - height
        XleftOrigin = XleftOrigin + width
        XrightOrigin = XrightOrigin + width
    grid = geopandas.GeoDataFrame({'geometry': polygons})
    grid = grid.set_crs(epsg=4326)
    return grid


def associate_data_with_grid(geopandas_df, geopandas_polygonset, vars):
    join = geopandas_df.sjoin(geopandas_polygonset, how='inner')
    join['associated_polygon'] = join['index_right'].apply(lambda x: geopandas_polygonset.loc[x, 'geometry'])
    join['associated_polygon'] = geopandas.GeoSeries(join.associated_polygon, crs='epsg:4326').convex_hull
    col_list = ['associated_polygon', 'index_right', 'claim_date', 'longitude', 'latitude', 'time'] + vars
    col_list = np.intersect1d(col_list, join.columns)
    df = join[col_list].rename(columns={'index_right': 'gridcell'})
    df = df.rename(columns={'associated_polygon': 'geometry'}).set_geometry('geometry')
    return df


def associate_claim_data_with_grid(geopandas_df_claims, geopandas_df_polygonset):
    join = geopandas_df_claims.sjoin(geopandas_df_polygonset, how='inner')
    join['associated_polygon'] = join['index_right'].apply(lambda x: geopandas_df_polygonset.loc[x, 'geometry'])
    join['associated_polygon'] = geopandas.GeoSeries(join.associated_polygon, crs='epsg:4326').convex_hull
    col_list = ['associated_polygon', 'index_right', 'claim_value', 'climada_dmg', 'MESHS', 'POH', 'claim_date', 'longitude', 'latitude']
    col_list = np.intersect1d(col_list, join.columns)
    df = join[col_list].rename(columns={'index_right': 'gridcell'})
    df = df.rename(columns={'associated_polygon': 'geometry'}).set_geometry('geometry')
    return df


def aggregate_claim_data(geopandas_df_claims, geopandas_df_polygonset):
    agg_df = associate_claim_data_with_grid(geopandas_df_claims, geopandas_df_polygonset)
    agg_df = agg_df.dissolve(by=['claim_date', 'gridcell'], aggfunc='sum')
    agg_df = agg_df.reset_index() \
        .assign(latitude=lambda x: x.geometry.centroid.y).assign(longitude=lambda x: x.geometry.centroid.x)
    return agg_df.set_index(['latitude', 'longitude', 'claim_date'])


def associate_exposure_data_with_grid(geopandas_df_exp, geopandas_df_polygonset):
    join = geopandas_df_exp.sjoin(geopandas_df_polygonset, how='inner')
    join['associated_polygon'] = join['index_right'].apply(lambda x: geopandas_df_polygonset.loc[x, 'geometry'])
    join['associated_polygon'] = geopandas.GeoSeries(join.associated_polygon, crs='epsg:4326').convex_hull
    df = join[['associated_polygon', 'index_right', 'value', 'longitude', 'latitude']].rename(columns={'index_right': 'gridcell'})
    df = df.rename(columns={'associated_polygon': 'geometry'}).set_geometry('geometry')
    return df


def aggregate_exposure_data(geopandas_df_exp, geopandas_df_polygonset):
    agg_df = associate_exposure_data_with_grid(geopandas_df_exp, geopandas_df_polygonset)
    agg_df = agg_df.dissolve(by=['gridcell'], aggfunc='sum')
    agg_df = agg_df.reset_index() \
        .assign(latitude=lambda x: x.geometry.centroid.y).assign(longitude=lambda x: x.geometry.centroid.x)
    return agg_df


def compute_spearman_correlation_over_grid(agg_df):
    cdf = agg_df.reset_index().drop(columns=['index_right'], errors='ignore').assign(latitude=lambda x: x['geometry'].centroid.y).assign(longitude=lambda x: x['geometry'].centroid.x)
    corr = cdf.set_index(['claim_date', 'latitude', 'longitude']).claim_value.unstack(['latitude', 'longitude']) \
        .corr(method='spearman', min_periods=15)
    return corr


def compute_extremal_correlation_over_grid(agg_df, dist=None):
    to_concat = []
    i = 0
    for (lat1, lon1), (lat2, lon2) in combinations(agg_df.groupby(['latitude', 'longitude']).count().index, 2):
        s1 = agg_df.loc[(lat1, lon1, slice(None))].claim_value.rename('s1')
        s2 = agg_df.loc[(lat2, lon2, slice(None))].claim_value.rename('s2')
        m = s1.to_frame().merge(s2.to_frame(), left_index=True, right_index=True)
        if len(m) >= 15:
            i += 1
            thresh1 = np.nanquantile(s1, 0.675)
            thresh2 = np.nanquantile(s2, 0.675)
            distance = distance_from_lonlat(lon1, lat1, lon2, lat2)
            spec_corr = len(m[(m['s1'] >= thresh1) & (m['s2'] >= thresh2)])
            to_concat.append(pd.DataFrame([[distance, spec_corr, len(m)]], columns=['distance', 'above', 'all']))
    df_dist = pd.concat(to_concat).assign(corr=lambda x: x['above'] / x['all'])
    if not dist:
        dist = np.linspace(0, 25, 8)
        dist = [int(i) for i in np.round(dist, 0)]
    df_dist['dist'] = df_dist.distance.apply(lambda x: dist[np.argmin(np.array(np.abs([j - x for j in dist])))])
    agg = df_dist[['corr', 'dist']].groupby('dist').mean()
    return agg


def link_distance_to_spearman_corr(corr):
    df_dist = []
    for (lat1, lon1), (lat2, lon2) in combinations(corr.columns, 2):
        spec_corr = corr.loc[(lat1, lon1), (lat2, lon2)]
        distance = distance_from_lonlat(lon1, lat1, lon2, lat2)
        df_dist.append(pd.DataFrame([[distance, spec_corr]], columns=['distance', 'corr']))
    df_dist = pd.concat(df_dist).sort_values('distance')
    return df_dist


def get_spearman_corr(gridded, dim='time', variable='count'):
    gridded = gridded.rename(columns={'claim_value': 'value'}).reset_index()
    gridded['count'] = 1
    agg = gridded.assign(month=lambda x: pd.to_datetime(x['claim_date']).dt.month) \
        .assign(season=lambda x: np.where((x['month'] >= 4) & (x['month'] <= 9), 'summer',
                                          np.where((x['month'] <= 4) & (x['month'] >= 1), 'spring', 'winter')))

    seasonal_df = agg[agg['season'] == 'summer']
    mapping_gridcell_lonlat = agg.groupby('gridcell').agg({'longitude': 'mean', 'latitude': 'mean'})
    stack_col = 'claim_date' if dim == 'time' else 'gridcell'
    dist_function = lambda x1, x2: (x2 - x1).days if dim == 'time' else distance_from_lonlat(*x1, *x2)
    us = seasonal_df.groupby(['claim_date', 'gridcell'])[variable].sum().unstack(stack_col)
    us = us.fillna(0.)  # if variable == 'count' else us
    corr = us.corr(method='spearman', min_periods=15)
    h = np.linspace(1, 100, 20)
    h = [int(i) for i in np.round(h, 0)]
    df = []
    for t1, t2 in combinations(us.columns, 2):
        if dim != 'time':
            x1, x2 = mapping_gridcell_lonlat.loc[t1].values, mapping_gridcell_lonlat.loc[t2].values
        else:
            x1, x2 = t1, t2
        if dist_function(x1, x2) <= np.max(h) + 4:
            spec_corr = corr.loc[t1, t2]
            distance = dist_function(x1, x2)
            df.append(pd.DataFrame([[distance, spec_corr]], columns=['lag', 'corr']))
    df = pd.concat(df).sort_values('lag')
    df['h'] = df.lag.apply(lambda x: h[np.argmin(np.array(np.abs([j - x for j in h])))])
    corragg = df.groupby('h').mean()
    q05 = df.groupby('h').quantile(0.1)['corr'].rename('q05')
    q95 = df.groupby('h').quantile(0.9)['corr'].rename('q95')
    return pd.concat([q05, corragg, q95], axis=1)


def get_extremal_corr(gridded, dim='time', variable='count'):
    def distance_gridcells(g1, g2):
        lon1, lat1 = gridded[gridded.gridcell == g1].reset_index()['longitude'].unique()[0], gridded[gridded.gridcell == g1].reset_index()['latitude'].unique()[0]
        lon2, lat2 = gridded[gridded.gridcell == g2].reset_index()['longitude'].unique()[0], gridded[gridded.gridcell == g2].reset_index()['latitude'].unique()[0]
        return distance_from_lonlat(lon1, lat1, lon2, lat2)

    aggfunc = 'sum' if variable == 'value' else 'count'
    if dim == 'time':
        extr = gridded.groupby('claim_date').agg(aggfunc).claim_value.to_frame(name='data').assign(days_since_start=lambda x: (x.index - x.index.min()).days)
        extr = extr.assign(threshold=np.exp(8.06) - 1) if variable == 'value' else extr.assign(threshold=15)
        to_return = extremogram_plot(extr, h_range=np.linspace(1, 100, 30)).to_frame(name='corr').reset_index().rename(columns={'index': 'h'}).set_index('h')
    else:
        tc = []
        for _ in range(100):
            random_gridcell = np.random.choice(gridded['gridcell'].unique(), 1)[0]
            extr = gridded.groupby('gridcell').agg(aggfunc).claim_value.to_frame(name='data').reset_index()
            extr['days_since_start'] = extr['gridcell'].apply(lambda x: distance_gridcells(x, random_gridcell))
            extr = extr.assign(threshold=np.exp(8.06) - 1) if variable == 'value' else extr.assign(threshold=15)
            extremal = extremogram_plot(extr, h_range=np.linspace(1, 100, 20)).to_frame(name='corr').reset_index().rename(columns={'index': 'h'}).set_index('h')
            tc.append(extremal['corr'].rename(_))
        tc = pd.concat(tc, axis=1)
        to_return = pd.concat([tc.mean(axis=1).rename('corr'), tc.quantile(0.1, axis=1).rename('q05'), tc.quantile(0.9, axis=1).rename('q95')], axis=1)
    return to_return
