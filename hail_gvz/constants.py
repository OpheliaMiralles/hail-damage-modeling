import os
import pathlib
from glob import glob

import cartopy.crs as ccrs
import geopandas
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# data
suffix = 'GVZ_emanuel'
DATA_ROOT = pathlib.Path(os.getenv('DATA_ROOT', ''))
PRED_ROOT = pathlib.Path(os.getenv('PRED_ROOT', ''))
FITS_ROOT = pathlib.Path(os.getenv('FITS_ROOT', ''))
PLOT_ROOT = pathlib.Path(os.getenv('PLOT_ROOT', ''))
EXPOSURES_ROOT = pathlib.Path(DATA_ROOT / 'GVZ_Exposure_202201').with_suffix('.csv')
HAILSTORM_ROOT_UNPROCESSED = pathlib.Path(DATA_ROOT / 'GVZ_Hail_Loss_200001_to_202203').with_suffix('.csv')
HAILSTORM_ROOT_PROCESSED = pathlib.Path(DATA_ROOT / 'GVZ_Hail_Loss_date_corrected7').with_suffix('.csv')
MODELLED_ROOT = pathlib.Path(DATA_ROOT / 'GVZ_imp_modelled').with_suffix('.csv')
TRANSLATION_EXPOSURE_COLUMNS = {'KoordinateNord': 'latitude', 'KoordinateOst': 'longitude', 'Versicherungssumme': 'value',
                                'Volumen': 'volume', 'VersicherungsID': 'id', 'Baujahr': 'construction_year',
                                'Nutzung': 'building_type', 'Nutzungscode': 'building_type_id',
                                'Adresse': 'address'}
TRANSLATION_HAILDATA_COLUMNS = {'KoordinateNord': 'latitude', 'KoordinateOst': 'longitude', 'Versicherungssumme': 'value',
                                'Volumen': 'volume', 'VersicherungsID': 'id', 'Baujahr': 'construction_year',
                                'Nutzung': 'building_type', 'Nutzungscode': 'building_type_id', 'Adresse': 'address',
                                'Schadennummer': 'claim_id', 'Schadendatum': 'claim_date', 'Schadensumme': 'claim_value'}

# constant datasets
claim_values = pd.read_csv(str(DATA_ROOT / 'processed.csv'), index_col=[0, 1, 2, 3], parse_dates=[3]).drop(columns=['time']).reset_index()
claim_values['geometry'] = geopandas.GeoSeries.from_wkt(claim_values['geometry'])
claim_values = claim_values.set_geometry('geometry')
geom_roots = glob(str(DATA_ROOT / 'SHAPEFILE/swissTLMRegio_Boundaries_LV95/swissTLMRegio_KANTONSGEBIET_LV95.shp'))
df_polygon = pd.concat([geopandas.read_file(geom_root) for geom_root in geom_roots]).to_crs(epsg='4326')
df_polygon = df_polygon[df_polygon.NAME == 'Zürich']
lakes = glob(str(DATA_ROOT / 'SHAPEFILE/swissTLMRegio_Product_LV95/Hydrography/swissTLMRegio_Lake.shp'))
df_lakes = geopandas.read_file(lakes[0]).to_crs(epsg='4326')

# splitting of observed data into 3 different sets
train_cond = lambda x: (int(pd.to_datetime(x).strftime('%Y')) >= 2008)
test_cond = lambda x: (int(pd.to_datetime(x).strftime('%Y')) < 2005)
valid_cond = lambda x: (int(pd.to_datetime(x).strftime('%Y')) < 2008) & (int(pd.to_datetime(x).strftime('%Y')) >= 2005)

# GPD model
threshold = 8.06
exp_threshold = np.exp(threshold) - 1

# negative binomial model
delta = 0.015
timestep = '1d'
scaling_factor = 1e2
pow_climada = 2
origin = (8.36278492095831, 47.163852336888695)

# prediction
nb_draws = 10
# confidence range
confidence = 1e-1

# plotting utilities
plt.rcParams["figure.constrained_layout.use"] = True
matplotlib.rcParams["text.usetex"] = True
params = {'legend.fontsize': 'x-large',
          'axes.facecolor': '#eeeeee',
          'axes.labelsize': 'xx-large', 'axes.titlesize': 20, 'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)
tol = 5e-2
quants = np.linspace(tol, 1, 200)  # franchise claims
CRS = 'EPSG:2056'
CCRS = ccrs.epsg(2056)

# chosen models
name_pot = '20230629_15:48'
name_beta = '20230629_14:56'
name_bern = '20230221_12:58'
