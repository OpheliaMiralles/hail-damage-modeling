from __future__ import annotations

import pathlib
from glob import glob

import cartopy
import geopandas
import geoplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model

from data.hailcount_data_processing import get_grid_mapping

PRED_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/prediction/')

keras = tf.keras
layers = keras.layers

PLOT_ROOT = pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/plots/')
METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
]

geom_roots = glob(str(pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/data/GVZ_Datenlieferung_an_ETH/') / 'SHAPEFILE/swissTLMRegio_Boundaries_LV95/swissTLMRegio_KANTONSGEBIET_LV95.shp'))
df_polygon = pd.concat([geopandas.read_file(geom_root) for geom_root in geom_roots]).to_crs(epsg='4326')
df_polygon = df_polygon[df_polygon.NAME == 'Zürich']
lakes = glob(str(pathlib.Path('/Users/Boubou/Documents/GitHub/hail_gvz/data/GVZ_Datenlieferung_an_ETH/') / 'SHAPEFILE/swissTLMRegio_Product_LV95/Hydrography/swissTLMRegio_Lake.shp'))
df_lakes = geopandas.read_file(lakes[0]).to_crs(epsg='4326')


def img_size(z):
    return z.shape[2]


def channels(z):
    return z.shape[-1]


def make_classifier(input_channels: int):
    input = layers.Input((input_channels), name="input")
    x = input
    # Extract useful info from transformations
    x = layers.Conv1D(4, kernel_size=n_days // 3, strides=4, activation=keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.L1(0.05))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(6, kernel_size=n_days // 12, strides=2, activation=keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.L1(0.05))(x)
    x = layers.LayerNormalization()(x)
    # Eliminate grid_size dimension -> (batch, variables)
    x = layers.Flatten()(x)
    x = layers.Dense(2, activation=keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.L1(0.05))(x)
    x = layers.Dense(8, activation=keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.L1(0.05))(x)
    x = layers.Dense(32, activation=keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.L1(0.05))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = layers.Dense(4, activation=keras.activations.relu)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = layers.Dense(1, use_bias=False)(x)
    x = layers.Activation(keras.activations.sigmoid)(x)
    return Model(inputs=input, outputs=x, name='damages')


def xy_from_pandas(data):
    covariates = ['gridcell', 'time', 'climadacnt', 'poh', 'meshs', 'wind_dir', 'wind_speed',
                  'obscnt']
    data_ = data.reset_index()[covariates].assign(climadacnt=lambda x: x.climadacnt / x.climadacnt.max()) \
        .assign(wind_dir=lambda x: x.wind_dir / x.wind_dir.max()) \
        .assign(meshs=lambda x: x.meshs / x.meshs.max()).assign(poh=lambda x: x.poh / 100).assign(time=lambda x: x.time / x.time.max()).assign(
        wind_speed=lambda x: x.wind_speed / x.wind_speed.max()).set_index(['time', 'gridcell'])
    x = np.moveaxis(data_.drop(columns=['obscnt']).to_xarray().to_array().to_numpy(), [0, 2, 1], [2, 1, 0])
    y = data_.obscnt.to_xarray().to_numpy()
    return x, y


def plot_map(data, y_pred_test, dates, mapping, title=""):
    gridcells = data.gridcell.unique()
    xr_y_pred = xr.DataArray(data=y_pred_test,
                             dims=['claim_date', 'gridcell'],
                             coords={'claim_date': dates, 'gridcell': gridcells})
    df_ypred = xr_y_pred.to_dataframe(name='y_pred_bin')
    dd = data.assign(y=lambda x: x.obscnt).assign(y_clim=lambda x: (x.climadacnt > 0).astype(int))
    dd = dd[dd.claim_date.isin(dates)].set_index(['claim_date', 'gridcell'])
    dd = dd.merge(df_ypred, left_index=True, right_index=True, how='left').reset_index().merge(mapping.drop_duplicates('gridcell')[['gridcell', 'geometry']], on='gridcell',
                                                                                               how='left').set_geometry('geometry')
    fig, axes = plt.subplots(ncols=2, nrows=len(dates), figsize=(8, 10), constrained_layout=True, subplot_kw={'projection': cartopy.crs.PlateCarree()})
    for (c, d), ax in zip(dd.groupby('claim_date'), axes):
        ax1, ax2 = ax.flatten()
        ax1.set_title(c)
        geoplot.choropleth(d[d.y > 0], hue='y', cmap='Reds', legend=False, ax=ax1)
        # geoplot.choropleth(d[d.y_clim > 0], hue='y_clim', cmap='Paired', legend=False, ax=ax2)
        geoplot.choropleth(d[d.y_pred_bin > 0], hue='y_pred_bin', cmap='Reds', legend=False, ax=ax2)
        for a in ax:
            df_polygon.plot(ax=a, facecolor='oldlace', edgecolor='grey')
            df_lakes.plot(ax=a, facecolor='royalblue', alpha=0.5)
            a.add_feature(cartopy.feature.BORDERS.with_scale('10m'), color='black')
            a.add_feature(cartopy.feature.COASTLINE.with_scale('10m'), color='black')
            a.set_extent([mapping.longitude.min(), mapping.longitude.max(), mapping.latitude.min(), mapping.latitude.max()])
    fig.suptitle(title, fontsize=14)
    fig.show()


class ShowMapCallback(keras.callbacks.Callback):
    def __init__(self, data, delay_epochs=10):
        super(ShowMapCallback, self).__init__()
        self.data = data
        self.delay_epochs = delay_epochs
        self.cooldown = 5
        indices_dates = np.array([85, 105, 238])
        self.dates = data.claim_date.unique()[indices_dates]
        # data_to_predict = self.data[self.data.claim_date.isin(self.dates)]
        x, y = xy_from_pandas(self.data)
        self.x_predict = x[indices_dates]
        self.mapping = get_grid_mapping(dl=False)

    def _run(self, epoch):
        y_pred = self.model.predict(self.x_predict)
        plot_map(self.data, y_pred, self.dates, self.mapping, title=f'Epoch {epoch}')

    def on_epoch_end(self, epoch, logs=None):
        self.cooldown -= 1
        if self.cooldown <= 0:
            self.cooldown = self.delay_epochs
            self._run(epoch=epoch)
