import numpy as np
import os
import re # regular expresions
import rasterio
import pandas as pd
import geopandas
import shapely.geometry as sg
from shapely.geometry import Point
import xarray as xr
import rioxarray as rxr
import pyproj
import matplotlib.pyplot as plt
from PIL import Image
import pathlib
import datacube


# file path 
path = pathlib.Path()
path = path.resolve()
ls_directory = path / 'data' / 'landsat' / 'landsat_complete_time_series' / 'ls-florida' / 'ls-florida'
raster_directory = ls_directory / 'ls_FiltFlorida2_LC08_009064_20220826.tif'
raster = rxr.open_rasterio(raster_directory, masked=True)
dataset = raster.to_dataset(dim='band', name=None, promote_attrs=False) 

# change band codes 
band_names_dict = {1:'Red',2:'Green', 3:'Blue', 4:'NIR', 5:'SWIR1', 6:'SWIR2', 7:'NDVI', 8:'NBR2', 9:'BSI',
                    10: 'BSI1', 11:'BSI2', 12:'BSI3', 13:'NDSI1', 14:'NDSI2', 15:'BI', 16:'MBI'} # check band names from GEE

dataset = dataset.rename(name_dict= band_names_dict)

# RGB PLOT 
# rgb_bands = ['Red', 'Green', 'Blue']
# rgb_dataset = dataset[rgb_bands]
# rgb_dataset.to_array(dim='variable').plot.imshow(robust=True) # hay que convertir to .to_array
# plt.show()


# TRAINING POINTS
#IMPORT SAMPLED POINTS

ls_directory = path / 'data' / 'landsat'/'bare_soil_points'
points_directory = ls_directory / 'class_points.gpkg'
class_points =  geopandas.read_file(points_directory)

class_points = class_points[['class','geometry']]
class_points.geom_type.head() # MultiPoint

class_points = class_points.explode(index_parts=False)
class_points.geom_type.head()
class_points.head()
class_points.shape

# TABULAR DATA

from landsat_py.extract_fun import extract_fun

col_names = ['Red','Green', 'Blue', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NBR2', 'BSI',
                    'BSI1', 'BSI2', 'BSI3', 'NDSI1', 'NDSI2', 'BI', 'MBI']


response = 'class'

dataset_sampled_class = extract_fun(dataset, class_points, col_names, response)
dataset_sampled_class.head()
dataset_sampled_class.shape

dataset_sampled_class.isna().sum()

#SAVE
output_directory = path / 'data' / 'landsat'/'training_data_bare_clasification'/'training_data_bare_clasification.shp'
dataset_sampled_class.to_file(output_directory)







# la idea es genrar un modelo que calsificque cada ecena como bare soil
# generar bare masks
# y filtrar cada escena con el mask
# promedia r cada escena y genra una sysi
# modelar el MO en base al sysi

