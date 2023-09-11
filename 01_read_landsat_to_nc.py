import numpy as np
import os
import re # regular expresions
import xarray as xr
import rioxarray as rxr
import pyproj
import rasterio
import pandas as pd
import geopandas
import shapely.geometry as sg
from shapely.geometry import Point

import matplotlib.pyplot as plt
from PIL import Image
import pathlib
import datacube
from datetime import datetime, date, time, timedelta

# file path 


# file path 
path = pathlib.Path()
path = path.resolve()
ls_directory = path / 'data' / 'landsat' / 'landsat_complete_time_series' / 'ls-florida'  / 'ls-florida' 
ls_files = list(ls_directory.glob('*.tif')) # load all file directories

# temporal sort

date_pattern = r"\d{8}"
dates = []
for file in ls_files:
    match = re.search(date_pattern, file.name)
    if match:
        date_str = match.group(0)
        date_obj = datetime.strptime(date_str, "%Y%m%d")  
        dates.append(date_obj)

dates[0]  

dates_list = [dt.strftime('%Y-%m-%d') for dt in dates]

len(dates_list) # 74 scenes

#FUNCTION TO CONSTRUCT THE XARRAY.DATASET

# LIST -> xarray.ARRAY
raster_list = [] 
for file in ls_files:
    raster = rxr.open_rasterio(file, masked=True)
    # raster_list.append(rxr.open_rasterio(file, masked=True))
    raster.attrs = {}
    raster_list.append(raster)


# xarray.ARRAY - > xarray.DATASET
dataset = xr.concat(raster_list, dim="time").assign_coords(time=dates)
dataset = dataset.to_dataset(dim='band', name=None, promote_attrs=False) #Dimensions:(time: 101, y: 888, x: 1089)

# change band codes 

band_names_dict = {1:'Red',2:'Green', 3:'Blue', 4:'NIR', 5:'SWIR1', 6:'SWIR2', 7:'NDVI', 8:'NBR2', 9:'BSI',
                    10: 'BSI1', 11:'BSI2', 12:'BSI3', 13:'NDSI1', 14:'NDSI2', 15:'BI', 16:'MBI'} # check band names from GEE

dataset = dataset.rename(name_dict= band_names_dict)

# SAVE DATASET
dataset_output = path / 'data' / 'landsat'/ 'nc_landsat' / 'ls_dataset.nc'
dataset.to_netcdf(dataset_output) # 2 GB

# PLOT RGB DATA TIME SERIES 

rgb_bands = ['Red', 'Green', 'Blue']
rgb = dataset.sel(time=dates_list[0:4])[rgb_bands] #, method='nearest'
rgb.to_array(dim='variable').plot.imshow(col='time', robust=True) # hay que convertir to .to_array
plt.show()




#  PIPELINE
# CONVERTIR  ESTO A UNA FUNCION QUE CON NBR2 THESHOLD COMO ARGUMENTO (probar varios valores) OK

# EL RESULTADO ES CAD VEZ Q SE APLICA LA FUNCION UNA COMPOSITE DIFERENTE
# USAR CADA VEZ UN COMPOSITE DIFERENTE 
# OVERLAY THE MO VALUES FOR EACH PIXEL OF THE COMPOSITE
# REGRESION DATAFRAME => RANDOM FOREST 
    # 10 CROSS VALIDATION
    # TUNING 
    # => RMSE PROMEDIO
# REPETIR
# FIN







# PARA CADA FECHA 
#   CACLULAR NDVI
#   CALCULAR NRB2
#   MASK IMAGEN  the following conditions were kept: no clouds, ∈ cropland, NDVI < 0.25, NBR2 < NBR2threshold.
#   usar ndvi y nrb2 de ejemplo (NDVI < 0.25, NBR2 < NBR2threshold (0.05).)
#   AGREGAR all IMAGES using mean() 
#   output => UNA SOLA IMAGEN (SYSI COMPOSITE)


# funcion para NDVI y NRB2 , aplicar 

# el principal valor de este esudio sería que se tuneo ambos NDVI y NRB2 en el pipeline
# PLOT ALL RASTERS IN A GRID  ( FIRST 20 OR SO)
# ESTA SERIA LA IDEA
# PARA CADA COMPOSITE
#      ELEGIR UN THESHOLE DE NDVI Y NRB2 EN UNA GRILLA
#       the following conditions were kept: no clouds, ∈ cropland, NDVI < 0.25, NBR2 < NBR2threshold. 
#       AJUSTAR UN MODELO (RF) 
#             PARA CADA MODELO HACER HYOERPAREMETER TUNING ( NRB2 es un parametro)
#             HALLAR EL ERROR CON CV
