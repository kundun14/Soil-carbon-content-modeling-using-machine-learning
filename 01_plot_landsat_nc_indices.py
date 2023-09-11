import os
import numpy as np
import rasterio
import xarray as xr
import rioxarray as rxr
import pyproj
import matplotlib.pyplot as plt
from PIL import Image
import pathlib
from math import ceil

data_directory = path / 'data' / 'landsat'/ 'nc_landsat' / 'ls_dataset.nc'
dataset = xr.open_dataset(data_directory) 

# DATE  2022-08-26

selected_time = '2022-08-26'
dataset = dataset.sel(time=selected_time)

# INDICES 

indices = ['NDVI', 'NBR2', 'BSI','BSI1', 'BSI2', 'BSI3', 'NDSI1', 'NDSI2', 'BI', 'MBI']

dataset = dataset[indices]

# PLOTING 

indices  = ['NDVI', 'NBR2', 'BSI', 'BSI1', 'BSI2', 'BSI3', 'NDSI1', 'NDSI2', 'BI', 'MBI']

data_arrays = [dataset[var_name].squeeze() for var_name in indices]

N=len(indices)
cols = 3


fig, axs = plt.subplots( ncols=cols, nrows=ceil(N/cols), layout='constrained',
                         figsize=(3.5 * 4, 3.5 * ceil(N/cols)) )

for index, ax in enumerate(axs.flat):
        
        raster = data_arrays[index]
        im = raster.plot(ax=ax, cmap='RdYlBu')
        filename = indices[index]
        ax.set_title(filename)
        
        # histogram inset
        ax_hist = ax.inset_axes([0.7, 0.7, 0.2, 0.2])  # esquina superior derecha 
        hist_values, hist_bins, _ = ax_hist.hist(raster.values.flatten(), bins=20, color="gray")
        ax_hist.grid() 

plt.show()



