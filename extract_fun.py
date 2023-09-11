import numpy as np
import os
import re # regular expresions
import numpy as np
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


def extract_fun(dataset, sampled_points, band_names, response):

    #dataset: xarrayDataset
    #sampled_points : geoapndas point 
    #band_names: list of strings for each condiered band name
    #response: string of response variable in sampled_points


    mo_x_coords = xr.DataArray(sampled_points.geometry.x, dims=["point"])
    mo_y_coords = xr.DataArray(sampled_points.geometry.y, dims=["point"])

    extracted_values = {}
    for band_name in band_names:
        band_data = getattr(dataset, band_name)  
        values = band_data.sel(x=mo_x_coords, y=mo_y_coords, method="nearest")
        extracted_values[band_name] = values

    df = pd.DataFrame(extracted_values)
    df["x"] = mo_x_coords
    df["y"] = mo_y_coords
    geometry = [Point(x, y) for x, y in zip(df["x"], df["y"])]
    geo_df = geopandas.GeoDataFrame(df, geometry=geometry, crs="EPSG:32718")
    merged_gdf = geo_df.merge(sampled_points[[response]], left_index=True, right_index=True)

    return merged_gdf




# agreagar if si solo se extrae un rater de una sola banda
