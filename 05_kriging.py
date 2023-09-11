import os
import pathlib
import datetime
import time
import joblib

import numpy as np
import pandas as pd

import xarray as xr
import rioxarray as rxr
import pyproj
import rasterio


# IMPORT HULL (AOI)



# SYSI   
path = pathlib.Path()
path = path.resolve()
sysi_directory = path / 'data' / 'landsat' / 'model' / 'sysi_median_composite.tif'
sysi = rxr.open_rasterio(sysi_directory, masked=True)

# SYSI + DEM 

# CREAR A STACK ?? DE COVARIABLESS ()
# load all tif inside a folder
# make list of all of them
# crop all to study region ( hull?)


#


# LOAD MO



# STEP 1 : LANDSAST COMPOSITE GENERETING 
    # STEP 1.1 LANDSAT SCENE CAISIFCCATION
    # STEP 1.2 MASKING
    # STEP 1.3 AGREGATION MEDIAN  
# STEP 2: KRIGING LANDSAT COMPOSITE
# STEP 3: DEM DERIVATIVES
# STEP 4 : RF MODELING OF MO
# STEP 5: EXXPERIMENT COMPARISON DIFERENT COVARIABLES ETC ?? SHAPE VALUES?
# RECORDAR QUE LAS PREDCCIONES FINALES SE HACEN DE REAJUSTANDO EL MEJOR MODELO A TAODA LA DATA ( VER ROBERT)



# CUALES SERIAN LAS FIGURA FINALES?
# 1 MAPA
# 2 DIAGRAMA SYSI (LANDSAT PROCE, MASK, KRIGING?)
# 3 DIAGRAMA MO MODELING (RF)
#  RESULTADOS 
# 4 MAPAS LANDSAT +  SYSI (ORIGINAL)  + SYSI(KRIGED)+ VARIOGRAMS ETC.
# 5. MAPAS DE COVARIABLES
# 6 ANALISIS SCATER PLOTS / HEATMAPS
# 7 SCATTER PLOTS PREDICCIONES 
# 8. MAPA DE SYSI (PROCESAMIENTO SYSI)
# 9. SHAPE VALUESS (INTEPRETABLE ML)
# 8. MAPA DE MO Y DE INCERTIDUMBRE (HACER )
