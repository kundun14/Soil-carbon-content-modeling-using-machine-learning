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

from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection  import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns

# HACER  COMPARTIVA DE FEATURES 
# que feature clasifica mejor el suelo desnudo?


# LOAD TRAINING DATA (LAND COVER CLASSES)  


data_directory = path / 'data' / 'landsat'/ 'training_data_bare_clasification' /'training_data_bare_clasification.shp'
data = geopandas.read_file(data_directory)

data.dtypes
data.head()
data.isna().sum() # nas

# EDA

# class labels
# 1 : soil
# 2 : vegetation
# 3 : cloud shadow

# SELECT MODELING COLUMNS 

columns_to_drop = ['x','y','geometry']
data = data.drop(columns=columns_to_drop)
data.head()

# SCATTER PLOTS

sns.set()
sns.pairplot(data.drop(columns=['Red','Green','Blue','NIR','SWIR1', 'SWIR2']), hue='class', kind='reg')
plt.savefig('output/scatterplots_indices.png')

#HEAT MAP 

# sns.set()
heatmap = sns.heatmap(data.drop(columns=['Red','Green','Blue','NIR','SWIR1', 'SWIR2']).corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
plt.show()


# MODELING

# MODEL WITHOUT FINE TUNING

data = data.drop(columns=['Red','Green','Blue','NIR','SWIR1', 'SWIR2'])
data.head()

TEST_RATIO = 0.2
RANDOM_STATE = 42
LABEL_NAME = 'class'
DATA_TYPE = np.int16

#starify spliting 

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['class'], axis='columns'),
    data['class'],
    random_state=RANDOM_STATE,
    train_size=1-TEST_RATIO,
    stratify=data['class'] 
)

# y_train.value_counts(), y_test.value_counts()


# 5 fold CV

kf = KFold(n_splits=5)
kf.get_n_splits(X_train)

# MODEL INITIAL PARAMETERS 

classifier = HistGradientBoostingClassifier(
    max_iter=100,  
    learning_rate=0.1,
    max_depth=2,
    random_state=42
)

# K FOLD FITTING
# todo esto es usanando el train data set

X_train.shape #(1003, 10)
y_train.shape # (1003,)
X_test.shape #(251, 10)
y_test.shape #(251,)


bestModel = None
bestModelScore = 0
scores = []

for trainIdx, testIdx in kf.split(X_train): 

    X_train_valid, X_test_valid = X_train.iloc[trainIdx], X_train.iloc[testIdx]
    y_train_valid, y_test_valid = y_train.iloc[trainIdx], y_train.iloc[testIdx]

    classifier.fit(X_train_valid, y_train_valid) 

    score = classifier.score(X_test_valid, y_test_valid)

    if score>=bestModelScore:
        bestModelScore = score
        bestModel = classifier
    
    test_predictions = classifier.predict(X_test_valid)
    print('Score: {}'.format(score))
    scores.append(score)
    del test_predictions, score

# scoreAvg = np.asarray(scores).mean()
# print('Average accuracy score: {}'.format(scoreAvg))
# print('Best accuracy score: {}'.format(bestModelScore))

classifier = bestModel
score = classifier.score(X_test, y_test)
score = round(score, 3)
score

#FEAURE IMPORTANCE 
# Permutation feature importance for train and test sets
# two analysis


train_perm_importance = permutation_importance(classifier, X_train, y_train, n_repeats=30, random_state=RANDOM_STATE)
test_perm_importance = permutation_importance(classifier, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE)


features = ['NDVI', 'NBR2', 'BSI', 'BSI1', 'BSI2', 'BSI3', 'NDSI1', 'NDSI2', 'BI', 'MBI']

r = train_perm_importance = permutation_importance(classifier, X_train, y_train, n_repeats=30, random_state=RANDOM_STATE)
for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{features[i]:<8}"
            f"{r.importances_mean[i]:.3f}"
            f" +/- {r.importances_std[i]:.3f}")


#PRIMERO HACER UNA NESTED HP TUNING OF THE CLASIFIER



# garbage 

del X_train, X_test, y_train, y_test

# SAVE MODEL

model_name = 'mw_{}_{}_{}_2.0.0_tuned_{}.sav'.format(
    score, hyperparameters['n_estimators'],
    'cpu', 
    datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))

model_save_path = path / 'data' / 'landsat'/ 'model' / model_name

joblib.dump(classifier, model_save_path, compress=3)


##########################3

# EL MODELO FUE GENERADO EN 03_nestedCV_tuning_selection.py

#LOAD MODEL

model_save_path = path / 'data' / 'landsat'/ 'model' / mw_0.996_20_cpu_2.0.0_tuned_2023_08_30_14_38.sav
classifier = joblib.load(model_save_path)



#INFERENCE USING RASTER DATA => MAPPING

# READ LANDSAT RASTER FROM NC FILE (PREVIOUS PROCESING WAS DONE IN read_landsat_to_nc.py)

data_directory = path / 'data' / 'landsat'/ 'nc_landsat' / 'ls_dataset.nc'
dataset = xr.open_dataset(data_directory) # son klo mismo OK
bands = ['Red','Green', 'Blue', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NBR2', 'BSI',
                    'BSI1', 'BSI2', 'BSI3', 'NDSI1', 'NDSI2', 'BI', 'MBI']



prediction_list = []

for time in dataset.time:
    dataset_array = dataset[bands].sel(time=time).to_array().values 
    transposed_data = dataset_array.transpose(1, 2, 0)
    reshaped_data = transposed_data.reshape(((429792, 8)))
    data_df = pd.DataFrame(reshaped_data, columns=bands, dtype=np.float32)
    predictions_flat = classifier.predict(data_df).astype(np.int16)
    prediction_2d = predictions_flat.reshape((592,726))
    y = dataset.sel(time=time).coords['y'].values
    x = dataset.sel(time=time).coords['x'].values
    prediction_xr = xr.DataArray(prediction_2d, coords={'y':y, 'x':x})
    prediction_list.append(prediction_xr)

predicted_dataset = xr.concat(prediction_list, dim="time").assign_coords(time=dataset.time.values)
mask = predicted_dataset <= 1
dataset_masked = dataset.where(mask, drop=False) 

# TEST EXPORT TESSS
# ras = dataset_masked.sel(time=dataset.time.values[11])
# ras.rio.to_raster('ras_masked.tif')


# dataset_masked_ = dataset_masked.to_dataset(dim='band', name=None, promote_attrs=False)

sysi_composite = dataset_masked.median(dim='time') 

# RESULTADO FINAL
sysi_composite.rio.to_raster('sysi_median_composite.tif')












###### TEST PREDIT RASTER   
# ESTE CODIGO FUNCIONA PARA UNA ESCENA 
# bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'ndvi', 'nrb2']
# time = dataset.time[11] # 2022-08-26
# dataset_array = dataset[bands].sel(time=time).to_array().values 

# transposed_data = dataset_array.transpose(1, 2, 0)
# reshaped_data = transposed_data.reshape(((429792, 8)))
# data_df = pd.DataFrame(reshaped_data, columns=bands, dtype=np.float32)

# # data_df.isna().sum()
# # data_df.loc[50]


# data_df.head()
# # blue        green       red
# #0.016645  ,  0.031385      0.02308
# #0.02242       0.0339       0.028525

# predictions_flat = classifier.predict(data_df).astype(np.int16)
# prediction_2d = predictions_flat.reshape((592,726))

# # prediction_2d.shape

# y = dataset.sel(time=time).coords['y'].values
# x = dataset.sel(time=time).coords['x'].values


# prediction_xr = xr.DataArray(prediction_2d, coords={'y':y, 'x':x})

# # prediction_xr.plot()
# # plt.show()

# # SAVE 

# prediction_xr.rio.to_raster('pred_ejemplo.tif')

