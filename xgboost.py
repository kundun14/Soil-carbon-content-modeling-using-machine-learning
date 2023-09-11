
# D:/miniconda/envs/geo/python.exe


import os
import pathlib
import datetime
import time
import joblib
import numpy as np
import pandas as pd
import geopandas
import itertools
import random
from tqdm import tqdm

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metricfrom lightgbm import LGBMClassifiers import make_scorer, f1_score

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import cohen_kappa_score

from openpyxl import Workbook

import matplotlib.pyplot as plt
import seaborn as sns


# data 
path = pathlib.Path()
path = path.resolve()
data_directory = path / 'data' / 'landsat'/ 'training_data_bare_clasification' /'training_data_bare_clasification.shp'
data = geopandas.read_file(data_directory)
data = data.drop(columns=['Red','Green','Blue','NIR','SWIR1', 'SWIR2', 'x','y','geometry'])
data['class'] = data['class'].astype(int)
data['binary_class'] = data['class'].apply(lambda x: 0 if x in [2, 3, 4, 5] else x)

# BAR PLOTS OF CLASSES 
# data['binary_class'].value_counts().plot.bar()
# plt.show()
# # SCATTER PLOTS
# sns.set()
# sns.pairplot(data, hue =='binary_class', kind='reg')
# plt.show()
# plt.savefig('output/scatterplots_indices.png')



###############################
############################### FEATURE SELECTION AND PAREMTER TUNING (NESTED CV)
###############################

TEST_RATIO = 0.2
RANDOM_STATE = 42
LABEL_NAME = 'class'
DATA_TYPE = np.int16


# XGBOOST DON'T SUPPORT NAS
data = data.dropna() 
data = data.drop(columns=['class'])
data.isna().sum()

# output_directory = path / 'data' / 'landsat'/'training_data_bare_clasification'/'training_data_bare_clasification_binary.shp'
# data.to_file(output_directory)


# LIGHTGBM

classifier = LGBMClassifier(feature_fraction=1.0, force_col_wise='true', random_state=RANDOM_STATE, objective='binary', verbose= -100)

param_names = ['learning_rate', 'n_estimators', 'max_depth', 'colsample_bytree', 'subsample', 'min_child_samples']

param_grid = {
    'learning_rate': [0.1, 0.01],
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 6],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'subsample': [0.7, 0.8, 0.9],
    'min_child_samples': [1, 5, 10]
    }

# alpha=1, lambda=1
# min_split_loss 0 , inf



# CV PARAMETERES

inner_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

model = GridSearchCV(
    estimator=classifier,
    param_grid=param_grid,
    cv=inner_cv,
    n_jobs=-1,
    scoring = 'f1_macro'
)

# FEATURE COMBINNATION 

feature_names = ['NDVI', 'NBR2', 'BSI', 'BSI1', 'BSI2', 'BSI3', 'NDSI1', 'NDSI2', 'BI', 'MBI']

feature_combinations = []
for L in range(len(feature_names) + 1):
    for subset in itertools.combinations(feature_names, L):
        feature_combinations.append(subset)

random.shuffle(feature_combinations)
feature_combinations = feature_combinations[:100]

for i, feature in enumerate(feature_combinations):
    if i < 10: 
        print(f'Selected Features: {list(feature)}')



# FEATURE COMBINATION PRIOR TO NESTED CROSS VALIDATION

# DATA SPLIT

X_lrn, X_test, y_lrn, y_test = train_test_split(
    data.drop(['binary_class'], axis='columns'),
    data['binary_class'],
    random_state=RANDOM_STATE,
    train_size=1-TEST_RATIO,
    stratify=data['binary_class'] 
)

# ALGORITMO: NESTED FEATURE SELECTION - PARM TUNING CV
results = []

# OUTER LOOP

    # FEATURE SELECTION

for features in tqdm(feature_combinations[1:100], desc="Feature Combinations"):  
    print(f" Selected Features = {list(features)}")
    X_lrn_selected = X_lrn[list(features)] 

    for fold_idx, (train_index, test_index) in enumerate(outer_cv.split(X_lrn_selected, y_lrn)):
        # OUTER SPLIT

        print(f"Outer Fold {fold_idx + 1}:")
        X_train_outer, X_test_outer = X_lrn_selected.iloc[train_index], X_lrn_selected.iloc[test_index]
        y_train_outer, y_test_outer = y_lrn.iloc[train_index], y_lrn.iloc[test_index]

        fold_results = []

        # INNER LOOP 
        # HYPERPARAEMTER TUNING
        print(" Tuning ... ")

        model.fit(X_train_outer, y_train_outer)
        inner_best_params = model.best_params_
        inner_best_classifier  = classifier.set_params(**inner_best_params)
        inner_best_classifier.fit(X_train_outer, y_train_outer)  # This line is important

        # MODEL EVALUATION

        print(" Evaluating ... ")

        X_test_selected = X_test_outer
        y_pred = inner_best_classifier.predict(X_test_selected)
        f1 = f1_score(y_test_outer, y_pred, average='macro')

        print(f" F1 = {f1}")

        combination_results = {
            'Selected Features': ', '.join(features),
            'F1 Score': f1,
        }

        for param_name in param_names:
            combination_results[param_name] = inner_best_params.get(param_name, None)
        
        fold_results.append(combination_results)
    
    results.extend(fold_results)    



results_df = pd.DataFrame(results) # CV results
results_df


#  GENERALIATION TEST 
# USE THE BEST PAREMTER AND FETURE CONFIRATION from CV AND REFITING TO THE ENTIRE DATASET

tuned_models = []
f1_scores = []

for result in results:
    
    # SELETED FEATURES AND TUNING PARAMETERS 

    selected_features = result['Selected Feature Names'].split(', ')

    best_hyperparameters = {
        'learning_rate': result['learning_rate'],
        'n_estimators': result['n_estimators'],
        'max_depth': result['max_depth'],
        'colsample_bytree': result['colsample_bytree'],
        'subsample': result['subsample'],
        'min_child_samples': result['min_child_samples']
    }

    X_test_selected = X_test[selected_features]

    # FIT MODEL WITH FINE TUNING PARAMETERS AND SELECTED FEATURES 
    classifier = XGBClassifier(**best_hyperparameters)
    classifier.fit(X_lrn[selected_features], y_lrn)
    y_pred = classifier.predict(X_test_selected)

    tuned_models.append(classifier) # SAVE TUNED MODELS 
    f1 = f1_score(y_test, y_pred, average='macro')
    f1_scores.append(f1)

# FINAL RESULTS 
# SORTING 

results_df['F1 Score test'] = f1_scores
results_df['Model specification'] = tuned_models
sorted_results_df = results_df.sort_values(by='F1 Score', ascending=False)
sorted_results_df

# SAVE F1 RESULTS DF

results_save_path = path /  'landsat_py'/ 'output' / 'cv_results.xlsx'
sorted_results_df.to_excel(results_save_path, index=False)  


#SAVE MODELS 


best_model_row = sorted_results_df.iloc[0]  # Get the first row, which corresponds to the best model

Model_specification = best_model_row['Model specification']


model_name = 'mw_{}_{}_{}_2.0.0_tuned_{}.sav'.format(
    score, hyperparameters['n_estimators'],
    'cpu', 
    datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))

model_save_path = path / 'data' / 'landsat'/ 'model' / model_name

joblib.dump(classifier, model_save_path, compress=3)
