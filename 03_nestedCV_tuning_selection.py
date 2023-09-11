import os
import pathlib
import datetime
import time
import joblib
import numpy as np
import pandas as pd
import geopandas
from itertools import combinations



from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt
import seaborn as sns

from openpyxl import Workbook



import os
import pathlib
import datetime
import time
import joblib
import numpy as np
import pandas as pd
import geopandas

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt
import seaborn as sns

from openpyxl import Workbook




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



###############################
############################### FEATURE SELECTION AND PAREMTER TUNING (NESTED CV)
###############################

TEST_RATIO = 0.2
RANDOM_STATE = 42
LABEL_NAME = 'class'
DATA_TYPE = np.int16
# NUM_TRIALS = 30


# SelectKBest does not accept missing values encoded as NaN natively.
data = data.dropna() 
data = data.drop(columns=['class'])
data.isna().sum()

# output_directory = path / 'data' / 'landsat'/'training_data_bare_clasification'/'training_data_bare_clasification_binary.shp'
# data.to_file(output_directory)


classifier = XGBClassifier()

param_grid = {
    'n_estimators': [100, 200],  # Number of boosting rounds, greater: overfiting
    'learning_rate': [0.1, 0.3, 1],
    'max_depth': [3, 6, 10 ],         # Maximum tree depth, greater : overtiting
    'subsample': [0.1, 0.5, 1.0],     # Fraction of samples used for tree building, 
    'reg_lambda': [1 , 10, 100 ]
}

# alpha=1, lambda=1
# min_split_loss 0 , inf


X_lrn, X_test, y_lrn, y_test = train_test_split(
    data.drop(['binary_class'], axis='columns'),
    data['binary_class'],
    random_state=RANDOM_STATE,
    train_size=1-TEST_RATIO,
    stratify=data['binary_class'] 
)

inner_cv = KFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
outer_cv = KFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)

model = GridSearchCV(
    estimator=classifier, 
    param_grid=param_grid, 
    cv=inner_cv, 
    n_jobs=-1,
    scoring = 'f1_macro'
)


N_FEATURES = 2
feature_names = ['NDVI', 'NBR2', 'BSI', 'BSI1', 'BSI2', 'BSI3', 'NDSI1', 'NDSI2', 'BI', 'MBI']
param_names = ['n_estimators', 'learning_rate', 'max_depth', 'subsample', 'reg_lambda']  


# ALGORITMO: NESTED FEATURE SELECTION - PARM TUNING CV
results = []

# OUTER LOOP

for fold_idx, (train_index, test_index) in enumerate(outer_cv.split(X_lrn, y_lrn)):

    print(f"Outer Fold {fold_idx + 1}:")

    # OUTER SPLIT

    X_train_outer, X_test_outer = X_lrn.iloc[train_index], X_lrn.iloc[test_index]
    y_train_outer, y_test_outer = y_lrn.iloc[train_index], y_lrn.iloc[test_index]

    fold_results = []

    # FEATURE SELECTION

    for n in range(1, N_FEATURES + 1):  

        print(f" Selected Features = {n}")

        selector = SelectKBest(f_classif, k=n) #ANOVA F-value between label/feature for classification tasks.
        X_train_selected = selector.fit_transform(X_train_outer, y_train_outer)
        selected_feature_indices = selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_feature_indices]

        # INNER LOOP 

        # HYPERPARAEMTER TUNING
        print(" Tuning ... ")

        model.fit(X_train_selected, y_train_outer)
        inner_best_params = model.best_params_
        inner_best_classifier  = classifier.set_params(**inner_best_params)
        inner_best_classifier.fit(X_train_selected, y_train_outer)  # This line is important

        # MODEL EVALUATION

        print(" Evaluating ... ")

        X_test_selected = selector.transform(X_test_outer)
        y_pred = inner_best_classifier.predict(X_test_selected)
        f1 = f1_score(y_test_outer, y_pred, average='macro')

        combination_results = {
            'Selected Features': n,
            'F1 Score': f1,
            'Selected Feature Names': ', '.join(selected_feature_names)
        }

        for param_name in param_names:
            combination_results[param_name] = inner_best_params.get(param_name, None)
        
        fold_results.append(combination_results)

    results.extend(fold_results)    

results_df = pd.DataFrame(results) # CV results



#  GENERALIATION TEST 
# USE THE BEST PAREMTER AND FETURE CONFIRATION from CV AND REFITING TO THE ENTIRE DATASET

tuned_models = []
f1_scores = []

for result in results:
    
    # SELETED FEATURES AND TUNING PARAMETERS 

    selected_features = result['Selected Feature Names'].split(', ')
    best_hyperparameters = {
        'n_estimators': result['n_estimators'],
        'learning_rate': result['learning_rate'],
        'max_depth': result['max_depth'],
        'subsample': result['subsample'],
        'reg_lambda': result['reg_lambda']
    }

    X_test_selected = X_test[selected_features]

    # FIT MODEL WITH FINE TUNING PARAMETERS AND SELECTED FEATURES 
    classifier = XGBClassifier(**best_hyperparameters)
    classifier.fit(X_lrn[selected_features], y_lrn)
    y_pred = classifier.predict(X_test_selected)

    tuned_models.append(classifier) # SAVE TUNED MODELS 
    f1 = f1_score(y_test, y_pred, average='macro')
    f1_scores.append(f1)

# tuned_models
# f1_scores



# FINAL RESULTS 
# SORTING 

results_df['F1 Score test'] = f1_scores
results_df['Model specification'] = tuned_models
sorted_results_df = results_df.sort_values(by='F1 Score', ascending=False)
sorted_results_df

# SAVE F1 RESULTS DF

results_save_path = path /  'landsat_py'/ 'output' / 'cv_resultsXX.xlsx'
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


#############################
#############################



# f1_scores_outer = []
# f1_scores_inner = []


# for train_index, test_index in outer_cv.split(X_lrn):
#     X_train, X_val = X_lrn[train_index], X_lrn[test_index]
#     y_train, y_val = y_lrn[train_index], y_lrn[test_index]
#     grid_search = GridSearchCV(classifier, param_grid=param_grid, cv=inner_cv, scoring='f1', verbose=1)
#     grid_search.fit(X_train, y_train)
#     best_classifier = grid_search.best_estimator_
#     y_pred_outer = best_classifier.predict(X_val)
#     f1_score_outer = f1_score(y_val, y_pred_outer)
#     f1_scores_outer.append(f1_score_outer)
#     y_pred_inner = cross_val_predict(best_classifier, X_train, y_train, cv=inner_cv)
#     f1_scores_inner.extend([f1_score(y_train, y_pred_inner)])
