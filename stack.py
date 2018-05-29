import warnings
warnings.filterwarnings("ignore")
from mlxtend.regressor import StackingRegressor
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
from sklearn.model_selection import KFold

RANDOM_STATE = 1992

train = pd.read_csv("data/train_ft_sel_31_v2.csv", encoding='utf-8', sep=',')

test = pd.read_csv("data/test_ft_sel_31_v2.csv", encoding='utf-8', sep=',')

train['SalePrice'] = 10 ** train['SalePrice']
train['SalePrice'] = np.log1p(train['SalePrice'])

X = train.loc[:, train.columns != 'SalePrice']
X = X.loc[:, X.columns != 'Id']
y = train['SalePrice']


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=RANDOM_STATE)

X_train = X_train.loc[:,X_train.columns != 'Id']
X_test = X_test.loc[:,X_test.columns != 'Id']
X_test = X_test.loc[:,X_train.columns]

models = [x for x in os.walk('models')][0][-1]
models_folder = ['models/' + model for model in models]


all_models = dict()

for i in range(len(models)):
    all_models[models[i]] = joblib.load(models_folder[i])

X_train_stack = pd.DataFrame()
X_test_stack = pd.DataFrame()
test_stack = pd.DataFrame()
X_stack = pd.DataFrame()

kfold = KFold(n_splits=7, shuffle=False, random_state=RANDOM_STATE)
models_selected = ['nn_model.pkl', 'sgd_model.pkl', 'knn_model.pkl', 'lasso_model.pkl']
for model in models_selected:
    t0 = time.time()
    print('Training {} model'.format(model))
    mdl = GridSearchCV(estimator=all_models[model],
                       param_grid={},
                       n_jobs=-1,
                       cv=kfold,
                       verbose=0,
                       scoring='neg_mean_squared_error')
    mdl.fit(X, y)
    print(mdl.cv_results_)
    X_stack[model] = mdl.predict(X)
    test_stack[model] = mdl.predict(test.loc[:, X.columns])
    print('Train finished {0} model in {1} min'.format(model, round((time.time()-t0)/60, 3)))

# for model in models:
#     X_train_stack[model] = all_models[model].predict(X_train)
#     X_test_stack[model] = all_models[model].predict(X_test)
#     test_stack[model] = all_models[model].predict(test.loc[:,X_train.columns])


# param_lasso = {'alpha': [0],
#              'tol': [1e-2],
#                'random_state': [RANDOM_STATE],
#                'warm_start': [True],
#                'max_iter': [10],
#                'precompute': [True],
#                'copy_X': [True],
#                'selection': ['random']
#              }
#
# estimator_lasso = Lasso()
# lasso_model = GridSearchCV(estimator=estimator_lasso,
#                              param_grid=param_lasso,
#                              n_jobs=-1,
#                              cv=5,
#                              verbose=0,
#                            scoring='neg_mean_squared_error')
#
# lasso_model.fit(X_train_stack , y_train)
# preds = lasso_model.predict(X_test_stack)
# rmse = mean_squared_error(y_test, preds)**0.5
# print('Lasso RMSE:', rmse)
# print(lasso_model.best_params_)

# for model in models:
#     mdl = joblib.load('models/'+model)
#     mdl.fit(X_train_stack, y_train)
#     preds = mdl.predict(X_test_stack)
#     rmse = mean_squared_error(y_test, preds)**0.5
#     print('{} RMSE:'.format(model), rmse)

preds_test = [np.mean(test_stack.iloc[i,:]) for i in range(len(test_stack))]
preds_test = np.expm1(preds_test)
submiss = pd.DataFrame({'Id': test['Id'], 'SalePrice': preds_test})
submiss.to_csv('submissions/Ensembling_NN_SGD_KNN_Lasso.csv', index=None)
# submiss.to_csv('submissions/stack_{}.csv'.format(round(rmse, 7)), index=None)
# ---------------------------

# param_nn = {'hidden_layer_sizes': [1024],
#              "solver": ['adam'],
#             "learning_rate": ['adaptive'],
#             "learning_rate_init": [0.001],
#             "max_iter": [10],
#             'random_state': [RANDOM_STATE],
#             'early_stopping': [True],
#             'warm_start': [True],
#             'beta_1': [0.8, 0.91,0.79],
#             'beta_2': [0.990,0.5,0.9],
#              }
#
# estimator_nn = MLPRegressor()
# nn_model = GridSearchCV(estimator=estimator_nn,
#                              param_grid=param_nn,
#                              n_jobs=-1,
#                              cv=5,
#                              verbose=0,
#                         scoring='neg_mean_squared_error')
#
# # nn_model.fit(X_train, y_train)
# nn_model.fit(X_stack, y)
# # preds = nn_model.predict(X_test_stack)
# # rmse = mean_squared_error(y_test, preds)**0.5
# # print('xgbr RMSE:', rmse)
# # print(nn_model.best_params_)
# preds_test = nn_model.predict(test_stack.loc[:, X_stack.columns])
# preds_test = np.expm1(preds_test)
# submiss = pd.DataFrame({'Id': test['Id'], 'SalePrice': preds_test})
# submiss.to_csv('submissions/NN_KFoldCV.csv', index=None)
# submiss.to_csv('submissions/rmse_{}.csv'.format(round(rmse, 7)), index=None)
# ---------------------------
# CREAR DIFERENTES CAPAS EN PLAN RED NEURONAL


# KERAS MODEL:
# def baseline_model():
#     # create model
#     model = Sequential()
#     model.add(Dense(8, input_dim=4, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal'))
#     # Compile model
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model
#
# def larger_model():
#     # create model
#     model = Sequential()
#     model.add(Dense(12, input_dim=4, kernel_initializer='normal', activation='relu'))
#     # model.add(Dense(4, kernel_initializer='normal', activation='relu'))
#     # model.add(Dense(128, kernel_initializer='normal', activation='relu'))
#     # model.add(Dense(4, kernel_initializer='normal', activation='relu'))
#     # model.add(Dense(2, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal'))
#     # Compile model
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model
#
# earlyStopping = keras.callbacks.EarlyStopping(monitor='loss',
#                                             patience=5,
#                                             verbose=0,
#                                             mode='auto')
#
# seed = 1992
# np.random.seed(seed)
# # evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn=baseline_model,
#                            epochs=1000,
#                            batch_size=5,
#                            verbose=1,
#                            callbacks=[earlyStopping])
# estimator.fit(X_stack, y)
# preds = estimator.predict(X_test_stack)
# rmse = mean_squared_error(y_test, preds)**0.5
# print('Keras RMSE:', rmse)



# preds_test = estimator.predict(test_stack.loc[:,X_stack.columns])
# preds_test = np.expm1(preds_test)
# submiss = pd.DataFrame({'Id': test['Id'], 'SalePrice': preds_test})
# submiss.to_csv('submissions/Keras_KFoldCV.csv', index=None)
# submiss.to_csv('submissions/Keras_{}.csv'.format(round(rmse, 7)), index=None)

