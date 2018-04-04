from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
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
from mlxtend.regressor import StackingRegressor
import lightgbm as lgb

RANDOM_STATE = 1992

train = pd.read_csv("data/train_ft_sel_31_v2.csv", encoding='utf-8', sep=',')

test = pd.read_csv("data/test_ft_sel_31_v2.csv", encoding='utf-8', sep=',')

train['SalePrice'] = 10 ** train['SalePrice']
train['SalePrice'] = np.log1p(train['SalePrice'])

X_train, X_test, y_train, y_test = train_test_split(train.loc[:,train.columns != 'SalePrice'],
                                                    train['SalePrice'],
                                                    test_size=0.3,
                                                    random_state=RANDOM_STATE)

X_train = X_train.loc[:,X_train.columns != 'Id']
X_test = X_test.loc[:,X_test.columns != 'Id']
X_test = X_test.loc[:,X_train.columns]



# param_xgb = {'objective': ['reg:linear'],
#              "learning_rate": [0.05],
#              "max_depth": [2],
#              "n_estimators": [1000],
#              'subsample': [0.85],
#              "colsample_bytree": [0.55],
#              "colsample_bylevel": [0.75],
#              # "lambda": [1, 0.8],
#              # "alpha": [0, 0.3],
#              }
#
# estimator_xgb = XGBRegressor()
# xgboost_model = GridSearchCV(estimator=estimator_xgb,
#                              param_grid=param_xgb,
#                              n_jobs=-1,
#                              cv=5,
#                              verbose=0)
#
# xgboost_model.fit(X_train, y_train)
# joblib.dump(xgboost_model, 'models/xgbr_model.pkl')
# preds = xgboost_model.best_estimator_.predict(X_test)
# rmse = mean_squared_error(y_test, preds)**0.5
# print('XGBoost RMSE:',rmse)
# print(xgboost_model.best_params_)
####################################################
#
# param_lr = {'fit_intercept': [True, False],
#              "normalize": [True, False],
#             'copy_X': [True, False]
#              }
#
# estimator_lr = LinearRegression()
# lr_model = GridSearchCV(estimator=estimator_lr,
#                              param_grid=param_lr,
#                              n_jobs=-1,
#                              cv=5,
#                              verbose=0)
#
# lr_model.fit(X_train, y_train)
# joblib.dump(lr_model, 'models/lr_model.pkl')
# preds = lr_model.predict(X_test)
# rmse = mean_squared_error(y_test, preds)**0.5
# print('LinearRegression RMSE:',rmse)
# print(lr_model.best_params_)
# ####################################################
#
# param_rf = {'n_estimators': [1000],
#              "max_features": [0.6],
#             'max_depth': [4],
#             "min_samples_leaf": [0.002],
#             "min_samples_split": [11],
#             "random_state": [RANDOM_STATE],
#              }
#
# estimator_rf = RandomForestRegressor()
# rf_model = GridSearchCV(estimator=estimator_rf,
#                              param_grid=param_rf,
#                              n_jobs=-1,
#                              cv=5,
#                              verbose=0)
#
# rf_model.fit(X_train, y_train)
# joblib.dump(rf_model, 'models/rf_model.pkl')
# preds = rf_model.predict(X_test)
# rmse = mean_squared_error(y_test, preds)**0.5
# print('RandomForest RMSE:',rmse)
# print(rf_model.best_params_)

# ####################################################
# params = {'boosting_type': 'gbdt',
#           'max_depth' : 2,
#           # 'nthread': 5,
#           # 'num_leaves': 64,
#           'learning_rate': 0.05,
#           'max_bin': 64,
#           # 'subsample_for_bin': 200,
#           'subsample': 0.8,
#           'subsample_freq': 1,
#           'colsample_bytree': 0.1,
#           'reg_alpha': 5,
#           'reg_lambda': 10,
#           'min_split_gain': 0.5,
#           'min_child_weight': 1,
#           'min_child_samples': 5,
#           # 'scale_pos_weight': 1,
#           # 'num_class' : 1,
#           'early_stopping_round': 5,
#           }
#
# param_lgbr = {
#     'learning_rate': [0.1],
#     'n_estimators': [1000],
#     'num_leaves': [6],
#     'boosting_type' : ['gbdt'],
#     'random_state' : [RANDOM_STATE],
#     'colsample_bytree' : [0.65],
#     'reg_alpha' : [0],
#     'reg_lambda' : [0.5],
#     'min_split_gain': [0.005],
#           'min_child_weight': [0],
#           'min_child_samples': [3],
#     }
# estimator_lgbr = lgb.LGBMRegressor(boosting_type= 'gbdt',
#                                     # objective = 'binary',
#                                     n_jobs = 1, # Updated from 'nthread'
#                                     silent = True,
#                                     max_depth = params['max_depth'],
#                                     max_bin = params['max_bin'],
#                                     subsample = params['subsample'],
#                                     subsample_freq = params['subsample_freq'],
#                                     min_split_gain = params['min_split_gain'],
#                                     min_child_weight = params['min_child_weight'],
#                                     min_child_samples = params['min_child_samples'],
#                                    metric = 'l2_root',
#                                    # early_stopping_round = params['early_stopping_round']
#                                    )
#
# lgbmr_model = GridSearchCV(estimator=estimator_lgbr,
#                              param_grid=param_lgbr,
#                              n_jobs=-1,
#                              cv=5,
#                              verbose=0)
#
# lgbmr_model.fit(X_train, y_train)
# joblib.dump(lgbmr_model, 'models/lgbmr_model.pkl')
# preds = lgbmr_model.predict(X_test)
# rmse = mean_squared_error(y_test, preds)**0.5
# print('lgbmr_model RMSE:',rmse)
# print(lgbmr_model.best_params_)

# ####################################################
#
# estimator_en = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
#                              n_jobs=-1,
#                              eps=0.0000005,
#                              n_alphas=5000,
#                              cv=7,
#                              max_iter=5000,
#                              random_state=RANDOM_STATE,
#                              )
#
# estimator_en.fit(X_train, y_train)
# joblib.dump(estimator_en, 'models/en_model.pkl')
# preds = estimator_en.predict(X_test)
# rmse = mean_squared_error(y_test, preds)**0.5
# print('ElasticNet RMSE:',rmse)
# ####################################################
#
# param_adab = {'n_estimators': [1000],
#              "learning_rate": [1.45],
#             "random_state": [RANDOM_STATE],
#               'loss': ['square']
#              }
#
# estimator_adab = AdaBoostRegressor()
# adab_model = GridSearchCV(estimator=estimator_adab,
#                              param_grid=param_adab,
#                              n_jobs=-1,
#                              cv=5,
#                              verbose=0)
#
# adab_model.fit(X_train, y_train)
# joblib.dump(adab_model, 'models/adab_model.pkl')
# preds = adab_model.predict(X_test)
# rmse = mean_squared_error(y_test, preds)**0.5
# print('AdaBoost RMSE:',rmse)
# print(adab_model.best_params_)
#
# # ####################################################
#
# param_sgd = {
# # 'max_iter': [20, 100, 200],
#              "power_t": [0.5],
#     'alpha': [1e-3],
#             "n_iter": [100000],
#              "eta0": [0.01],
#             "random_state": [RANDOM_STATE],
#     'loss': ['huber'],
#     'penalty': ['l2'],
#     'fit_intercept': [True],
#     'l1_ratio': [0.01],
#     'shuffle': [True],
#     'epsilon': [0.009],
#     'learning_rate': ['invscaling'],
#     'warm_start': [True],
#     'average': [True]
#              }
#
# estimator_sgd = SGDRegressor()
# sgd_model = GridSearchCV(estimator=estimator_sgd,
#                              param_grid=param_sgd,
#                              n_jobs=-1,
#                              cv=5,
#                              verbose=0)
#
# sgd_model.fit(X_train, y_train)
# joblib.dump(sgd_model, 'models/sgd_model.pkl')
# preds = sgd_model.predict(X_test)
# rmse = mean_squared_error(y_test, preds)**0.5
# print('SGDRegressor RMSE:',rmse)
# print(sgd_model.best_params_)

#
# # ####################################################
#
# estimator_svr = SVR()
#
# param_svr = {
#     'C': [1.781],
#     'epsilon': [0.0001],
#     'coef0': [1],
#     'shrinking': [False],
#     'tol': [1e-5],
#     'max_iter': [2300]
# }
#
# svr_model = GridSearchCV(estimator=estimator_svr,
#                              param_grid=param_svr,
#                              n_jobs=-1,
#                              cv=5,
#                              verbose=0)
#
# svr_model.fit(X_train, y_train)
# joblib.dump(svr_model, 'models/svr_model.pkl')
# preds = svr_model.predict(X_test)
# rmse = mean_squared_error(y_test, preds)**0.5
# print('SVR RMSE:',rmse)
# print(svr_model.best_params_)

# # ####################################################
#
# param_knn = {'n_neighbors': [13],
#              "leaf_size": [1],
#              'weights': ['distance'],
#              'algorithm': ['auto'],
#              'p': [1],
#              }
#
# estimator_knn = KNeighborsRegressor()
# knn_model = GridSearchCV(estimator=estimator_knn,
#                              param_grid=param_knn,
#                              n_jobs=-1,
#                              cv=5,
#                              verbose=0)
#
# knn_model.fit(X_train, y_train)
# joblib.dump(knn_model, 'models/knn_model.pkl')
# preds = knn_model.predict(X_test)
# rmse = mean_squared_error(y_test, preds)**0.5
# print('kNNRegressor RMSE:',rmse)
# print(knn_model.best_params_)

# # ####################################################
#
# param_nn = {'hidden_layer_sizes': [1024],
#              "solver": ['adam'],
#             "learning_rate": ['adaptive'],
#             "learning_rate_init": [0.01],
#             "max_iter": [90],
#             'random_state': [RANDOM_STATE],
#             'early_stopping': [True],
#             'warm_start': [True],
#             'beta_1': [0.8],
#             'beta_2': [0.990],
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
# nn_model.fit(X_train, y_train)
# joblib.dump(nn_model, 'models/nn_model.pkl')
# preds = nn_model.predict(X_test)
# rmse = mean_squared_error(y_test, preds)**0.5
# print('NeuralNetwork RMSE:',rmse)
# print(nn_model.best_params_)
#
# # ####################################################
# # def baseline_model():
# #     # create model
# #     model = Sequential()
# #     model.add(Dense(32, input_dim=93, kernel_initializer='normal', activation='relu'))
# #     model.add(Dense(64, activation='relu'))
# #     model.add(Dense(128, activation='relu'))
# #     model.add(Dense(256, activation='relu'))
# #     model.add(Dense(1))
# #     # Compile model
# #     model.compile(loss='mean_squared_error', optimizer='rmsprop')
# #     return model
# #
# # estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=1, verbose=1)
# #
# # estimator.fit(X_train, y_train)
# # preds = estimator.predict(X_test)
# # rmse = mean_squared_error(y_test, preds)
# # print('Keras RMSE:',rmse)
#
#
# # ####################################################
#
# ###################################################
#
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
# lasso_model.fit(X_train, y_train)
# joblib.dump(lasso_model, 'models/lasso_model.pkl')
# preds = lasso_model.predict(X_test)
# rmse = mean_squared_error(y_test, preds)**0.5
# print('Lasso RMSE:', rmse)
# print(lasso_model.best_params_)
#
# ###################################################
#
# param_ridge = {'alpha': [0.99],
#              'tol': [1e-2],
#                'random_state': [RANDOM_STATE],
#                'max_iter': [1],
#                'solver': ['svd']
#              }
#
# estimator_ridge = Ridge()
# ridge_model = GridSearchCV(estimator=estimator_ridge,
#                              param_grid=param_ridge,
#                              n_jobs=-1,
#                              cv=5,
#                              verbose=0,
#                            scoring='neg_mean_squared_error')
#
# ridge_model.fit(X_train, y_train)
# joblib.dump(ridge_model, 'models/ridge_model.pkl')
# preds = ridge_model.predict(X_test)
# rmse = mean_squared_error(y_test, preds)**0.5
# print('Ridge RMSE:', rmse)
# print(ridge_model.best_params_)


# model = joblib.load('models/xgbr_model.pkl')
# preds = model.predict(X_test)
# rmse = mean_squared_error(y_test, preds)
# print('Ridge RMSE:', rmse)
# preds_test = model.predict(test.loc[:,X_train.columns])
# preds_test = 10 ** preds_test
# submiss = pd.DataFrame({'Id': test['Id'], 'SalePrice': preds_test})
# submiss.to_csv('submissions/rmse_{}.csv'.format(round(rmse, 7)), index=None)

# x_plot = list(range(len(X_test)))
# plt.plot(x_plot,preds, '--', x_plot, y_test, '--k')
# plt.show()