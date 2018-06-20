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
from sklearn.linear_model import Lasso
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import os
from datetime import date


if __name__ == "__main__":

    def save_model(model, name, score):
        score = round(score, 4)
        today = date.today()
        day = str(today.day)
        month = str(today.month)
        year = str(today.year)
        dir = 'models/'+year+'-'+month+'-'+day
        if not os.path.exists(dir):
            os.makedirs(dir)
        joblib.dump(model, dir+'/'+name+'_'+str(score)+'.pkl')

    def submit(model, name):
        preds_test = np.expm1(model.predict(test[X_train.columns]))
        submiss = pd.DataFrame({'Id': test['Id'], 'SalePrice': preds_test})
        submiss.to_csv('submissions/{0}_{1}.csv'.format(name, round(rmse, 4)), index=None)

    RANDOM_STATE = 1992

    train = pd.read_csv("data/numeric_train_ft_eng.csv", encoding='utf-8', sep=',')
    test = pd.read_csv("data/numeric_test_ft_eng.csv", encoding='utf-8', sep=',')

    X_train, X_test, y_train, y_test = train_test_split(train.loc[:,train.columns != 'SalePrice'],
                                                        train['SalePrice'],
                                                        test_size=0.3,
                                                        random_state=RANDOM_STATE)

    X_train = X_train.loc[:,X_train.columns != 'Id']
    # X_test = X_test.loc[:,X_test.columns != 'Id']
    X_test = X_test.loc[:,X_train.columns]



    # param_xgb = {'objective': ['reg:linear'],
    #              "learning_rate": [0.05, 0.1, 0.3],
    #              "max_depth": [5, 8, 10],
    #              "n_estimators": [300, 400, 600],
    #              'subsample': [0.6, 0.8, 0.5],
    #              "colsample_bytree": [0.8, 0.6],
    #              "colsample_bylevel": [0.8, 0.6],
    #              # "lambda": [1, 1.2],
    #              # "alpha": [0, 0.3],
    #              }
    #
    # estimator_xgb = XGBRegressor()
    # xgboost_model = GridSearchCV(estimator=estimator_xgb,
    #                              param_grid=param_xgb,
    #                              n_jobs=-1,
    #                              cv=5,
    #                              verbose=1)
    #
    # xgboost_model.fit(X_train, y_train)
    # preds = xgboost_model.best_estimator_.predict(X_test)
    # rmse = mean_squared_error(y_test, preds)
    # save_model(xgboost_model, 'xgbr', rmse)
    # print('XGBoost RMSE:',rmse)
    ####################################################
    # print('Entrenando Linear Regression')
    # param_lr = {'fit_intercept': [True, False],
    #              "normalize": [True, False]
    #              }
    #
    # estimator_lr = LinearRegression()
    # # lr_model = GridSearchCV(estimator=estimator_lr,
    # #                              param_grid=param_lr,
    # #                              n_jobs=-1,
    # #                              cv=5,
    # #                              verbose=0)
    #
    # estimator_lr.fit(X_train, y_train)
    # preds = estimator_lr.predict(X_test)
    # rmse = np.sqrt(mean_squared_error(y_test, preds))
    # save_model(estimator_lr, 'lr', rmse)
    # print('LinearRegression RMSE:',rmse)

    ####################################################
    # print('Entrenando Lasso')
    #
    # estimator_lasso = Lasso(alpha=0.001, max_iter=100, tol=0.01)
    # # lr_model = GridSearchCV(estimator=estimator_lr,
    # #                              param_grid=param_lr,
    # #                              n_jobs=-1,
    # #                              cv=5,
    # #                              verbose=0)
    #
    # estimator_lasso.fit(X_train, y_train)
    # preds = estimator_lasso.predict(X_test)
    # rmse = np.sqrt(mean_squared_error(y_test, preds))
    # # save_model(estimator_lasso, 'lasso', rmse)
    # print('Lasso RMSE:',rmse)
    ####################################################

    # print('Training Random Forest model')
    # # param_rf = {'n_estimators': [10, 50, 100, 200],
    # #              "max_features": [0.6, 0.8],
    # #             "min_samples_leaf": [0.2, 0.3, 0.1],
    # #             "min_samples_split": [2, 3],
    # #             "random_state": [RANDOM_STATE],
    # #              }
    # #
    # # estimator_rf = RandomForestRegressor()
    # # rf_model = GridSearchCV(estimator=estimator_rf,
    # #                              param_grid=param_rf,
    # #                              n_jobs=-1,
    # #                              cv=5,
    # #                              verbose=1)
    #
    #
    # rf_model = RandomForestRegressor(n_estimators=1000,
    #                                  criterion='mse',
    #                                  max_depth=20,
    #                                  min_samples_split=2,
    #                                  min_samples_leaf=1,
    #                                  min_weight_fraction_leaf=0.0,
    #                                  max_features=50,
    #                                  max_leaf_nodes=None,
    #                                  min_impurity_decrease=0.0,
    #                                  min_impurity_split=None,
    #                                  bootstrap=True,
    #                                  oob_score=False,
    #                                  n_jobs=1, random_state=RANDOM_STATE, verbose=1, warm_start=False)
    # rf_model.fit(X_train, y_train)
    # preds = rf_model.predict(X_test)
    # rmse = np.sqrt(mean_squared_error(y_test, preds))
    # save_model(rf_model, 'rf', rmse)
    # print('RandomForest RMSE:',rmse)
    # ####################################################
    #
    # print('Training EslaticNetCV')
    # estimator_en = ElasticNetCV(l1_ratio=0.1,
    #                              n_jobs=-1,
    #                              eps=0.00005,
    #                              n_alphas=5000,
    #                              cv=5,
    #                              max_iter=50,
    #                              random_state=RANDOM_STATE,
    #                              )
    #
    # estimator_en.fit(X_train, y_train)
    # preds = estimator_en.predict(X_test)
    # rmse = np.sqrt(mean_squared_error(y_test, preds))
    # save_model(estimator_en, 'en', rmse)
    # print('ElasticNet RMSE:',rmse)
    # submit(estimator_en, 'EN')
    # ####################################################

    # print('Training ADABoost')
    # # param_adab = {'n_estimators': [10, 50,100],
    # #              "learning_rate": [1, 0.5, 0.01],
    # #             "random_state": [RANDOM_STATE],
    # #              }
    # #
    # # estimator_adab = AdaBoostRegressor()
    # # adab_model = GridSearchCV(estimator=estimator_adab,
    # #                              param_grid=param_adab,
    # #                              n_jobs=-1,
    # #                              cv=5,
    # #                              verbose=0)
    #
    # adab_model = AdaBoostRegressor(base_estimator=estimator_lasso, n_estimators=500,
    #                                learning_rate=1, loss='linear', random_state=RANDOM_STATE)
    # adab_model.fit(X_train, y_train)
    # preds = adab_model.predict(X_test)
    # rmse = np.sqrt(mean_squared_error(y_test, preds))
    # save_model(adab_model, 'adaboost', rmse)
    # print('AdaBoost RMSE:',rmse)

    # # ####################################################
    #
    # param_sgd = {
    # # 'max_iter': [20, 100, 200],
    #              "power_t": [0.1, 0.25, 0.5],
    #             "n_iter": [50, 60, 100],
    #              "eta0": [0.01, 0.5],
    #             "random_state": [RANDOM_STATE],
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
    # rmse = mean_squared_error(y_test, preds)
    # print('SGDRegressor RMSE:',rmse)
    #
    # # ####################################################
    #
    estimator_svr = SVR(kernel='rbf', degree=6, gamma='auto',
                        coef0=0.0, tol=0.001, C=1.0, epsilon=0.1,
                        shrinking=True, cache_size=200, verbose=False, max_iter=-1)

    estimator_svr.fit(X_train, y_train)
    preds = estimator_svr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    save_model(estimator_svr, 'svr', rmse)
    print('SVR RMSE:',rmse)

    #
    # # ####################################################
    #
    # param_knn = {'n_neighbors': [5, 10, 20, 100],
    #              "leaf_size": [20, 10, 30],
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
    # rmse = mean_squared_error(y_test, preds)
    # print('kNNRegressor RMSE:',rmse)
    #
    # # ####################################################
    #
    # param_nn = {'hidden_layer_sizes': [20, 50, 100, 500],
    #              "solver": ['sgd', 'adam'],
    #             "learning_rate": ['adaptive'],
    #             "learning_rate_init": [0.01, 0.5],
    #             "max_iter": [20, 50, 100],
    #             'random_state': [RANDOM_STATE]
    #              }
    #
    # estimator_nn = MLPRegressor()
    # nn_model = GridSearchCV(estimator=estimator_nn,
    #                              param_grid=param_nn,
    #                              n_jobs=-1,
    #                              cv=5,
    #                              verbose=0)
    #
    # nn_model.fit(X_train, y_train)
    # joblib.dump(nn_model, 'models/nn_model.pkl')
    # preds = nn_model.predict(X_test)
    # rmse = mean_squared_error(y_test, preds)
    # print('NeuralNetwork RMSE:',rmse)

    # # ####################################################
    # def baseline_model():
    #     # create model
    #     model = Sequential()
    #     model.add(Dense(32, input_dim=30, kernel_initializer='normal', activation='relu'))
    #     model.add(Dense(64, activation='relu'))
    #     model.add(Dense(128, activation='relu'))
    #     model.add(Dense(256, activation='relu'))
    #     model.add(Dense(1))
    #     # Compile model
    #     model.compile(loss='mean_squared_error', optimizer='rmsprop')
    #     return model
    #
    # estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=1, verbose=1)
    #
    # estimator.fit(X_train, y_train)
    # preds = estimator.predict(X_test)
    # rmse = mean_squared_error(y_test, preds)
    # print('Keras RMSE:',rmse)


    # # SUBMISSIONS

    # submit(estimator_lasso, 'Lasso')

    # x_plot = list(range(len(X_test)))
    # plt.plot(x_plot,preds, '--', x_plot, y_test, '--k')
    # plt.show()