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

RANDOM_STATE = 1992

train = pd.read_csv("data/train_ft_eng.csv", encoding='utf-8', sep=',')

test = pd.read_csv("data/test_ft_eng.csv", encoding='utf-8', sep=',')

X_train, X_test, y_train, y_test = train_test_split(train.loc[:,train.columns != 'SalePrice'],
                                                    train['SalePrice'],
                                                    test_size=0.3,
                                                    random_state=RANDOM_STATE)

X_train = X_train.loc[:,X_train.columns != 'Id']
X_test = X_test.loc[:,X_test.columns != 'Id']
X_test = X_test.loc[:,X_train.columns]



param = {'objective': ['reg:linear'],  # Specify multiclass classification
             # 'num_class': 12,  # Number of possible output classes
             # 'tree_method': 'gpu_hist',  # Use GPU accelerated algorithm
             # 'verbose': [False],
             # 'gpu_id': [0],
             # 'silent': [1],
             # 'predictor': 'gpu_predictor',
             # 'n_jobs': -1,
             "learning_rate": [0.05, 0.1],
             "max_depth": [20, 10, 5],
             "n_estimators": [300, 400, 600],
             'subsample': [0.6, 0.8, 0.5],
             "colsample_bytree": [0.8],
             "colsample_bylevel": [0.8],
             # 'gamma': 0.2,
             }

# gpu_res = {}  # Store accuracy result
tmp = time.time()
# Train model
# xgboost_model = XGBRegressor(**param)
estimator = XGBRegressor()
xgboost_model = GridSearchCV(estimator=estimator,
                             param_grid=param,
                             n_jobs=-1,
                             cv=5,
                             verbose=0)
####################################################
# xgboost_model = LinearRegression()
# xgboost_model = RandomForestRegressor(n_estimators=2000,
#                                       criterion='mse',
#                                       max_depth=10,
#                                     min_samples_split=5,
#                                     min_samples_leaf=1,
#                                     min_weight_fraction_leaf=0.0,
#                                     max_features='auto',
#                                       max_leaf_nodes=10,
#                                       n_jobs=-1,
#                                       random_state=RANDOM_STATE,
#                                       verbose=1
#                                       )

# xgboost_model = ElasticNetCV(l1_ratio=0.01,
#                              n_jobs=-1,
#                              eps=0.0005,
#                              n_alphas=2000,
#                              cv=5,
#                              max_iter=200,
#                              random_state=RANDOM_STATE,
#                              )
##################################################
xgboost_model.fit(X_train, y_train)
print("Training Time: %s seconds" % (str(time.time() - tmp)))

preds = xgboost_model.predict(X_test)

rmse = mean_squared_error(y_test, preds)
print(rmse)

preds_test = xgboost_model.predict(test.loc[:,X_train.columns])
preds_test = 10 ** preds_test

submiss = pd.DataFrame({'Id': test['Id'], 'SalePrice': preds_test})
submiss.to_csv('submissions/rmse_{}.csv'.format(round(rmse, 7)), index=None)

# x_plot = list(range(len(X_test)))
# plt.plot(x_plot,preds, '--', x_plot, y_test, '--k')
# plt.show()