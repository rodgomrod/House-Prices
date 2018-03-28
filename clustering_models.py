import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import os
from sklearn.externals import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


RANDOM_STATE = 1992

if not os.path.exists('kmeans'):
    os.makedirs('kmeans')

train = pd.read_csv("data/pro_train_v2.csv", encoding='utf-8', sep=',')

test = pd.read_csv("data/pro_test_v2.csv", encoding='utf-8', sep=',')


Y = train.loc[:, ['SalePrice']]
X_train = train.loc[:, train.columns != 'Id']
X_train = X_train.loc[:, X_train.columns != 'SalePrice']
X_test = test.loc[:, test.columns != 'Id']

n_clusters = [8, 16, 32, 64, 128, 256, 512]

for N in n_clusters:
    print('Init {} clusters'.format(N))
    parameter = {
        'n_init': [10, 15, 20],
    'max_iter': [200, 300,],
    'tol': [1e-2, 1e-4, 1e-3],
    }

    km_model = KMeans(n_clusters=N,
                      init='k-means++',
                      precompute_distances='auto',
                      n_jobs=1,
                      )

    km_cv = GridSearchCV(estimator= km_model,
                           param_grid=parameter,
                           n_jobs=-1,
                           cv = 7,
                           verbose=0,
                         scoring='neg_mean_squared_error')


    km_cv.fit(X_train, Y)

    joblib.dump(km_cv.best_estimator_,
                'kmeans/km_{}.pkl'.format(N))

    print('End {} clusters\n'.format(N))


for N in n_clusters:
    model = joblib.load('kmeans/km_{}.pkl'.format(N))
    train_preds = model.predict(X_train)
    print(X_train.columns.tolist())
    test_preds = model.predict(X_test)

    train['kn_{}'.format(N)] = pd.Series(train_preds)
    test['kn_{}'.format(N)] = pd.Series(test_preds)


train.to_csv('data/km_train_extra_v2.csv', sep=',', index=False)
test.to_csv('data/km_test_extra_v2.csv', sep=',', index=False)
