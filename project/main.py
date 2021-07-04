import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

# data loading

data = pd.read_csv('house_data.csv',
                   usecols=['Id', 'LotArea', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                            'GrLivArea', 'GarageArea',
                            'WoodDeckSF', 'OpenPorchSF', 'SalePrice'])
data.set_index('Id')

# print(data.head(10))

nndata = data.copy()
data.dropna(inplace=True)

data['MasVnrArea'] = pd.to_numeric(data['MasVnrArea'], errors='coerce')
data['MasVnrArea'] = data['MasVnrArea'].astype('int64')

print(data.dtypes)

# podela podataka na cenu i ostale info o nekretnini

infos = data[['LotArea', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea',
              'WoodDeckSF', 'OpenPorchSF']]
prices = data['SalePrice']

infos = infos.values
prices = prices.values

info_train, info_test, price_train, price_test = train_test_split(infos, prices, test_size=0.2, random_state=0)

# linear regression

linear_regression = LinearRegression()
linear_regression.fit(info_train, price_train)
lr_prediction = linear_regression.predict(info_test)

# decision tree

decision_tree = DecisionTreeRegressor()
decision_tree.fit(info_train, price_train)
tree_prediction = decision_tree.predict(info_test)

# random forest

random_forest = RandomForestRegressor(bootstrap=True, max_samples=300)
random_forest.fit(info_train, price_train)
forest_prediction = random_forest.predict(info_test)

# extra trees

extra_tree = ExtraTreesRegressor(bootstrap=True, max_samples=300)
extra_tree.fit(info_train, price_train)
extra_prediction = extra_tree.predict(info_test)

# neural network

infos = nndata[['LotArea', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea',
                'WoodDeckSF', 'OpenPorchSF']]
prices = nndata['SalePrice']

infos = infos.fillna(infos.mean())
info_train2, info_test2, price_train2, price_test2 = train_test_split(infos, prices, test_size=0.2, random_state=0)

model = tf.keras.Sequential()
model.add(layers.Dense(64, input_dim=10, activation='relu', kernel_initializer='normal'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, kernel_initializer='normal'))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(info_train2, price_train2, epochs=150)

predictions = model.predict(info_test2)

print('-------------------------------------------------------------------------------')
print('Average price deviation of LINEAR REGRESSION model is {}'.format(
    sum(price_test) / len(price_test) - sum(lr_prediction) / len(lr_prediction)))
print('R-Squared of LINEAR REGRESSION model is {}'.format(r2_score(price_test, lr_prediction)))

print('-------------------------------------------------------------------------------')
print('Average price deviation of DECISION TREE model is {}'.format(
    sum(price_test) / len(price_test) - sum(tree_prediction) / len(tree_prediction)))
print('R-Squared of DECISION TREE model is {}'.format(r2_score(price_test, tree_prediction)))

print('-------------------------------------------------------------------------------')
print('Average price deviation of RANDOM FOREST model is {}'.format(
    sum(price_test) / len(price_test) - sum(forest_prediction) / len(forest_prediction)))
print('R-Squared of RANDOM FOREST model is {}'.format(r2_score(price_test, forest_prediction)))

print('-------------------------------------------------------------------------------')
print(
    'Average price deviation of EXTRA TREE model is {}'.format(
        sum(price_test) / len(price_test) - sum(extra_prediction) / len(extra_prediction)))
print('R-Squared of EXTRA TREE model is {}'.format(r2_score(price_test, extra_prediction)))

print('-------------------------------------------------------------------------------')

print(
    'Average price deviation Score of NEURAL NETWORK model is {}'.format(
        sum(price_test) / len(price_test) - sum(predictions) / len(predictions)))
print('R-Squared of NEURAL NETWORK model is {}'.format(r2_score(price_test2, predictions)))
