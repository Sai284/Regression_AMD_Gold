
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import math
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from scipy import stats
from scipy.stats import kurtosis, skew

path = r"/Users/saipolukonda/Documents/Projects/LinearRegressionModel/AMD_GOLD.xlsx"
price_data = pd.read_excel(path)

new_column_names = {'AMD_Close':'AMD_price', 'GC=F_Close':'Gold_price'}

price_data = price_data.rename(columns=new_column_names)

price_data = price_data.dropna()
price_data = price_data.drop(columns=['Date'])
price_data.isna().any()

#Scatter Graph
x = price_data['AMD_price']
y = price_data['Gold_price']

plt.plot(x,y,'o',color='cadetblue', label='Daily Price')

plt.title("AMD Vs Gold")
plt.xlabel("AMD")
plt.ylabel("Gold")
plt.legend()

plt.show()

print(price_data.corr())
print(price_data.describe())

#Histogram

price_data.hist(grid=False, color='cadetblue')
plt.show()

amd_kurtosis = kurtosis(price_data['AMD_price'], fisher = True)
gold_kurtosis = kurtosis(price_data['Gold_price'], fisher = True)

amd_skew = skew(price_data['AMD_price'])
gold_skew = skew(price_data['Gold_price'])

print("AMD Excess Kurtosis: {:.2}".format(amd_kurtosis))
print("Gold Excess Kurtosis: {:.2}".format(gold_kurtosis))

print("AMD Skew: {:.2}".format(amd_skew))
print("Gold Skew: {:.2}".format(gold_skew))

#kurtosis test
print('AMD')
print(stats.kurtosis(price_data['AMD_price']))
print('Gold')
print(stats.kurtosis(price_data['Gold_price']))

#skew test
print('AMD')
print(stats.skew(price_data['AMD_price']))
print('Gold')
print(stats.skew(price_data['Gold_price']))

y = price_data.drop('Gold_price', axis = 1)
x = price_data[['Gold_price']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state =1)

regression_model = LinearRegression()

regression_model.fit(x_train, y_train)

intercept = regression_model.intercept_[0]
coefficient = regression_model.coef_[0][0]

print("Intercept: {:.4}".format(intercept))
print("Coefficient: {:.2}".format(coefficient))

prediction = regression_model.predict([[1900]])
predicted_value = prediction[0][0]
print("Predicted value: {:.4}".format(predicted_value))


y_predict = regression_model.predict(x_test)
print(y_predict[:5])

x2 = sm.add_constant(x)
model = sm.OLS(y,x2)
est = model.fit()

#confidence intervals
est.conf_int()

#hypothesis testing
print(est.pvalues)

#Mean squared error (MSE)
model_mse = mean_squared_error(y_test, y_predict)

#Mean absolute Error (MAE)
model_mae = mean_absolute_error(y_test, y_predict)

#Root Mean Squared Error (RMSE)
model_rmse = math.sqrt(model_mse)

print("MSE: {:.3}".format(model_mse))
print("MAE: {:.3}".format(model_mae))
print("RMSE: {:.3}".format(model_rmse))

#R-Squared

model_r2 = r2_score(y_test, y_predict)
print("R2: {:.2}".format(model_r2))

print(est.summary())

(y_test - y_predict).hist(grid = False, color = 'royalblue')
plt.title("Model Residuals")
plt.show()

#Plotting the Line

plt.scatter(x_test, y_test, color = 'gainsboro', label = 'Price')
plt.plot(x_test, y_predict, color='royalblue', linewidth=3, linestyle='-', label = 'Regression Line')

plt.title("Linear Regression AMD Vs Gold")
plt.xlabel("Gold")
plt.ylabel("AMD")
plt.legend()
plt.show()

# The coefficients
print('Gold coefficient:' + '\033[1m' + '{:.2}''\033[0m'.format(regression_model.coef_[0][0]))

# The mean squared error
print('Mean squared error: ' + '\033[1m' + '{:.4}''\033[0m'.format(model_mse))

# The mean squared error
print('Root Mean squared error: ' + '\033[1m' + '{:.4}''\033[0m'.format(math.sqrt(model_mse)))

# Explained variance score
print('R2 score: '+ '\033[1m' + '{:.2}''\033[0m'.format(r2_score(y_test,y_predict)))

with open('my_linear_regression.sav', 'wb') as f:
    pickle.dump(regression_model, f)

# load it back in.
with open('my_linear_regression.sav', 'rb') as pickle_file:
    regression_model_3 = pickle.load(pickle_file)

# make a new prediction.
regression_model_3.predict([[1900]])