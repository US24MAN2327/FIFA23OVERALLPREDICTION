import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, StandardScaler

raw_data = pd.read_csv('FIFA23.csv')
frequency_table = raw_data.groupby(['Value(£)', 'Wage(£)']).size().reset_index(name='Frequency')

print(frequency_table)

# Plotting a histogram
plt.scatter(raw_data['Value(£)'], raw_data['Wage(£)'])
plt.xlabel('Value')
plt.ylabel('Wage')
plt.title('Scatter Plot of VALUE vs. WAGE')
plt.show()

plt.scatter(raw_data['Overall'], raw_data['Wage(£)'])
plt.xlabel('Overall')
plt.ylabel('Wage')
plt.title('Scatter Plot of OVERALL vs. WAGE')
plt.show()

pd.get_dummies(raw_data)


xtrain, xtest, ytrain, ytest = train_test_split(raw_data, raw_data['Wage(£)'], test_size=0.2, random_state=0)
print(xtrain.shape)
#plt.scatter(xtrain, ytrain,  color='blue')
#plt.xlabel('Value')
#plt.ylabel('Wage')
#plt.title('Scatter Plot of VALUE vs. WAGE')
#plt.show()



regress = linear_model.LinearRegression()
x = np.asanyarray(xtrain.values.reshape(-1, 1))
y = np.asanyarray(ytrain.values.reshape(-1, 1))
regress.fit(x,y)

print('Coefficients: ', regress.coef_)
print('Intercept: ', regress.intercept_)

y_ha = regress.coef_ * xtrain.values.reshape(-1, 1) + regress.intercept_

#plt.scatter(xtrain, ytrain, color='blue')
#plt.plot(xtrain, y_ha, color='red')
#plt.show()

y_pred = regress.predict(xtest.values.reshape(-1, 1))
print(y_pred[0:5])

print('Mean absolute error: %.2f' % np.mean(np.absolute(y_pred - ytest.values.reshape(-1, 1))))

print('Mean sum of squares (MSE): %.2f' % np.mean((y_pred - ytest.values.reshape(-1, 1)) ** 2))

print('R2-score: %.2f' % r2_score(y_pred, ytest.values.reshape(-1, 1)))









