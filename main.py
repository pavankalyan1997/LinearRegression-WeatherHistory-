#importing data sets

#This file does two tasks
"""
1. Given humidity predict temperature
2. Given humidity predict apparent temperature
"""
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

#impoting dataset
dataset=pd.read_csv('weatherHistory.csv')
humidity=dataset.iloc[0:10000,5:6].values
temperature=dataset.iloc[0:10000:,3:4].values
apparentTemperature=dataset.iloc[0:10000:,4:5].values


#replace missing data with mean using imputer library
from sklearn.preprocessing import Imputer
imputerHumidity=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputerHumidity=imputerHumidity.fit(humidity)
humidity=imputerHumidity.transform(humidity)

imputerTemperature=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputerTemperature=imputerTemperature.fit(temperature)
temperature=imputerTemperature.transform(temperature)

imputerappTemp=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputerappTemp=imputerappTemp.fit(apparentTemperature)
apparentTemperature=imputerappTemp.transform(apparentTemperature)


#split the data into training and testing
from sklearn.cross_validation import train_test_split
humidity_train,humidity_test,temperature_train,temperature_test=train_test_split(humidity,temperature,test_size=0.3,random_state=0)
humidity_train,humidity_test,apptemp_train,apptemp_test=train_test_split(humidity,apparentTemperature,test_size=0.3,random_state=0)
humidity_train=humidity_train
humidity_test=humidity_test
temperature_train=temperature_train.flatten()
temperature_test=temperature_test.flatten()
apptemp_train=apptemp_train.flatten()
apptemp_test=apptemp_test.flatten()

#Task 1 To predict Temperature given Humidity
from LinearRegression import LinearRegression
lr1=LinearRegression(humidity_train,temperature_train)
print(lr1.computeCostFunction())
theta1=lr1.returnTheta()
theta1,cost_history1,theta_history1=lr1.performGradientDescent(10000,0.01)
temperature_predict,temperature_error=lr1.predict(humidity_test,temperature_test)
temperature_pred_normal,error_temp_normal=lr1.predictUsingNormalEquation(humidity_test,temperature_test)

plt.scatter(humidity_test,temperature_test)
plt.plot(humidity_test,temperature_pred_normal,'r')
plt.title('Humidity vs Temperature using normal equation')
plt.xlabel('Humidity')
plt.ylabel('Temperature')
plt.show()


plt.scatter(humidity_test,temperature_test)
plt.plot(humidity_test,temperature_predict,'r')
plt.title('Humidity vs Temperature using gradient descent')
plt.xlabel('Humidity')
plt.ylabel('Temperature')
plt.show()

lr2=LinearRegression(humidity_train,apptemp_train)
print(lr2.computeCostFunction())
theta2=lr2.returnTheta()
theta2,cost_history2,theta_history2=lr2.performGradientDescent(100000,0.01)
apptemp_predict,apptemp_error=lr2.predict(humidity_test,apptemp_test)
apptemp_pred_normal,error_apptemp_normal=lr2.predictUsingNormalEquation(humidity_test,apptemp_test)

plt.scatter(humidity_test,apptemp_test)
plt.plot(humidity_test,apptemp_pred_normal,'r')
plt.title('Humidity vs Apparent Temperature using normal equation')
plt.xlabel('Humidity')
plt.ylabel('Apparent Temperature')
plt.show()


plt.scatter(humidity_test,temperature_test)
plt.plot(humidity_test,temperature_predict,'r')
plt.title('Humidity vs Apparent Temperature using gradient descent')
plt.xlabel('Humidity')
plt.ylabel('Apparent Temperature')
plt.show()