#Organizing imports here--------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv('housing.csv')
data
data.dropna(inplace=True)
data.hist(figsize=(20,10))
plt.figure(figsize=(20,5))
sns.heatmap(data.corr(), annot=True, cmap="cividis")
data['total_rooms']=np.log(data['total_rooms'])
data.hist(figsize=(20,8))
data.ocean_proximity.value_counts()
pd.get_dummies(data.ocean_proximity)
data=data.join(pd.get_dummies(data.ocean_proximity)).drop(['ocean_proximity'],axis=1)
print(data)
print(data.corr())
plt.figure(figsize=(20,5))
sns.heatmap(data.corr(), annot=True, cmap="cividis")
data['bedroom_ratio']=data['total_bedrooms']/data['total_rooms']
data['household_rooms']=data['total_rooms']/data['households']
plt.figure(figsize=(20,8))
sns.heatmap(data.corr(), annot=True, cmap="cividis")
X=data.drop(['median_house_value'],axis=1)
y=data['median_house_value']
#-------------Linear_regression---------------- 
model = LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_predict))
r2 = metrics.r2_score(y_test, y_predict)
print(rmse)
print(r2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
