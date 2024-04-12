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
