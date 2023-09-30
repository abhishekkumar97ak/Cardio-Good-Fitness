#!pip install matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
###############################################
data=pd.read_csv('CardioGoodFitness.csv',na_values=[0])

#EDA
data.head()
data.tail()
data.info()
data.describe()
data.columns
data.isna().sum()


sns.heatmap(data.corr(),annot=True)

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['Product']=encoder.fit_transform(data['Product'])
data['Gender']=encoder.fit_transform(data['Gender'])
data['MaritalStatus']=encoder.fit_transform(data['MaritalStatus'])

                                            
                                            
x=data.drop(['Miles'],axis=1)
y=data['Miles']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                        test_size=0.20,
                                        random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(x_train,y_train)
regressor.coef_
regressor.intercept_


y_pred=regressor.predict(x_test)
y_pred

from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test,y_pred))
metrics.r2_score(y_test, y_pred) 
metrics.mean_absolute_error(y_test,y_pred)

