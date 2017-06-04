
# coding: utf-8

# In[3]:

get_ipython().system('pip install seaborn')


# In[1]:

import sys
sys.path.append('C:/Users/Sam/Desktop/mlnn/')
from utils.bikeshare import download_bikeshare_data

download_bikeshare_data(2016, 1, 'C:/Users/Sam/Desktop/mlnn/data/')


# In[4]:

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd
bikes = pd.read_csv('C:/Users/Sam/Desktop/mlnn/data/2016-Q1-cabi-trip-history-data.csv')
bikes.head()
bikes['start'] = pd.to_datetime(bikes['Start date'], infer_datetime_format=True)
bikes['end'] = pd.to_datetime(bikes['End date'], infer_datetime_format=True)

bikes.head()


# In[5]:

bikes['hour_of_day'] = (bikes.start.dt.hour + (bikes.start.dt.minute/60).round(2))


# In[7]:

hours = bikes.groupby('hour_of_day').agg('count')
hours['hour'] = hours.index

hours.start.plot()
import seaborn as sns

sns.lmplot(x='hour', y='start', data=hours, aspect=1.5, scatter_kws={'alpha':0.2})


# In[ ]:

#1. Create 3 models fit to hour_of_day with varying polynomial degrees


# In[20]:

import numpy as np
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# In[28]:

x = hours[['hour']]
y = hours.start


# In[29]:

linear = linear_model.LinearRegression()

linear.fit(x, y)

linear.coef_, linear.intercept_


# In[33]:

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=15)

x_15 = poly.fit_transform(x)


# In[34]:

linear = linear_model.LinearRegression()

linear.fit(x_15, y)

(linear.coef_, linear.intercept_)


# In[35]:

#MODEL 1

plt.scatter(x,y)
plt.plot(x, np.dot(x_15, linear.coef_) + linear.intercept_, c='b')


# In[47]:

poly = PolynomialFeatures(degree=5)

x_5 = poly.fit_transform(x)


# In[48]:

linear = linear_model.LinearRegression()

linear.fit(x_5, y)

(linear.coef_, linear.intercept_)


# In[49]:

#MODEL 2

plt.scatter(x,y)
plt.plot(x, np.dot(x_5, linear.coef_) + linear.intercept_, c='b')


# In[43]:

poly = PolynomialFeatures(degree=25)

x_25 = poly.fit_transform(x)


# In[44]:

linear = linear_model.LinearRegression()

linear.fit(x_25, y)

(linear.coef_, linear.intercept_)


# In[45]:

#MODEL 3

plt.scatter(x,y)
plt.plot(x, np.dot(x_25, linear.coef_) + linear.intercept_, c='b')


# In[ ]:

#2. Choose one of the polynomial models and create 3 new models fit to hour_of_day with different 
#Ridge Regression $\alpha$ (alpha) Ridge Coefficient values

#Choosing Model 1 (x_15)


# In[51]:

poly = PolynomialFeatures(degree=15)

x_15 = poly.fit_transform(x)
linear = linear_model.LinearRegression()

linear.fit(x_15, y)

(linear.coef_, linear.intercept_)


# In[57]:

ridge = linear_model.Ridge(alpha=.5)

ridge.fit(x_15, y)

ridge.coef_, ridge.intercept_


# In[58]:

plt.scatter(x,y)
plt.plot(x, np.dot(x_15, ridge.coef_) + ridge.intercept_, c='r')


# In[59]:

ridge = linear_model.Ridge(alpha=20)

ridge.fit(x_15, y)

ridge.coef_, ridge.intercept_


# In[60]:

plt.scatter(x,y)
plt.plot(x, np.dot(x_15, ridge.coef_) + ridge.intercept_, c='r')


# In[61]:

ridge = linear_model.Ridge(alpha=3)

ridge.fit(x_15, y)

ridge.coef_, ridge.intercept_


# In[62]:

plt.scatter(x,y)
plt.plot(x, np.dot(x_15, ridge.coef_) + ridge.intercept_, c='r')


# In[63]:

plt.scatter(x,y)
plt.plot(x, np.dot(x_15, linear.coef_) + linear.intercept_, c='b')
plt.plot(x, np.dot(x_15, ridge.coef_) + ridge.intercept_, c='r')

