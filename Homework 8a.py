#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# In[14]:


df = pd.read_csv("GlobalLandTemperaturesByState.csv")

df = df[['dt', 'AverageTemperature', 'State']]

df['dt'] = pd.to_datetime(df['dt'])

df = df[df['dt'].dt.year > 2000]

df = df[df['State'].isin(['Wyoming', 'Nebraska', 'South Dakota'])]

print(df.shape)
print(df)


# In[15]:


df_avg_temp = df.groupby('dt')['AverageTemperature'].mean().reset_index()

df_avg_temp.columns = ['date', 'average_temperature']

print(df_avg_temp)


# In[16]:


plt.figure(figsize=(10, 6))
plt.plot(df_avg_temp['date'], df_avg_temp['average_temperature'], color='blue')

plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.title('Average Temperature Across Three States Over Time')

plt.xticks(rotation=45)

plt.grid(True)
plt.tight_layout()
plt.show()


# In[17]:


df_avg_temp['date'] = pd.to_datetime(df_avg_temp['date'])

df_avg_temp['year'] = df_avg_temp['date'].dt.year
df_avg_temp['month'] = df_avg_temp['date'].dt.month
df_avg_temp['day'] = df_avg_temp['date'].dt.day

df_avg_temp['numerical_date'] = df_avg_temp['year'] * 10000 + df_avg_temp['month'] * 100 + df_avg_temp['day']

print(df_avg_temp.head())


# In[18]:


def model(x, A, B, C, D):
    return A * np.sin(B * x + C) + D
range_of_data = 21.449667
mean_or_median_of_data = 0.590333
initial_guess = [range_of_data / 2, 2 * np.pi / np.pi, 0, mean_or_median_of_data]
print(initial_guess)


# In[19]:


x_data = df_avg_temp['numerical_date']
y_data = df_avg_temp['average_temperature']

def model(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

initial_guess = [range_of_data / 2, 2 * np.pi / np.pi, 0, mean_or_median_of_data]
params, covariance = curve_fit(model, x_data, y_data, p0=initial_guess)
fitted_curve = model(x_data, *params)

plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'bo', label='Original Data')
plt.plot(x_data, fitted_curve, 'r-', label='Fitted Curve')
plt.xlabel('Numerical Date')
plt.ylabel('Average Temperature')
plt.title('Fitting Sinusoidal Model to Data')
plt.legend()
plt.grid(True)
plt.show()

print("Parameters:", params)
print("Covariance Matrix:", covariance)


# In[20]:


x_data = df_avg_temp['numerical_date']
y_data = df_avg_temp['average_temperature']

def model(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

initial_guess = [range_of_data / 2, 2 * np.pi / np.pi, 0, mean_or_median_of_data]
params, covariance = curve_fit(model, x_data, y_data, p0=initial_guess)
fitted_curve = model(x_data, *params)

plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'bo', label='Original Data')
plt.plot(x_data, fitted_curve, 'r-', label='Fitted Curve')
plt.xlabel('Numerical Date')
plt.ylabel('Average Temperature')
plt.title('Fitting Sinusoidal Model to Data')
plt.legend()
plt.grid(True)
plt.show()


# In[21]:


errors = np.sqrt(np.diag(covariance))
for i, error in enumerate(errors):
    print(f"Parameter {i+1}'s error: {error}")


# In[22]:


for i, (param, error) in enumerate(zip(params, errors)):
    print(f"Parameter {i+1}: {param:.2f} +/- {error:.2f}")
print(f"Final Equation: y = {params[0]:.2f} * sin({params[1]:.2f} * x + {params[2]:.2f}) + {params[3]:.2f}")


# In[ ]:




