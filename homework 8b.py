#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy as ast
from scipy.optimize import curve_fit


# In[19]:


data_table = Table.read('global_CCl4_MM.dat', format='ascii')
(data_table)


# In[20]:


df = data_table.to_pandas()

df = df.rename(columns={'CCl4ottoyr': 'year', 'CCl4ottoGLm': 'Global Mean', 'CCl4ottoGLsd': 'Global Standard Deviation'})
df


# In[21]:


plt.errorbar(df['year'], df['Global Mean'], yerr=df['Global Standard Deviation'], fmt='o')
plt.xlabel('year')
plt.ylabel('Global Mean Concentration')
plt.title('Global Mean Concentration Over Time with Error Bars')
plt.show()


# In[28]:


valid_indices = np.isfinite(df['Global Mean'])
x_data = df['year'][valid_indices]
y_data = df['Global Mean'][valid_indices]
params, covariance = curve_fit(linear_model, x_data, y_data)
fitted_line = linear_model(x_data, *params)
plt.scatter(df['year'], df['Global Mean'], label='Original Data')

plt.plot(x_data, fitted_line, color='red', label='Fitted Line')

plt.xlabel('Year')
plt.ylabel('Global Mean')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()


# In[30]:


residuals = df['Global Mean'] - fitted_line

N = len(df['year'])
k = 2 
chi_squared = np.sum((residuals / df['Global Mean'])**2)
red_chi_squared = chi_squared / (N - k)
red_chi_squared


# In[31]:


m, b = params
m_error, b_error = np.sqrt(np.diag(covariance))
print(f"Slope (m): {m:.2f} +/- {m_error:.2f}")
print(f"Intercept (b): {b:.2f} +/- {b_error:.2f}")

print(f"Final Equation: y = {m:.2f} * x + {b:.2f}")

print(f"Reduced Chi-Squared Value: {red_chi_squared:.2f}")


# In[32]:


plt.scatter(df['year'], residuals)
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()


# In[ ]:




