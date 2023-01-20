#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:16:07 2023

@author: aakhashd
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Import ploting Libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import itertools as iter

# Reading the Csv file
def read_csv(filename):
    # Load the dataset
    data = pd.read_csv(filename)
    return data,data.T    
               
data, data_transposed = read_csv("Ass3data.csv")
data1=data
num_vars = data.select_dtypes(exclude="O").columns.to_list()
data_2=data.drop(['Country_Name'], axis = 1)
# GDP data
def plot_yearly_gdp(country_name):
    sns.set_style('darkgrid')
    plt.figure(figsize=(15,5))
    sns.lineplot(data=data[data.Country_Name == country_name],
                 y='GDP',x='Year')
    plt.title(f"{country_name}: Yearly GDP")
    plt.xticks(data['Year'].unique())
    plt.show()
plot_yearly_gdp('Canada')
#Bilatral Analysis
sns.pairplot(data, hue='Country_Name')
plt.figure(figsize=(15,40))
nrows = len(num_vars)
ncols = 1
c = 1 
for var in num_vars:
    if var != 'Year':
        plt.subplot(nrows,ncols, c)
        data.groupby("Country_Name").mean()[var].sort_values(
            ascending =False).plot(kind='bar')
        plt.title(f"{var} for last 10 Years")
    c=c+1
plt.tight_layout()
plt.show()

#K means clustering
def clusterring(data):
    # Normalize the data using GDP per capita
    data['gdp_per_capita'] = data['GDP'] / data['total_population']
    # Perform clustering using KMeans
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data[['gdp_per_capita']])
    # Predict the cluster for each country
    data['cluster'] = kmeans.predict(data[['gdp_per_capita']])
    # Plot the cluster membership and cluster centers
    plt.scatter(data['gdp_per_capita'], data['cluster'], c=data['cluster'])
    plt.scatter(kmeans.cluster_centers_[:, 0], [0, 1, 2], c='red', marker='x')
    plt.xlabel('GDP per capita')
    plt.ylabel('Cluster')
    plt.legend()
    plt.show()
    x_data=data['GDP']
    y_data=data['Year']
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    plt.plot(x_data, y_data, 'o')
    plt.title("GDP of years")
    plt.xlabel("Year")
    plt.ylabel("GDP")
    plt.show()
    

'''function to calculate the error limits'''
def func(x,a,b,c):
    return a * np.exp(-(x-b)**2 / c)

'''adding an exponential function'''
def expoFunc(x,a,b):
    return a**(x+b)

"""
Calculates the upper and lower limits for the function, parameters and
sigmas for single value or array x. Functions values are calculated for 
all combinations of +/- sigma and the minimum and maximum is determined.
Can be used for all number of parameters and sigmas >=1.
This routine can be used in assignment programs.
"""  
def err_ranges(x, func, param, sigma):
  
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))        
    pmix = list(iter.product(*uplow))    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)        
    return lower, upper

#Curve fitting
def curvefit(data_2):
    '''plot for scattering'''
    plt.scatter(data_2['exports_percentGDP'],data_2['imports_percentGDP'])
    plt.title('Scatter plot before curve fitting')
    plt.ylabel('Import in GDP')
    plt.xlabel('export in GDP')
    plt.show()
    x_data = data_2['exports_percentGDP']
    y_data = data_2['imports_percentGDP']
    #curve fitting for export and import in GDP
    popt, pcov = curve_fit(expoFunc,data_2['exports_percentGDP'],
                           data_2['imports_percentGDP'],p0=[1,0],
                           maxfev=500000)
    a_opt, b_opt = popt
    x_mod = np.linspace(min(x_data),max(x_data),100)
    y_mod = expoFunc(x_mod,a_opt,b_opt)
    '''plot for scattering after fitting the curve'''
    plt.scatter(x_data,y_data)
    plt.plot(x_mod,y_mod,color = 'r')
    plt.title('Scatter plot after curve fitting')
    plt.ylabel('Import in GDP')
    plt.xlabel('Export in GDP')
    plt.legend()
    plt.savefig("curvefit_after.png")
    plt.show()

#Clustering of data
clusterring(data1)
curvefit(data1)
