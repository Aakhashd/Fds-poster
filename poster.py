#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:16:07 2023

@author: aakhashd
"""

from ipywidgets import interact,widgets # for interactive visualization
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Import ploting Libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit


def read_csv(filename):
    # Load the dataset
    data = pd.read_csv(filename)
    return data,data.T                   
data, data_transposed = read_csv("Ass3data.csv")
data1=data
# Check the number of Countries we have:
print("Number of Unique Countries:",len(data["Country_Name"].unique()))
print("Number of Unique Years:", len(data["Year"].unique()))

num_vars = data.select_dtypes(exclude="O").columns.to_list()
print("Numerical variables:", num_vars)
# GDP data

def plot_yearly_gdp(country_name):
    sns.set_style('darkgrid')
    plt.figure(figsize=(15,5))
    sns.lineplot(data=data[data.Country_Name == country_name], y='GDP',x='Year')
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
        data.groupby("Country_Name").mean()[var].sort_values(ascending =False).plot(kind='bar')
        plt.title(f"{var} for last 10 Years")
    c=c+1
plt.tight_layout()
plt.show()
#clustering
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
    plt.show()
def Gaussian_fun(x, a, b):
    y_res = a*np.exp(-1*b*x**2)
    return y_res
def func(x, a, b):
    return a*np.exp(b*x)
# Define the error_range function
def err_ranges(params, cov, x):
    return np.sqrt(np.diag(cov))*np.abs(x - params)
def curvefit(data):
    x_data=data['GDP']
    y_data=data['Year']
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    plt.plot(y_data, x_data, 'o')
    plt.legend()
    plt.xlabel('GDP')
    plt.ylabel('Years')
    plt.show()
    

def curvefit1(data):
    x_data=data['GDP']
    y_data=data['life_expectancy']
    params, cov = curve_fit(Gaussian_fun, x_data, y_data)
    fitA = params[0]
    fitB = params[1]
    fity = Gaussian_fun(x_data, fitA, fitB)
    x_pred = np.linspace(2022, 2030, 2)
    y_pred = func(x_pred, *params)
    # Compute the lower and upper limits of the confidence range
    x_err = err_ranges(params, cov, x_pred)
    y_err = err_ranges(params, cov, y_pred)
    y_upper = y_pred + y_err
    y_lower = y_pred - y_err
    #Plot the results
    plt.scatter(data['life_expectancy'], data['GDP'], c='blue')
    plt.plot(x_pred, y_pred, '-r')
    plt.fill_between(x_pred, y_lower, y_upper, color='gray', alpha=0.2)  
    plt.xlabel('imports_percentGDP')
    plt.ylabel('exports_percentGDP')
    plt.show()
    plt.plot(x_data, y_data, '*', label='data')
    plt.plot(x_data, fity, '-', label='fit')
    plt.legend()
clusterring(data1)
curvefit(data1)
curvefit1(data1)
