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


# Load the dataset
data = pd.read_csv('Ass3dataset.csv')
# print the dataset
print("We have 150 Rows and 12 Features")
data.head()


# Check the number of Countries we have:
print("Number of Unique Countries:",len(data["Country_Name"].unique()))
print("Number of Unique Years:", len(data["Year"].unique()))

num_vars = data.select_dtypes(exclude="O").columns.to_list()
print("Numerical variables:", num_vars)
# GDP data
country = widgets.Dropdown(
                    options=data.Country_Name.unique(),
                    value='India',
                    description='Number:',
                    disabled=False,
                )
@interact(country_name=country)
def plot_yearly_gdp(country_name):
    
    sns.set_style('darkgrid')
    plt.figure(figsize=(15,5))
    sns.lineplot(data=data[data.Country_Name == country_name], y='GDP',x='Year')
    plt.title(f"{country_name}: Yearly GDP")
    plt.xticks(data['Year'].unique())
    plt.show()

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