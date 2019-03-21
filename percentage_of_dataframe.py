#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 22:30:24 2019

@author: rainfall
"""
import numpy as np
import pandas as pd
import copy
import os
from collections import Counter


path = '/media/DATA/tmp/datasets/brazil/brazil_qgis/csv/'
file = 'yrly_br_under_c1_over_c3c4.csv'
#for file in os.listdir(path):
print('reading file: ' + path + file)
df_rain = pd.read_csv(os.path.join(path, file), sep=',', decimal='.', encoding="utf8", header = 0)

# ----------------------------------------
# SUBSET BY SPECIFIC CLASS (UNDERSAMPLING)
# ----------------------------------------
n = 0.90
to_remove = np.random.choice(
        df_rain[df_rain['CLASSE']=='C1'].index,
        size=int(df_rain[df_rain['CLASSE']=='C1'].shape[0]*n),
        replace=False)

df_rain = df_rain.drop(to_remove)

# Saving the new output DB's (rain and no rain):
file_name = os.path.splitext(file)[0] + "_10pct.csv"
df_rain_2.to_csv(os.path.join(path, file_name), index=False, sep=",", decimal='.')
print("The file ", file_name, " was genetared!")
