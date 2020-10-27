# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:38:04 2020

@author: srinivasan.c
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

dataset = pd.read_csv('data/Market_Basket_Optimisation.csv', header = None)
transaction = []

for i in range(0, 7501):
    transaction.append([str(dataset.values[i,j]) for j in range(0,20)])
    
from apyori import apriori
rules = apriori(transaction, min_support= .003, min_confidence=1/5, min_lift=3, min_length=2)

res = list(rules)