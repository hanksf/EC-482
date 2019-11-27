## imports
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import PS2_func as P2
import pandas as pd

#%%
## Question 1
## columns are ouput, prices, unemployment, Commodity price, M2, interest rate the rest don't matter
df = pd.read_excel(r'SZdata.xlsx')
df = df.drop('date', axis = 1)
Data = df.to_numpy()
Data = Data[:,1:7]

P2.part_1(Data,13,0.2,1)
