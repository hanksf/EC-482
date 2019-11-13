## imports
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import PS2_func
import pandas as pd


## Question 1
## columns are ouput, prices, unemployment, Commodity price, M2, interest rate the rest don't matter
df = pd.read_excel(r'SZdata.xlsx')
Data = df.as_matrix()
Data = Data[:,1:7]
print(Data)
