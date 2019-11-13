## imports
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import PS2_func
import pandas as pd


## Question 1
## columns are ouput, prices, unemployment, Pcom, M2, interest rate
df = pd.read_excel(r'SZdata.xlsx')
Data = df.as_matrix()
print(Data)
