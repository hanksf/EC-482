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
Data1 = Data[:,0:6]

B, a0, inv_hess = P2.part_1(Data1,13,0.2,1)

IRFS = P2.IRF(B,P2.a2A(a0),13,6,100)

fig1, axes1 = plt.subplots(2, 3, figsize=(10, 12))

axes1[0,0].plot(IRFS[0,:])

axes1[0,1].plot(IRFS[1,:])
axes1[0,2].plot(IRFS[2,:])
axes1[1,0].plot(IRFS[3,:])
axes1[1,1].plot(IRFS[4,:])
axes1[1,2].plot(IRFS[5,:])
axes1[0,0].set(title='Y')
axes1[0,1].set(title='Prices')
axes1[0,2].set(title='unemployment')
axes1[1,0].set(title='Commodity Prices')
axes1[1,1].set(title='Money Supply')
axes1[1,2].set(title='Interest Rate')

#%%
IRF_errors = P2.part_3(Data1,13,0.2,1,1000,0.65,100,6)

IRF_quantiles = np.zeros((6,100,2))
for i in range(6):
    for j in range(100):
        IRF_quantiles[i,j,0] = np.quantile(IRF_errors[i,j,:],0.05)
        IRF_quantiles[i,j,1] = np.quantile(IRF_errors[i,j,:],0.95)


fig, axes = plt.subplots(2, 3, figsize=(10, 12))

axes[0,0].plot(IRF_quantiles[0,:,0],linestyle = '--')
axes[0,0].plot(IRF_quantiles[0,:,1],linestyle = '--')
axes[0,0].plot(IRFS[0,:])


axes[0,1].plot(IRF_quantiles[1,:,0],linestyle = '--')
axes[0,1].plot(IRF_quantiles[1,:,1],linestyle = '--')
axes[0,1].plot(IRFS[1,:])

axes[0,2].plot(IRF_quantiles[2,:,0],linestyle = '--')
axes[0,2].plot(IRF_quantiles[2,:,1],linestyle = '--')
axes[0,2].plot(IRFS[2,:])

axes[1,0].plot(IRF_quantiles[3,:,0],linestyle = '--')
axes[1,0].plot(IRF_quantiles[3,:,1],linestyle = '--')
axes[1,0].plot(IRFS[3,:])

axes[1,1].plot(IRF_quantiles[4,:,0],linestyle = '--')
axes[1,1].plot(IRF_quantiles[4,:,1],linestyle = '--')
axes[1,1].plot(IRFS[4,:])

axes[1,2].plot(IRF_quantiles[5,:,0],linestyle = '--')
axes[1,2].plot(IRF_quantiles[5,:,1],linestyle = '--')
axes[1,2].plot(IRFS[5,:])
axes[0,0].set(title='Y')
axes[0,1].set(title='Prices')
axes[0,2].set(title='unemployment')
axes[1,0].set(title='Commodity Prices')
axes[1,1].set(title='Money Supply')
axes[1,2].set(title='Interest Rate')
#%%
#2.1
Data2 = np.hstack((Data[:,6].reshape(-1, 1),Data[:,1:6]))
B, a0, inv_hess = P2.part_1(Data2,13,0.2,1)
IRFS = P2.IRF(B,P2.a2A(a0),13,6,100)
IRF_errors = P2.part_3(Data2,13,0.2,1,1000,0.65,100,6)
IRF_quantiles = np.zeros((6,100,2))
for i in range(6):
    for j in range(100):
        IRF_quantiles[i,j,0] = np.quantile(IRF_errors[i,j,:],0.05)
        IRF_quantiles[i,j,1] = np.quantile(IRF_errors[i,j,:],0.95)

fig, axes = plt.subplots(2, 3, figsize=(10, 12))

axes[0,0].plot(IRF_quantiles[0,:,0],linestyle = '--')
axes[0,0].plot(IRF_quantiles[0,:,1],linestyle = '--')
axes[0,0].plot(IRFS[0,:])


axes[0,1].plot(IRF_quantiles[1,:,0],linestyle = '--')
axes[0,1].plot(IRF_quantiles[1,:,1],linestyle = '--')
axes[0,1].plot(IRFS[1,:])

axes[0,2].plot(IRF_quantiles[2,:,0],linestyle = '--')
axes[0,2].plot(IRF_quantiles[2,:,1],linestyle = '--')
axes[0,2].plot(IRFS[2,:])

axes[1,0].plot(IRF_quantiles[3,:,0],linestyle = '--')
axes[1,0].plot(IRF_quantiles[3,:,1],linestyle = '--')
axes[1,0].plot(IRFS[3,:])

axes[1,1].plot(IRF_quantiles[4,:,0],linestyle = '--')
axes[1,1].plot(IRF_quantiles[4,:,1],linestyle = '--')
axes[1,1].plot(IRFS[4,:])

axes[1,2].plot(IRF_quantiles[5,:,0],linestyle = '--')
axes[1,2].plot(IRF_quantiles[5,:,1],linestyle = '--')
axes[1,2].plot(IRFS[5,:])
axes[0,0].set(title='emp/pop')
axes[0,1].set(title='Prices')
axes[0,2].set(title='unemployment')
axes[1,0].set(title='Commodity Prices')
axes[1,1].set(title='Money Supply')
axes[1,2].set(title='Interest Rate')

# %%
