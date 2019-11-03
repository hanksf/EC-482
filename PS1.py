#%%
#imports
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import tables
from scipy.io import loadmat
#%%
#Question 1 functions definitions
#holy shit the loops
def Sims_Uhlig(n_sims=10000, series_length=100, true_prior = [0.8, 1.1, 31],y_0=0):
    rho_grid = np.linspace(true_prior[0],true_prior[1],true_prior[2])
    epsilon = np.random.normal(size=(series_length,n_sims))
    estimates = np.zeros((rho_grid.size,n_sims))
    for i in range(rho_grid.size):
        y_matrix = np.zeros((series_length+1,n_sims))
        rho = rho_grid[i]
        y_matrix[0,:]=y_0
        y_matrix[1,:] = epsilon[0,:]
        for j in range(series_length-1):
            y_matrix[j+2,:] = rho*y_matrix[j+1,:]+epsilon[j+1,:]
        estimates[i,:] = np.diagonal(y_matrix[1:].T@y_matrix[:100])/np.diagonal(y_matrix[:100].T@y_matrix[:100])
    return estimates, rho_grid

def empirical_densities(n_sims=10000, series_length=100, true_prior = [0.8, 1.1, 31],y_0=0):
    draw_estimates, rho_grid = Sims_Uhlig(n_sims=n_sims, series_length=series_length, true_prior = true_prior,y_0=y_0)
    fig, axes = plt.subplots(2, 2, figsize=(10,15))
    #fig, axes = plt.subplots(2,2)
    axes[0,0].hist(draw_estimates[10,:], bins=20, density=True)
    axes[0,0].set(title='rho = 0.9')
    axes[1,0].hist(draw_estimates[20,:], bins =20, density=True)
    axes[1,0].set(title='rho = 1')
    #creating estimates of density conditional on rho_hat
    index1  = (draw_estimates>0.895)*(draw_estimates<0.905)
    density1 = np.sum(index1,axis=1)/np.sum(index1)
    axes[0,1].bar(rho_grid,density1,0.01)
    axes[0,1].set(title='rho_hat = 0.9')
    index2 = (draw_estimates>0.995)*(draw_estimates<1.005)
    density2 = np.sum(index2,axis=1)/np.sum(index2)
    axes[1,1].bar(rho_grid,density2,0.01)
    axes[1,1].set(title='rho_hat = 1')
    plt.show()

#%%
#test = Sims_Uhlig()
#print(test)

#%%
#empirical_densities()

#%%
#Question 2 functions

def create_X(Data,lags):
    x = np.zeros((np.size(Data,0)-lags,np.size(Data,1)*lags+1))
    # set first column to 1
    x[:,0]=1
    for i in range(np.size(Data,0)-lags):
        
        x[i,1:]=np.flip(Data[i:(i+lags),:],0).flatten('C')
    X = np.kron(np.identity(np.size(Data,1)),x)
    return X

def Var(Data,lags):
    Y = Data[lags:,:].flatten('F')
    X = create_X(Data,lags)
    Beta = np.linalg.inv(X.T@X)@(X.T@Y)
    return np.reshape(Beta,(np.size(Data,1),1+np.size(Data,1)*lags))


def forecast(Data,lags, coef,periods_ahead):
    predictors = np.insert(np.flip(Data[(np.size(Data,0)-lags):,:],0).flatten('C'),0,1)
    for t in range(periods_ahead):
        y_forward = coef@predictors
        if t<periods_ahead-1:
            predictors = np.insert(predictors[:predictors.size-y_forward.size],1,y_forward)
    return y_forward


#%%
Matlab_file= loadmat('dataVARmedium.mat')
Dataset = Matlab_file['y']

test = np.array([[0,0],[1,2],[1.5,3],[2,1],[3,2],[1,4],[2,5],[2,2]])
values = Var(test,1)
print(values)
print(forecast(test,1,values,2))
print(values[1,0]+values[1,2]*(values[1,0]+values[1,1]*2+values[1,2]*2)+values[1,1]*(values[0,0]+values[0,1]*2+values[0,2]*2))



    


