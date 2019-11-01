#%%
#imports
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

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
    return estimates

def empirical_densities(n_sims=10000, series_length=100, true_prior = [0.8, 1.1, 31],y_0=0):
    estimates = Sims_Uhlig(n_sims=n_sims, series_length=series_length, true_prior = true_prior,y_0=y_0)
    fig, axes = plt.subplots(plot_values.size, 1,figsize)
    #fig, axes = plt.subplots(plot_values.size, plot_values.size,figsize)
    axes[0,0].hist(estimates[10,:])
    axes[0,0].set(title='rho = 0.9')
    axes[1,0].hist(estimates[20,:])
    axes[1,0].set(title='rho = 1')
    plt.plot()

#%%
#test = Sims_Uhlig()
#print(test)

#%%
empirical_densities()

# %%
