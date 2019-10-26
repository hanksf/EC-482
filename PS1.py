#%%
#imports
import numpy as np
import scipy as sci

#%%
#Question 1 functions definitions
def Sims_Uhlig(n_sims=10000, series_length=100, true_prior = [0.8, 1.1, 31],y_0=0,sig_epsilon):
    rho_grid = np.linspace(true_prior[0],true_prior[1],true_prior[2])
    epsilon = np.random.normal(size=(n_sims,series_length))
    