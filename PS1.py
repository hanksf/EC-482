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
        #note the y_matrix has 101 elements so doing series_length cuts off the last element of the y_matrix
        estimates[i,:] = np.diagonal(y_matrix[1:].T@y_matrix[:series_length])/np.diagonal(y_matrix[:series_length].T@y_matrix[:series_length])
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

def create_x(Data,lags):
    x = np.zeros((np.size(Data,0)-lags,np.size(Data,1)*lags+1))
    # set first column to 1
    x[:,0]=1
    for i in range(np.size(Data,0)-lags):
        x[i,1:]=np.flip(Data[i:(i+lags),:],0).flatten('C')
    return x

def create_X(Data,lags):
    x = create_x(Data,lags)
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


def part_a(Data,lags,sample_end=60):
    quarter_1_gdp = np.zeros(np.size(Data,0)-sample_end-1)
    quarter_4_gdp = np.zeros(np.size(Data,0)-sample_end-5)
    quarter_1_infl = np.zeros(np.size(Data,0)-sample_end-1)
    quarter_4_infl = np.zeros(np.size(Data,0)-sample_end-5)
    for t in range(np.size(Data,0)-sample_end-2):
        sample = Data[:t+2+60,:]
        coefficients = Var(sample,lags)
        forecast_1 = forecast(sample,lags,coefficients,1)
        quarter_1_gdp[t] = forecast_1[0]-Data[t+2+60,0]
        quarter_1_infl[t] = forecast_1[1]-Data[t+2+60,1]
        if t+5+60<=199:
            forecast_4 = forecast(sample,lags,coefficients,4)
            quarter_4_gdp[t] = (forecast_4[0]-Data[t+5+60,0])/4
            quarter_4_infl[t] = (forecast_4[1]-Data[t+5+60,0])/4
    MSFE_gdp_1 = np.sum(quarter_1_gdp**2)/np.size(quarter_1_gdp)
    MSFE_gdp_4 = np.sum(quarter_4_gdp**2)/np.size(quarter_4_gdp)  
    MSFE_infl_1 = np.sum(quarter_1_infl**2)/np.size(quarter_1_infl)  
    MSFE_infl_4 = np.sum(quarter_4_infl**2)/np.size(quarter_4_infl) 
    return MSFE_gdp_1, MSFE_gdp_4, MSFE_infl_1, MSFE_infl_4

def AR_1(Data):
    Estimates = np.zeros(np.size(Data,1))
    for j in range(np.size(Data,1)):
        ar_coeff = ((Data[1:,j].T)@Data[:np.size(Data,0)-1,j])/((Data[:np.size(Data,0)-1,j].T)@Data[:np.size(Data,0)-1,j])
        errors = Data[:np.size(Data,0)-1,j] - ar_coeff* Data[1:,j]
        Estimates[j] = np.sum(errors**2)/(np.size(Data,0)-1)
    return Estimates
        

def minnesota_prior(Data,lags, lambd):
    #create b
    #note need to convert to 2D vector from 1D
    b = (np.vstack([np.zeros(np.size(Data,1)),np.identity(np.size(Data,1)),np.zeros(((lags-1)*np.size(Data,1),np.size(Data,1)))])).flatten('F')
    Omega = np.zeros((1+lags*np.size(Data,1),1+lags*np.size(Data,1)))
    Omega[0,0]= 10**6
    Omega[1:,1:] = np.kron(np.diag(np.linspace(1,lags,num=lags)),np.diag(AR_1(Data)**(-1)))*lambd**2
    return np.reshape(b,(np.size(Data,1),1+np.size(Data,1)*lags)), Omega

def b_Var(Data,lags,b, Omega):
    x = create_x(Data,lags)
    B = np.inv(x.T@x+np.inv(Omega))@(x.T@Data[lags:,:]+np.inv(Omega)@b)
    return B




#def part_b(Data,lags,sample_end=60):
            

#%%


#%%
Matlab_file = loadmat('dataVARmedium.mat')
Dataset = Matlab_file['y']

#print(minnesota_prior(Dataset, 5, 0.2))

print(part_a(Dataset,5))
    


