#%%
#imports
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import tables
from scipy.io import loadmat
import math as math
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
    fig, axes = plt.subplots(2, 2, figsize=(10,15), sharex=True)
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
empirical_densities()

#%%
#Question 2 functions

def create_x(Data,lags):
    x = np.zeros((np.size(Data,0)-lags,np.size(Data,1)*lags+1))
    # set first column to 1
    x[:,0]=1
    for i in range(np.size(Data,0)-lags):
        #flatten row by row of flipped data
        x[i,1:]=np.flip(Data[i:(i+lags),:],0).flatten('C')
    return x

def create_X(Data,lags):
    x = create_x(Data,lags)
    X = np.kron(np.identity(np.size(Data,1)),x)
    return X

def Var(Data,lags):
    #flattern column by column
    Y = Data[lags:,:].flatten('F')
    X = create_X(Data,lags)
    #OLS formula
    Beta = np.linalg.inv(X.T@X)@(X.T@Y)
    #reshape from beta to B
    return np.reshape(Beta,(np.size(Data,1),1+np.size(Data,1)*lags))


def forecast(Data,lags, coef,periods_ahead):
    #flattern row by row
    #then puts a one in the front for the constant
    predictors = np.hstack([np.ones(1),np.flip(Data[(np.size(Data,0)-lags):,:],0).flatten()])
    for t in range(periods_ahead):
        y_forward = coef@predictors
        if t<periods_ahead-1:
            predictors = np.insert(predictors[:predictors.size-y_forward.size],1,y_forward)
    return y_forward


def part_a(Data,lags,sample_end=64):
    quarter_1_gdp = np.zeros(np.size(Data,0)-sample_end)
    quarter_4_gdp = np.zeros(np.size(Data,0)-sample_end-3)
    quarter_1_infl = np.zeros(np.size(Data,0)-sample_end)
    quarter_4_infl = np.zeros(np.size(Data,0)-sample_end-3)
    for t in range(np.size(Data,0)-sample_end):
        sample = Data[:t+sample_end,:]
        coefficients = Var(sample,lags)
        forecast_1 = forecast(sample,lags,coefficients,1)
        quarter_1_gdp[t] = forecast_1[0]-Data[t+sample_end,0]
        quarter_1_infl[t] = forecast_1[1]-Data[t+sample_end,1]
        if t+3+sample_end<=199:
            forecast_4 = forecast(sample,lags,coefficients,4)
            quarter_4_gdp[t] = (forecast_4[0]-Data[t+3+sample_end,0])/4
            quarter_4_infl[t] = (forecast_4[1]-Data[t+3+sample_end,1])/4
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
    # the diagonal linspace is s
    Omega[1:,1:] = np.kron(np.diag(np.float_power(np.linspace(1,lags,num=lags),-2)),np.diag(AR_1(Data)**(-1)))*lambd**2
    return np.reshape(b,(np.size(Data,1),1+np.size(Data,1)*lags)).T, Omega

def b_Var(Data,lags,b, Omega):
    x = create_x(Data,lags)
    B = (np.linalg.inv(x.T@x+np.linalg.inv(Omega))@(x.T@Data[lags:,:]+np.linalg.inv(Omega)@b)).T
    return B




def part_b(Data,lags,sample_end=64,lambd=0.2):
    quarter_1_gdp = np.zeros(np.size(Data,0)-sample_end)
    quarter_4_gdp = np.zeros(np.size(Data,0)-sample_end-3)
    quarter_1_infl = np.zeros(np.size(Data,0)-sample_end)
    quarter_4_infl = np.zeros(np.size(Data,0)-sample_end-3)
    for t in range(np.size(Data,0)-sample_end):
        sample = Data[:t+sample_end,:]
        b, Omega = minnesota_prior(Data,lags, lambd)
        coefficients = b_Var(sample,lags, b, Omega)
        forecast_1 = forecast(sample,lags,coefficients,1)
        quarter_1_gdp[t] = forecast_1[0]-Data[t+sample_end,0]
        quarter_1_infl[t] = forecast_1[1]-Data[t+sample_end,1]
        if t+3+sample_end<=199:
            forecast_4 = forecast(sample,lags,coefficients,4)
            quarter_4_gdp[t] = (forecast_4[0]-Data[t+3+sample_end,0])/4
            quarter_4_infl[t] = (forecast_4[1]-Data[t+3+sample_end,1])/4
    MSFE_gdp_1 = np.sum(quarter_1_gdp**2)/np.size(quarter_1_gdp)
    MSFE_gdp_4 = np.sum(quarter_4_gdp**2)/np.size(quarter_4_gdp)  
    MSFE_infl_1 = np.sum(quarter_1_infl**2)/np.size(quarter_1_infl)  
    MSFE_infl_4 = np.sum(quarter_4_infl**2)/np.size(quarter_4_infl) 
    return MSFE_gdp_1, MSFE_gdp_4, MSFE_infl_1, MSFE_infl_4   

def errors(Data,lags,coef):
    X = create_X(Data,lags)
    Y = Data[lags:,:].flatten('F')
    epsilon_hat = Y-X@(coef.flatten())
    epsilon_hat = np.reshape(epsilon_hat,(np.size(Data,0)-lags,np.size(Data,1)))
    return epsilon_hat.T@epsilon_hat

def optimal_lambda(Data, lags, n, T, d, grid_points=50):
    lambda_grid = np.geomspace(0.03,2,num=grid_points)
    postierior_lambda = np.zeros(grid_points)
    for j in range(grid_points):
        lambd = lambda_grid[j]
        b, Omega = minnesota_prior(Data,lags, lambd)
        coefficients = b_Var(Data,lags, b, Omega)
        error = errors(Data, lags,coefficients)
        phi = np.diag(AR_1(Data))
        x = create_x(Data,lags)
        # test1 = np.log(np.linalg.det(x.T@x+np.linalg.inv(Omega))**(-n/2))
        # test3 = np.log(np.linalg.det(Omega)**(-n/2))
        # test2 = np.log(np.linalg.det(phi+error+((coefficients-b.T))@np.linalg.inv(Omega)@((coefficients-b.T).T))**(-(T+d)/2))
        postierior_lambda[j] = (-n/2)*np.log(np.linalg.det(Omega)) + (-n/2)*np.log(np.linalg.det(x.T@x+np.linalg.inv(Omega)))+ np.log(np.linalg.det(phi+error+((coefficients-b.T))@np.linalg.inv(Omega)@((coefficients-b.T).T)))*(-(T+d)/2)
    opt_lambda = lambda_grid[np.argmax(postierior_lambda)]
    return opt_lambda

def part_c(Data,lags,sample_end=64):
    quarter_1_gdp = np.zeros(np.size(Data,0)-sample_end)
    quarter_4_gdp = np.zeros(np.size(Data,0)-sample_end-3)
    quarter_1_infl = np.zeros(np.size(Data,0)-sample_end)
    quarter_4_infl = np.zeros(np.size(Data,0)-sample_end-3)
    lambda_path = np.zeros(np.size(Data,0)-sample_end)
    for t in range(np.size(Data,0)-sample_end):
        sample = Data[:t+sample_end,:]
        lambd = optimal_lambda(sample, lags, np.size(sample,1), np.size(sample,0)-lags,np.size(sample,1)+2)
        lambda_path[t] = lambd
        b, Omega = minnesota_prior(sample,lags, lambd)
        coefficients = b_Var(sample,lags, b, Omega)
        forecast_1 = forecast(sample,lags,coefficients,1)
        quarter_1_gdp[t] = forecast_1[0]-Data[t+sample_end,0]
        quarter_1_infl[t] = forecast_1[1]-Data[t+sample_end,1]
        if t+3+sample_end<=199:
            forecast_4 = forecast(sample,lags,coefficients,4)
            quarter_4_gdp[t] = (forecast_4[0]-Data[t+3+60,0])/4
            quarter_4_infl[t] = (forecast_4[1]-Data[t+3+60,1])/4
        MSFE_gdp_1 = np.sum(quarter_1_gdp**2)/np.size(quarter_1_gdp)
    MSFE_gdp_4 = np.sum(quarter_4_gdp**2)/np.size(quarter_4_gdp)  
    MSFE_infl_1 = np.sum(quarter_1_infl**2)/np.size(quarter_1_infl)  
    MSFE_infl_4 = np.sum(quarter_4_infl**2)/np.size(quarter_4_infl) 
    return MSFE_gdp_1, MSFE_gdp_4, MSFE_infl_1, MSFE_infl_4, lambda_path   

#%%


#%%
Matlab_file = loadmat('dataVARmedium.mat')
Dataset = Matlab_file['y']






print(part_a(Dataset,5))
print(part_b(Dataset,5))
MSFE1, MSFE2, MSFE3, MSFE4, path =part_c(Dataset,5)
print(MSFE1, MSFE2, MSFE3, MSFE4)
fig, ax = plt.subplots()
ax.plot(path, 'b-', label='Optimal lambda', linewidth=2,alpha=0.6)
ax.set_title('Optimal lambda over time')
ax.set(xlabel='', ylabel='lambda')
plt.show()

