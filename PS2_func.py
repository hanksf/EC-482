##Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci


## Question 1

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

def initial_average(Data,lags):
    y_0 = np.mean(Data[:lags,:],axis=0)
    return y_0

def SoC_dummy(Data,lags,mu):
    """
    Create Y and X with the dummy observations included
    """
    #Creating Y
    bar_y0 = initial_average(Data,lags)
    dummy_obs = np.identiy(np.size(Data,0))@bar_y0/mu
    Y = np.vstack(Data[lags:,:],dummy_obs)
    #Creating X
    x = create_x(Data,lags)
    dummy_x = np.hstack(np.zeros((np.size(Data,1),1)),np.repeat(dummy_obs,lags))
    X = np.kron(np.identity(np.size(Data,1)+1),np.vstack(x,dummy_x))
    return Y,X, np.vstack(x,dummy_x)


def b_Var(Y,x,lags,b, Omega):
    B = (np.linalg.inv(x.T@x+np.linalg.inv(Omega))@(x.T@Y+np.linalg.inv(Omega)@b)).T
    return B

def S(Y,B,x)

def postior_mode_A0()

def part_1(Data,lags,lambd,mu):
    b , omega = minnesota_prior(Data,lags, lambd)
    Yp, Xp, xp = SoC_dummy(Data,lags, mu)
    B = b_Var(Yp,xp,lags,b,omega)


