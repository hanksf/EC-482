##Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


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
        y = Data[1:,j]
        x = Data[:np.size(Data,0)-1,j]
        ar_coeff = (x.T@y)/(x.T@x)
        #ar_coeff = ((Data[1:,j].T)@Data[:np.size(Data,0)-1,j])/((Data[:np.size(Data,0)-1,j].T)@Data[:np.size(Data,0)-1,j])
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
    dummy_obs = np.diag(bar_y0/mu)
    Y = np.vstack((Data[lags:,:],dummy_obs))
    #Creating X
    x = create_x(Data,lags)
    dummy_x = np.hstack((np.zeros((np.size(Data,1),1)),np.tile(dummy_obs,lags)))
    X = np.kron(np.identity(np.size(Data,1)+1),np.vstack((x,dummy_x)))
    return Y,X, np.vstack((x,dummy_x))


def b_Var(Y,x,lags,b, Omega):
    B = (np.linalg.inv(x.T@x+np.linalg.inv(Omega))@(x.T@Y+np.linalg.inv(Omega)@b)).T
    return B

def S(Y,B,x):
    residuals = Y-x@B.T
    residuals = residuals**2
    SoSR = np.sum(residuals)
    return SoSR

def a2A(a):
    A = np.array([[a[0],0,0,0,0,0],[a[1],a[2],0,0,0,0],[a[3],a[4],a[5],0,0,0],[a[6],a[7],a[8],a[9],a[10],a[11]],[a[12],a[13],0,0,a[14],a[15]],[0,0,0,0,a[16],a[17]]])
    return A



def postior_mode_A0(B,b,S,Omega,T,p,n):
    def log_post(a):
        A0 = a2A(a)
        if np.linalg.det(A0)<0:
            return 999999999999999999999999999999
        piece_1 = (T-p+n)*np.log(np.linalg.det(A0))
        piece_2 = -0.5*np.trace((S+((B.T-b).T)@np.linalg.inv(Omega)@(B.T-b))@(A0.T@A0))
        return -piece_1-piece_2
    result = optimize.minimize(log_post,np.array([5,1,5,1,1,5,1,1,1,5,1,1,1,1,5,1,1,5])*0.205)
    mode = result.x
    inv_hessian = result.hess_inv
    return mode,inv_hessian


def part_1(Data,lags,lambd,mu):
    b , omega = minnesota_prior(Data,lags, lambd)
    Yp, Xp, xp = SoC_dummy(Data,lags, mu)
    B = b_Var(Yp,xp,lags,b,omega)
    residuals = S(Yp,B,xp)
    mode, inv_hess = postior_mode_A0(B,b,residuals,omega,np.size(Data,0)-lags,lags,np.size(Data,1))
    return B, mode, inv_hess

def IRF(B, A0, lags, variable, length):
    IRF = np.zeros((np.size(B,axis=0),length))
    IRF[variable-1,0] = np.linalg.inv(A0)[variable-1,variable-1] 
    initial_Y = np.zeros((np.size(B,axis=0),lags))
    initial_Y[:,0] = np.linalg.inv(A0)[:,variable-1]
    predictors = np.hstack([np.ones(1),initial_Y.flatten('F')])
    predictors_ns = np.hstack([np.ones(1),np.zeros(initial_Y.size)])
    for t in range(length-1):
        y_forward = B@predictors
        y_forward_ns = B@predictors_ns
        IRF[:,t+1] = y_forward - y_forward_ns
        predictors = np.insert(predictors[:predictors.size-y_forward.size],1,y_forward)
        predictors_ns = np.insert(predictors_ns[:predictors_ns.size-y_forward.size],1,y_forward_ns)
    return IRF
    


def MCMC_draw(postior,theta,c,inv_hess):
    candidate = np.random.normal(theta,(c**2)*inv_hess)
    if np.random.uniform()<=(postior(candidate)/postior(theta)):
        return candidate
    else:
        return theta

def draw_beta(B,a,x,omega):
    beta = B.flatten()
    A0=a2A(a)
    draw = np.random.normal(beta,np.kron(np.linalg.inv(A0)@np.linalg.inv(A0).T,np.linalg.inv(x.T@x+np.linalg.inv(omega))))
    return draw

def part_3(Data,lags,lambd,mu,draws,c,length,variable):
    b , omega = minnesota_prior(Data,lags, lambd)
    Yp, Xp, xp = SoC_dummy(Data,lags, mu)
    B = b_Var(Yp,xp,lags,b,omega)
    residuals = S(Yp,B,xp)
    mode, inv_hess = postior_mode_A0(B,b,residuals,omega,np.size(Data,0)-lags,lags,np.size(Data,1))
    IRF_errors=np.zeros((np.size(B,axis=0),length,draws))
    A_value = mode
    def log_postierior(a):
        A0 = a2A(a)
        if np.linalg.det(A0)<0:
            return 999999999999999999999999999999
        piece_1 = (T-p+n)*np.log(np.linalg.det(A0))
        piece_2 = -0.5*np.trace((S+((B.T-b).T)@np.linalg.inv(Omega)@(B.T-b))@(A0.T@A0))
        return -piece_1-piece_2
    for n in range(draws):
        A_value = MCMC_draw(log_postierior,value,c,inv_hess)
        B_value = draw_beta(B,A_value,xp,omega)
        B_value = np.reshape(B_value,(np.size(Data,1),1+np.size(Data,1)*lags)).T
        IRF_errors[:,:,n] = IRF(B_value,a2A(A_value),lags,variable,length)
    return IRF_errors

# def MCMC(B,b,S,Omega,T,p,n,draws=5000, burnout=100,)

# def part_3(Data,lags,lambd,mu):
#     b , omega = minnesota_prior(Data,lags, lambd)
#     Yp, Xp, xp = SoC_dummy(Data,lags, mu)
#     B = b_Var(Yp,xp,lags,b,omega)
#     residuals = S(Yp,B,xp)
#     draws = MCMC()

