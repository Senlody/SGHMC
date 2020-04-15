import numpy as _np
from scipy import linalg as _la
from multiprocessing import Pool as _Pool
from functools import partial as _pt

def sghmc(grad_log_den_data, grad_log_den_prior, data, V_hat, eps, theta_0, C, heatup, epoches, batch_size, Minv = None):
    '''
    Implementation of Stochastic Gradient Hamiltonian Monte Carlo.
    (See details in Chen et al., 2014)
    
    Dimensions in sampling procdure:
        p: dimension of parameters(theta)
        n: number of observed data.
        m: dimension of data.
    
    INPUT:            
        grad_log_den_data: function with parameters (data,theta)
            to compute $\\nabla log(p(data|theta))$ (gradient with respect to theta) of a set of data.
            
        grad_log_den_prior: function with parameter (theta)
            to compute $\\nabla log(p(theta))$.
            
        data: np.array with shape (n,m)
            representing observed data 
            
        V_hat: np.array with shape (p,p)
            a matrix of estimated Fisher Information 
            
        eps: float or double
            learning rate
            
        theta_0: np.array with shape (p,)
            initial point of sampling.
            
        C: np.array with shape (p,p)
            a matrix representing friction, see paper for details. 
            C-0.5*eps*V_hat must be positive definite.
            
        heatup: int
            iteration to dump before storing sampling points.
            
        epoches: int
            iterations to run. Must be greater than heatup.
        
        batch_size: int
            size of a minibatch in an iteration, hundreds recommended
            
        Minv: np.array with shape (p,p)
            if default(NULL), will be identical. (See paper for details)
            
    OUT:
        sample: np.array with shape (epoches - heatup, p)
            sampled posterior thetas.
    '''
    
    def gradU(grad_log_den_data, grad_log_den_prior, batch, theta, n):
        '''
        inner function to compute $\nabla \tilde{U}$ defined in paper.
        '''
        return(-(n*grad_log_den_data(batch,theta)/batch.shape[0]+grad_log_den_prior(theta)))
    
    n,m = data.shape
    p = theta_0.shape[0]
    
    if(Minv is None):
        sqrtM = _np.eye(p)
        prer = eps
        fric = eps*C
    else:
        sqrtM = _la.sqrtm(_la.inv(Minv))
        prer = eps*Minv
        fric = eps*C@Minv

    sqrt_noise = _la.sqrtm(2*(C-0.5*eps*V_hat)*eps)
    
    samples = _np.zeros((epoches - heatup, p))
    batches = _np.int(_np.ceil(n/batch_size))
    
    theta = theta_0
    for t in range(epoches):
        if(Minv is None):
            r = _np.random.normal(size=(p))
        else:
            r = sqrtM@_np.random.normal(size=(p))
        
        split = _np.split(data,batches)
        for i in range(batches):
            batch = split[i]
            theta = theta + (prer*r if Minv is None else prer@r)
            gU = gradU(grad_log_den_data,grad_log_den_prior,batch,theta,n)
            r = r - eps*gU - fric@r + sqrt_noise@_np.random.normal(size=(p))
        theta = theta + (prer*r if Minv is None else prer@r)
        
        if(t>=heatup):
            samples[t-heatup] = theta
    
    return(samples)

def _single_chain(seed, theta_0, epoches, heatup, p, n, Minv, sqrtM, data, batches, prer, gradU, grad_log_den_data, grad_log_den_prior, eps, fric, sqrt_noise):
    '''
    private function to simulate a single chain
    '''
    _np.random.seed(seed)
    theta = theta_0
    samples = _np.zeros((epoches - heatup, p))
    for t in range(epoches):
        if(Minv is None):
            r = _np.random.normal(size=(p))
        else:
            r = sqrtM@_np.random.normal(size=(p))

        split = _np.split(data,batches)
        for i in range(batches):
            batch = split[i]
            theta = theta + (prer*r if Minv is None else prer@r)
            gU = gradU(grad_log_den_data,grad_log_den_prior,batch,theta,n)
            r = r - eps*gU - fric@r + sqrt_noise@_np.random.normal(size=(p))
        theta = theta + (prer*r if Minv is None else prer@r)

        if(t>=heatup):
            samples[t-heatup] = theta

    return(samples)

def _gradU(grad_log_den_data, grad_log_den_prior, batch, theta, n):
    '''
    inner function to compute $\nabla \tilde{U}$ defined in paper.
    '''
    return(-(n*grad_log_den_data(batch,theta)/batch.shape[0]+grad_log_den_prior(theta)))

def sghmc_chains(grad_log_den_data, grad_log_den_prior, data, V_hat, eps, theta_0, C, heatup, epoches, batch_size, chain = 1, Minv = None):
    '''
    Implementation of Stochastic Gradient Hamiltonian Monte Carlo.
    (See details in Chen et al., 2014)
    
    This is a multiprocess version of sghmc (only works on linux).
    It will run multiple(number = chain) simulations simutaneously. And returns a list of simulations
    
    Dimensions in sampling procdure:
        p: dimension of parameters(theta)
        n: number of observed data.
        m: dimension of data.
    
    INPUT:            
        grad_log_den_data: function with parameters (data,theta)
            to compute $\\nabla log(p(data|theta))$ (gradient with respect to theta) of a set of data.
            
        grad_log_den_prior: function with parameter (theta)
            to compute $\\nabla log(p(theta))$.
            
        data: np.array with shape (n,m)
            representing observed data 
            
        V_hat: np.array with shape (p,p)
            a matrix of estimated Fisher Information 
            
        eps: float or double
            learning rate
            
        theta_0: np.array with shape (p,)
            initial point of sampling.
            
        C: np.array with shape (p,p)
            a matrix representing friction, see paper for details. 
            C-0.5*eps*V_hat must be positive definite.
            
        heatup: int
            iteration to dump before storing sampling points.
            
        epoches: int
            iterations to run. Must be greater than heatup.
        
        batch_size: int
            size of a minibatch in an iteration, hundreds recommended
            
        chain: int
            number of chains to run. Each chain is a simulation.
            
        Minv: np.array with shape (p,p)
            if default(NULL), will be identical. (See paper for details)
            
    OUT:
        sample: a list (number = chain) of np.array with shape (epoches - heatup, p)
            sampled posterior thetas.
    '''

    n,m = data.shape
    p = theta_0.shape[0]
    sqrt_noise = _la.sqrtm(2*(C-0.5*eps*V_hat)*eps)
    batches = _np.int(_np.ceil(n/batch_size))
    
    if(Minv is None):
        sqrtM = None
        prer = eps
        fric = eps*C
    else:
        sqrtM = _la.sqrtm(_la.inv(Minv))
        prer = eps*Minv
        fric = eps*C@Minv

    sp = _pt(_single_chain,
            theta_0 = theta_0, epoches = epoches, heatup = heatup, p=p, n=n, 
             Minv=Minv, sqrtM=sqrtM, data=data, batches=batches, prer=prer, 
             gradU=_gradU, grad_log_den_data=grad_log_den_data, grad_log_den_prior=grad_log_den_prior, 
             eps=eps, fric=fric, sqrt_noise=sqrt_noise)
    
    with _Pool(processes=chain) as pool:
        seedss = list(_np.random.randint(0,10000,chain))
        res = pool.map(sp, seedss)
    
    return(res)
