import numpy as np
import random
from scipy import linalg as la

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
            to compute $\nabla log(p(data|theta))$ (gradient with respect to theta) of a set of data.
            
        grad_log_den_prior: function with parameter (theta)
            to compute $\nabla log(p(theta))$.
            
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
        sqrtM = np.eye(p)
        prer = eps
        fric = eps*C
    else:
        sqrtM = la.sqrtm(la.inv(Minv))
        prer = eps*Minv
        fric = eps*C@Minv

    sqrt_noise = la.sqrtm(2*(C-0.5*eps*V_hat)*eps)
    
    samples = np.zeros((epoches - heatup, p))
    batches = np.int(np.ceil(n/batch_size))
    
    theta = theta_0
    for t in range(epoches):
        if(Minv is None):
            r = np.random.normal(size=(p))
        else:
            r = sqrtM@np.random.normal(size=(p))
        
        reorder = random.sample(range(n),n)
        for i in range(batches):
            batch = data[reorder[(i*batch_size):np.min([(i+1)*batch_size,n])]]
            theta = theta + (prer*r if Minv is None else prer@r)
            gU = gradU(grad_log_den_data,grad_log_den_prior,batch,theta,n)
            r = r - eps*gU - fric@r + sqrt_noise@np.random.normal(size=(p))
        theta = theta + (prer*r if Minv is None else prer@r)
        
        if(t>=heatup):
            samples[t-heatup] = theta
    
    return(samples)
