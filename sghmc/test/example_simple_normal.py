## An example of 2-dimensional normal:
## 	$p(y|\theta)\sim N(\theta,I)$
## 	$p(\theta)\sim N(0,I)$

import numpy as np
from sghmc.sghmcCore import sghmc

def grad_log_den_data(data,theta):
    return(np.sum(data-theta,axis=0))

def grad_log_den_prior(theta):
    return(np.array([0,0])-theta)

data = np.array([-10,10])+np.random.normal(size=(100,2))
V_hat=np.eye(2)
eps=0.01
theta_0=np.zeros(2)
C=np.eye(2)
heatup=100
epoches=200
batch_size=20

sghmc(grad_log_den_data, grad_log_den_prior, data, V_hat, eps, theta_0, C, heatup, epoches, batch_size, Minv = None)

sghmc(grad_log_den_data, grad_log_den_prior, data, V_hat, eps, theta_0, C, heatup, epoches, batch_size, Minv = np.eye(2))