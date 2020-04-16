import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from sghmc.module import sghmc,sghmc_chains

# This example tests the basic idea of SGHMC (refer to original paper)
# where the potential energy is $U(\theta)=-2\theta^2+\theta^4$
# and the grad of potential energy follows a normal distribution:
# $\nabla \tilde{U}=\nabla U + N(0,4)$
# The simulated distribution of theta is saved in simpleU.png

def su_glpdf(y,theta):
    '''compute grad log pdf'''
    g = -(-4*theta+4*theta**3)
    
    return g/50

def su_glpr(theta):
    '''dummy prior'''
    return 0

eps=0.1
batch_size = 1
simsu = sghmc(su_glpdf, su_glpr, np.zeros((50*batch_size,1)), V_hat = np.eye(1)*0, eps = eps, 
            theta_0 = np.array([0]), C = np.eye(1)*2*eps, 
            heatup = 100, epoches = 20000, batch_size = batch_size)

sns.kdeplot(simsu[:,0])
plt.savefig('simpleU.png')