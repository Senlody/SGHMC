import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from sghmc.module import sghmc,sghmc_chains

# This example simulates data~0.5*Normal(0,mu_1)+0.5*Normal(0,mu_2) 
# where mu_1=-3 and mu_2 = 3
# and sample posterior mu_1 and mu_2 through sghmc and sghmc_chain
# The sampled density figure are mixnorm1.png and mixnorm2.png respectively

mu = np.array([-3, 3]).reshape(2,1) # true value for mu
n = 200 # number of observations
y = np.r_[np.random.normal(mu[0], 1, n),np.random.normal(mu[1], 1, n)]

def mn_glpdf(y,mu):
    '''compute grad log pdf'''
    exp1=np.exp(-0.5*(y-mu[0])**2)
    exp2=np.exp(-0.5*(y-mu[1])**2)
    
    v = np.c_[exp1*(y-mu[0]),exp2*(y-mu[1])]
    return np.sum(v/(exp1+exp2),axis=0)

def mn_glpr(mu):
    '''compute grad log prior'''
    return -(np.sum(mu)/10)

sim1 = sghmc(mn_glpdf, mn_glpr, y[:,None], V_hat = np.eye(2), eps = 0.01, 
            theta_0 = np.array([0,0]), C = np.eye(2), 
            heatup = 100, epoches = 200, batch_size = 80)

kdeplt2 = sns.kdeplot(sim1[:,0],sim1[:,1])  
plt.title('kernel density plot run by sghmc')
plt.savefig('mixnom1.png')

sim2 = sghmc_chains(mn_glpdf, mn_glpr, y[:,None], V_hat = np.eye(2), eps = 0.01, 
            theta_0 = np.array([0,0]), C = np.eye(2), 
            heatup = 100, epoches = 200, batch_size = 80,chain = 20)
sim2 = np.r_[tuple([asim for asim in sim2])]

plt.clf()
kdeplt2 = sns.kdeplot(sim2[:,0],sim2[:,1])  
plt.title('kernel density plot run by sghmc_chains')
plt.savefig('mixnom2.png')