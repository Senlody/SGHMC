import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from sghmc.module import sghmc,sghmc_chains
import pandas as pd

# This example fit a bayesian neural network with sghmc
# The bnn does a regression problem on MPG data
# The bnn has an input layer, a hidden layer with 10 nodes, 
# and an output layer of 1 node
# The figure of fitted data vs true data on test set
# is saved in bnnReg.png
# The figure of rmse among iterations 
# is saved in bnnRMSE.png

print('reading data')

# codes loading MPG comes from keras tutorial
datapd = pd.read_csv('auto-mpg.data', names=['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'],
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)
datapd.tail()
datapd = datapd.dropna()
origin = datapd.pop('Origin')
datapd['USA'] = (origin == 1)*1.0
datapd['Europe'] = (origin == 2)*1.0
datapd['Japan'] = (origin == 3)*1.0

#standardize
sd = np.max(datapd[['Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year']],axis=0)
datapd[['Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year']]/=sd

train_dataset = datapd.sample(n=300,random_state=0)
test_dataset = datapd.drop(train_dataset.index)

print('performing sghmc to fit bnn')

def generate_grad_lpdf(xdim,nnodes,sgmsq):
    '''
    This function generates the function to compute grad log pdf of p(data|weights)
    '''
    def expit(x):
        return 1/(1+np.exp(-x))
    
    def dexpit(x):
        epx=expit(x)
        return epx*(1-epx)
    
    def grad_lpdf(data,theta):
        Beta=np.c_[tuple(np.split(theta[nnodes+1:],nnodes))]
        Y=data[:,0]
        X=np.c_[np.ones(data.shape[0]),data[:,1:]]
        IN=X@Beta
        V=expit(IN)
        betau=theta[:nnodes+1]
        u=betau[0]+V@betau[1:]
        dIN=dexpit(-IN)
        buidexp=dIN*betau[1:]
        gU = np.c_[tuple([buidexp[:,[i]]*X for i in range(nnodes)])]
        gU = np.c_[np.ones(data.shape[0]),V,gU]
        gU = (Y-u)[:,None]/sgmsq*gU
        return np.sum(gU,axis=0)
        
    return grad_lpdf

def generate_grad_lprior(xdim,nnodes,sgmsqbh,sgmsqbu):
    '''
    This function generates the function to compute grad log pdf of prior p(weights)
    '''
    def grad_lprior(theta):
        return -np.r_[theta[:nnodes+1]/sgmsqbu,theta[nnodes+1:]/sgmsqbh]
    return grad_lprior

def generate_predict_func(xdim,nnodes):
    '''
    This function generates the function to compute prediction given data and weights
    '''
    def expit(x):
        return 1/(1+np.exp(-x))
    
    def a_predict(adata,theta):
        Beta=np.c_[tuple(np.split(theta[nnodes+1:],nnodes))]
        X=np.r_[1,adata]
        IN = Beta.T@X
        V=expit(IN)
        betau=theta[:nnodes+1]
        u=betau[0]+np.dot(V,betau[1:])
        return u
    
    def m_predict(data,theta):
        return np.array([a_predict(adata,theta) for adata in data])
        
    def predict(data,theta_sim):
        return np.c_[tuple([m_predict(data,theta) for theta in theta_sim])]
        
    return predict

def form_data(y,x):
    '''
    To make data matrix
    '''
    return np.c_[y,x]

xdim=9
nnodes=10
sgmsq=1
sgmsqbh=1
sgmsqbu=1

data = form_data(train_dataset.MPG,train_dataset.drop('MPG',axis=1))

nn_glpdf=generate_grad_lpdf(xdim,nnodes,sgmsq)
nn_glprior=generate_grad_lprior(xdim,nnodes,sgmsqbh,sgmsqbu)
predict=generate_predict_func(xdim,nnodes)

thetadim=nnodes+1+nnodes*(xdim+1)
sim = sghmc(nn_glpdf, nn_glprior, data[:300], V_hat = np.eye(thetadim), eps = 0.01, 
            theta_0 = np.zeros(thetadim), C = np.eye(thetadim), 
            heatup = 100, epoches = 2000, batch_size = 100)

print('sghmc done')
print('predicting (may take 1~2 minutes)')

pred = predict(np.array(test_dataset.drop('MPG',axis=1)),sim)

predtrain = predict(np.array(train_dataset.drop('MPG',axis=1)),sim)

print('prediction done, drawing figures')

predmean=np.mean(pred,axis=1)
plt.plot(np.array([10,35]),np.array([10,35]),c='r')
plt.scatter(test_dataset.MPG,predmean)
plt.title('fitted vs. true on test set')
plt.ylabel('fitted mpg')
plt.xlabel('true mpg')
plt.savefig('bnnReg.png')
plt.clf()

def rmse(u,v):
    ''' Compute rmse '''
    return np.sqrt(np.mean((u-v)**2))

nsim = sim.shape[0]
jump=10
rmsetest=np.zeros(np.int(nsim/jump))
rmsetrain=np.zeros(np.int(nsim/jump))

for i in range(np.int(nsim/jump)):
    rmsetest[i]=rmse(test_dataset.MPG,np.mean(pred[:,:i*jump+1],axis=1))
    rmsetrain[i]=rmse(train_dataset.MPG,np.mean(predtrain[:,:i*jump+1],axis=1))
    
plt.plot(np.arange(len(rmsetest))*jump,rmsetest,c='r',label='test')
plt.plot(np.arange(len(rmsetrain))*jump,rmsetrain,c='b',label='train')
plt.legend()
plt.title('RMSE plot')
plt.ylabel('rmse')
plt.xlabel('iteration')
plt.savefig('bnnRMSE.png')

print('all done')