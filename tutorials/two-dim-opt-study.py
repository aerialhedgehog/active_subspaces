
# coding: utf-8

## USE CASE: Optimization for two-dimensional active subspace.

# A quadratic function to study. The user provides this. It is usually a call to some simulation code---not a python function.

# In[1]:

import numpy as np
import active_subspaces as ac
import matplotlib.pyplot as plt
from scipy.spatial import convex_hull_plot_2d

class MyQuadratic():
    def __init__(self):
        self.e = np.exp(np.array([2,1,-2,-3,-4,-5,-6,-7,-8,-9.]))
        self.W = np.linalg.qr(np.random.normal(size=(10,10)))[0]
        self.A = np.dot(self.W,np.dot(np.diag(self.e),self.W.T))
    def __call__(self,x):
        return 0.5*np.dot(x,np.dot(self.A,x.T)),np.dot(self.A,x.T)
    def integral(self):
        return (1./6.)*np.sum(np.diag(self.A))

fun = MyQuadratic()


# Discover the active subspace of the function.

# In[2]:

M,m = 100,10
X = np.random.uniform(-1.,1.,size=(M,m))

# sample the function. often done offline. 'sample_function' is just for convenience
F,dF = ac.sample_function(X,fun,dflag=True)

# compute the active subspace components
k = 6
e,W,e_br,sub_br = ac.compute_active_subspace(dF,k,n_boot=1000)

# make some plots
asp = ac.ActiveSubspacePlotter()
asp.eigenvalues(e,e_br=e_br)
asp.subspace_errors(sub_br)
asp.eigenvectors(W[:,:6])
y = np.dot(X,W[:,0])
asp.sufficient_summary(y,F)


# Get the response surface design and plot its points.

# In[3]:

# Get a design on the active variable
n = 2 # dimension of the active subspace
N = [10] # number points in the active variable
NMC = 10 # number of monte carlo samples
XX,ind,Y = ac.response_surface_design(W,n,N,NMC,bflag=1)

# look at the zonotope and the design
zt = ac.Zonotope(W[:,:n])
convex_hull_plot_2d(zt.convhull)
plt.plot(Y[:,0],Y[:,1],'ro')
plt.xlabel('Active variable 1')
plt.ylabel('Active variable 2')
plt.show()


# In[4]:

# sample the function at the design points
FF = ac.sample_function(XX,fun,dflag=True)[0]

# compute conditional expectations
G,V = ac.conditional_expectations(FF,ind)

# build the response surface on the active variable
gp = ac.GaussianProcess(2)
gp.train(Y,G,e=e,gl=0.0,gu=10.0,v=V)


# Optimize on the active variables.

# In[5]:

# set up the functions for the optimization
import pdb
def fun(x):
    n = x.size
    return gp.predict(x.reshape((1,n)))[0]
def dfun(x):
    n = x.size
    return gp.predict(x.reshape((1,n)),compgrad=True)[1]

# get constraints from the zonotope
cons = zt.constraints()

# run optimization
from scipy.optimize import minimize
y0 = np.random.normal(size=(1,n))
res = minimize(fun,y0,constraints=cons,method='SLSQP',
                options={'disp':True,'maxiter':1e4,'ftol':1e-9})
print y0
print res

