import numpy as np
from utils import process_inputs
from scipy.optimize import fminbound

class ResponseSurface():
    def train(self,X,f):
        raise NotImplementedError()
        
    def predict(self,X,compgrad=False,compvar=False):
        raise NotImplementedError()
        
    def gradient(self,X):
        return self.predict(X,compgrad=True)[1]
        
    def __call__(self,X):
        return self.predict(X)[0]

class PolynomialRegression(ResponseSurface):
    def __init__(self,N=2):
        self.N = N
        
    def train(self,X,f):
        X,M,m = process_inputs(X)
         
        B,indices = polynomial_bases(X,self.N)
        Q,R = np.linalg.qr(B)
        poly_weights = np.linalg.solve(R,np.dot(Q.T,f))
        
        # store data
        self.X,self.f = X,f
        self.poly_weights = poly_weights
        self.Q,self.R = Q,R
        
        # organize linear and quadratic coefficients
        self.g = poly_weights[1:m+1].copy()
        if self.N>1:
            H = np.zeros((m,m))
            for i in range(m+1,indices.shape[0]):
                ind = indices[i,:]
                loc = np.nonzero(ind!=0)[0]
                if loc.size==1:
                    H[loc,loc] = 2.0*poly_weights[i]
                elif loc.size==2:
                    H[loc[0],loc[1]] = poly_weights[i]
                    H[loc[1],loc[0]] = poly_weights[i]
                else:
                    raise Exception('Error creating quadratic coefficients.')
            self.H = H
        
    def predict(self,X,compgrad=False,compvar=False):
        X,M,m = process_inputs(X)
        
        B = polynomial_bases(X,self.N)[0]
        f = np.dot(B,self.poly_weights)
        f = f.reshape((M,1))
        
        if compgrad:
            dB = grad_polynomial_bases(X,self.N)
            df = np.zeros((M,m))
            for i in range(m):
                df[:,i] = np.dot(dB[:,:,i],self.poly_weights).reshape((M))
            df = df.reshape((M,m))
        else:
            df = None
        
        if compvar:
            R = np.linalg.solve(self.R.T,B.T)
            v = np.var(self.f)*np.diag(np.dot(R.T,R))
            v = v.reshape((M,1))
        else:
            v = None
            
        return f,df,v

class GaussianProcess():
    def __init__(self,N=2,e=None,gl=0.0,gu=10.0,v=None):
        self.N = N
        self.e = e
        self.gl,self.gu = gl,gu
        self.v = v
        
    def train(self,X,f):
        X,M,m = process_inputs(X)
            
        if self.e is None:
            e = np.hstack((np.ones(m),np.array([np.var(f)])))
        else:
            e = self.e
        g = fminbound(negative_log_likelihood,self.gl,self.gu,args=(X,f,e,self.N,self.v,))
        
        # set parameters
        sig = g*np.sum(e)
        ell = sig/e[:m]
        
        # covariance matrix of observations
        K = exponential_squared_covariance(X,X,sig,ell)
        if self.v is None:
            K += g*np.sum(e[m:])*np.eye(M)
        else:
            K += np.diag(self.v)
        radial_weights = np.linalg.solve(K,f)
        
        # coefficients of polynomial basis
        B = polynomial_bases(X,self.N)[0]
        A = np.dot(B.T,np.linalg.solve(K,B))
        poly_weights = np.linalg.solve(A,np.dot(B.T,radial_weights))
        
        # store parameters
        self.X,self.f = X,f
        self.sig,self.ell = sig,ell
        self.radial_weights,self.poly_weights = radial_weights,poly_weights
        self.K,self.A,self.B = K,A,B
        
    def predict(self,X,compgrad=False,compvar=False):
        X,M,m = process_inputs(X)

        # predict without polys
        K = exponential_squared_covariance(self.X,X,self.sig,self.ell)
        f = np.dot(K.T,self.radial_weights)
        
        # update with polys
        P = np.linalg.solve(self.K,K)
        B = polynomial_bases(X,self.N)[0]
        R = B - np.dot(P.T,self.B)
        f += np.dot(R,self.poly_weights)
        f = f.reshape((M,1))
        
        if compgrad:
            dK = grad_exponential_squared_covariance(self.X,X,self.sig,self.ell)
            dB = grad_polynomial_bases(X,self.N)
            df = np.zeros((M,m))
            for i in range(m):
                dP = np.linalg.solve(self.K,dK[:,:,i])
                dR = dB[:,:,i] - np.dot(dP.T,self.B)
                df[:,i] = (np.dot(dK[:,:,i].T,self.radial_weights) \
                    + np.dot(dR,self.poly_weights)).reshape((M))
                df = df.reshape((M,m))
        else:
            df = None
        
        if compvar:
            V = exponential_squared_covariance(X,X,self.sig,self.ell)
            v = np.diag(V) - np.sum(P*P,axis=0)
            v += np.diag(np.dot(R,np.linalg.solve(self.A,R.T)))
            v = v.reshape((M,1))
        else:
            v = None
            
        return f,df,v

def negative_log_likelihood(g,X,f,e,N,v):
    M,m = X.shape
    sig = g*np.sum(e)
    ell = sig/e[:m]
    
    # covariance matrix
    K = exponential_squared_covariance(X,X,sig,ell)
    if v is None:
        K += g*np.sum(e[m:])*np.eye(M)
    else:
        K += np.diag(v)
    L = np.linalg.cholesky(K)
    
    # polynomial basis
    B = polynomial_bases(X,N)[0]
    A = np.dot(B.T,np.linalg.solve(K,B))
    AL = np.linalg.cholesky(A)
    
    # negative log likelihood
    z = np.linalg.solve(K,f)
    Bz = np.dot(B.T,z)
    r = np.dot(f.T,z) \
        - np.dot(Bz.T,np.linalg.solve(A,Bz)) \
        + np.sum(np.log(np.diag(L))) \
        + np.sum(np.log(np.diag(AL))) \
        + (M-B.shape[1])*np.log(2*np.pi)
    return 0.5*r

def exponential_squared_covariance(X1,X2,sigma,ell):
    m = X1.shape[0]
    n = X2.shape[0]
    c = -1.0/ell.flatten()
    C = np.zeros((m,n))
    for i in range(n):
        x2 = X2[i,:]
        B = X1 - x2
        C[:,i] = sigma*np.exp(np.dot(B*B,c))
    return C

def grad_exponential_squared_covariance(X1,X2,sigma,ell):
    m,d = X1.shape
    n = X2.shape[0]
    c = -1.0/ell.flatten()
    C = np.zeros((m,n,d))
    for k in range(d):
        for i in range(n):
            x2 = X2[i,:]
            B = X1 - x2
            C[:,i,k] = sigma*(-2.0*c[k]*B[:,k])*np.exp(np.dot(B*B,c))
    return C

def polynomial_bases(X,N):
    M,m = X.shape
    I = index_set(N,m)
    n = I.shape[0]
    B = np.zeros((M,n))
    for i in range(n):
        ind = I[i,:]
        B[:,i] = np.prod(np.power(X,ind),axis=1)
    return B,I
    
def grad_polynomial_bases(X,N):
    M,m = X.shape
    I = index_set(N,m)
    n = I.shape[0]
    B = np.zeros((M,n,m))
    for k in range(m):
        for i in range(n):
            ind = I[i,:]
            indk = ind[k]
            if indk==0:
                B[:,i,k] = np.zeros(M)
            else:
                ind[k] -= 1
                B[:,i,k] = indk*np.prod(np.power(X,ind),axis=1)
    return B

def full_index_set(n,d):
    if d == 1:
        I = np.array([[n]])
    else:
        II = full_index_set(n,d-1)
        m = II.shape[0]
        I = np.hstack((np.zeros((m,1)),II))
        for i in range(1,n+1):
            II = full_index_set(n-i,d-1)
            m = II.shape[0]
            T = np.hstack((i*np.ones((m,1)),II))
            I = np.vstack((I,T))
    return I
    
def index_set(n,d):
    I = np.zeros((1,d))
    for i in range(1,n+1):
        II = full_index_set(i,d)
        I = np.vstack((I,II))
    return I[:,::-1]