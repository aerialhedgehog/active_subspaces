import numpy as np
import active_subspaces as asm

def quad_fun(x):
    A = np.array([[ 0.2406659045776698, -0.3159904335007421, -0.1746908591702878],
       [-0.3159904335007421,  0.5532215729009683,  0.3777995408101305],
       [-0.1746908591702878,  0.3777995408101305,  0.3161125225213613]])
    f = 0.5*np.dot(x.T,np.dot(A,x))
    return f

def quad_grad(x):
    A = np.array([[ 0.2406659045776698, -0.3159904335007421, -0.1746908591702878],
       [-0.3159904335007421,  0.5532215729009683,  0.3777995408101305],
       [-0.1746908591702878,  0.3777995408101305,  0.3161125225213613]])
    df = np.dot(A,x)
    return df

if __name__ == '__main__':
    print 'Here we go'
    m = 3
    bflag = True
    
    # initialize the model
    model = asm.ActiveSubspaceModel(bflag=bflag)
    
    # build the model from an interface
    model.build_from_interface(m,quad_fun)
    model.build_from_interface(m,quad_fun,dfun=quad_grad)
    
    # build the model from data
    qd = np.load('quad_data.npz')
    X,f,df = qd['X'],qd['f'],qd['df']
    model.build_from_data(X,f)
    model.build_from_data(X,f,df=df)
    
    # make some plots
    #model.diagnostics()
    
    # trying 2d
    model.subspace.partition(n=2)
    
    # just for tests -- these ran without error
    #model.set_domain()
    #model.set_response_surface()
    
    
    # check response surface predictions
    qd = np.load('quad_predict.npz')
    XX = qd['XX']
    ff,dff,vv = model.predict(XX)
    ff,dff,vv = model.predict(XX,compvar=True)
    ff,dff,vv = model.predict(XX,compgrad=True)
    ff,dff,vv = model.predict(XX,compgrad=True,compvar=True)
    print 'Predictions'
    print ff
    print 'Gradients'
    print dff
    print 'Variances'
    print vv
    
    # check statistics
    mu = model.mean()
    print 'Mean'
    print mu
    sig = model.variance()
    print 'Variance'
    print sig
    np.random.seed(seed=1000)
    p = model.probability(0.0,0.05)
    print 'Probability'
    print p
    
    # trying optimization
    fmin,xmin = model.minimum()
    print 'Minimum'
    print fmin
    print 'Argmin'
    print xmin
    
    
    
    

