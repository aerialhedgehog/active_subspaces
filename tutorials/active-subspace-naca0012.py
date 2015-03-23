# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>
# Active Subspace Analysis Notebook for NACA0012 Airfoil Design
# <markdowncell>

# **DESCRIPTION**
# 
# This data set comes from a simulation of the NACA0012 airfoil. The inputs are parameters of the free-form deformation boxes for the airfoil geometry. The outputs are the lift and drag of the wing. Gradients are available from an adjoint solver. All simulations are performed with [SU2](http://su2.stanford.edu/).
# 
# **INPUTS (18)**
# 
# Variable | Lower bound | Upper bound | Density
# --- | --- | --- | ---
# x1-x18 | -0.01 | 0.01 | Uniform
# 
# **OUTPUTS (2)**
# 
# Variable | Description
# --- | --- 
# Lift | The lift of the airfoil.
# Drag | The drag of the airfoil.
# 
# **REFERENCES**
# 
# + Lukaczyk, Palacios, Alonso, and Constantine. [Active Subspaces for Shape Optimization](http://arc.aiaa.org/doi/abs/10.2514/6.2014-1171)
# + Constantine, Dow, and Wang. [Active Subspace Methods in Theory and Practice: Applications to Kriging Surfaces](http://epubs.siam.org/doi/abs/10.1137/130916138)
# 
# **CONTACT**
# 
# Questions or comments? Contact [Paul Constantine](mailto:pconstan@mines.edu)
# <codecell>

import numpy as np
import pandas as pn
import active_subspaces as ac

# <markdowncell>
# Import the data set. Distinguish inputs (X), outputs (F), and gradients (G).
# <codecell>

df = pn.DataFrame.from_csv('NACA0012.txt')
data = df.as_matrix()
X = data[:,:18]/0.01
f_lift = data[:,18]
f_drag = data[:,19]
df_lift = data[:,20:38]
df_drag = data[:,38:]
M,m = X.shape
labels = df.keys()
out_labels = labels[18:20]

# <markdowncell>
# Build the active subspace.
# <codecell>

model = ac.ActiveSubspaceModel(bflag=True)
model.build_from_data(X,f_drag,df=df_drag)
model.subspaces.partition(n=4)

# <markdowncell>
# Show the active subspace components.
# <codecell>

model.diagnostics()

# <markdowncell>
# Try minimization.
# <codecell>

[xmin,_],fmin = model.minimum()
print 'Minimum: %6.4f' % fmin
print 'Argminimum:'
print xmin

# <markdowncell>
# Train response surface with the barrier.
# <codecell>

chi = f_lift<0.2
h = f_drag + chi*np.log(-(f_lift-0.2)+1)
Y = np.dot(X,model.subspaces.W1)
model.rs.train(Y,h)
#model.plotter.sufficient_summary(Y[:,0],h,out_label='Drag + barrier')

# <markdowncell>
# Minimize with the barrier.
# <codecell>

[xmin,_],fmin = model.minimum()
print 'fmin: %6.4f' % fmin
print 'xmin:'
print xmin
print 'ymin:'
print np.dot(xmin,model.subspaces.W1[:,0])

