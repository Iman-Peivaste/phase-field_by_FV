#Solve a phase-field (Allen-Cahn) problem in one-dimension.


import time 
import numpy as np
from fipy import *
import matplotlib.pyplot as plt



strart_time = time.time ()

#At first, we need to have a mesh

N = 64
L = 2.5 * N / 100.
dL = L / N
mesh = PeriodicGrid2D(dx=dL, dy=dL, nx=N, ny=N)

# steps for checking mesh
u = mesh.faceCenters() #(1, 401)
# u1 = mesh.cellCenters() #(1, 400)
# u3 = mesh.x() # (400, )

# the second step is to create a variable, the one that we need to calculate over the mesh

phase  = CellVariable(name = "Phase", mesh =mesh, value = 1.0 )  #(400, )
#phase  = ModularVariable(name = "Phase", mesh =mesh, value = 1.0 )  #(400, )
# theta = ModularVariable(name='theta',mesh=mesh,value=1.,hasOld=1)

# print(theta.shape)

x,y = mesh.cellCenters #  (400, )




#Intitial value
phase.setValue(1.0)

a = L/2.0
b = L/2.0

segment = (x - a)**2 + (y - b)**2 < (L / 6.)**2 

# # phase.setValue(1.0)
phase.setValue(0., where= segment)

# plot intial value over the domain
viewr = Viewer(vars= phase)

# if we don't set BCs, it is n.(dphi/d(x)=0)
###################################################################

kappa = 0.0025
W = 1.
Lv = 1.
Tm = 1.
T = Tm
enthalpy = Lv * (T - Tm) / Tm

analyticalArray = 0.5*(1 - numerix.tanh((x - L/2)/(2*numerix.sqrt(kappa/W))))

mPhi = -((1 - 2 * phase) * W + 30 * phase * (1 - phase) * enthalpy)
S0 = mPhi * phase * (1 - phase)


eq = TransientTerm() == S0 + DiffusionTerm(coeff = kappa)


#the eq is time dependent, so we need steps and time steps
#the eq is not non-liner so we don't need to update and sweep

steps = 200
dt  = .1
A = []
for i in range(steps):
    eq.solve(var = phase, dt = dt)

    phase.setValue(0.9999, where = phase>0.9999)
    phase.setValue(0.0001, where = phase<0.0001)
    ff= np.array(phase)
    fff=np.where(ff > 0.1, 0, ff)
    A.append(np.sum(fff))
    viewr.plot()

























