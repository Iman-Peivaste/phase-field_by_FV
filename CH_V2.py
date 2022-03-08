from fipy import *
import glob
import json
import numpy as np
import os
import sys
from fipy.tools import numerix
import time 
import numpy as np 

from fipy import LinearLUSolver as Solver

strart_time = time.time ()



nx = 64
dx = 0.25



mesh = PeriodicGrid2D(nx=nx, ny=nx, dx=dx, dy=dx)



kappa = 1.0
M = 1.0
c_0 = 0.4
epsilon = 0.01
A = 1.0

c_var = CellVariable(mesh=mesh, name="consentration", hasOld=True)
c_var.mesh.cellCenters()


noise = GaussianNoiseVariable(mesh=mesh, mean=c_0, variance=0.01).value

c_var [:] = noise

viewer = Viewer(vars=(c_var,), datamin=0., datamax=1.)


def f_0(c):
    return A * (c )**2 * (1-c)**2
def f_0_var(c_var):
    return 2 * A * ((0 - c_var)**2 + 4*(0 - c_var)*(1 - c_var) + (1 - c_var)**2)
# free energy
def f(c):
    return f_0(c)+ .5*kappa*(c.grad.mag)**2




eqn = TransientTerm(coeff=1.) == DiffusionTerm(M * f_0_var(c_var)) - DiffusionTerm((M, kappa))




elapsed = 0.0
steps = 0
dt = 1e-2
dt_max = 1.0
total_sweeps = 2
tolerance = 1e-1

total_steps = 500
checkpoint = 50

duration = 100.0

c_var.updateOld()
solver = Solver()
dexp = -5
elapsed = 0.

F=[]


while elapsed < duration and steps < total_steps:
    res0 = eqn.sweep(c_var, dt=dt, solver=solver)

    for sweeps in range(total_sweeps):
        res = eqn.sweep(c_var, dt=dt, solver=solver)

    if res < res0 * tolerance:
        # anything in this loop will only be executed every $checkpoint steps
        # if (steps % checkpoint == 0):
        #     save_data(elapsed, c_var, f(c_var).cellVolumeAverage*mesh.numberOfCells*dx*dx, steps)
        dt = min(100, numerix.exp(dexp))
        elapsed += dt
        dexp += 0.01  
        
        
        ff = (f(c_var))
        cons = np.array(ff)
    
        f_sum = np.sum(cons)
        F.append(f_sum)
        
        steps += 1
        
      
        
        c_var.updateOld()
        
        viewer.plot()
    else:
        dt *= 0.8
        c_var[:] = c_var.old


end_time = time.time()

etime = -(strart_time-end_time)

print ("compute time is :%d seconds"%etime)























