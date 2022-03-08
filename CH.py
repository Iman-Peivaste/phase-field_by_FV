import time 
import numpy as np
from fipy import *
# import matplotlib.pyplot as plt
from vtk import *
strart_time = time.time ()
nx = ny = 64
dx = dy = .25
noise = 0.4

mesh = PeriodicGrid2D(nx=nx, ny=ny, dx=0.25, dy=0.25)
con = CellVariable(name="con", mesh=mesh)
si = CellVariable(name="si", mesh=mesh)
noise = GaussianNoiseVariable(mesh=mesh,
                              mean=noise,
                              variance=0.01).value
con[:] = noise


M = a = 1.0
kappa = 1.0

viewer = Viewer(vars=(con,), datamin=0., datamax=1.)
vtk(nx, ny, dx, dy, 1, np.reshape(con, (nx, ny)))

dfdcon = a * con * (1 - con) * (1 - 2 * con)
dfdcon_ = a * (1 - con) * (1 - 2 * con)

d2fdcon2 = a**2 * (1 - 6 * con * (1 - con))
eq1 = (TransientTerm(var=con) == DiffusionTerm(coeff=M, var=si))

eq2 = (ImplicitSourceTerm(coeff=1., var=si)
        == ImplicitSourceTerm(coeff=d2fdcon2, var=con) - d2fdcon2 * con + dfdcon
        - DiffusionTerm(coeff=kappa, var=con))

eq3 = (ImplicitSourceTerm(coeff=1., var=si)
        == ImplicitSourceTerm(coeff=dfdcon_, var=con)
        - DiffusionTerm(coeff=kappa, var=con))

eq = eq1 & eq3

def f_0(con):
    return a * (con )**2 * (1-con)**2
def f(con):
    return f_0(con)+ .5*kappa*(con.grad.mag)**2





duration = 200.
dt = 1e-2
dexp = -5
elapsed = 0.
F=[]
while elapsed < duration:
    dt = min(100, numerix.exp(dexp))
    elapsed += dt
    dexp += 0.01
    eq.solve(dt=dt)
  
    ff = (f(con))
    cons = np.array(ff)
    
    f_sum = np.sum(cons)
    F.append(f_sum)
    
    viewer.plot()
    print(elapsed)


end_time = time.time()

etime = -(strart_time-end_time)

print ("compute time is :%d seconds"%etime)














