import numpy as np
import os
import pyfftw
import time
import sys
from Classes import gridclass, FFTclass, variables,utilitiesClass
from savehook import savehook
utilities = utilitiesClass()
myFFT = FFTclass(N1,N2,N3,nthreads)
grid = gridclass(N1,N2,N3,x,y,z)
main = variables(turb_model,grid,uhat,vhat,what,t,dt,nu)


if not os.path.exists('3DSolution'):
   os.makedirs('3DSolution')

def advanceQ_RK4(main,grid,myFFT):
  Q0 = np.zeros(np.shape(main.Q),dtype='complex')
  Q0[:,:,:] = main.Q[:,:,:]
  rk4const = np.array([1./4,1./3,1./2,1.])
  for i in range(0,4):
    main.computeRHS(main,grid,myFFT)
    main.Q[:,:,:] = Q0[:,:,:] + main.dt*rk4const[i]*main.Q[:,:,:]


t0 = time.time()
main.U2Q()

iteration = 0
Energy = np.zeros(1)

Energy[0] = utilities.computeEnergy(main.uhat,main.vhat,main.what,grid)
while t <= et:
  main.t = t
  advanceQ_RK4(main,grid,myFFT)
  t += dt
  if (iteration%save_freq == 0):
    savehook(main.uhat,main.vhat,main.what,grid,iteration)
  iteration += 1
  Energy = np.append(Energy, utilities.computeEnergy(main.uhat,main.vhat,main.what,grid) )
  sys.stdout.write("Wall Time= " + str(time.time() - t0) + "   t=" + str(t) + \
                   "   Energy = " + str(np.real(Energy[-1]))  + "\n")
  sys.stdout.flush()
 
t1 = time.time()
print('time = ' + str(t1 - t0))
Dissipation = 1./dt*(Energy[1::] - Energy[0:-1])
np.savez('3DSolution/stats',Energy=Energy,Dissipation=Dissipation,t=np.linspace(0,t+dt,np.size(Energy)))

