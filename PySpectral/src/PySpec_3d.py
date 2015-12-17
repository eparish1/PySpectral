import numpy as np
import os
import pyfftw
import time
if not os.path.exists('3DSolution'):
   os.makedirs('3DSolution')

def advanceQ_RK4(dt,Q,uhat,vhat,what,nu,grid,myFFT):
  Q0 = np.zeros(np.shape(Q),dtype='complex')
  Q0[:,:,:] = Q
  rk4const = np.array([1./4,1./3,1./2,1.])
  for i in range(0,4):
    RHS = computeRHS(Q,uhat,vhat,what,nu,grid,myFFT)
    Q = Q0 + dt*rk4const[i]*RHS
  return Q


t0 = time.time()
Q = np.zeros((3*N1,3*N2,3*(N3/2+1)),dtype='complex')
Q = U2Q(Q,uhat,vhat,what)

iteration = 0
Energy = np.zeros(1)

Energy[0] = utilities.computeEnergy(uhat,vhat,what,grid)
while t <= et:
  Q = advanceQ_RK4(dt,Q,uhat,vhat,what,nu,grid,myFFT)
  t += dt
  if (iteration%save_freq == 0):
    savehook(uhat,vhat,what,grid,iteration)
  iteration += 1
  Energy = np.append(Energy, utilities.computeEnergy(uhat,vhat,what,grid) )
  sys.stdout.write("Wall Time= " + str(time.time() - t0) + "   t=" + str(t) + \
                   "   Energy = " + str(np.real(Energy[-1]))  + "\n")
  sys.stdout.flush()
 
t1 = time.time()
print('time = ' + str(t1 - t0))
Dissipation = 1./dt*(Energy[1::] - Energy[0:-1])
numpy.savez('3DSolution/stats',Energy=Energy,Dissipation=Dissipation,t=np.linspace(0,t+dt,np.size(Energy)))

