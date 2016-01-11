import numpy as np
import os
import pyfftw
import time
import sys
from Classes import gridclass, FFTclass, variables,utilitiesClass
utilities = utilitiesClass()
myFFT = FFTclass(N1,N2,N3,nthreads)
grid = gridclass(N1,N2,N3,x,y,z,kc)
main = variables(turb_model,grid,uhat,vhat,what,t,dt,nu)

# Make Solution Directory if it does not exist
if not os.path.exists('3DSolution'):
   os.makedirs('3DSolution')

# Save the grid information
np.savez('3DSolution/grid',k1=grid.k1,k2=grid.k2,k3=grid.k3,x=grid.x,y=grid.y,z=grid.z)
# Save the run information
np.savez('3DSolution/runinfo',turb_model=turb_model,dt=dt,save_freq=save_freq)

#RK4 time advancement function. Note that to save memory the computeRHS 
#function puts the RHS array into the Q array
def advanceQ_RK4(main,grid,myFFT):
  Q0 = np.zeros(np.shape(main.Q),dtype='complex')
  Q0[:,:,:] = main.Q[:,:,:]
  rk4const = np.array([1./4,1./3,1./2,1.])
  for i in range(0,4):
    main.computeRHS(main,grid,myFFT)
    main.Q[:,:,:] = Q0[:,:,:] + main.dt*rk4const[i]*main.Q[:,:,:]


t0 = time.time() #start the timer
main.U2Q() #distribute u variables to Q

iteration = 0 #time step iteration
Energy = np.zeros(1) #initialize an array for energy
Energy[0] = utilities.computeEnergy(main,grid)
Energy_resolved = np.zeros(1) #initialize an array for resolved energy
Energy_resolved[0] = utilities.computeEnergy_resolved(main,grid)

#========== MAIN TIME INTEGRATION LOOP =======================
while t <= et:
  main.t = t 
  advanceQ_RK4(main,grid,myFFT) 
  t += dt
  if (iteration%save_freq == 0): #call the savehook routine every save_freq iterations
    savehook(main,grid,iteration)
  iteration += 1
  Energy = np.append(Energy, utilities.computeEnergy(main,grid) ) #add to the energy array
  Energy_resolved = np.append(Energy_resolved, utilities.computeEnergy_resolved(main,grid) ) #add to the energy array
  print(Energy - Energy_resolved)
  #print out stats
  sys.stdout.write("Wall Time= " + str(time.time() - t0) + "   t=" + str(t) + \
                   "   Energy = " + str(np.real(Energy[-1]))  + "\n")
  sys.stdout.flush()

#=================================================================
t1 = time.time()
print('time = ' + str(t1 - t0))
Dissipation = 1./dt*(Energy[1::] - Energy[0:-1])
Dissipation_resolved = 1./dt*(Energy_resolved[1::] - Energy_resolved[0:-1])
np.savez('3DSolution/stats',Energy=Energy,Dissipation=Dissipation,Energy_resolved=Energy_resolved,Dissipation_resolved=Dissipation_resolved,t=np.linspace(0,t+dt,np.size(Energy)))

