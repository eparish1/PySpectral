import numpy as np
import os
import pyfftw
import time
import sys
from Classes import gridclass, FFTclass, variables,utilitiesClass


## Check if variables exist
#==============================================
if 'weave' in globals():	             #|
  pass					     #|
else:				             #|
  weave = 0				     #|
if 'rotate' in globals():	             #|
  if ('Om1' in globals()\
  and 'Om2' in globals()\
  and 'Om3' in globals()):
    pass
  else:
   print('Error, rotate is on but not \
          all Om1,2,3 are defined')	     #|
else:				             #|
  rotate = 0                                 #|
  Om1 = 0                                    #|
  Om2 = 0                                    #|
  Om3 = 0                                    #|
                                             #|
if 'cfl' in globals():			     #|
  pass					     #|
else:				             #|
  cfl = -dt				     #|
if 'Ct' in globals():                        #|
  pass					     #|
else:					     #|
  Ct = -10				     #|
if 'dt0' in globals():                       #|
  pass					     #|
else:					     #|
  dt0 = -10				     #|
if 'dt1' in globals():  	 	     #|
  pass                                       #|
else:                                        #|
  dt1 = -10                                  #|
                                             #|
if 'dt0_subintegrations' in globals():       #|
  pass                                       #|
else:                                        #|
  dt0_subintegrations = -10                  #|
                                             #|
if 'dt1_subintegrations' in globals():       #|
  pass                                       #| 
else:                                        #| 
  dt1_subintegrations = -10                  #| 
#==============================================

# Initialize Classes. 
#=====================================================================
utilities = utilitiesClass()
myFFT = FFTclass(N1,N2,N3,nthreads)
grid = gridclass(N1,N2,N3,x,y,z,kc)
main = variables(weave,turb_model,rotate,Om1,Om2,Om3,grid,uhat,vhat,what,t,dt,nu,Ct,dt0,\
                 dt0_subintegrations,dt1,dt1_subintegrations,cfl)
#====================================================================

# Make Solution Directory if it does not exist
if not os.path.exists('3DSolution'):
   os.makedirs('3DSolution')

# Save the grid information
np.savez('3DSolution/grid',k1=grid.k1,k2=grid.k2,k3=grid.k3,x=grid.x,y=grid.y,z=grid.z)
# Save the run information
np.savez('3DSolution/runinfo',turb_model=turb_model,dt=dt,save_freq=save_freq)

#RK4 time advancement function. Note that to save memory the computeRHS 
#function puts the RHS array into the Q array
def advanceQ_RK4(main,grid,myFFT,utilities):
  Q0 = np.zeros(np.shape(main.Q),dtype='complex')
  Q0[:,:,:] = main.Q[:,:,:]
  rk4const = np.array([1./4,1./3,1./2,1.])
  utilities.preAdvanceQ_hook(main,grid,myFFT) #pass by default. 
  utilities.compute_dt(main,grid)
  for i in range(0,4):
    main.computeRHS(main,grid,myFFT)
    main.Q = Q0 + main.dt*rk4const[i]*main.Q
  utilities.postAdvanceQ_hook(main,grid,myFFT)

t0 = time.time() #start the timer
main.U2Q() #distribute u variables to Q

iteration = 0 #time step iteration
Energy = np.zeros(1) #initialize an array for energy
Energy[0] = utilities.computeEnergy(main,grid)
Enstrophy = np.zeros(1) #initialize an array for energy
Enstrophy[0] = utilities.computeEnstrophy(main,grid)

Energy_resolved = np.zeros(1) #initialize an array for resolved energy
Energy_resolved[0] = utilities.computeEnergy_resolved(main,grid)
t_hist = np.zeros(1)
t_hist[0] = 0
#========== MAIN TIME INTEGRATION LOOP =======================
while t <= et:
  main.t = t 
  main.iteration = iteration
  advanceQ_RK4(main,grid,myFFT,utilities) 
  t += main.dt
  if (iteration%save_freq == 0): #call the savehook routine every save_freq iterations
    savehook(main,grid,utilities,iteration)
  iteration += 1
  enstrophy,energy,dissipation,lambda_k,tau_k,Re_lambda = utilities.computeAllStats(main,grid)
  Energy = np.append(Energy,energy) #add to the energy array
  Enstrophy = np.append(Enstrophy,enstrophy) #add to the energy array
  Energy_resolved = np.append(Energy_resolved, utilities.computeEnergy_resolved(main,grid) ) #add to the energy array
  t_hist = np.append(t_hist,t)
  #print out stats
  sys.stdout.write("===================================================================================== \n")
  sys.stdout.write("Wall Time= " + str(time.time() - t0) + "   t=" + str(t) + \
                   "   Energy = " + str(np.real(energy)) + "  eps = " + str(np.real(dissipation)) + " \n")
  sys.stdout.write("  tau_k/dt = " + str(np.real(tau_k/main.dt)) + \
                   "   Re_lambda = " + str(np.real(Re_lambda))   + "  lam/dx = " + \
                   str(np.real(lambda_k/grid.dx)) +  "\n")
  sys.stdout.flush()

#=================================================================
t1 = time.time()
print('time = ' + str(t1 - t0))
Dissipation = 1./(t_hist[1::] - t_hist[0:-1])*(Energy[1::] - Energy[0:-1])
Dissipation_resolved = 1./(t_hist[1::] - t_hist[0:-1])*(Energy_resolved[1::] - Energy_resolved[0:-1])
np.savez('3DSolution/stats',Energy=Energy,Dissipation=Dissipation,Energy_resolved=Energy_resolved,Dissipation_resolved=Dissipation_resolved,t=t_hist,Enstrophy=Enstrophy)

