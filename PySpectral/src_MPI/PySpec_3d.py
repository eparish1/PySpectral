import numpy as np
import os
import time
import sys
from RHSfunctions import *
from Classes import gridclass, FFTclass, variables, utilitiesClass

## Check if variables exist
#==============================================
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
if 'initDomain' in globals():                #|
  pass                                       #| 
else:                                        #| 
  initDomain = 'Physical'                    #| 


#==============================================

# Initialize Classes. 
#=====================================================================
myFFT = FFTclass(N1,N2,N3,nthreads,fft_type,Npx,Npy,num_processes,comm,mpi_rank)
grid = gridclass(N1,N2,N3,x,y,z,kc,num_processes,L1,L2,L3,mpi_rank,comm,turb_model)
main = variables(turb_model,rotate,Om1,Om2,Om3,grid,u,v,w,uhat,vhat,what,t,dt,nu,myFFT,mpi_rank,initDomain)
utilities = utilitiesClass()
#====================================================================

# Make Solution Directory if it does not exist
if not os.path.exists('3DSolution'):
   os.makedirs('3DSolution')

# Save the grid information
np.savez('3DSolution/grid',k1=grid.k1,k2=grid.k2,k3=grid.k3,x=grid.x,y=grid.y,z=grid.z)
# Save the run information
np.savez('3DSolution/runinfo',turb_model=turb_model,dt=dt,save_freq=save_freq,nu=nu)

#RK4 time advancement function. Note that to save memory the computeRHS 
#function puts the RHS array into the Q array
def advanceQ_RK4(main,grid,myFFT):
  main.Q0[:] = main.Q[:]
  rk4const = np.array([1./4,1./3,1./2,1.])
  for i in range(0,4):
    main.computeRHS(main,grid,myFFT)
    main.Q = main.Q0 + main.dt*rk4const[i]*main.Q

t0 = time.time() #start the timer
main.U2Q() #distribute u variables to Q
main.iteration = 0 #time step iteration
#========== MAIN TIME INTEGRATION LOOP =======================
while main.t <= et:
  if (main.iteration%save_freq == 0): #call the savehook routine every save_freq iterations
    enstrophy,energy,dissipation,lambda_k,tau_k,Re_lambda,kspec,spectrum = utilities.computeAllStats(main,grid,myFFT)
    ktrans,transfer = utilities.computeTransfer(main,grid,myFFT)
    ktrans,transfer_res = utilities.computeTransfer_resolved(main,grid,myFFT)
    kspecr,spectrum_res = utilities.computeSpectrum_resolved(main,grid)
    E_res = utilities.computeEnergy_resolved(main,grid)
    myFFT.myifft3D(main.uhat,main.u)
    myFFT.myifft3D(main.vhat,main.v)
    myFFT.myifft3D(main.what,main.w)
    #myFFT.myifft3D(1j*grid.k1[:,None,None]*main.vhat - 1j*grid.k2[None,:,None]*main.uhat,main.w)
    uGlobal = allGather_physical(main.u,comm,mpi_rank,grid.N1,grid.N2,grid.N3,num_processes,Npy)
    vGlobal = allGather_physical(main.v,comm,mpi_rank,grid.N1,grid.N2,grid.N3,num_processes,Npy)
    wGlobal = allGather_physical(main.w,comm,mpi_rank,grid.N1,grid.N2,grid.N3,num_processes,Npy)
    if (mpi_rank == 0):
      string = '3DSolution/PVsol' + str(main.iteration)
      string2 = '3DSolution/npsol' + str(main.iteration)
      string3 = '3DSolution/npspec' + str(main.iteration)
      sys.stdout.write("===================================================================================== \n")
      sys.stdout.write("Wall Time= " + str(time.time() - t0) + "   t=" + str(main.t) + \
                       "   Energy = " + str(np.real(energy)) + "  eps = " + str(np.real(dissipation)) + " \n")
      sys.stdout.write("  tau_k/dt = " + str(np.real(tau_k/main.dt)) + \
                       "   Re_lambda = " + str(np.real(Re_lambda))   + "  lam/dx = " + \
                       str(np.real(lambda_k/grid.dx)) +  "\n")
      sys.stdout.flush()


      #gridToVTK(string, grid.xG,grid.yG,grid.zG, pointData = {"u" : np.real(uGlobal.transpose()) , \
      #    "v" : np.real(vGlobal.transpose()) , "w" : np.real(wGlobal.transpose())  } )
      np.savez(string2,u=uGlobal,v=vGlobal,w=wGlobal)
      np.savez(string3,k = kspec,spec = spectrum,kt = ktrans,T = transfer,spec_res = spectrum_res, \
               T_res = transfer_res,Re_lambda=Re_lambda,eps=np.real(dissipation),t=main.t,Energy = \
               energy,space_res = np.real(lambda_k/grid.dx),time_res = np.real(tau_k/main.dt))
  main.iteration += 1
  advanceQ_RK4(main,grid,myFFT) 
  main.t += main.dt



#=================================================================
#t1 = time.time()
#print('time = ' + str(t1 - t0))
#Dissipation = 1./(t_hist[1::] - t_hist[0:-1])*(Energy[1::] - Energy[0:-1])
#Dissipation_resolved = 1./(t_hist[1::] - t_hist[0:-1])*(Energy_resolved[1::] - Energy_resolved[0:-1])
#np.savez('3DSolution/stats',Energy=Energy,Dissipation=Dissipation,Energy_resolved=Energy_resolved,Dissipation_resolved=Dissipation_resolved,t=t_hist,Enstrophy=Enstrophy)

