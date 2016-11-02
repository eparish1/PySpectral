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
if 'time_scheme' in globals():               #|
  if (mpi_rank == 0):
    print('Using ' + time_scheme )           #|
else:                                        #| 
  time_scheme = 'RK4'                        #| 
  if (mpi_rank == 0):
    print('time_scheme not specified, using RK4')  #|
if 'IO' in globals():
  pass
else:
  IO = 'MPI'

if 'folderLoc' in globals():
  pass
else:
  folderLoc = '3DSolution'
#==============================================

# Make Solution Directory if it does not exist
if (mpi_rank == 0):
  if not os.path.exists(folderLoc):
     os.makedirs(folderLoc)


# Initialize Classes. 
#=====================================================================
myFFT = FFTclass(N1,N2,N3,nthreads,fft_type,Npx,Npy,num_processes,comm,mpi_rank)
grid = gridclass(N1,N2,N3,x,y,z,kc,num_processes,L1,L2,L3,mpi_rank,comm,turb_model)
main = variables(turb_model,rotate,Om1,Om2,Om3,grid,u,v,w,uhat,vhat,what,t,dt,nu,myFFT,mpi_rank,initDomain,time_scheme)
utilities = utilitiesClass()
#====================================================================

# Make Solution Directory if it does not exist
solloc = folderLoc + '/rank_' + str(mpi_rank)
if not os.path.exists(solloc):
   os.makedirs(solloc)

# Save the grid information
#np.savez('3DSolution/grid',k1=grid.k1,k2=grid.k2,k3=grid.k3,x=grid.x,y=grid.y,z=grid.z)
# Save the run information
if (mpi_rank == 0):
  string = folderLoc + '/runinfo'
  np.savez(string,turb_model=turb_model,dt=dt,save_freq=save_freq,nu=nu,N1=N1,N2=N2,N3=N3,num_processes=num_processes,et=et)


t0 = time.time() #start the timer
main.U2Q() #distribute u variables to Q
main.iteration = 0 #time step iteration
#========== MAIN TIME INTEGRATION LOOP =======================
while main.t <= et:
  if (mpi_rank == 0):
    sys.stdout.write("===================================================================================== \n")
    sys.stdout.write("Wall Time= " + str(time.time() - t0) + "   t=" + str(main.t) + "\n")
    sys.stdout.write("===================================================================================== \n")
    sys.stdout.flush()
  if (main.iteration%save_freq == 0): #call the savehook routine every save_freq iterations
    enstrophy,energy,dissipation,lambda_k,tau_k,Re_lambda,kspec,spectrum = utilities.computeAllStats(main,grid,myFFT)
    ktrans,transfer = utilities.computeTransfer(main,grid,myFFT)
    ktrans,transfer_res = utilities.computeTransfer_resolved(main,grid,myFFT)
    kspecr,spectrum_res = utilities.computeSpectrum_resolved(main,grid)
    E_res = utilities.computeEnergy_resolved(main,grid)
    myFFT.myifft3D(main.uhat,main.u)
    myFFT.myifft3D(main.vhat,main.v)
    myFFT.myifft3D(main.what,main.w)
    if (IO == 'serial'):
      uGlobal = allGather_physical(main.u,comm,mpi_rank,grid.N1,grid.N2,grid.N3,num_processes,Npy)
      vGlobal = allGather_physical(main.v,comm,mpi_rank,grid.N1,grid.N2,grid.N3,num_processes,Npy)
      wGlobal = allGather_physical(main.w,comm,mpi_rank,grid.N1,grid.N2,grid.N3,num_processes,Npy)

    if (mpi_rank == 0):
      string  = folderLoc + '/PVsol' + str(main.iteration)
      string2 = folderLoc + '/npsol' + str(main.iteration)
      string3 = folderLoc + '/npspec' + str(main.iteration)
      sys.stdout.write("===================================================================================== \n")
      sys.stdout.write("Wall Time= " + str(time.time() - t0) + "   t=" + str(main.t) + \
                       "   Energy = " + str(np.real(energy)) + "  eps = " + str(np.real(dissipation)) + " \n")
      sys.stdout.write("  tau_k/dt = " + str(np.real(tau_k/main.dt)) + \
                       "   Re_lambda = " + str(np.real(Re_lambda))   + "  lam/dx = " + \
                       str(np.real(lambda_k/grid.dx)) +  "\n")
      sys.stdout.flush()
      #gridToVTK(string, grid.xG,grid.yG,grid.zG, pointData = {"u" : np.real(uGlobal.transpose()) , \
      #    "v" : np.real(vGlobal.transpose()) , "w" : np.real(wGlobal.transpose())  } )
      if (IO == 'serial'):
        np.savez(string2,u=uGlobal,v=vGlobal,w=wGlobal)
      np.savez(string3,k = kspec,spec = spectrum,kt = ktrans,T = transfer,spec_res = spectrum_res, \
               T_res = transfer_res,Re_lambda=Re_lambda,eps=np.real(dissipation),t=main.t,Energy = \
               energy,space_res = np.real(lambda_k/grid.dx),time_res = np.real(tau_k/main.dt), \
               Energy_res = E_res)

    if (IO == 'MPI'):
      string2 = solloc + '/npsol' + str(main.iteration)
      if (main.turb_model == 'Orthogonal Dynamics'):
        utilities.computeF(main,grid,myFFT)
        np.savez(string2,u=u,v=v,w=w,F = main.F/eps)
      else:
        w0_u,w0_v,w0_w = utilities.computeSGS_DNS(main,grid,myFFT) 
        PLQLU = utilities.computePLQLU(main,grid,myFFT)
        Qcri = utilities.computeQcriterion(main,grid,myFFT)
        np.savez(string2,u=u,v=v,w=w,Q=Qcri,w0_u=w0_u,w0_v=w0_v,w0_w=w0_w,PLQLU=PLQLU)


  main.iteration += 1
  main.advanceQ(main,grid,myFFT) 
  main.t += main.dt



#=================================================================
#t1 = time.time()
#print('time = ' + str(t1 - t0))
#Dissipation = 1./(t_hist[1::] - t_hist[0:-1])*(Energy[1::] - Energy[0:-1])
#Dissipation_resolved = 1./(t_hist[1::] - t_hist[0:-1])*(Energy_resolved[1::] - Energy_resolved[0:-1])
#np.savez('3DSolution/stats',Energy=Energy,Dissipation=Dissipation,Energy_resolved=Energy_resolved,Dissipation_resolved=Dissipation_resolved,t=t_hist,Enstrophy=Enstrophy)

