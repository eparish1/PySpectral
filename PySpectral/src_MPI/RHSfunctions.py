import numpy as np
import sys
from mpi4py import MPI
from padding import *
def allGather_physical(tmp_local,comm,mpi_rank,N1,N2,N3,num_processes,Npy):
  data = comm.gather(tmp_local,root = 0)
  if (mpi_rank == 0):
    tmp_global = np.empty((N1,N2,N3))
    for j in range(0,num_processes):
      tmp_global[:,j*Npy:(j+1)*Npy,:] = data[j][:,:,:]
    return tmp_global

def allGather_spectral(tmp_local,comm,mpi_rank,N1,N2,N3,num_processes,Npx):
  data = comm.gather(tmp_local,root = 0)
  if (mpi_rank == 0):
    tmp_global = np.empty((N1,N2,N3/2+1),dtype='complex')
    for j in range(0,num_processes):
      tmp_global[j*Npx:(j+1)*Npx,:,:] = data[j][:,:,:]
    return tmp_global


def computeRHS_NOSGS(main,grid,myFFT):
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    mpi_rank = comm.Get_rank()

    main.Q2U()
    
    main.uhat = myFFT.dealias(main.uhat,grid)
    main.vhat = myFFT.dealias(main.vhat,grid)
    main.what = myFFT.dealias(main.what,grid)

    myFFT.myifft3D(main.uhat,main.u)
    myFFT.myifft3D(main.vhat,main.v)
    myFFT.myifft3D(main.what,main.w)

    myFFT.myfft3D(main.u*main.u,main.NL[0])
    myFFT.myfft3D(main.v*main.v,main.NL[1])
    myFFT.myfft3D(main.w*main.w,main.NL[2])
    myFFT.myfft3D(main.u*main.v,main.NL[3])
    myFFT.myfft3D(main.u*main.w,main.NL[4])
    myFFT.myfft3D(main.v*main.w,main.NL[5])

    phat  = grid.ksqr_i*( -grid.k1[:,None,None]*grid.k1[:,None,None]*main.NL[0] - grid.k2[None,:,None]*grid.k2[None,:,None]*main.NL[1] - \
             grid.k3[None,None,:]*grid.k3[None,None,:]*main.NL[2] - 2.*grid.k1[:,None,None]*grid.k2[None,:,None]*main.NL[3] - \
             2.*grid.k1[:,None,None]*grid.k3[None,None,:]*main.NL[4] - 2.*grid.k2[None,:,None]*grid.k3[None,None,:]*main.NL[5] )

    if (main.rotate == 1):
      phat[:,:,:] = phat[:,:,:] - 2.*grid.ksqr_i*1j*( grid.k1[:,None,None]*(main.vhat*main.Om3 - main.what*main.Om2) + 
                    grid.k2[None,:,None]*(main.what*main.Om1 - main.uhat*main.Om3) + \
                    grid.k3[None,None,:]*(main.uhat*main.Om2 - main.vhat*main.Om1))

    #==================== RK4  ====================================
    if (main.time_scheme == 'RK4'):
      main.Q[0] = myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[0] - 1j*grid.k2[None,:,None]*main.NL[3] - 1j*grid.k3[None,None,:]*main.NL[4] - \
                                           1j*grid.k1[:,None,None]*phat - main.nu*grid.ksqr*main.uhat ,grid)
  
      main.Q[1] = myFFT.dealias(-1j*grid.k1[:,None,None]*main.NL[3] - 1j*grid.k2[None,:,None]*main.NL[1] - 1j*grid.k3[None,None,:]*main.NL[5] - \
                                           1j*grid.k2[None,:,None]*phat - main.nu*grid.ksqr*main.vhat ,grid)
  
      main.Q[2] = myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[4] - 1j*grid.k2[None,:,None]*main.NL[5] - 1j*grid.k3[None,None,:]*main.NL[2] - \
                                           1j*grid.k3[None,None,:]*phat - main.nu*grid.ksqr*main.what ,grid)
  
      if (main.rotate == 1):
        main.Q[0] = main.Q[0] + 2.*(main.vhat*main.Om3 - main.what*main.Om2)
        main.Q[1] = main.Q[1] + 2.*(main.what*main.Om1 - main.uhat*main.Om3)
        main.Q[2] = main.Q[2] + 2.*(main.uhat*main.Om2 - main.vhat*main.Om1)
    #=========================================================================
    #================== Crank Nicolson/Adams Bashforth ==============
    if (main.time_scheme == 'Semi-Implicit'):
      main.H_old[:] = main.H[:]
      main.H[0] = myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[0] - 1j*grid.k2[None,:,None]*main.NL[3] - 1j*grid.k3[None,None,:]*main.NL[4] - \
                                           1j*grid.k1[:,None,None]*phat,grid)
  
      main.H[1] = myFFT.dealias(-1j*grid.k1[:,None,None]*main.NL[3] - 1j*grid.k2[None,:,None]*main.NL[1] - 1j*grid.k3[None,None,:]*main.NL[5] - \
                                           1j*grid.k2[None,:,None]*phat,grid)
  
      main.H[2] = myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[4] - 1j*grid.k2[None,:,None]*main.NL[5] - 1j*grid.k3[None,None,:]*main.NL[2] - \
                                           1j*grid.k3[None,None,:]*phat,grid)
 
      main.viscous_term[0] = -main.nu*grid.ksqr*main.uhat 
      main.viscous_term[1] = -main.nu*grid.ksqr*main.vhat 
      main.viscous_term[2] = -main.nu*grid.ksqr*main.what 

      if (main.rotate == 1):
        main.H[0] = main.H[0] + 2.*(main.vhat*main.Om3 - main.what*main.Om2)
        main.H[1] = main.H[1] + 2.*(main.what*main.Om1 - main.uhat*main.Om3)
        main.H[2] = main.H[2] + 2.*(main.uhat*main.Om2 - main.vhat*main.Om1)






def computeRHS_MHD(main,grid,myFFT):
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    mpi_rank = comm.Get_rank()

    main.Q2U()
    
    main.uhat = myFFT.dealias(main.uhat,grid)
    main.vhat = myFFT.dealias(main.vhat,grid)
    main.what = myFFT.dealias(main.what,grid)
    main.B1hat = myFFT.dealias(main.B1hat,grid)
    main.B2hat = myFFT.dealias(main.B2hat,grid)
    main.B3what = myFFT.dealias(main.B3hat,grid)

    myFFT.myifft3D(main.uhat,main.u)
    myFFT.myifft3D(main.vhat,main.v)
    myFFT.myifft3D(main.what,main.w)
    myFFT.myifft3D(main.B1hat,main.B1)
    myFFT.myifft3D(main.B2hat,main.B2)
    myFFT.myifft3D(main.B3hat,main.B3)


    myFFT.myfft3D(main.u*main.u,main.NL[0])
    myFFT.myfft3D(main.v*main.v,main.NL[1])
    myFFT.myfft3D(main.w*main.w,main.NL[2])
    myFFT.myfft3D(main.u*main.v,main.NL[3])
    myFFT.myfft3D(main.u*main.w,main.NL[4])
    myFFT.myfft3D(main.v*main.w,main.NL[5])

    ## now compute nonlinear terms for MHD
    myFFT.myfft3D(main.v*main.B3 - main.w*main.B2,main.u_x_B[0])
    myFFT.myfft3D(main.w*main.B1 - main.u*main.B3,main.u_x_B[1])
    myFFT.myfft3D(main.u*main.B2 - main.v*main.B1,main.u_x_B[2])

    myFFT.myfft3D(main.B1*main.B1,main.MHDNL[0])
    myFFT.myfft3D(main.B2*main.B2,main.MHDNL[1])
    myFFT.myfft3D(main.B3*main.B3,main.MHDNL[2])
    myFFT.myfft3D(main.B1*main.B2,main.MHDNL[3])
    myFFT.myfft3D(main.B1*main.B3,main.MHDNL[4])
    myFFT.myfft3D(main.B2*main.B3,main.MHDNL[5])

    phat1  = grid.ksqr_i*( -grid.k1[:,None,None]*grid.k1[:,None,None]*main.NL[0] - grid.k2[None,:,None]*grid.k2[None,:,None]*main.NL[1] - \
             grid.k3[None,None,:]*grid.k3[None,None,:]*main.NL[2] - 2.*grid.k1[:,None,None]*grid.k2[None,:,None]*main.NL[3] - \
             2.*grid.k1[:,None,None]*grid.k3[None,None,:]*main.NL[4] - 2.*grid.k2[None,:,None]*grid.k3[None,None,:]*main.NL[5] )

    phat2  = grid.ksqr_i*( -grid.k1[:,None,None]*grid.k1[:,None,None]*main.MHDNL[0] - grid.k2[None,:,None]*grid.k2[None,:,None]*main.MHDNL[1] - \
             grid.k3[None,None,:]*grid.k3[None,None,:]*main.MHDNL[2] - 2.*grid.k1[:,None,None]*grid.k2[None,:,None]*main.MHDNL[3] - \
             2.*grid.k1[:,None,None]*grid.k3[None,None,:]*main.MHDNL[4] - 2.*grid.k2[None,:,None]*grid.k3[None,None,:]*main.MHDNL[5] )
 
    phat = phat1 - phat2



    #==================== RK4  ====================================
    if (main.time_scheme == 'RK4'):
      main.Q[0] = myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[0] - 1j*grid.k2[None,:,None]*main.NL[3] - 1j*grid.k3[None,None,:]*main.NL[4] + \
                                  1j*grid.k1[:,None,None]*main.MHDNL[0] + 1j*grid.k2[None,:,None]*main.MHDNL[3] + 1j*grid.k3[None,None,:]*main.MHDNL[4] - \
                                           1j*grid.k1[:,None,None]*phat - main.nu*grid.ksqr*main.uhat + main.force[0],grid)  
  
      main.Q[1] = myFFT.dealias(-1j*grid.k1[:,None,None]*main.NL[3] - 1j*grid.k2[None,:,None]*main.NL[1] - 1j*grid.k3[None,None,:]*main.NL[5] + \
                                 1j*grid.k1[:,None,None]*main.MHDNL[3] + 1j*grid.k2[None,:,None]*main.MHDNL[1] + 1j*grid.k3[None,None,:]*main.MHDNL[5] - \
                                           1j*grid.k2[None,:,None]*phat - main.nu*grid.ksqr*main.vhat + main.force[1] ,grid)
  
      main.Q[2] = myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[4] - 1j*grid.k2[None,:,None]*main.NL[5] - 1j*grid.k3[None,None,:]*main.NL[2] + \
                                  1j*grid.k1[:,None,None]*main.MHDNL[4] + 1j*grid.k2[None,:,None]*main.MHDNL[5] + 1j*grid.k3[None,None,:]*main.MHDNL[2] - \
                                           1j*grid.k3[None,None,:]*phat - main.nu*grid.ksqr*main.what + main.force[2],grid)

      main.Q[3] = myFFT.dealias( (1j*grid.k2[None,:,None]*main.u_x_B[2] - 1j*grid.k3[None,None,:]*main.u_x_B[1]) - \
                                           main.lam*grid.ksqr*main.B1hat ,grid)

      main.Q[4] = myFFT.dealias( (1j*grid.k3[None,None,:]*main.u_x_B[0] - 1j*grid.k1[:,None,None]*main.u_x_B[2]) - \
                                           main.lam*grid.ksqr*main.B2hat ,grid)

      main.Q[5] = myFFT.dealias( (1j*grid.k1[:,None,None]*main.u_x_B[1] - 1j*grid.k2[None,:,None]*main.u_x_B[0]) - \
                                           main.lam*grid.ksqr*main.B3hat ,grid)

    #=========================================================================
    else:
      if (mpi_rank == 0):
        print('Error, time scheme' + main.time_scheme + ' not implemented for MHD equations. PySpectral Quitting')
      sys.exit()



def computeRHS_Ortho(main,grid,myFFT):
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    mpi_rank = comm.Get_rank()

    main.Q2U()
    
    main.uhat = myFFT.dealias(main.uhat,grid)
    main.vhat = myFFT.dealias(main.vhat,grid)
    main.what = myFFT.dealias(main.what,grid)

    myFFT.myifft3D(main.uhat,main.u)
    myFFT.myifft3D(main.vhat,main.v)
    myFFT.myifft3D(main.what,main.w)

    myFFT.myfft3D(main.u*main.u,main.NL[0])
    myFFT.myfft3D(main.v*main.v,main.NL[1])
    myFFT.myfft3D(main.w*main.w,main.NL[2])
    myFFT.myfft3D(main.u*main.v,main.NL[3])
    myFFT.myfft3D(main.u*main.w,main.NL[4])
    myFFT.myfft3D(main.v*main.w,main.NL[5])

    phat  = grid.ksqr_i*( -grid.k1[:,None,None]*grid.k1[:,None,None]*main.NL[0] - grid.k2[None,:,None]*grid.k2[None,:,None]*main.NL[1] - \
             grid.k3[None,None,:]*grid.k3[None,None,:]*main.NL[2] - 2.*grid.k1[:,None,None]*grid.k2[None,:,None]*main.NL[3] - \
             2.*grid.k1[:,None,None]*grid.k3[None,None,:]*main.NL[4] - 2.*grid.k2[None,:,None]*grid.k3[None,None,:]*main.NL[5] )

    if (main.rotate == 1):
      phat[:,:,:] = phat[:,:,:] - 2.*grid.ksqr_i*1j*( grid.k1[:,None,None]*(main.vhat*main.Om3 - main.what*main.Om2) + 
                    grid.k2[None,:,None]*(main.what*main.Om1 - main.uhat*main.Om3) + \
                    grid.k3[None,None,:]*(main.uhat*main.Om2 - main.vhat*main.Om1))

    #==================== RK4  ====================================
    if (main.time_scheme == 'RK4'):
      main.Q[0] = myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[0] - 1j*grid.k2[None,:,None]*main.NL[3] - 1j*grid.k3[None,None,:]*main.NL[4] - \
                                           1j*grid.k1[:,None,None]*phat - main.nu*grid.ksqr*main.uhat ,grid)
  
      main.Q[1] = myFFT.dealias(-1j*grid.k1[:,None,None]*main.NL[3] - 1j*grid.k2[None,:,None]*main.NL[1] - 1j*grid.k3[None,None,:]*main.NL[5] - \
                                           1j*grid.k2[None,:,None]*phat - main.nu*grid.ksqr*main.vhat ,grid)
  
      main.Q[2] = myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[4] - 1j*grid.k2[None,:,None]*main.NL[5] - 1j*grid.k3[None,None,:]*main.NL[2] - \
                                           1j*grid.k3[None,None,:]*phat - main.nu*grid.ksqr*main.what ,grid)
  
      if (main.rotate == 1):
        main.Q[0] = main.Q[0] + 2.*(main.vhat*main.Om3 - main.what*main.Om2)
        main.Q[1] = main.Q[1] + 2.*(main.what*main.Om1 - main.uhat*main.Om3)
        main.Q[2] = main.Q[2] + 2.*(main.uhat*main.Om2 - main.vhat*main.Om1)
    #=========================================================================

    uhat_f = grid.filter(main.uhat)
    vhat_f = grid.filter(main.vhat)
    what_f = grid.filter(main.what)
    u_f = np.zeros(np.shape(main.u))
    v_f = np.zeros(np.shape(main.v))
    w_f = np.zeros(np.shape(main.w))
    myFFT.myifft3D(uhat_f,u_f)
    myFFT.myifft3D(vhat_f,v_f)
    myFFT.myifft3D(what_f,w_f)

    myFFT.myfft3D(u_f*u_f,main.NL[0])
    myFFT.myfft3D(v_f*v_f,main.NL[1])
    myFFT.myfft3D(w_f*w_f,main.NL[2])
    myFFT.myfft3D(u_f*v_f,main.NL[3])
    myFFT.myfft3D(u_f*w_f,main.NL[4])
    myFFT.myfft3D(v_f*w_f,main.NL[5])

    phat_f  = grid.ksqr_i*( -grid.k1[:,None,None]*grid.k1[:,None,None]*main.NL[0] - grid.k2[None,:,None]*grid.k2[None,:,None]*main.NL[1] - \
             grid.k3[None,None,:]*grid.k3[None,None,:]*main.NL[2] - 2.*grid.k1[:,None,None]*grid.k2[None,:,None]*main.NL[3] - \
             2.*grid.k1[:,None,None]*grid.k3[None,None,:]*main.NL[4] - 2.*grid.k2[None,:,None]*grid.k3[None,None,:]*main.NL[5] )

    if (main.rotate == 1):
      phat_f[:,:,:] = phat_f[:,:,:] - 2.*grid.ksqr_i*1j*( grid.k1[:,None,None]*(vhat_f*main.Om3 - what_f*main.Om2) + 
                    grid.k2[None,:,None]*(what_f*main.Om1 - uhat_f*main.Om3) + \
                    grid.k3[None,None,:]*(uhat_f*main.Om2 - vhat_f*main.Om1))

    #==================== RK4  ====================================
    if (main.time_scheme == 'RK4'):
      main.Q[0] = main.Q[0] - myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[0] - 1j*grid.k2[None,:,None]*main.NL[3] - 1j*grid.k3[None,None,:]*main.NL[4] - \
                                           1j*grid.k1[:,None,None]*phat_f - main.nu*grid.ksqr*uhat_f ,grid)
  
      main.Q[1] = main.Q[1] - myFFT.dealias(-1j*grid.k1[:,None,None]*main.NL[3] - 1j*grid.k2[None,:,None]*main.NL[1] - 1j*grid.k3[None,None,:]*main.NL[5] - \
                                           1j*grid.k2[None,:,None]*phat_f - main.nu*grid.ksqr*vhat_f ,grid)
  
      main.Q[2] = main.Q[2] - myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[4] - 1j*grid.k2[None,:,None]*main.NL[5] - 1j*grid.k3[None,None,:]*main.NL[2] - \
                                           1j*grid.k3[None,None,:]*phat_f - main.nu*grid.ksqr*what_f ,grid)
  
      if (main.rotate == 1):
        main.Q[0] = main.Q[0] - 2.*(vhat_f*main.Om3 - what_f*main.Om2)
        main.Q[1] = main.Q[1] - 2.*(what_f*main.Om1 - uhat_f*main.Om3)
        main.Q[2] = main.Q[2] - 2.*(uhat_f*main.Om2 - vhat_f*main.Om1)

      #main.F[:] = main.Q[:]
    #=========================================================================
      
