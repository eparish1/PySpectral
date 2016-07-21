import numpy as np
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



#
#def computeTransfer_2D(self,main,grid,myFFT):
#    main.Q2U()
#    main.uhat = myFFT.dealias(main.uhat,grid)
#    main.vhat = myFFT.dealias(main.vhat,grid)
#    main.what = myFFT.dealias(main.what,grid)
#    myFFT.myifft3D(main.uhat,main.u)
#    myFFT.myifft3D(main.vhat,main.v)
#    myFFT.myifft3D(main.what,main.w)
#
#    myFFT.myfft3D(main.u*main.u,main.NL[0])
#    myFFT.myfft3D(main.v*main.v,main.NL[1])
#    myFFT.myfft3D(main.w*main.w,main.NL[2])
#    myFFT.myfft3D(main.u*main.v,main.NL[3])
#    myFFT.myfft3D(main.u*main.w,main.NL[4])
#    myFFT.myfft3D(main.v*main.w,main.NL[5])
#
#
#    phat  = grid.ksqr_i*( -grid.k1[:,None,None]*grid.k1[:,None,None]*main.NL[0] - grid.k2[None,:,None]*grid.k2[None,:,None]*main.NL[1] - \
#             grid.k3[None,None,:]*grid.k3[None,None,:]*main.NL[2] - 2.*grid.k1[:,None,None]*grid.k2[None,:,None]*main.NL[3] - \
#             2.*grid.k1[:,None,None]*grid.k3[None,None,:]*main.NL[4] - 2.*grid.k2[None,:,None]*grid.k3[None,None,:]*main.NL[5] )
#
#    if (main.rotate == 1):
#      phat[:,:,:] = phat[:,:,:] - 2.*grid.ksqr_i*1j*( grid.k1[:,None,None]*(main.vhat*main.Om3 - main.what*main.Om2) +
#                    grid.k2[None,:,None]*(main.what*main.Om1 - main.uhat*main.Om3) + \
#                    grid.k3[None,None,:]*(main.uhat*main.Om2 - main.vhat*main.Om1))
#
#    main.Q[0] = myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[0] - 1j*grid.k2[None,:,None]*main.NL[3] - 1j*grid.k3[None,None,:]*main.NL[4] - \
#                                         1j*grid.k1[:,None,None]*phat - main.nu*grid.ksqr*main.uhat ,grid)
#
#    main.Q[1] = myFFT.dealias(-1j*grid.k1[:,None,None]*main.NL[3] - 1j*grid.k2[None,:,None]*main.NL[1] - 1j*grid.k3[None,None,:]*main.NL[5] - \
#                                         1j*grid.k2[None,:,None]*phat - main.nu*grid.ksqr*main.vhat ,grid)
#
#    main.Q[2] = myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[4] - 1j*grid.k2[None,:,None]*main.NL[5] - 1j*grid.k3[None,None,:]*main.NL[2] - \
#                                         1j*grid.k3[None,None,:]*phat - main.nu*grid.ksqr*main.what ,grid)
#
#
#
#
# 910 
# 911 
# 912     RHSu[:,:,:] = np.conj(main.uhat)*(-1j*grid.k1*uuhat - 1j*grid.k2*uvhat - 1j*grid.k3*uwhat - \
# 913                                        1j*grid.k1*phat )
# 914 
# 915     RHSv[:,:,:] = np.conj(main.vhat)*(-1j*grid.k1*uvhat - 1j*grid.k2*vvhat - 1j*grid.k3*vwhat - \
# 916                                        1j*grid.k2*phat)
# 917 
# 918     RHSw[:,:,:] = np.conj(main.what)*(-1j*grid.k1*uwhat - 1j*grid.k2*vwhat - 1j*grid.k3*wwhat - \
# 919                                       1j*grid.k3*phat)
# 920     kmag = np.sqrt(grid.k1**2 + grid.k2**2)
# 921     k_m, indices1 = np.unique((np.rint(kmag[:,:,0].flatten())), return_inverse=True)
# 922     kmax = np.int(np.round(np.amax(k_m)))
# 923     kdata = np.linspace(0,kmax,kmax+1)
# 924     spectrum = np.zeros((kmax+1,3),dtype='complex')
# 925     np.add.at( spectrum[:,0],np.int8(k_m[indices1]),RHSu[:,:,0].flatten())
# 926     np.add.at( spectrum[:,1],np.int8(k_m[indices1]),RHSv[:,:,0].flatten())
# 927     Transfer = (spectrum[:,0] + spectrum[:,1] + spectrum[:,2] ) / (grid.N1*grid.N2*grid.N3)
# 928     return kdata,Transfer

