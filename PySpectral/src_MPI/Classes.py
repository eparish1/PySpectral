import numpy as np
import sys
from RHSfunctions import *

class variables:
  def __init__(self,turb_model,rotate,Om1,Om2,Om3,grid,u,v,w,t,dt,nu,myFFT):
    self.turb_model = turb_model
    self.rotate = rotate
    self.Om1 = Om1
    self.Om2 = Om2
    self.Om3 = Om3
    self.t = t
    self.kc = np.amax(grid.k1)
    self.dt = dt
    self.nu = nu
    self.NL = np.empty((6,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    self.u = u
    self.v = v
    self.w = w
    self.uhat = np.zeros((grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    self.vhat = np.zeros((grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    self.what = np.zeros((grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    myFFT.myfft3D(self.u,self.uhat)
    myFFT.myfft3D(self.v,self.vhat)
    myFFT.myfft3D(self.w,self.what)
    #self.cfl = cfl
    ##============ DNS MODE ========================
    if (turb_model == 0):
      sys.stdout.write('Not using any SGS \n')
      self.Q = np.zeros( (3,grid.Npx,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.Q0 = np.zeros( (3,grid.Npx,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.nvars = 3
      self.Q[0] = self.uhat[:,:,:]
      self.Q[1] = self.vhat[:,:,:]
      self.Q[2] = self.what[:,:,:]
      def U2Q():
        self.Q[0] = self.uhat[:,:,:]
        self.Q[1] = self.vhat[:,:,:]
        self.Q[2] = self.what[:,:,:]
      def Q2U():
        self.uhat[:,:,:] = self.Q[0]
        self.vhat[:,:,:] = self.Q[1]
        self.what[:,:,:] = self.Q[2]
      self.computeRHS = computeRHS_NOSGS
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================


class gridclass:
  def __init__(self,N1,N2,N3,x,y,z,kc,num_processes,L1,L3,mpi_rank,comm,turb_model):
    self.Npx = int(float(N1 / num_processes))
    self.Npy = int(float(N2 / num_processes))
    self.N1 = N1
    self.N2 = N2
    self.N3 = N3
    self.x = x
    self.y = y
    self.z = z
    self.xG = allGather_physical(self.x,comm,mpi_rank,self.N1,self.N2,self.N3,num_processes,self.Npy)
    self.yG = allGather_physical(self.y,comm,mpi_rank,self.N1,self.N2,self.N3,num_processes,self.Npy)
    self.zG = allGather_physical(self.z,comm,mpi_rank,self.N1,self.N2,self.N3,num_processes,self.Npy)
    self.dx = x[1,0,0] - x[0,0,0]
    self.dy = y[0,1,0] - y[0,0,0]
    self.dz = z[0,0,1] - z[0,0,0]
    self.k1 = np.fft.fftshift( np.linspace(-N1/2,N1/2-1,N1) )[mpi_rank*self.Npx:(mpi_rank+1)*self.Npx] *2.*np.pi/L1
    self.k2 = np.fft.fftshift( np.linspace(-N2/2,N2/2-1,N2) ) *2.*np.pi/L1
    self.k3 = np.linspace( 0,N3/2,N3/2+1 ) *2.*np.pi/L1
    self.ksqr = self.k1[:,None,None]*self.k1[:,None,None] + self.k2[None,:,None]*self.k2[None,:,None] + self.k3[None,None,:]*self.k3[None,None,:] + 1.e-50
    self.ksqr_i = 1./self.ksqr
    self.kc = kc
    self.Delta = np.pi/self.kc
    self.dealias_x = np.ones(self.Npx)
    self.dealias_y = np.ones(self.N2)
    self.dealias_z = np.ones(self.N3/2+1)
    for i in range(0,self.Npx):
      if (abs(self.k1[i]) >= (self.N1/2)*2./3.*2.*np.pi/L1):
        self.dealias_x[i] = 0.
    self.dealias_y[int( (self.N2/2)*2./3. )] = 0.
    self.dealias_z[int( (self.N3/2)*2./3. )] = 0.

class FFTclass:
  def __init__(self,N1,N2,N3,nthreads,fft_type,Npx,Npy,num_processes,comm,mpi_rank):
    self.nthreads = nthreads
    self.Uc_hat = np.zeros((Npx,N2,N3/2+1),dtype='complex')
    self.Uc_hatT = np.zeros((N1,Npy,N3/2+1) ,dtype='complex')
    self.U_mpi = np.zeros((num_processes,Npx,Npy,N3/2+1),dtype='complex')

    def myifft3D(uhat,u):
      self.Uc_hat[:,:,:] = np.fft.ifft(uhat,axis=1)
      self.U_mpi[:] = np.rollaxis(self.Uc_hat.reshape(Npx, num_processes, Npy, N3/2+1) ,1)
      comm.Alltoall(self.U_mpi,self.Uc_hatT)
      u[:] = np.fft.irfft2(self.Uc_hatT,axes=(0,2) ) * N1 * N3
      return u

    def myfft3D(u,uhat):
      self.Uc_hatT[:,:,:] = np.fft.rfft2(u,axes=(0,2) ) / (N1 * N3)
      comm.Alltoall(self.Uc_hatT, self.U_mpi )
      self.Uc_hat[:,:,:] = np.rollaxis(self.U_mpi,1).reshape(self.Uc_hat.shape)
      uhat[:] = np.fft.fft(self.Uc_hat,axis=1)
      return uhat
    self.myfft3D = myfft3D
    self.myifft3D = myifft3D

    def dealias(uhat,grid):
      uhat[:,:,:] = grid.dealias_x[:,None,None]*grid.dealias_y[None,:,None]*grid.dealias_z[None,None,:]
      return uhat 
    self.dealias = dealias
