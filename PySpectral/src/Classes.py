import numpy as np
import pyfftw
from RHSfunctions import *
class variables:
  def __init__(self,turb_model,grid,uhat,vhat,what,t,dt,nu,dt0,\
                dt0_subintegrations,dt1,dt1_subintegrations,cfl):
    self.turb_model = turb_model
    self.t = t
    self.kc = np.amax(grid.k1)
    self.dt = dt
    self.nu = nu
    self.tauhat = np.zeros((grid.N1,grid.N2,grid.N3/2+1,6),dtype='complex')
    self.uhat = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    self.uhat[:,:,:] = uhat[:,:,:]
    self.vhat = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    self.vhat[:,:,:] = vhat[:,:,:]
    self.what = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    self.what[:,:,:] = what[:,:,:]
    self.cfl = cfl
    ##============ DNS MODE ========================
    if (turb_model == 0):
      print('Not using any SGS')
      self.Q = np.zeros( (3*grid.N1,3*grid.N2,3*(grid.N3/2+1)),dtype='complex')
      self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
      self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
      self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def U2Q():
        self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
        self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
        self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def Q2U():
        self.uhat[:,:,:] = self.Q[0::3,0::3,0::3]
        self.vhat[:,:,:] = self.Q[1::3,1::3,1::3]
        self.what[:,:,:] = self.Q[2::3,2::3,2::3]
      self.computeRHS = computeRHS_NOSGS
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================

    ##============ SMAGORINSKY ====================
    if (turb_model == 1):
      print('Using Smagorinsky SGS')
      self.Q = np.zeros( (3*grid.N1,3*grid.N2,3*(grid.N3/2+1)),dtype='complex')
      self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
      self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
      self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def U2Q():
        self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
        self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
        self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def Q2U():
        self.uhat[:,:,:] = self.Q[0::3,0::3,0::3]
        self.vhat[:,:,:] = self.Q[1::3,1::3,1::3]
        self.what[:,:,:] = self.Q[2::3,2::3,2::3]
      self.computeRHS = computeRHS_SMAG
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================

    ##============ t-model ========================
    if (turb_model == 2):
      print('Using the t-model')
      self.PLQLu = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.PLQLv = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.PLQLw = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.Q = np.zeros( (3*grid.N1,3*grid.N2,3*(grid.N3/2+1)),dtype='complex')
      self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
      self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
      self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def U2Q():
        self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
        self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
        self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def Q2U():
        self.uhat[:,:,:] = self.Q[0::3,0::3,0::3]
        self.vhat[:,:,:] = self.Q[1::3,1::3,1::3]
        self.what[:,:,:] = self.Q[2::3,2::3,2::3]
      self.computeRHS = computeRHS_tmodel
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================

    ##============ FM1 model ========================
    if (turb_model == 3):

      print('Using the First Order Finite Memory Model')
      if dt0 == -10:
        print('Did not assign dt0 for FM1 Model, using default dt0=0.1')
        self.dt0 = 0.1
      else:
        print('Assigning dt0 = ' + str(dt0))
        self.dt0 = dt0

      if dt0_subintegrations == -10:
        print('Did not assign dt0_subintegrations for FM1 Model, using default dt_subintegrations=1')
        self.dt0_subintegrations = 1
      else:
        print('Assigning dt0_subintegrations = ' + str(dt0_subintegrations))
        self.dt0_subintegrations = dt0_subintegrations

      self.PLQLu = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.PLQLv = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.PLQLw = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.w0_u = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.w0_v = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.w0_w = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.nvars = 3 + 3*self.dt0_subintegrations
      self.Q = np.zeros( (self.nvars*grid.N1,self.nvars*grid.N2,self.nvars*(grid.N3/2+1)),dtype='complex')
      self.Q[0::self.nvars,0::self.nvars,0::self.nvars] = self.uhat[:,:,:]
      self.Q[1::self.nvars,1::self.nvars,1::self.nvars] = self.vhat[:,:,:]
      self.Q[2::self.nvars,2::self.nvars,2::self.nvars] = self.what[:,:,:]
      j = 3
      for i in range(0,self.dt0_subintegrations):
        self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars] = self.w0_u[:,:,:,i]
        self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars] = self.w0_v[:,:,:,i]
        self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars] = self.w0_w[:,:,:,i]
        j += 3

      def U2Q():
        self.Q[0::self.nvars,0::self.nvars,0::self.nvars] = self.uhat[:,:,:]
        self.Q[1::self.nvars,1::self.nvars,1::self.nvars] = self.vhat[:,:,:]
        self.Q[2::self.nvars,2::self.nvars,2::self.nvars] = self.what[:,:,:]
        j = 3
        for i in range(0,self.dt0_subintegrations):
          self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars] = self.w0_u[:,:,:,i]
          self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars] = self.w0_v[:,:,:,i]
          self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars] = self.w0_w[:,:,:,i]
          j += 3
      def Q2U():
        self.uhat[:,:,:] = self.Q[0::self.nvars,0::self.nvars,0::self.nvars]
        self.vhat[:,:,:] = self.Q[1::self.nvars,1::self.nvars,1::self.nvars]
        self.what[:,:,:] = self.Q[2::self.nvars,2::self.nvars,2::self.nvars]
        j = 3
        for i in range(0,self.dt0_subintegrations):
          self.w0_u[:,:,:,i] = self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars]
          self.w0_v[:,:,:,i] = self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars]
          self.w0_w[:,:,:,i] = self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars]
          j += 3
      self.computeRHS = computeRHS_FM1
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================

    ##============ FM2 model ========================
    if (turb_model == 4):
      print('Using the Second Order Finite Memory Model')
      if dt0 == -10:
        print('Did not assign dt0 for FM2 Model, using default dt0=0.1')
        self.dt0 = 0.1
      else:
        print('Assigning dt0 = ' + str(dt0))
        self.dt0 = dt0
      if dt1 == -10:
        print('Did not assign dt1 for FM2 Model, using default dt1=0.05')
        self.dt1 = 0.05
      else:
        print('Assigning dt1 = ' + str(dt1))
        self.dt1 = dt1
      if dt0_subintegrations == -10:
        print('Did not assign dt0_subintegrations for FM2 Model, using default dt_subintegrations=1')
        self.dt0_subintegrations = 1
      else:
        print('Assigning dt0_subintegrations = ' + str(dt0_subintegrations))
        self.dt0_subintegrations = dt0_subintegrations
      if dt1_subintegrations == -10:
        print('Did not assign dt1_subintegrations for FM2 Model, using default dt_subintegrations=1')
        self.dt1_subintegrations = 1
      else:
        print('Assigning dt1_subintegrations = ' + str(dt1_subintegrations))
        self.dt1_subintegrations = dt1_subintegrations


      self.nvars = 3 + 3*self.dt0_subintegrations + 3*self.dt1_subintegrations

      self.w0_v = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.w0_u = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.w0_w = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.w1_v = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt1_subintegrations),dtype='complex')
      self.w1_u = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt1_subintegrations),dtype='complex')
      self.w1_w = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt1_subintegrations),dtype='complex')
      self.Q = np.zeros( (self.nvars*grid.N1,self.nvars*grid.N2,self.nvars*(grid.N3/2+1)),dtype='complex')
      self.Q[0::self.nvars,0::self.nvars,0::self.nvars] = self.uhat[:,:,:]
      self.Q[1::self.nvars,1::self.nvars,1::self.nvars] = self.vhat[:,:,:]
      self.Q[2::self.nvars,2::self.nvars,2::self.nvars] = self.what[:,:,:]
      j = 3
      for i in range(0,self.dt0_subintegrations):
        self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars] = self.w0_u[:,:,:,i]
        self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars] = self.w0_v[:,:,:,i]
        self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars] = self.w0_w[:,:,:,i]
        if (i < self.dt1_subintegrations-1):
          j += 6
        else: 
          j += 3
      j = 6
      for i in range(0,self.dt1_subintegrations):
        print(i,j)
        self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars] = self.w1_u[:,:,:,i]
        print(i,j+1)
        self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars] = self.w1_v[:,:,:,i]
        print(i,j+2)
        self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars] = self.w1_w[:,:,:,i]
        print('Hi')
        if (i < self.dt0_subintegrations-1):
          j += 6
        else:
          j += 3

      def U2Q():
        self.Q[0::self.nvars,0::self.nvars,0::self.nvars] = self.uhat[:,:,:]
        self.Q[1::self.nvars,1::self.nvars,1::self.nvars] = self.vhat[:,:,:]
        self.Q[2::self.nvars,2::self.nvars,2::self.nvars] = self.what[:,:,:]
        j = 3
        for i in range(0,self.dt0_subintegrations):
          self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars] = self.w0_u[:,:,:,i]
          self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars] = self.w0_v[:,:,:,i]
          self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars] = self.w0_w[:,:,:,i]
          if (i < self.dt1_subintegrations - 1):
            j += 6
          else: 
            j += 3
        j = 6
        for i in range(0,self.dt1_subintegrations):
          self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars] = self.w1_u[:,:,:,i]
          self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars] = self.w1_v[:,:,:,i]
          self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars] = self.w1_w[:,:,:,i]
          if (i < self.dt0_subintegrations - 1):
            j += 6
          else:
            j += 3

      def Q2U():
        self.uhat[:,:,:] = self.Q[0::self.nvars,0::self.nvars,0::self.nvars]
        self.vhat[:,:,:] = self.Q[1::self.nvars,1::self.nvars,1::self.nvars]
        self.what[:,:,:] = self.Q[2::self.nvars,2::self.nvars,2::self.nvars]
        j = 3
        for i in range(0,self.dt0_subintegrations):
          self.w0_u[:,:,:,i] = self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars] 
          self.w0_v[:,:,:,i] = self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars] 
          self.w0_w[:,:,:,i] = self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars]
          if (i < self.dt1_subintegrations - 1):
            j += 6
          else: 
            j += 3
        j = 6
        for i in range(0,self.dt1_subintegrations):
          self.w1_u[:,:,:,i] = self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars] 
          self.w1_v[:,:,:,i] = self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars] 
          self.w1_w[:,:,:,i] = self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars] 
          if (i < self.dt0_subintegrations - 1):
            j += 6
          else:
            j += 3

      self.computeRHS = computeRHS_FM2
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================



    ##============ FM1 model -two term trapezoidal ========================
    ## ONLY NEED FOR VERIFICATION NOW. TURB MODELS 3,4 SHOULD HAVE AN N TERM TRAP RULE BUILT IN
    if (turb_model == 5):
      print('Using the Second Order Finite Memory Model')
      if dt0 == -10:
        print('Did not assign dt0 for FM1 Model, using default dt0=0.1')
        self.dt0 = 0.1
      else:
        print('Assigning dt0 = ' + str(dt0))
        self.dt0 = dt0
      self.w0_v = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.w0_u = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.w0_w = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.w01_v = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.w01_u = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.w01_w = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')

      self.Q = np.zeros( (9*grid.N1,9*grid.N2,9*(grid.N3/2+1)),dtype='complex')
      self.Q[0::9,0::9,0::9] = self.uhat[:,:,:]
      self.Q[1::9,1::9,1::9] = self.vhat[:,:,:]
      self.Q[2::9,2::9,2::9] = self.what[:,:,:]
      self.Q[3::9,3::9,3::9] = self.w0_u[:,:,:]
      self.Q[4::9,4::9,4::9] = self.w0_v[:,:,:]
      self.Q[5::9,5::9,5::9] = self.w0_w[:,:,:]
      self.Q[6::9,6::9,6::9] = self.w01_u[:,:,:]
      self.Q[7::9,7::9,7::9] = self.w01_v[:,:,:]
      self.Q[8::9,8::9,8::9] = self.w01_w[:,:,:]
      def U2Q():
        self.Q[0::9,0::9,0::9] = self.uhat[:,:,:]
        self.Q[1::9,1::9,1::9] = self.vhat[:,:,:]
        self.Q[2::9,2::9,2::9] = self.what[:,:,:]
        self.Q[3::9,3::9,3::9] = self.w0_u[:,:,:]
        self.Q[4::9,4::9,4::9] = self.w0_v[:,:,:]
        self.Q[5::9,5::9,5::9] = self.w0_w[:,:,:]
        self.Q[6::9,6::9,6::9] = self.w01_u[:,:,:]
        self.Q[7::9,7::9,7::9] = self.w01_v[:,:,:]
        self.Q[8::9,8::9,8::9] = self.w01_w[:,:,:]
      def Q2U():
        self.uhat[:,:,:] = self.Q[0::9,0::9,0::9]
        self.vhat[:,:,:] = self.Q[1::9,1::9,1::9]
        self.what[:,:,:] = self.Q[2::9,2::9,2::9]
        self.w0_u[:,:,:] = self.Q[3::9,3::9,3::9]
        self.w0_v[:,:,:] = self.Q[4::9,4::9,4::9]
        self.w0_w[:,:,:] = self.Q[5::9,5::9,5::9]
        self.w01_u[:,:,:] = self.Q[6::9,6::9,6::9]
        self.w01_v[:,:,:] = self.Q[7::9,7::9,7::9]
        self.w01_w[:,:,:] = self.Q[8::9,8::9,8::9]

      self.computeRHS = computeRHS_FM1_2term
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================



class gridclass:
  def __init__(self,N1,N2,N3,x,y,z,kc):
    self.N1 = N1
    self.N2 = N2
    self.N3 = N3
    self.x = np.zeros(np.shape(x))
    self.x[:,:,:] = x[:,:,:]
    self.y = np.zeros(np.shape(y))
    self.y[:,:,:] = y[:,:,:]
    self.z = np.zeros(np.shape(z))
    self.z[:,:,:] = z[:,:,:]
    self.dx = x[1,0,0] - x[0,0,0]
    self.dy = y[0,1,0] - y[0,0,0]
    self.dz = z[0,0,1] - z[0,0,0]
    k1 = np.fft.fftshift( np.linspace(-N1/2,N1/2-1,N1) )
    k2 = np.fft.fftshift( np.linspace(-N2/2,N2/2-1,N2) )
    k3 = np.linspace( 0,N3/2,N3/2+1 )
    k1f = np.fft.fftshift( np.linspace(-N1,N1-1,2.*N1) )
    k2f = np.fft.fftshift( np.linspace(-N2,N2-1,2.*N2) )
    k3f = np.linspace( 0,N3,N3+1 )

    self.k2,self.k1,self.k3 = np.meshgrid(k2,k1,k3)
    self.k2f,self.k1f,self.k3f = np.meshgrid(k2f,k1f,k3f)
  

    self.ksqr = self.k1*self.k1 + self.k2*self.k2 + self.k3*self.k3 + 1.e-50
    self.ksqr_i = 1./self.ksqr
    self.ksqrf = self.k1f*self.k1f + self.k2f*self.k2f + self.k3f*self.k3f + 1.e-50
    self.ksqrf_i = 1./self.ksqrf
    self.kc = kc
    self.Delta = np.pi/self.kc
    self.Gf = np.zeros(np.shape(self.k1)) #Sharp Spectral cutoff (we cutoff the oddball frequency)
    self.Gf[0:self.kc,0:self.kc,0:self.kc] = 1 # get first quardants
    self.Gf[0:self.kc,self.N2-self.kc+1::,0:self.kc] = 1 #0:kc in k1 and -kc:0 in k2
    self.Gf[self.N1-self.kc+1::,0:self.kc,0:self.kc] = 1 #-kc:0 in k1 and 0:kc in k2
    self.Gf[self.N1-self.kc+1::,self.N2-self.kc+1::,0:self.kc] = 1 #-kc:0 in k1 and k2

class FFTclass:
  def __init__(self,N1,N2,N3,nthreads):
    self.scale = np.sqrt( (3./2.)**3*np.sqrt(N1*N2*N3) ) #scaling for FFTS
    ## Inverse transforms of uhat,vhat,what are of the truncated padded variable. 
    ## Input is complex truncate,output is real untruncated
    self.invalT =    pyfftw.n_byte_align_empty((int(3./2.*N1),int(3./2.*N2),int(3./4.*N3+1)), 16, 'complex128')
    self.outvalT=    pyfftw.n_byte_align_empty((int(3./2.*N1),int(3./2.*N2),int(3./2*N3   )), 16, 'float64')
    self.ifftT_obj = pyfftw.FFTW(self.invalT,self.outvalT,axes=(0,1,2,),\
                     direction='FFTW_BACKWARD',threads=nthreads)
    ## Fourier transforms of padded vars like u*u.
    ## Input is real full, output is imag truncated 
    self.inval =   pyfftw.n_byte_align_empty((int(3./2.*N1),int(3./2.*N2),int(3./2.*N3) ), 16, 'float64')
    self.outval=   pyfftw.n_byte_align_empty((int(3./2.*N1),int(3./2.*N2),int(3./4*N3+1)), 16, 'complex128')
    self.fft_obj = pyfftw.FFTW(self.inval,self.outval,axes=(0,1,2,),\
                    direction='FFTW_FORWARD', threads=nthreads)

    self.invalT2 =    pyfftw.n_byte_align_empty((int(2.*N1),int(2.*N2),int(N3+1)), 16, 'complex128')
    self.outvalT2 =   pyfftw.n_byte_align_empty((int(2.*N1),int(2.*N2),int(2*N3)), 16, 'float64')
    self.ifftT_obj2 = pyfftw.FFTW(self.invalT2,self.outvalT2,axes=(0,1,2,),\
                     direction='FFTW_BACKWARD',threads=nthreads)

    self.inval2 =   pyfftw.n_byte_align_empty((int(2.*N1),int(2.*N2),int(2.*N3) ), 16, 'float64')
    self.outval2=   pyfftw.n_byte_align_empty((int(2.*N1),int(2.*N2),int(N3+1)), 16, 'complex128')
    self.fft_obj2 = pyfftw.FFTW(self.inval2,self.outval2,axes=(0,1,2,),\
                    direction='FFTW_FORWARD', threads=nthreads)

    self.invalT3 =    pyfftw.n_byte_align_empty((int(3.*N1),int(3.*N2),int(3*N3/2+1)), 16, 'complex128')
    self.outvalT3 =   pyfftw.n_byte_align_empty((int(3.*N1),int(3.*N2),int(3*N3)), 16, 'float64')
    self.ifftT_obj3 = pyfftw.FFTW(self.invalT3,self.outvalT3,axes=(0,1,2,),\
                     direction='FFTW_BACKWARD',threads=nthreads)

    self.inval3 =   pyfftw.n_byte_align_empty((int(3.*N1),int(3.*N2),int(3.*N3) ), 16, 'float64')
    self.outval3=   pyfftw.n_byte_align_empty((int(3.*N1),int(3.*N2),int(3*N3/2+1)), 16, 'complex128')
    self.fft_obj3 = pyfftw.FFTW(self.inval3,self.outval3,axes=(0,1,2,),\
                    direction='FFTW_FORWARD', threads=nthreads)





class utilitiesClass():
  def computeEnergy(self,main,grid):
      uE = np.sum(main.uhat[:,:,1:grid.N3/2]*np.conj(main.uhat[:,:,1:grid.N3/2]*2) ) + \
           np.sum(main.uhat[:,:,0]*np.conj(main.uhat[:,:,0])) 
      vE = np.sum(main.vhat[:,:,1:grid.N3/2]*np.conj(main.vhat[:,:,1:grid.N3/2]*2) ) + \
           np.sum(main.vhat[:,:,0]*np.conj(main.vhat[:,:,0])) 
      wE = np.sum(main.what[:,:,1:grid.N3/2]*np.conj(main.what[:,:,1:grid.N3/2]*2) ) + \
           np.sum(main.what[:,:,0]*np.conj(main.what[:,:,0]))
      return np.real(0.5*(uE + vE + wE)/(grid.N1*grid.N2*grid.N3))


  def computeEnergy_resolved(self,main,grid):
      uFilt = grid.Gf*main.uhat
      vFilt = grid.Gf*main.vhat
      wFilt = grid.Gf*main.what
      uE = np.sum(uFilt[:,:,1:grid.N3/2]*np.conj(uFilt[:,:,1:grid.N3/2]*2) ) + \
           np.sum(uFilt[:,:,0]*np.conj(uFilt[:,:,0])) 
      vE = np.sum(vFilt[:,:,1:grid.N3/2]*np.conj(vFilt[:,:,1:grid.N3/2]*2) ) + \
           np.sum(vFilt[:,:,0]*np.conj(vFilt[:,:,0])) 
      wE = np.sum(wFilt[:,:,1:grid.N3/2]*np.conj(wFilt[:,:,1:grid.N3/2]*2) ) + \
           np.sum(wFilt[:,:,0]*np.conj(wFilt[:,:,0]))
      return np.real(0.5*(uE + vE + wE)/(grid.N1*grid.N2*grid.N3))

  def compute_dt(self,main,grid):
    if (main.cfl > 0):
      u = np.fft.irfftn(main.uhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
      v = np.fft.irfftn(main.vhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
      w = np.fft.irfftn(main.what)*np.sqrt(grid.N1*grid.N2*grid.N3)
      max_vel = np.amax( abs(u)/grid.dx + abs(v)/grid.dy + abs(w)/grid.dz)
      main.dt = main.cfl/(max_vel + 1e-10)
      #CFL = c * dt / dx  -> dt = CFL*dx/c
      if (main.nu > 0):
        main.dt=np.minimum(main.dt,0.634/(1./grid.dx**2+1./grid.dy**2+1./grid.dz**2)/main.nu*main.cfl/1.35)
    else:
      main.dt = -main.cfl

  def computeEnstrophy(self,main,grid):
      omega1 = 1j*grid.k2*main.what - 1j*grid.k3*main.vhat
      omega2 = 1j*grid.k3*main.uhat - 1j*grid.k1*main.what
      omega3 = 1j*grid.k1*main.vhat - 1j*grid.k2*main.uhat
      om1E = np.sum(omega1[:,:,1:grid.N3/2]*np.conj(omega1[:,:,1:grid.N3/2]*2) ) + \
           np.sum(omega1[:,:,0]*np.conj(omega1[:,:,0])) 
      om2E = np.sum(omega2[:,:,1:grid.N3/2]*np.conj(omega2[:,:,1:grid.N3/2]*2) ) + \
           np.sum(omega2[:,:,0]*np.conj(omega2[:,:,0])) 
      om3E = np.sum(omega3[:,:,1:grid.N3/2]*np.conj(omega3[:,:,1:grid.N3/2]*2) ) + \
           np.sum(omega3[:,:,0]*np.conj(omega3[:,:,0]))
      return np.real(0.5*(om1E + om2E + om3E)/(grid.N1*grid.N2*grid.N3))

  def computeSpectrum(self,main,grid):
      k_m, indices1 = np.unique(np.rint(np.sqrt(grid.ksqr[:,:,1:grid.N3/2].flatten())), return_inverse=True)
      k_0, indices2 = np.unique(np.rint(np.sqrt(grid.ksqr[:,:,0].flatten())), return_inverse=True)
      spectrum = np.zeros((np.size(k_m),3),dtype='complex')
      np.add.at(spectrum[:,0],indices1,2*main.uhat[:,:,1:grid.N3/2].flatten()*np.conj(main.uhat[:,:,1:grid.N3/2].flatten()))
      np.add.at(spectrum[1::,0],indices2,main.uhat[:,:,0].flatten()*np.conj(main.uhat[:,:,0].flatten()))
      np.add.at(spectrum[:,1],indices1,2*main.vhat[:,:,1:grid.N3/2].flatten()*np.conj(main.vhat[:,:,1:grid.N3/2].flatten()))
      np.add.at(spectrum[1::,1],indices2,main.vhat[:,:,0].flatten()*np.conj(main.vhat[:,:,0].flatten()))
      np.add.at(spectrum[:,2],indices1,2*main.what[:,:,1:grid.N3/2].flatten()*np.conj(main.what[:,:,1:grid.N3/2].flatten()))
      np.add.at(spectrum[1::,2],indices2,main.what[:,:,0].flatten()*np.conj(main.what[:,:,0].flatten()))
      spectrum = spectrum/(grid.N1*grid.N2*grid.N3)
      return k_m,spectrum

  def computeSpectrum_resolved(self,main,grid):
      k_m, indices1 = np.unique(np.rint(np.sqrt(grid.ksqr[:,:,1:grid.N3/2].flatten())), return_inverse=True)
      k_0, indices2 = np.unique(np.rint(np.sqrt(grid.ksqr[:,:,0].flatten())), return_inverse=True)
      uFilt = grid.Gf*main.uhat
      vFilt = grid.Gf*main.vhat
      wFilt = grid.Gf*main.what
      spectrum = np.zeros((np.size(k_m),3),dtype='complex')
      np.add.at(spectrum[:,0],indices1,2*uFilt[:,:,1:grid.N3/2].flatten()*np.conj(uFilt[:,:,1:grid.N3/2].flatten()))
      np.add.at(spectrum[1::,0],indices2,uFilt[:,:,0].flatten()*np.conj(uFilt[:,:,0].flatten()))
      np.add.at(spectrum[:,1],indices1,2*vFilt[:,:,1:grid.N3/2].flatten()*np.conj(vFilt[:,:,1:grid.N3/2].flatten()))
      np.add.at(spectrum[1::,1],indices2,vFilt[:,:,0].flatten()*np.conj(vFilt[:,:,0].flatten()))
      np.add.at(spectrum[:,2],indices1,2*wFilt[:,:,1:grid.N3/2].flatten()*np.conj(wFilt[:,:,1:grid.N3/2].flatten()))
      np.add.at(spectrum[1::,2],indices2,wFilt[:,:,0].flatten()*np.conj(wFilt[:,:,0].flatten()))
      spectrum = spectrum/(grid.N1*grid.N2*grid.N3)
      return k_m,spectrum 
