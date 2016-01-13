import numpy as np
import pyfftw
from RHSfunctions import *
class variables:
  def __init__(self,turb_model,grid,uhat,vhat,what,t,dt,nu):
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

    ##============ DNS MODE ========================
    if (turb_model == 0):
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
      self.dt0 = 0.5
      self.w0_v = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.w0_u = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.w0_w = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.Q = np.zeros( (6*grid.N1,6*grid.N2,6*(grid.N3/2+1)),dtype='complex')
      self.Q[0::6,0::6,0::6] = self.uhat[:,:,:]
      self.Q[1::6,1::6,1::6] = self.vhat[:,:,:]
      self.Q[2::6,2::6,2::6] = self.what[:,:,:]
      self.Q[3::6,3::6,3::6] = self.w0_u[:,:,:]
      self.Q[4::6,4::6,4::6] = self.w0_v[:,:,:]
      self.Q[5::6,5::6,5::6] = self.w0_w[:,:,:]
      def U2Q():
        self.Q[0::6,0::6,0::6] = self.uhat[:,:,:]
        self.Q[1::6,1::6,1::6] = self.vhat[:,:,:]
        self.Q[2::6,2::6,2::6] = self.what[:,:,:]
        self.Q[3::6,3::6,3::6] = self.w0_u[:,:,:]
        self.Q[4::6,4::6,4::6] = self.w0_v[:,:,:]
        self.Q[5::6,5::6,5::6] = self.w0_w[:,:,:]
      def Q2U():
        self.uhat[:,:,:] = self.Q[0::6,0::6,0::6]
        self.vhat[:,:,:] = self.Q[1::6,1::6,1::6]
        self.what[:,:,:] = self.Q[2::6,2::6,2::6]
        self.w0_u[:,:,:] = self.Q[3::6,3::6,3::6]
        self.w0_v[:,:,:] = self.Q[4::6,4::6,4::6]
        self.w0_w[:,:,:] = self.Q[5::6,5::6,5::6]
      self.computeRHS = computeRHS_FM1
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
    self.Gf[0:self.kc+1,0:self.kc+1,0:self.kc+1] = 1 # get first quardants
    self.Gf[0:self.kc+1,self.N2-self.kc::,0:self.kc+1] = 1 #0:kc in k1 and -kc:0 in k2
    self.Gf[self.N1-self.kc::,0:self.kc+1,0:self.kc+1] = 1 #-kc:0 in k1 and 0:kc in k2
    self.Gf[self.N1-self.kc::,self.N2-self.kc::,0:self.kc+1] = 1 #-kc:0 in k1 and k2

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

