import numpy as np
import pyfftw
from padding import *
### Generate the variables ("main") class from a solution file
class variablesFromFile:
  def __init__(self,gridFile,solutionFile,infoFile,kc):
    self.turb_model = infoFile['turb_model'].mean()  #turb_model is a 1 element array for some reason, take mean to get rid of array
    self.kc = kc
    self.uhat = np.zeros(np.shape(solutionFile['uhat']),dtype='complex')
    self.uhat[:,:,:] = solutionFile['uhat'][:,:,:]
    self.vhat = np.zeros(np.shape(solutionFile['vhat']),dtype='complex')
    self.vhat[:,:,:] = solutionFile['vhat'][:,:,:]
    self.what = np.zeros(np.shape(solutionFile['what']),dtype='complex')
    self.what[:,:,:] = solutionFile['what'][:,:,:]
    if (self.turb_model != 0):
      self.w0_u = solutionFile['w0_u'][:,:,:,0]
      self.w0_v = solutionFile['w0_v'][:,:,:,0]
      self.w0_w = solutionFile['w0_w'][:,:,:,0]

  def setFieldsFromFile(self,solutionFile):
    self.uhat[:,:,:] = solutionFile['uhat'][:,:,:]
    self.vhat[:,:,:] = solutionFile['vhat'][:,:,:]
    self.what[:,:,:] = solutionFile['what'][:,:,:]
    if (self.turb_model != 0):
      self.w0_u = solutionFile['w0_u'][:,:,:,0]
      self.w0_v = solutionFile['w0_v'][:,:,:,0]
      self.w0_w = solutionFile['w0_w'][:,:,:,0]

### Same for a grid class
class gridFromFile:
  def __init__(self,gridFile,solutionFile,infoFile,kc):
    N1,N2,N3 = np.shape(solutionFile['uhat'])
    N3 = (N3-1)*2
    self.N1 = N1
    self.N2 = N2
    self.N3 = N3
    self.k1 = np.zeros((N1,N2,N3/2+1))
    self.k2 = np.zeros((N1,N2,N3/2+1))
    self.k3 = np.zeros((N1,N2,N3/2+1))
    
    self.k1[:,:,:] = gridFile['k1']
    self.k2[:,:,:] = gridFile['k2']
    self.k3[:,:,:] = gridFile['k3']

    self.ksqr = self.k1*self.k1 + self.k2*self.k2 + self.k3*self.k3 + 1.e-50
    self.ksqr_i = 1./self.ksqr
    self.kc = kc
    self.Delta = np.pi/self.kc
    self.Gf = np.zeros(np.shape(self.k1)) #Sharp Spectral cutoff (we cutoff the oddball frequency)
    self.Gf[0:self.kc,0:self.kc,0:self.kc] = 1 # get first quardants
    self.Gf[0:self.kc,self.N2-self.kc+1::,0:self.kc] = 1 #0:kc in k1 and -kc:0 in k2
    self.Gf[self.N1-self.kc+1::,0:self.kc,0:self.kc] = 1 #-kc:0 in k1 and 0:kc in k2
    self.Gf[self.N1-self.kc+1::,self.N2-self.kc+1::,0:self.kc] = 1 #-kc:0 in k1 and k2


## And same for FFT class
class FFTclass:
  def __init__(self,gridFile,nthreads):
    N1,N2,N3 = np.shape(gridFile['k1'])
    N3 = (N3-1)*2
    self.nthreads = nthreads
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


