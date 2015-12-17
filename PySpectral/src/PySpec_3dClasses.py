import numpy as np
import pyfftw
class gridclass:
  def initialize(self,N1,N2,N3):
    self.N1 = N1
    self.N2 = N2
    self.N3 = N3
    dx = 2*np.pi/(N1-1)
    dy = 2*np.pi/(N2-1)
    dz = 2*np.pi/(N3-1)
    x = np.linspace(0,2*np.pi-dx,N1)
    y = np.linspace(0,2*np.pi-dy,N2)
    z = np.linspace(0,2*np.pi-dz,N3) 
    self.y,self.x,self.z = np.meshgrid(y,x,z)
    k1 = np.fft.fftshift( np.linspace(-N1/2,N1/2-1,N1) )
    k2 = np.fft.fftshift( np.linspace(-N2/2,N2/2-1,N2) )
    k3 = np.linspace( 0,N3/2,N3/2+1 )
    self.k2,self.k1,self.k3 = np.meshgrid(k2,k1,k3)
    self.ksqr = self.k1*self.k1 + self.k2*self.k2 + self.k3*self.k3 + 1.e-50
    self.ksqr_i = 1./self.ksqr
    self.kc = np.amax(k1)
    self.Delta = np.pi/np.amax(np.abs(k1))

class FFTclass:
  def initialize(self,N1,N2,N3,nthreads):
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
