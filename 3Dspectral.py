from pylab import *
import numpy
import scipy.signal 
import os
import pyfftw
from evtk.hl import gridToVTK
import time
close("all")
if not os.path.exists('3DSolution'):
   os.makedirs('3DSolution')

def unpad(uhat_pad,arrange):
  N1 = shape(uhat_pad)[0]/2
  N2 = shape(uhat_pad)[1]/2
  N3 = shape(uhat_pad)[2]/2
  uhat = zeros((N1,N2,N3),dtype = 'complex')
  if (arrange == 0):
    ## Remove padding from the outsides of the 3D cube
    uhat[:,:] = uhat_pad[N1/2:N1/2 + N1,N2/2:N2/2 + N2,N3/2:N3/2+N3]
  if (arrange == 1):
    ## Remove padding from the middle of the 3D cube
    uhat[0:N1/2,0:N2/2,0:N3/2] = uhat_pad[0:N1/2      ,0:N2/2      ,0:N3/2     ] #left lower back (0,0,0)
    uhat[0:N1/2,N2/2::,0:N3/2] = uhat_pad[0:N1/2      ,2*N2-N2/2:: ,0:N3/2     ] #left upper back (0,1,0)
    uhat[0:N1/2,0:N2/2,N3/2::] = uhat_pad[0:N1/2      ,0:N2/2      ,2*N3-N3/2::] #left lower forward (0,0,1)
    uhat[0:N1/2,N2/2::,N3/2::] = uhat_pad[0:N1/2      ,2*N2-N2/2:: ,2*N3-N3/2::] #left upper forward (0,1,1)

    uhat[N1/2::,0:N2/2,0:N3/2] = uhat_pad[2*N1-N1/2:: ,0:N2/2      ,0:N3/2     ] #right lower back (1,0,0)
    uhat[N1/2::,N2/2::,0:N3/2] = uhat_pad[2*N1-N1/2:: ,2*N2-N2/2:: ,0:N3/2     ] #right upper back (1,1,0)
    uhat[N1/2::,0:N2/2,N3/2::] = uhat_pad[2*N1-N1/2:: ,0:N2/2      ,2*N3-N3/2::] # right lower forward (1,0,1)
    uhat[N1/2::,N2/2::,N3/2::] = uhat_pad[2*N1-N1/2:: ,2*N2-N2/2:: ,2*N3-N3/2::] #right upper forward (1,1,1)
  return uhat


def pad(uhat,arrange):
  N1,N2,N3 = shape(uhat)
  if (arrange == 0):
    ## Add padding to the outsides of the 3D cube
    uhat_pad = zeros((2*N1,2*N2,2*N3),dtype = 'complex')
    uhat_pad[N1/2:N1/2 + N1,N2/2:N2/2 + N2,N3/2:N3/2+N3] = uhat
  if (arrange == 1):
    ## Add padding to the middle of the 3D cube
    uhat_pad = zeros((2*N1,2*N2,2*N3),dtype = 'complex')
    uhat_pad[0:N1/2      ,0:N2/2      ,0:N3/2     ] = uhat[0:N1/2,0:N2/2,0:N3/2] #left lower back (0,0,0)
    uhat_pad[0:N1/2      ,2*N2-N2/2:: ,0:N3/2     ] = uhat[0:N1/2,N2/2::,0:N3/2] #left upper back (0,1,0)
    uhat_pad[0:N1/2      ,0:N2/2      ,2*N3-N3/2::] = uhat[0:N1/2,0:N2/2,N3/2::] #left lower forward (0,0,1)
    uhat_pad[0:N1/2      ,2*N2-N2/2:: ,2*N3-N3/2::] = uhat[0:N1/2,N2/2::,N3/2::] #left upper forward (0,1,1)

    uhat_pad[2*N1-N1/2:: ,0:N2/2      ,0:N3/2     ] = uhat[N1/2::,0:N2/2,0:N3/2] #right lower back (1,0,0)
    uhat_pad[2*N1-N1/2:: ,2*N2-N2/2:: ,0:N3/2     ] = uhat[N1/2::,N2/2::,0:N3/2] #right upper back (1,1,0)
    uhat_pad[2*N1-N1/2:: ,0:N2/2      ,2*N3-N3/2::] = uhat[N1/2::,0:N2/2,N3/2::] #right lower forward (1,0,1)
    uhat_pad[2*N1-N1/2:: ,2*N2-N2/2:: ,2*N3-N3/2::] = uhat[N1/2::,N2/2::,N3/2::] #right upper forward (1,1,1)

  return uhat_pad

def Q2U(Q,uhat,vhat,what):
  uhat[:,:] = Q[0::3,0::3,0::3]
  vhat[:,:] = Q[1::3,1::3,1::3]
  what[:,:] = Q[2::3,2::3,2::3]
  return uhat,vhat,what

def U2Q(Q,uhat,vhat,what):
  Q[0::3,0::3,0::3] = uhat[:,:]
  Q[1::3,1::3,1::3] = vhat[:,:]
  Q[2::3,2::3,2::3] = what[:,:]
  return Q

def imag2Real_pad(hatvar):
  realvar = ifftn(pad(hatvar,1))*sqrt(N1*N2*N3)*8
  return realvar

def multReal2imag(real1,real2):
  hat12 = unpad( fftn(real1,real2)/(sqrt(N1*N2*N3)*8),1)
  return hat12

def computeRHS(Q,uhat,vhat,what,grid,myFFT):
  uhat,vhat,what = Q2U(Q,uhat,vhat,what)
  RHS = zeros((3*grid.N1,3*grid.N2,3*grid.N3),dtype='complex')
  uhat_pad = pad(uhat,1)
  vhat_pad = pad(vhat,1)
  what_pad = pad(what,1)
  scale = sqrt(8*sqrt(grid.N1*grid.N2*grid.N3))
  myFFT.inval[:,:,:] = uhat_pad[:,:,:]
  ureal = myFFT.ifft_obj()*scale
  myFFT.inval[:,:,:] = vhat_pad[:,:,:]
  vreal = myFFT.ifft_obj()*scale
  myFFT.inval[:,:,:] = what_pad[:,:,:]
  wreal = myFFT.ifft_obj()*scale

  myFFT.inval[:,:,:] = ureal[:,:,:]*ureal[:,:,:]
  uuhat = unpad( myFFT.fft_obj(),1)
  myFFT.inval[:,:,:] = vreal[:,:,:]*vreal[:,:,:]
  vvhat = unpad( myFFT.fft_obj(),1)
  myFFT.inval[:,:,:] = wreal[:,:,:]*wreal[:,:,:]
  wwhat = unpad( myFFT.fft_obj(),1)
  myFFT.inval[:,:,:] = ureal[:,:,:]*vreal[:,:,:]
  uvhat = unpad( myFFT.fft_obj(),1)
  myFFT.inval[:,:,:] = ureal[:,:,:]*wreal[:,:,:]
  uwhat = unpad( myFFT.fft_obj(),1)
  myFFT.inval[:,:,:] = vreal[:,:,:]*wreal[:,:,:]
  vwhat = unpad( myFFT.fft_obj(),1)
  phat  = -grid.k1*grid.k1*grid.ksqr_i*uuhat - grid.k2*grid.k2*grid.ksqr_i*vvhat - \
           grid.k3*grid.k3*grid.ksqr_i*wwhat - 2.*grid.k1*grid.k2*grid.ksqr_i*uvhat - \
           2.*grid.k1*grid.k3*grid.ksqr_i*uwhat - 2.*grid.k2*grid.k3*grid.ksqr_i*vwhat
  RHS[0:3*N1:3,0:3*N2:3,0:3*N3:3] = -1j*grid.k1*uuhat - 1j*grid.k2*uvhat - 1j*grid.k3*uwhat - \
                                         1j*grid.k1*phat - nu*grid.ksqr*uhat

  RHS[1:3*N1:3,1:3*N2:3,1:3*N3:3] = -1j*grid.k1*uvhat - 1j*grid.k2*vvhat - 1j*grid.k3*vwhat - \
                                         1j*grid.k2*phat - nu*grid.ksqr*vhat

  RHS[2:3*N1:3,2:3*N2:3,2:3*N3:3] = -1j*grid.k1*uwhat - 1j*grid.k2*vwhat - 1j*grid.k3*wwhat - \
                                         1j*grid.k3*phat - nu*grid.ksqr*what
  return RHS
def computePressure(uuhat,vvhat,wwhat,uvhat,uwhat,vwhat,k1,k2,i3):
  ksqr = k1**2 + k2**2 + k3**2 + 1.e-30
  phat = -k1*k1*ksqr_i*uuhat - k2*k2*ksqr_i*vvhat - k3*k3*ksqr_i  \
      - 2.*k1*k2*ksqr_i*uvhat - 2.*k1*k3*ksqr_i*uwhat - 2.*k2*k3*ksqr_i*vwhat
  return phat


rk4const = array([1./4,1./3,1./2,1.])
def advanceQ_RK4(dt,Q,uhat,vhat,what,grid,myFFT):
  Q0 = zeros(shape(Q),dtype='complex')
  Q0[:,:,:] = Q
  for i in range(0,4):
    RHS = computeRHS(Q,uhat,vhat,what,grid,myFFT)
    Q = Q0 + dt*rk4const[i]*RHS
    uhat,vhat,what = Q2U(Q,uhat,vhat,what)
  return Q


class gridclass:
  def initialize(self,N1,N2,N3):
    self.N1 = N1
    self.N2 = N2
    self.N3 = N3
    dx = 2*pi/(N1-1)
    dy = 2*pi/(N2-1)
    dz = 2*pi/(N3-1)
    x = linspace(0,2*pi-dx,N1)
    y = linspace(0,2*pi-dy,N2)
    z = linspace(0,2*pi-dz,N3) 
    self.y,self.x,self.z = meshgrid(y,x,z)
    k1 = fftshift( linspace(-N1/2,N1/2-1,N1) )
    k2 = fftshift( linspace(-N2/2,N2/2-1,N2) )
    k3 = fftshift( linspace(-N3/2,N3/2-1,N3) )
    self.k2,self.k1,self.k3 = meshgrid(k2,k1,k3)
    self.ksqr = self.k1*self.k1 + self.k2*self.k2 + self.k3*self.k3 + 1.e-50
    self.ksqr_i = 1./self.ksqr

class FFTclass:
  def initialize(self,N1,N2,N3):
    self.inval = pyfftw.n_byte_align_empty((2*N1,2*N2,2*N3), 64, 'complex128')
    self.outval = pyfftw.n_byte_align_empty((2*N1,2*N2,2*N3), 64, 'complex128')
    self.fft_obj = pyfftw.FFTW(self.inval,self.outval, axes=(0,1,2), threads=16)
    self.ifft_obj = pyfftw.FFTW(self.inval,self.outval, axes=(0,1,2),  direction='FFTW_BACKWARD',threads=16)

N1 = 64
N2 = 64
N3 = 64
myFFT = FFTclass() 
myFFT.initialize(N1,N2,N3)
grid = gridclass()
grid.initialize(N1,N2,N3)
nu = 0.01

u =  cos(grid.x)*sin(grid.y)*cos(grid.z)
v = -sin(grid.x)*cos(grid.y)*cos(grid.z)
w = zeros((N1,N2,N3))
uhat =  numpy.fft.fftn(u) / sqrt(N1*N2*N3) 
vhat =  numpy.fft.fftn(v) / sqrt(N1*N2*N3)
what =  numpy.fft.fftn(w) / sqrt(N1*N2*N3)



#t = 0.2: 0.108
t0 = time.time()
t = 0
et = 1.e-1
dt = 1.e-2
Q = zeros((3*N1,3*N2,3*N3),dtype='complex')
Q = U2Q(Q,uhat,vhat,what)
iteration = 0
save_freq = 10
Energy = zeros(1)
Energy[0] = real( mean(0.5*uhat*conj(uhat) + 0.5*vhat*conj(vhat) + 0.5*what*conj(what)) )
print(Energy[0])
while t <= et:
  Q = advanceQ_RK4(dt,Q,uhat,vhat,what,grid,myFFT)
  t += dt
  if (iteration%save_freq == 0):
    string = '3DSolution/sol' + str(iteration)
    u = ifftn(uhat)*sqrt(N1*N2*N3)
    v = ifftn(vhat)*sqrt(N1*N2*N3)
    w = ifftn(what)*sqrt(N1*N2*N3)
    gridToVTK(string, grid.x,grid.y,grid.z, pointData = {"u" : real(u.transpose()) , \
      "v" : real(v.transpose()), \
      "w" : real(w.transpose())} ) 
  iteration += 1
  #Energy = append(Energy,mean(0.5*uhat*conj(uhat) + 0.5*vhat*conj(vhat) + 0.5*what*conj(what)))
  print(t,Energy[-1])
t1 = time.time()
print('time = ' + str(t1 - t0))
