import numpy as np
import os
import pyfftw
import time
if not os.path.exists('3DSolution'):
   os.makedirs('3DSolution')

def unpad(uhat_pad,arrange):
  N1 = int( np.shape(uhat_pad)[0]*2./3. )
  N2 = int( np.shape(uhat_pad)[1]*2./3. )
  N3 = int( np.shape(uhat_pad)[2]*2./3. + 1 )
  uhat = np.zeros((N1,N2,N3),dtype = 'complex')
  ## Remove padding from the middle of the 3D cube
  uhat[0:N1/2,0:N2/2,0:N3-1] = uhat_pad[0:N1/2               ,0:N2/2               ,0:N3-1              ] #left     lower back (0,0,0)
  uhat[0:N1/2,N2/2+1::,0:N3-1] = uhat_pad[0:N1/2               ,int(3./2.*N2)-N2/2+1:: ,0:N3-1              ] #l    eft upper back (0,1,0)

  uhat[N1/2+1::,0:N2/2,0:N3-1] = uhat_pad[int(3./2.*N1)-N1/2+1:: ,0:N2/2               ,0:N3-1              ] #r    ight lower back (1,0,0)
  uhat[N1/2+1::,N2/2+1::,0:N3-1] = uhat_pad[int(3./2.*N1)-N1/2+1:: ,int(3./2.*N2)-N2/2+1::  ,0:N3-1                  ] #right upper back (1,1,0)
  return uhat


def pad(uhat,arrange):
  N1,N2,N3 = np.shape(uhat)
  ## Add padding to the middle of the 3D cube
  uhat_pad = np.zeros((int(3./2.*N1),int(3./2.*N2),N3 + (N3-1)/2 ),dtype = 'complex')
  uhat_pad[0:N1/2               ,0:N2/2               ,0:N3-1              ] = uhat[0:N1/2,0:N2/2,0:N3-1] #left     lower back (0,0,0)
  uhat_pad[0:N1/2               ,int(3./2.*N2)-N2/2+1:: ,0:N3-1              ] = uhat[0:N1/2,N2/2+1::,0:N3-1] #l    eft upper back (0,1,0)

  uhat_pad[int(3./2.*N1)-N1/2+1:: ,0:N2/2               ,0:N3-1              ] = uhat[N1/2+1::,0:N2/2,0:N3-1] #r    ight lower back (1,0,0)
  uhat_pad[int(3./2.*N1)-N1/2+1:: ,int(3./2.*N2)-N2/2+1:: ,0:N3-1             ] = uhat[N1/2+1::,N2/2+1::,0:N3-1]     #right upper back (1,1,0)
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

def computeRHS(Q,uhat,vhat,what,grid,myFFT):
  uhat,vhat,what = Q2U(Q,uhat,vhat,what)
  RHS = np.zeros((3*grid.N1,3*grid.N2,3*(grid.N3/2+1) ),dtype='complex')
  #uhat_pad = pad(uhat,1)
  #vhat_pad = pad(vhat,1)
  #what_pad = pad(what,1)
  scale = np.sqrt( (3./2.)**3*np.sqrt(grid.N1*grid.N2*grid.N3) )
  ureal = np.zeros( (int(3./2.*N1),int(3./2.*N2),int(3./2.*N3)) )
  vreal = np.zeros( (int(3./2.*N1),int(3./2.*N2),int(3./2.*N3)) )
  wreal = np.zeros( (int(3./2.*N1),int(3./2.*N2),int(3./2.*N3)) )

  ureal[:,:,:] = myFFT.ifftT_obj(pad(uhat,1))*scale
  vreal[:,:,:] = myFFT.ifftT_obj(pad(vhat,1))*scale
  wreal[:,:,:] = myFFT.ifftT_obj(pad(what,1))*scale

  uuhat = unpad( myFFT.fft_obj(ureal*ureal),1)
  vvhat = unpad( myFFT.fft_obj(vreal*vreal),1)
  wwhat = unpad( myFFT.fft_obj(wreal*wreal),1)
  uvhat = unpad( myFFT.fft_obj(ureal*vreal),1)
  uwhat = unpad( myFFT.fft_obj(ureal*wreal),1)
  vwhat = unpad( myFFT.fft_obj(vreal*wreal),1)

  phat  = -grid.k1*grid.k1*grid.ksqr_i*uuhat - grid.k2*grid.k2*grid.ksqr_i*vvhat - \
           grid.k3*grid.k3*grid.ksqr_i*wwhat - 2.*grid.k1*grid.k2*grid.ksqr_i*uvhat - \
           2.*grid.k1*grid.k3*grid.ksqr_i*uwhat - 2.*grid.k2*grid.k3*grid.ksqr_i*vwhat
  RHS[0:3*N1:3,0:3*N2:3,0::3] = -1j*grid.k1*uuhat - 1j*grid.k2*uvhat - 1j*grid.k3*uwhat - \
                                         1j*grid.k1*phat - nu*grid.ksqr*uhat

  RHS[1:3*N1:3,1:3*N2:3,1::3] = -1j*grid.k1*uvhat - 1j*grid.k2*vvhat - 1j*grid.k3*vwhat - \
                                         1j*grid.k2*phat - nu*grid.ksqr*vhat

  RHS[2:3*N1:3,2:3*N2:3,2::3] = -1j*grid.k1*uwhat - 1j*grid.k2*vwhat - 1j*grid.k3*wwhat - \
                                         1j*grid.k3*phat - nu*grid.ksqr*what
  return RHS


def advanceQ_RK4(dt,Q,uhat,vhat,what,grid,myFFT):
  Q0 = np.zeros(np.shape(Q),dtype='complex')
  Q0[:,:,:] = Q
  for i in range(0,4):
    RHS = computeRHS(Q,uhat,vhat,what,grid,myFFT)
    Q = Q0 + dt*rk4const[i]*RHS
  return Q


t0 = time.time()
Q = np.zeros((3*N1,3*N2,3*(N3/2+1)),dtype='complex')
Q = U2Q(Q,uhat,vhat,what)

iteration = 0
Energy = np.zeros(1)
Energy[0] = np.real( np.mean(0.5*uhat*np.conj(uhat) + 0.5*vhat*np.conj(vhat) + 0.5*what*np.conj(what)) )
EnergyScale = np.mean( 0.5*(u*u +v*v + w*w) )/Energy[0] 
while t <= et:
  Q = advanceQ_RK4(dt,Q,uhat,vhat,what,grid,myFFT)
  t += dt
  if (iteration%save_freq == 0):
    savehook(uhat,vhat,what,grid,iteration)
  iteration += 1
  Energy = np.append(Energy,EnergyScale*np.mean(0.5*uhat*np.conj(uhat) +\
           0.5*vhat*np.conj(vhat) + 0.5*what*np.conj(what)))
  sys.stdout.write("Wall Time= " + str(time.time() - t0) + "   t=" + str(t) + \
                   "   Energy = " + str(np.real(Energy[-1]))  + "\n")
  sys.stdout.flush()
 
t1 = time.time()
print('time = ' + str(t1 - t0))
