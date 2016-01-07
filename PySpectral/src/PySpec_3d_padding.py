import numpy as np
def unpad(uhat_pad,arrange):
  N1 = int( np.shape(uhat_pad)[0]*2./3. )
  N2 = int( np.shape(uhat_pad)[1]*2./3. )
  N3 = int( np.shape(uhat_pad)[2]*2./3. + 1 )
  uhat = np.zeros((N1,N2,N3),dtype = 'complex')
  ## Remove padding from the middle of the 3D cube
  uhat[0:N1/2   , 0:N2/2   , 0:N3-1] = uhat_pad[0:N1/2                 ,0:N2/2                  ,0:N3-1 ] #left     lower back (0,0,0)
  uhat[0:N1/2   , N2/2+1:: , 0:N3-1] = uhat_pad[0:N1/2                 ,int(3./2.*N2)-N2/2+1::  ,0:N3-1 ] #left     upper back (0,1,0)

  uhat[N1/2+1:: , 0:N2/2   , 0:N3-1] = uhat_pad[int(3./2.*N1)-N1/2+1:: ,0:N2/2                  ,0:N3-1 ] #right     lower back (1,0,0)
  uhat[N1/2+1:: , N2/2+1:: , 0:N3-1] = uhat_pad[int(3./2.*N1)-N1/2+1:: ,int(3./2.*N2)-N2/2+1::  ,0:N3-1 ] #right     upper back (1,1,0)
  return uhat


def pad(uhat,arrange):
  N1,N2,N3 = np.shape(uhat)
  ## Add padding to the middle of the 3D cube
  uhat_pad = np.zeros((int(3./2.*N1),int(3./2.*N2),N3 + (N3-1)/2 ),dtype = 'complex')
  uhat_pad[0:N1/2                 , 0:N2/2                 , 0:N3-1              ] = uhat[0:N1/2   , 0:N2/2   ,     0:N3-1] #left     lower back (0,0,0)
  uhat_pad[0:N1/2                 , int(3./2.*N2)-N2/2+1:: , 0:N3-1              ] = uhat[0:N1/2   , N2/2+1:: ,     0:N3-1] #l    eft upper back (0,1,0)

  uhat_pad[int(3./2.*N1)-N1/2+1:: , 0:N2/2                 , 0:N3-1              ] = uhat[N1/2+1:: , 0:N2/2   ,     0:N3-1] #r    ight lower back (1,0,0)
  uhat_pad[int(3./2.*N1)-N1/2+1:: , int(3./2.*N2)-N2/2+1:: , 0:N3-1              ] = uhat[N1/2+1:: , N2/2+1:: ,     0:N3-1]     #right upper back (1,1,0)
  return uhat_pad


def unpad_2x(uhat_pad,arrange):
  N1 = np.shape(uhat_pad)[0]/2
  N2 = np.shape(uhat_pad)[1]/2
  N3 = (np.shape(uhat_pad)[2] + 1)/2
  uhat = np.zeros((N1,N2,N3),dtype = 'complex')
  ## Remove padding from the middle of the 3D cube
  uhat[0:N1/2,0:N2/2,0:N3-1] = uhat_pad[0:N1/2      ,0:N2/2      ,0:N3-1] #left lower back (0,0,0)
  uhat[0:N1/2,N2/2::,0:N3-1] = uhat_pad[0:N1/2      ,2*N2-N2/2:: ,0:N3-1] #left upper back (0,1,0)
  uhat[N1/2::,0:N2/2,0:N3-1] = uhat_pad[2*N1-N1/2:: ,0:N2/2      ,0:N3-1] #right lower back (1,0,0)
  uhat[N1/2::,N2/2::,0:N3-1] = uhat_pad[2*N1-N1/2:: ,2*N2-N2/2:: ,0:N3-1] #right upper back (1,1,0)
  return uhat


def pad_2x(uhat,arrange):
  N1,N2,N3 = np.shape(uhat)
  ## Add padding to the middle of the 3D cube
  uhat_pad = np.zeros((2*N1,2*N2,2*N3-1),dtype = 'complex')
  uhat_pad[0:N1/2      ,0:N2/2      ,0:N3-1] = uhat[0:N1/2,0:N2/2,0:N3-1] #left lower back (0,0,0)
  uhat_pad[0:N1/2      ,2*N2-N2/2:: ,0:N3-1] = uhat[0:N1/2,N2/2::,0:N3-1] #left upper back (0,1,0)
  uhat_pad[2*N1-N1/2:: ,0:N2/2      ,0:N3-1] = uhat[N1/2::,0:N2/2,0:N3-1] #right lower back (1,0,0)
  uhat_pad[2*N1-N1/2:: ,2*N2-N2/2:: ,0:N3-1] = uhat[N1/2::,N2/2::,0:N3-1] #right upper back (1,1,0)
  return uhat_pad

## Funciton to take in uhat with modes in F U G and to sepearte them into modes in F and modes in G. q are the modes in G and p are the modes in F to be consistent with notes
def seperateModes(uhat_pad,arrange):
  N1 = np.shape(uhat_pad)[0]/2
  N2 = np.shape(uhat_pad)[1]/2
  N3 = (np.shape(uhat_pad)[2] + 1)/2
  u_p = np.zeros((2*N1,2*N2,2*N3-1),dtype = 'complex')
  u_q = np.zeros((2*N1,2*N2,2*N3-1),dtype = 'complex') 
  u_q[N1/2:2*N1-N1/2, N2/2:2*N2-N2/2, N3/2::] = uhat_pad[N1/2:2*N1-N1/2 ,N2/2:2*N2-N2/2 ,N3/2::] 
  u_p = uhat_pad - u_q
  return u_p,u_q

## test comment
