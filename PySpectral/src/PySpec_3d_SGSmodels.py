import numpy as np
from PySpec_3d_importmodule import pad,unpad,getnut
def computeSGS_EV_REALDOMAIN(grid,uhat,vhat,what,myFFT):
  S11hat = 1j*grid.k1*uhat 
  S22hat = 1j*grid.k2*vhat 
  S33hat = 1j*grid.k3*what
  S12hat = 0.5*(1j*grid.k2*uhat + 1j*grid.k1*vhat)
  S13hat = 0.5*(1j*grid.k3*uhat + 1j*grid.k1*what)
  S23hat = 0.5*(1j*grid.k3*vhat + 1j*grid.k2*what)
  scale =  (3./2.)**3*np.sqrt(grid.N1*grid.N2*grid.N3)
  scale_i = 1./scale
  S11real = myFFT.ifftT_obj(pad(S11hat,1))*scale
  S22real = myFFT.ifftT_obj(pad(S22hat,1))*scale
  S33real = myFFT.ifftT_obj(pad(S33hat,1))*scale
  S12real = myFFT.ifftT_obj(pad(S12hat,1))*scale
  S13real = myFFT.ifftT_obj(pad(S13hat,1))*scale
  S23real = myFFT.ifftT_obj(pad(S23hat,1))*scale

  nutreal = getnut(grid,S11real,S22real,S33real,S12real,S13real,S23real)
  tau11real = -2.*nutreal*S11real
  tau22real = -2.*nutreal*S22real
  tau33real = -2.*nutreal*S33real
  tau12real = -2.*nutreal*S12real
  tau13real = -2.*nutreal*S13real
  tau23real = -2.*nutreal*S23real

  tau11imag = unpad( myFFT.fft_obj(tau11real)*scale_i,1)
  tau22imag = unpad( myFFT.fft_obj(tau22real)*scale_i,1)
  tau33imag = unpad( myFFT.fft_obj(tau33real)*scale_i,1)
  tau12imag = unpad( myFFT.fft_obj(tau12real)*scale_i,1)
  tau13imag = unpad( myFFT.fft_obj(tau13real)*scale_i,1)
  tau23imag = unpad( myFFT.fft_obj(tau23real)*scale_i,1)
  return tau11imag,tau22imag,tau33imag,tau12imag,tau13imag,tau23imag



