import numpy as np
from PySpec_3d_importmodule import Q2U,U2Q,pad,unpad
from PySpec_3d_importmodule import computeSGS
def computeRHS_NOSGS(Q,uhat,vhat,what,nu,grid,myFFT):
    uhat,vhat,what = Q2U(Q,uhat,vhat,what)
    RHS = np.zeros((3*grid.N1,3*grid.N2,3*(grid.N3/2+1) ),dtype='complex')
    #uhat_pad = pad(uhat,1)
    #vhat_pad = pad(vhat,1)
    #what_pad = pad(what,1)
    scale = np.sqrt( (3./2.)**3*np.sqrt(grid.N1*grid.N2*grid.N3) )
    ureal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    vreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    wreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )

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
    RHS[0::3,0::3,0::3] = -1j*grid.k1*uuhat - 1j*grid.k2*uvhat - 1j*grid.k3*uwhat - \
                                         1j*grid.k1*phat - nu*grid.ksqr*uhat

    RHS[1::3,1::3,1::3] = -1j*grid.k1*uvhat - 1j*grid.k2*vvhat - 1j*grid.k3*vwhat - \
                                         1j*grid.k2*phat - nu*grid.ksqr*vhat

    RHS[2::3,2::3,2::3] = -1j*grid.k1*uwhat - 1j*grid.k2*vwhat - 1j*grid.k3*wwhat - \
                                         1j*grid.k3*phat - nu*grid.ksqr*what
    return RHS



### For SGS models where the stress can be instantaneously found from current field.
## Use this for any EV model
def computeRHS_MARKOVIAN(Q,uhat,vhat,what,nu,grid,myFFT):
    uhat,vhat,what = Q2U(Q,uhat,vhat,what)
    RHS = np.zeros((3*grid.N1,3*grid.N2,3*(grid.N3/2+1) ),dtype='complex')
    #uhat_pad = pad(uhat,1)
    #vhat_pad = pad(vhat,1)
    #what_pad = pad(what,1)
    scale = np.sqrt( (3./2.)**3*np.sqrt(grid.N1*grid.N2*grid.N3) )
    ureal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    vreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    wreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )

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

    tauhat11,tauhat22,tauhat33,tauhat12,tauhat13,tauhat23 = computeSGS(grid,uhat,vhat,what,myFFT)
    RHS[0::3,0::3,0::3] = -1j*grid.k1*uuhat - 1j*grid.k2*uvhat - 1j*grid.k3*uwhat - \
                           1j*grid.k1*phat - nu*grid.ksqr*uhat - 1j*grid.k1*tauhat11 - \
                           1j*grid.k2*tauhat12 - 1j*grid.k3*tauhat13

    RHS[1::3,1::3,1::3] = -1j*grid.k1*uvhat - 1j*grid.k2*vvhat - 1j*grid.k3*vwhat - \
                           1j*grid.k2*phat - nu*grid.ksqr*vhat - 1j*grid.k1*tauhat12 - \
                           1j*grid.k2*tauhat22 - 1j*grid.k3*tauhat23
 

    RHS[2::3,2::3,2::3] = -1j*grid.k1*uwhat - 1j*grid.k2*vwhat - 1j*grid.k3*wwhat - \
                           1j*grid.k3*phat - nu*grid.ksqr*what - 1j*grid.k1*tauhat13 - \
                           1j*grid.k2*tauhat23 - 1j*grid.k3*tauhat33

    return RHS

