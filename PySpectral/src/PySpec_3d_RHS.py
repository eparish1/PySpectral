import numpy as np
from PySpec_3d_importmodule import Q2U,U2Q,pad,unpad,pad_2x,unpad_2x,seperateModes
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



### RHS function for the t-model
def computeRHS_tmodel(Q,uhat,vhat,what,nu,grid,myFFT):
    uhat,vhat,what = Q2U(Q,uhat,vhat,what)
    RHS = np.zeros((3*grid.N1,3*grid.N2,3*(grid.N3/2+1) ),dtype='complex')
    ## in the t-model, do 2x padding because we want to have convolutions where 
    ## the modes in G support twice the modes in F.
    scale = np.sqrt( (2.)**3*np.sqrt(grid.N1*grid.N2*grid.N3) )
    ureal = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(2.*grid.N3)) )
    vreal = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(2.*grid.N3)) )
    wreal = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(2.*grid.N3)) )

    ureal = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(2*grid.N3)) )
    vreal = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(2*grid.N3)) )
    wreal = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(2*grid.N3)) )

    PLu_qreal = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(2*grid.N3)) )
    PLv_qreal = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(2*grid.N3)) )
    PLw_qreal = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(2*grid.N3)) )

    PLu_p = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(grid.N3+1)) ,dtype = 'complex')
    PLv_p = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(grid.N3+1)) ,dtype = 'complex')
    PLw_p = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(grid.N3+1)) ,dtype = 'complex')
    PLu_q = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(grid.N3+1)) ,dtype = 'complex')
    PLv_q = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(grid.N3+1)) ,dtype = 'complex')
    PLw_q = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(grid.N3+1)) ,dtype = 'complex')

    uhat_pad = pad_2x(uhat,1)
    vhat_pad = pad_2x(vhat,1)
    what_pad = pad_2x(what,1)
    ureal[:,:,:] = myFFT.ifftT_obj2(uhat_pad)*scale
    vreal[:,:,:] = myFFT.ifftT_obj2(vhat_pad)*scale
    wreal[:,:,:] = myFFT.ifftT_obj2(what_pad)*scale

    uuhat = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    vvhat = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    wwhat = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    uvhat = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    uwhat = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    vwhat = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')



    uuhat[:,:,:] = myFFT.fft_obj2(ureal[:,:,:]*ureal[:,:,:])
    vvhat[:,:,:] = myFFT.fft_obj2(vreal[:,:,:]*vreal[:,:,:])
    wwhat[:,:,:] = myFFT.fft_obj2(wreal[:,:,:]*wreal[:,:,:])
    uvhat[:,:,:] = myFFT.fft_obj2(ureal[:,:,:]*vreal[:,:,:])
    uwhat[:,:,:] = myFFT.fft_obj2(ureal[:,:,:]*wreal[:,:,:])
    vwhat[:,:,:] = myFFT.fft_obj2(vreal[:,:,:]*wreal[:,:,:])

    uuhat2 = unpad_2x(uuhat,1)
    vvhat2 = unpad_2x(vvhat,1)
    wwhat2 = unpad_2x(wwhat,1)
    uvhat2 = unpad_2x(uvhat,1)
    uwhat2 = unpad_2x(uwhat,1)
    vwhat2 = unpad_2x(vwhat,1)

    phat  = -grid.k1f*grid.k1f*grid.ksqrf_i*uuhat - grid.k2f*grid.k2f*grid.ksqrf_i*vvhat - \
             grid.k3f*grid.k3f*grid.ksqrf_i*wwhat - 2.*grid.k1f*grid.k2f*grid.ksqrf_i*uvhat - \
             2.*grid.k1f*grid.k3f*grid.ksqrf_i*uwhat - 2.*grid.k2f*grid.k3f*grid.ksqrf_i*vwhat

    PLu = -1j*grid.k1f*uuhat - 1j*grid.k2f*uvhat - 1j*grid.k3f*uwhat - \
                                         1j*grid.k1f*phat - nu*grid.ksqrf*pad_2x(uhat,1)

    PLv = -1j*grid.k1f*uvhat - 1j*grid.k2f*vvhat - 1j*grid.k3f*vwhat - \
                                         1j*grid.k2f*phat - nu*grid.ksqrf*pad_2x(vhat,1)

    PLw = -1j*grid.k1f*uwhat - 1j*grid.k2f*vwhat - 1j*grid.k3f*wwhat - \
                                         1j*grid.k3f*phat - nu*grid.ksqrf*pad_2x(what,1)

    PLu_p[:,:,:],PLu_q[:,:,:] = seperateModes(PLu,1)
    PLv_p[:,:,:],PLv_q[:,:,:] = seperateModes(PLv,1)
    PLw_p[:,:,:],PLw_q[:,:,:] = seperateModes(PLw,1)

    PLu_qreal[:,:,:] = myFFT.ifftT_obj2(PLu_q)*scale
    PLv_qreal[:,:,:] = myFFT.ifftT_obj2(PLv_q)*scale
    PLw_qreal[:,:,:] = myFFT.ifftT_obj2(PLw_q)*scale
    
    up_PLuq = unpad_2x( myFFT.fft_obj2(ureal*PLu_qreal),1)
    vp_PLuq = unpad_2x( myFFT.fft_obj2(vreal*PLu_qreal),1)
    wp_PLuq = unpad_2x( myFFT.fft_obj2(wreal*PLu_qreal),1)

    up_PLvq = unpad_2x( myFFT.fft_obj2(ureal*PLv_qreal),1)
    vp_PLvq = unpad_2x( myFFT.fft_obj2(vreal*PLv_qreal),1)
    wp_PLvq = unpad_2x( myFFT.fft_obj2(wreal*PLv_qreal),1)

    up_PLwq = unpad_2x( myFFT.fft_obj2(ureal*PLw_qreal),1)
    vp_PLwq = unpad_2x( myFFT.fft_obj2(vreal*PLw_qreal),1)
    wp_PLwq = unpad_2x( myFFT.fft_obj2(wreal*PLw_qreal),1)

    t1 = grid.k1*up_PLuq + grid.k2*vp_PLuq + grid.k3*wp_PLuq
    t2 = grid.k1*up_PLvq + grid.k2*vp_PLvq + grid.k3*wp_PLvq
    t3 = grid.k1*up_PLwq + grid.k2*vp_PLwq + grid.k3*wp_PLwq 

    PLQLu = -1j*grid.k1*up_PLuq - 1j*grid.k2*vp_PLuq - 1j*grid.k3*wp_PLuq - \
            1j*grid.k1*up_PLuq - 1j*grid.k1*vp_PLvq - 1j*grid.k1*wp_PLwq + \
            1j*grid.k1*grid.ksqr_i*(2.*grid.k1*(t1) ) + \
            1j*grid.k2*grid.ksqr_i*(2.*grid.k1*(t2) ) + \
            1j*grid.k3*grid.ksqr_i*(2.*grid.k1*(t3) ) 

    PLQLv = -1j*grid.k1*up_PLvq - 1j*grid.k2*vp_PLvq - 1j*grid.k3*wp_PLvq - \
            1j*grid.k2*up_PLuq - 1j*grid.k2*vp_PLvq - 1j*grid.k2*wp_PLwq + \
            1j*grid.k1*grid.ksqr_i*(2.*grid.k2*(t1) ) + \
            1j*grid.k2*grid.ksqr_i*(2.*grid.k2*(t2) ) + \
            1j*grid.k3*grid.ksqr_i*(2.*grid.k2*(t3) ) 

    PLQLw = -1j*grid.k1*up_PLwq - 1j*grid.k2*vp_PLwq - 1j*grid.k3*wp_PLwq -\
            1j*grid.k3*up_PLuq - 1j*grid.k3*vp_PLvq - 1j*grid.k3*wp_PLwq + \
            1j*grid.k1*grid.ksqr_i*(2.*grid.k3*(t1) ) + \
            1j*grid.k2*grid.ksqr_i*(2.*grid.k3*(t2) ) + \
            1j*grid.k3*grid.ksqr_i*(2.*grid.k3*(t3) ) 

    RHS[0::3,0::3,0::3] = unpad_2x(PLu,1) + 0.1*grid.t*PLQLu

    RHS[1::3,1::3,1::3] = unpad_2x(PLv,1) + 0.1*grid.t*PLQLv

    RHS[2::3,2::3,2::3] = unpad_2x(PLw,1) + 0.1*grid.t*PLQLw

    return RHS
#

