import numpy as np
from padding import pad,unpad,pad_2x,unpad_2x,seperateModes
def computeRHS_NOSGS(main,grid,myFFT):
    main.Q2U()
    scale = np.sqrt( (3./2.)**3*np.sqrt(grid.N1*grid.N2*grid.N3) )
    ureal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    vreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    wreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )

    ureal[:,:,:] = myFFT.ifftT_obj(pad(main.uhat,1))*scale
    vreal[:,:,:] = myFFT.ifftT_obj(pad(main.vhat,1))*scale
    wreal[:,:,:] = myFFT.ifftT_obj(pad(main.what,1))*scale

    uuhat = unpad( myFFT.fft_obj(ureal*ureal),1)
    vvhat = unpad( myFFT.fft_obj(vreal*vreal),1)
    wwhat = unpad( myFFT.fft_obj(wreal*wreal),1)
    uvhat = unpad( myFFT.fft_obj(ureal*vreal),1)
    uwhat = unpad( myFFT.fft_obj(ureal*wreal),1)
    vwhat = unpad( myFFT.fft_obj(vreal*wreal),1)


    phat  = -grid.k1*grid.k1*grid.ksqr_i*uuhat - grid.k2*grid.k2*grid.ksqr_i*vvhat - \
             grid.k3*grid.k3*grid.ksqr_i*wwhat - 2.*grid.k1*grid.k2*grid.ksqr_i*uvhat - \
             2.*grid.k1*grid.k3*grid.ksqr_i*uwhat - 2.*grid.k2*grid.k3*grid.ksqr_i*vwhat

    main.Q[0::3,0::3,0::3] = -1j*grid.k1*uuhat - 1j*grid.k2*uvhat - 1j*grid.k3*uwhat - \
                                         1j*grid.k1*phat - main.nu*grid.ksqr*main.uhat

    main.Q[1::3,1::3,1::3] = -1j*grid.k1*uvhat - 1j*grid.k2*vvhat - 1j*grid.k3*vwhat - \
                                         1j*grid.k2*phat - main.nu*grid.ksqr*main.vhat

    main.Q[2::3,2::3,2::3] = -1j*grid.k1*uwhat - 1j*grid.k2*vwhat - 1j*grid.k3*wwhat - \
                                         1j*grid.k3*phat - main.nu*grid.ksqr*main.what



### For SGS models where the stress can be instantaneously found from current field.
## Use this for any EV model
def computeRHS_SMAG(main,grid,myFFT):
    main.Q2U()
    scale = np.sqrt( (3./2.)**3*np.sqrt(grid.N1*grid.N2*grid.N3) )
    ureal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    vreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    wreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )

    ureal[:,:,:] = myFFT.ifftT_obj(pad(main.uhat,1))*scale
    vreal[:,:,:] = myFFT.ifftT_obj(pad(main.vhat,1))*scale
    wreal[:,:,:] = myFFT.ifftT_obj(pad(main.what,1))*scale

    uuhat = unpad( myFFT.fft_obj(ureal*ureal),1)
    vvhat = unpad( myFFT.fft_obj(vreal*vreal),1)
    wwhat = unpad( myFFT.fft_obj(wreal*wreal),1)
    uvhat = unpad( myFFT.fft_obj(ureal*vreal),1)
    uwhat = unpad( myFFT.fft_obj(ureal*wreal),1)
    vwhat = unpad( myFFT.fft_obj(vreal*wreal),1)

    phat  = -grid.k1*grid.k1*grid.ksqr_i*uuhat - grid.k2*grid.k2*grid.ksqr_i*vvhat - \
             grid.k3*grid.k3*grid.ksqr_i*wwhat - 2.*grid.k1*grid.k2*grid.ksqr_i*uvhat - \
             2.*grid.k1*grid.k3*grid.ksqr_i*uwhat - 2.*grid.k2*grid.k3*grid.ksqr_i*vwhat

    S11hat = 1j*grid.k1*main.uhat
    S22hat = 1j*grid.k2*main.vhat
    S33hat = 1j*grid.k3*main.what
    S12hat = 0.5*(1j*grid.k2*main.uhat + 1j*grid.k1*main.vhat)
    S13hat = 0.5*(1j*grid.k3*main.uhat + 1j*grid.k1*main.what)
    S23hat = 0.5*(1j*grid.k3*main.vhat + 1j*grid.k2*main.what)
    scale =  (3./2.)**3*np.sqrt(grid.N1*grid.N2*grid.N3)
    scale_i = 1./scale
    S11real = myFFT.ifftT_obj(pad(S11hat,1))*scale
    S22real = myFFT.ifftT_obj(pad(S22hat,1))*scale
    S33real = myFFT.ifftT_obj(pad(S33hat,1))*scale
    S12real = myFFT.ifftT_obj(pad(S12hat,1))*scale
    S13real = myFFT.ifftT_obj(pad(S13hat,1))*scale
    S23real = myFFT.ifftT_obj(pad(S23hat,1))*scale
 
    S_magreal = np.sqrt( 2.*(S11real*S11real + S22real*S22real + S33real*S33real + \
              2.*S12real*S12real + 2.*S13real*S13real + 2.*S23real*S23real ) )
    nutreal = 0.16*grid.Delta*0.16*grid.Delta*np.abs(S_magreal)
    
    tau11real = -2.*nutreal*S11real
    tau22real = -2.*nutreal*S22real
    tau33real = -2.*nutreal*S33real
    tau12real = -2.*nutreal*S12real
    tau13real = -2.*nutreal*S13real
    tau23real = -2.*nutreal*S23real
    
    main.tauhat[:,:,:,0] = unpad( myFFT.fft_obj( -2.*nutreal*S11real )*scale_i,1)  #11
    main.tauhat[:,:,:,1] = unpad( myFFT.fft_obj( -2.*nutreal*S22real )*scale_i,1)  #22
    main.tauhat[:,:,:,2] = unpad( myFFT.fft_obj( -2.*nutreal*S33real )*scale_i,1)  #33
    main.tauhat[:,:,:,3] = unpad( myFFT.fft_obj( -2.*nutreal*S12real )*scale_i,1)  #12
    main.tauhat[:,:,:,4] = unpad( myFFT.fft_obj( -2.*nutreal*S13real )*scale_i,1)  #13
    main.tauhat[:,:,:,5] = unpad( myFFT.fft_obj( -2.*nutreal*S23real )*scale_i,1)  #23

    main.Q[0::3,0::3,0::3] = -1j*grid.k1*uuhat - 1j*grid.k2*uvhat - 1j*grid.k3*uwhat - \
                           1j*grid.k1*phat - main.nu*grid.ksqr*main.uhat - 1j*grid.k1*main.tauhat[:,:,:,0] - \
                           1j*grid.k2*main.tauhat[:,:,:,3] - 1j*grid.k3*main.tauhat[:,:,:,4]

    main.Q[1::3,1::3,1::3] = -1j*grid.k1*uvhat - 1j*grid.k2*vvhat - 1j*grid.k3*vwhat - \
                           1j*grid.k2*phat - main.nu*grid.ksqr*main.vhat - 1j*grid.k1*main.tauhat[:,:,:,3] - \
                           1j*grid.k2*main.tauhat[:,:,:,1] - 1j*grid.k3*main.tauhat[:,:,:,5]
 

    main.Q[2::3,2::3,2::3] = -1j*grid.k1*uwhat - 1j*grid.k2*vwhat - 1j*grid.k3*wwhat - \
                           1j*grid.k3*phat - main.nu*grid.ksqr*main.what - 1j*grid.k1*main.tauhat[:,:,:,4] - \
                           1j*grid.k2*main.tauhat[:,:,:,5] - 1j*grid.k3*main.tauhat[:,:,:,2]


def computeRHS_tmodel(main,grid,myFFT):
    main.Q2U()
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

    uhat_pad = pad_2x(main.uhat,1)
    vhat_pad = pad_2x(main.vhat,1)
    what_pad = pad_2x(main.what,1)
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
                                         1j*grid.k1f*phat - main.nu*grid.ksqrf*pad_2x(main.uhat,1)

    PLv = -1j*grid.k1f*uvhat - 1j*grid.k2f*vvhat - 1j*grid.k3f*vwhat - \
                                         1j*grid.k2f*phat - main.nu*grid.ksqrf*pad_2x(main.vhat,1)

    PLw = -1j*grid.k1f*uwhat - 1j*grid.k2f*vwhat - 1j*grid.k3f*wwhat - \
                                         1j*grid.k3f*phat - main.nu*grid.ksqrf*pad_2x(main.what,1)

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

    main.PLQLu = -1j*grid.k1*up_PLuq - 1j*grid.k2*vp_PLuq - 1j*grid.k3*wp_PLuq - \
            1j*grid.k1*up_PLuq - 1j*grid.k1*vp_PLvq - 1j*grid.k1*wp_PLwq + \
            1j*grid.k1*grid.ksqr_i*(2.*grid.k1*(t1) ) + \
            1j*grid.k2*grid.ksqr_i*(2.*grid.k1*(t2) ) + \
            1j*grid.k3*grid.ksqr_i*(2.*grid.k1*(t3) )

    main.PLQLv = -1j*grid.k1*up_PLvq - 1j*grid.k2*vp_PLvq - 1j*grid.k3*wp_PLvq - \
            1j*grid.k2*up_PLuq - 1j*grid.k2*vp_PLvq - 1j*grid.k2*wp_PLwq + \
            1j*grid.k1*grid.ksqr_i*(2.*grid.k2*(t1) ) + \
            1j*grid.k2*grid.ksqr_i*(2.*grid.k2*(t2) ) + \
            1j*grid.k3*grid.ksqr_i*(2.*grid.k2*(t3) )

    main.PLQLw = -1j*grid.k1*up_PLwq - 1j*grid.k2*vp_PLwq - 1j*grid.k3*wp_PLwq -\
            1j*grid.k3*up_PLuq - 1j*grid.k3*vp_PLvq - 1j*grid.k3*wp_PLwq + \
            1j*grid.k1*grid.ksqr_i*(2.*grid.k3*(t1) ) + \
            1j*grid.k2*grid.ksqr_i*(2.*grid.k3*(t2) ) + \
            1j*grid.k3*grid.ksqr_i*(2.*grid.k3*(t3) )

    main.Q[0::3,0::3,0::3] = unpad_2x(PLu,1) + 0.35*main.t*main.PLQLu

    main.Q[1::3,1::3,1::3] = unpad_2x(PLv,1) + 0.35*main.t*main.PLQLv

    main.Q[2::3,2::3,2::3] = unpad_2x(PLw,1) + 0.35*main.t*main.PLQLw

def computeRHS_FM1(main,grid,myFFT):
    main.Q2U()
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

    uhat_pad = pad_2x(main.uhat,1)
    vhat_pad = pad_2x(main.vhat,1)
    what_pad = pad_2x(main.what,1)
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
                                         1j*grid.k1f*phat - main.nu*grid.ksqrf*pad_2x(main.uhat,1)

    PLv = -1j*grid.k1f*uvhat - 1j*grid.k2f*vvhat - 1j*grid.k3f*vwhat - \
                                         1j*grid.k2f*phat - main.nu*grid.ksqrf*pad_2x(main.vhat,1)

    PLw = -1j*grid.k1f*uwhat - 1j*grid.k2f*vwhat - 1j*grid.k3f*wwhat - \
                                         1j*grid.k3f*phat - main.nu*grid.ksqrf*pad_2x(main.what,1)

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

    main.Q[0::6,0::6,0::6] = unpad_2x(PLu,1) + main.w0_u
    main.Q[1::6,1::6,1::6] = unpad_2x(PLv,1) + main.w0_v
    main.Q[2::6,2::6,2::6] = unpad_2x(PLw,1) + main.w0_w
    main.Q[3::6,3::6,3::6] = -2./main.dt0*main.w0_u + PLQLu 
    main.Q[4::6,4::6,4::6] = -2./main.dt0*main.w0_v + PLQLv
    main.Q[5::6,5::6,5::6] = -2./main.dt0*main.w0_w + PLQLw

    
