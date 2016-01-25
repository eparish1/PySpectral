import numpy as np
from padding import pad,unpad,pad_2x,unpad_2x,seperateModes
def computeRHS_NOSGS(main,grid,myFFT):
    main.Q2U()
    scale = np.sqrt( (3./2.)**3*np.sqrt(grid.N1*grid.N2*grid.N3) )
    ureal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    vreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    wreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )

    main.uhat = unpad(pad(main.uhat,1),1)
    main.vhat = unpad(pad(main.vhat,1),1)
    main.what = unpad(pad(main.what,1),1)

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
    
    main.tauhat[:,:,:,0] = unpad( myFFT.fft_obj( -2.*nutreal*S11real ),1)  #11
    main.tauhat[:,:,:,1] = unpad( myFFT.fft_obj( -2.*nutreal*S22real ),1)  #22
    main.tauhat[:,:,:,2] = unpad( myFFT.fft_obj( -2.*nutreal*S33real ),1)  #33
    main.tauhat[:,:,:,3] = unpad( myFFT.fft_obj( -2.*nutreal*S12real ),1)  #12
    main.tauhat[:,:,:,4] = unpad( myFFT.fft_obj( -2.*nutreal*S13real ),1)  #13
    main.tauhat[:,:,:,5] = unpad( myFFT.fft_obj( -2.*nutreal*S23real ),1)  #23

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

    main.uhat = unpad(pad(main.uhat,1),1)
    main.vhat = unpad(pad(main.vhat,1),1)
    main.what = unpad(pad(main.what,1),1)


    uhat_pad = pad_2x(main.uhat,1)
    vhat_pad = pad_2x(main.vhat,1)
    what_pad = pad_2x(main.what,1)
#    print(np.linalg.norm(uhat_pad),np.linalg.norm(vhat_pad),np.linalg.norm(what_pad))
    ureal[:,:,:] = myFFT.ifftT_obj2(uhat_pad*scale)
    vreal[:,:,:] = myFFT.ifftT_obj2(vhat_pad*scale)
    wreal[:,:,:] = myFFT.ifftT_obj2(what_pad*scale)
#    print(np.linalg.norm(uhat_pad),np.linalg.norm(vhat_pad),np.linalg.norm(what_pad))

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

    PLu_qreal[:,:,:] = myFFT.ifftT_obj2(PLu_q*scale)
    PLv_qreal[:,:,:] = myFFT.ifftT_obj2(PLv_q*scale)
    PLw_qreal[:,:,:] = myFFT.ifftT_obj2(PLw_q*scale)

    up_PLuq = unpad_2x( myFFT.fft_obj2(ureal*PLu_qreal),1)
    vp_PLuq = unpad_2x( myFFT.fft_obj2(vreal*PLu_qreal),1)
    wp_PLuq = unpad_2x( myFFT.fft_obj2(wreal*PLu_qreal),1)

    up_PLvq = unpad_2x( myFFT.fft_obj2(ureal*PLv_qreal),1)
    vp_PLvq = unpad_2x( myFFT.fft_obj2(vreal*PLv_qreal),1)
    wp_PLvq = unpad_2x( myFFT.fft_obj2(wreal*PLv_qreal),1)

    up_PLwq = unpad_2x( myFFT.fft_obj2(ureal*PLw_qreal),1)
    vp_PLwq = unpad_2x( myFFT.fft_obj2(vreal*PLw_qreal),1)
    wp_PLwq = unpad_2x( myFFT.fft_obj2(wreal*PLw_qreal),1)


    pterm = 2.*grid.ksqr_i*( grid.k1*grid.k1*up_PLuq + grid.k2*grid.k2*vp_PLvq + grid.k3*grid.k3*wp_PLwq + \
                          grid.k1*grid.k2*(up_PLvq + vp_PLuq) + grid.k1*grid.k3*(up_PLwq + wp_PLuq) + \
                          grid.k2*grid.k3*(vp_PLwq + wp_PLvq) )

    main.PLQLu = -1j*grid.k1*up_PLuq - 1j*grid.k2*vp_PLuq - 1j*grid.k3*wp_PLuq - \
            1j*grid.k1*up_PLuq - 1j*grid.k2*up_PLvq - 1j*grid.k3*up_PLwq + \
            1j*grid.k1*pterm

    main.PLQLv = -1j*grid.k1*up_PLvq - 1j*grid.k2*vp_PLvq - 1j*grid.k3*wp_PLvq - \
            1j*grid.k1*vp_PLuq - 1j*grid.k2*vp_PLvq - 1j*grid.k3*vp_PLwq + \
            1j*grid.k2*pterm

    main.PLQLw = -1j*grid.k1*up_PLwq - 1j*grid.k2*vp_PLwq - 1j*grid.k3*wp_PLwq -\
            1j*grid.k1*wp_PLuq - 1j*grid.k2*wp_PLvq - 1j*grid.k3*wp_PLwq + \
            1j*grid.k3*pterm

    main.Q[0::3,0::3,0::3] = unpad_2x(PLu,1) + 0.05*main.t*main.PLQLu

    main.Q[1::3,1::3,1::3] = unpad_2x(PLv,1) + 0.05*main.t*main.PLQLv

    main.Q[2::3,2::3,2::3] = unpad_2x(PLw,1) + 0.05*main.t*main.PLQLw

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

    main.uhat = unpad(pad(main.uhat,1),1)
    main.vhat = unpad(pad(main.vhat,1),1)
    main.what = unpad(pad(main.what,1),1)


    uhat_pad = pad_2x(main.uhat,1)
    vhat_pad = pad_2x(main.vhat,1)
    what_pad = pad_2x(main.what,1)
    ureal[:,:,:] = myFFT.ifftT_obj2(uhat_pad*scale)
    vreal[:,:,:] = myFFT.ifftT_obj2(vhat_pad*scale)
    wreal[:,:,:] = myFFT.ifftT_obj2(what_pad*scale)

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

    PLu_qreal[:,:,:] = myFFT.ifftT_obj2(PLu_q*scale)
    PLv_qreal[:,:,:] = myFFT.ifftT_obj2(PLv_q*scale)
    PLw_qreal[:,:,:] = myFFT.ifftT_obj2(PLw_q*scale)

    up_PLuq = unpad_2x( myFFT.fft_obj2(ureal*PLu_qreal),1)
    vp_PLuq = unpad_2x( myFFT.fft_obj2(vreal*PLu_qreal),1)
    wp_PLuq = unpad_2x( myFFT.fft_obj2(wreal*PLu_qreal),1)

    up_PLvq = unpad_2x( myFFT.fft_obj2(ureal*PLv_qreal),1)
    vp_PLvq = unpad_2x( myFFT.fft_obj2(vreal*PLv_qreal),1)
    wp_PLvq = unpad_2x( myFFT.fft_obj2(wreal*PLv_qreal),1)

    up_PLwq = unpad_2x( myFFT.fft_obj2(ureal*PLw_qreal),1)
    vp_PLwq = unpad_2x( myFFT.fft_obj2(vreal*PLw_qreal),1)
    wp_PLwq = unpad_2x( myFFT.fft_obj2(wreal*PLw_qreal),1)

    pterm = 2.*grid.ksqr_i*( grid.k1*grid.k1*up_PLuq + grid.k2*grid.k2*vp_PLvq + grid.k3*grid.k3*wp_PLwq + \
                          grid.k1*grid.k2*(up_PLvq + vp_PLuq) + grid.k1*grid.k3*(up_PLwq + wp_PLuq) + \
                          grid.k2*grid.k3*(vp_PLwq + wp_PLvq) )

    main.PLQLu = -1j*grid.k1*up_PLuq - 1j*grid.k2*vp_PLuq - 1j*grid.k3*wp_PLuq - \
            1j*grid.k1*up_PLuq - 1j*grid.k2*up_PLvq - 1j*grid.k3*up_PLwq + \
            1j*grid.k1*pterm

    main.PLQLv = -1j*grid.k1*up_PLvq - 1j*grid.k2*vp_PLvq - 1j*grid.k3*wp_PLvq - \
            1j*grid.k1*vp_PLuq - 1j*grid.k2*vp_PLvq - 1j*grid.k3*vp_PLwq + \
            1j*grid.k2*pterm

    main.PLQLw = -1j*grid.k1*up_PLwq - 1j*grid.k2*vp_PLwq - 1j*grid.k3*wp_PLwq -\
            1j*grid.k1*wp_PLuq - 1j*grid.k2*wp_PLvq - 1j*grid.k3*wp_PLwq + \
            1j*grid.k3*pterm

    main.Q[0::6,0::6,0::6] = unpad_2x(PLu,1) + main.w0_u
    main.Q[1::6,1::6,1::6] = unpad_2x(PLv,1) + main.w0_v
    main.Q[2::6,2::6,2::6] = unpad_2x(PLw,1) + main.w0_w
    main.Q[3::6,3::6,3::6] = -2./main.dt0*main.w0_u + 2.*main.PLQLu 
    main.Q[4::6,4::6,4::6] = -2./main.dt0*main.w0_v + 2.*main.PLQLv
    main.Q[5::6,5::6,5::6] = -2./main.dt0*main.w0_w + 2.*main.PLQLw


def computeRHS_FM2(main,grid,myFFT):
    main.Q2U()
    ## do 2x padding because we want to have convolutions where 
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

    main.uhat = unpad(pad(main.uhat,1),1)
    main.vhat = unpad(pad(main.vhat,1),1)
    main.what = unpad(pad(main.what,1),1)


    uhat_pad = pad_2x(main.uhat,1)
    vhat_pad = pad_2x(main.vhat,1)
    what_pad = pad_2x(main.what,1)
    ureal[:,:,:] = myFFT.ifftT_obj2(uhat_pad*scale)
    vreal[:,:,:] = myFFT.ifftT_obj2(vhat_pad*scale)
    wreal[:,:,:] = myFFT.ifftT_obj2(what_pad*scale)

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

    PLu_qreal[:,:,:] = myFFT.ifftT_obj2(PLu_q*scale)
    PLv_qreal[:,:,:] = myFFT.ifftT_obj2(PLv_q*scale)
    PLw_qreal[:,:,:] = myFFT.ifftT_obj2(PLw_q*scale)

    up_PLuq = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    vp_PLuq = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    wp_PLuq = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    up_PLvq = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    vp_PLvq = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    wp_PLvq = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    up_PLwq = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    vp_PLwq = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    wp_PLwq = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')

    up_PLuq[:,:,:] = myFFT.fft_obj2(ureal*PLu_qreal)
    vp_PLuq[:,:,:] = myFFT.fft_obj2(vreal*PLu_qreal)
    wp_PLuq[:,:,:] = myFFT.fft_obj2(wreal*PLu_qreal)

    up_PLvq[:,:,:] = myFFT.fft_obj2(ureal*PLv_qreal)
    vp_PLvq[:,:,:] = myFFT.fft_obj2(vreal*PLv_qreal)
    wp_PLvq[:,:,:] = myFFT.fft_obj2(wreal*PLv_qreal)

    up_PLwq[:,:,:] = myFFT.fft_obj2(ureal*PLw_qreal)
    vp_PLwq[:,:,:] = myFFT.fft_obj2(vreal*PLw_qreal)
    wp_PLwq[:,:,:] = myFFT.fft_obj2(wreal*PLw_qreal)

    pterm = 2.*grid.ksqrf_i*( grid.k1f*grid.k1f*up_PLuq + grid.k2f*grid.k2f*vp_PLvq + grid.k3f*grid.k3f*wp_PLwq + \
                          grid.k1f*grid.k2f*(up_PLvq + vp_PLuq) + grid.k1f*grid.k3f*(up_PLwq + wp_PLuq) + \
                          grid.k2f*grid.k3f*(vp_PLwq + wp_PLvq) )

    PLQLu = -1j*grid.k1f*up_PLuq - 1j*grid.k2f*vp_PLuq - 1j*grid.k3f*wp_PLuq - \
            1j*grid.k1f*up_PLuq - 1j*grid.k2f*up_PLvq - 1j*grid.k3f*up_PLwq + \
            1j*grid.k1f*pterm

    PLQLv = -1j*grid.k1f*up_PLvq - 1j*grid.k2f*vp_PLvq - 1j*grid.k3f*wp_PLvq - \
            1j*grid.k1f*vp_PLuq - 1j*grid.k2f*vp_PLvq - 1j*grid.k3f*vp_PLwq + \
            1j*grid.k2f*pterm

    PLQLw = -1j*grid.k1f*up_PLwq - 1j*grid.k2f*vp_PLwq - 1j*grid.k3f*wp_PLwq -\
            1j*grid.k1f*wp_PLuq - 1j*grid.k2f*wp_PLvq - 1j*grid.k3f*wp_PLwq + \
            1j*grid.k3f*pterm

#    print(np.linalg.norm(PLu),np.linalg.norm(PLQLv),np.linalg.norm(PLQLw))
    ### Now get second order model
    PLQLu_p,PLQLu_q = seperateModes(PLQLu,1)
    PLQLv_p,PLQLv_q = seperateModes(PLQLv,1)
    PLQLw_p,PLQLw_q = seperateModes(PLQLw,1)
 
#    PLu_real = zeros(np.shape(PLu_qreal)) 
    PLu_real  = myFFT.ifftT_obj2(PLu[:,:,:]*scale)
    PLv_real  = myFFT.ifftT_obj2(PLv[:,:,:]*scale)
    PLw_real  = myFFT.ifftT_obj2(PLw[:,:,:]*scale)
 #   print(np.linalg.norm(PLu),np.linalg.norm(PLQLv),np.linalg.norm(PLQLw))

#
    ### Now get second order model
    PLQLu_p,PLQLu_q = seperateModes(PLQLu,1)
    PLQLv_p,PLQLv_q = seperateModes(PLQLv,1)
    PLQLw_p,PLQLw_q = seperateModes(PLQLw,1)

    ## We need to pad for the convolutions with p \in F U G 
    scale2= np.sqrt( (3.)**3*np.sqrt(grid.N1*grid.N2*grid.N3) ) #2*3/2=3

    PLu_real  = myFFT.ifftT_obj3(pad(PLu[:,:,:],1)*scale2)
    PLv_real  = myFFT.ifftT_obj3(pad(PLv[:,:,:],1)*scale2)
    PLw_real  = myFFT.ifftT_obj3(pad(PLw[:,:,:],1)*scale2)

    PLu_qreal = myFFT.ifftT_obj3(pad(PLu_q,1)*scale2)
    PLv_qreal = myFFT.ifftT_obj3(pad(PLv_q,1)*scale2)
    PLw_qreal = myFFT.ifftT_obj3(pad(PLw_q,1)*scale2)

    PLu_PLuq = unpad_2x( unpad( myFFT.fft_obj3(PLu_real*PLu_qreal) , 1 ) , 1)
    PLv_PLuq = unpad_2x( unpad( myFFT.fft_obj3(PLv_real*PLu_qreal) , 1 ) , 1)
    PLw_PLuq = unpad_2x( unpad( myFFT.fft_obj3(PLw_real*PLu_qreal) , 1 ) , 1)
    PLu_PLvq = unpad_2x( unpad( myFFT.fft_obj3(PLu_real*PLv_qreal) , 1 ) , 1)
    PLv_PLvq = unpad_2x( unpad( myFFT.fft_obj3(PLv_real*PLv_qreal) , 1 ) , 1)
    PLw_PLvq = unpad_2x( unpad( myFFT.fft_obj3(PLw_real*PLv_qreal) , 1 ) , 1)
    PLu_PLwq = unpad_2x( unpad( myFFT.fft_obj3(PLu_real*PLw_qreal) , 1 ) , 1)
    PLv_PLwq = unpad_2x( unpad( myFFT.fft_obj3(PLv_real*PLw_qreal) , 1 ) , 1)
    PLw_PLwq = unpad_2x( unpad( myFFT.fft_obj3(PLw_real*PLw_qreal) , 1 ) , 1)

    PLQLu_qreal = myFFT.ifftT_obj2(PLQLu_q*scale)
    PLQLv_qreal = myFFT.ifftT_obj2(PLQLv_q*scale)
    PLQLw_qreal = myFFT.ifftT_obj2(PLQLw_q*scale)

    PLQLuq_up = unpad_2x( myFFT.fft_obj2(PLQLu_qreal*ureal) , 1)
    PLQLvq_up = unpad_2x( myFFT.fft_obj2(PLQLv_qreal*ureal) , 1)
    PLQLwq_up = unpad_2x( myFFT.fft_obj2(PLQLw_qreal*ureal) , 1)
    PLQLuq_vp = unpad_2x( myFFT.fft_obj2(PLQLu_qreal*vreal) , 1)
    PLQLvq_vp = unpad_2x( myFFT.fft_obj2(PLQLv_qreal*vreal) , 1)
    PLQLwq_vp = unpad_2x( myFFT.fft_obj2(PLQLw_qreal*vreal) , 1)
    PLQLuq_wp = unpad_2x( myFFT.fft_obj2(PLQLu_qreal*wreal) , 1)
    PLQLvq_wp = unpad_2x( myFFT.fft_obj2(PLQLv_qreal*wreal) , 1)
    PLQLwq_wp = unpad_2x( myFFT.fft_obj2(PLQLw_qreal*wreal) , 1)

    pterm = grid.ksqr_i*( grid.k1*grid.k1*PLu_PLuq + grid.k2*grid.k2*PLv_PLvq + grid.k3*grid.k3*PLw_PLwq + \
                          grid.k1*grid.k2*(PLv_PLuq + PLu_PLvq) + grid.k1*grid.k3*(PLu_PLwq + PLw_PLuq) + \
                          grid.k2*grid.k3*(PLv_PLwq + PLw_PLvq) )
  
    pterm2= grid.ksqr_i*( grid.k1*grid.k1*PLQLuq_up + grid.k2*grid.k2*PLQLvq_vp + grid.k3*grid.k3*PLQLwq_wp + \
                          grid.k1*grid.k2*(PLQLvq_up + PLQLuq_vp) + grid.k1*grid.k3*(PLQLwq_up + PLQLuq_wp) + \
                          grid.k2*grid.k3*(PLQLwq_vp + PLQLvq_wp) )

    PLQLu_q = unpad_2x( PLQLu_q , 1)
    PLQLv_q = unpad_2x( PLQLv_q , 1)
    PLQLw_q = unpad_2x( PLQLw_q , 1)

    PLQLQLu = 2.*-1j*grid.k1*PLu_PLuq - 1j*grid.k2*PLv_PLuq - 1j*grid.k3*PLw_PLuq - 1j*grid.k2*PLu_PLvq - 1j*grid.k3*PLu_PLwq + 2.*1j*grid.k1*pterm + \
             -2.*-1j*grid.k1*PLQLuq_up - 1j*grid.k2*PLQLvq_up - 1j*grid.k3*PLQLwq_up - 1j*grid.k2*PLQLuq_vp - 1j*grid.k3*PLQLuq_wp + \
              2.* 1j*grid.k1*pterm2 - main.nu*grid.ksqr*PLQLu_q

    PLQLQLv = -1j*grid.k1*PLu_PLvq - 2.*1j*grid.k2*PLv_PLvq - 1j*grid.k3*PLw_PLvq - 1j*grid.k1*PLv_PLuq - 1j*grid.k3*PLv_PLwq + 2.*1j*grid.k2*pterm + \
              -1j*grid.k1*PLQLuq_vp - 2.*1j*grid.k2*PLQLvq_vp - 1j*grid.k3*PLQLwq_vp - 1j*grid.k1*PLQLvq_up - 1j*grid.k3*PLQLvq_wp + \
              2.* 1j*grid.k2*pterm2 - main.nu*grid.ksqr*PLQLv_q

    PLQLQLw = -1j*grid.k1*PLu_PLwq - 1j*grid.k2*PLv_PLwq - 2.*1j*grid.k3*PLw_PLwq - 1j*grid.k1*PLw_PLuq - 1j*grid.k2*PLw_PLvq + 2.*1j*grid.k3*pterm + \
              -1j*grid.k1*PLQLuq_wp - 1j*grid.k2*PLQLvq_wp - 2.*1j*grid.k3*PLQLwq_wp - 1j*grid.k1*PLQLwq_up - 1j*grid.k3*PLQLwq_vp + \
              2.* 1j*grid.k3*pterm2 - main.nu*grid.ksqr*PLQLw_q

    PLQLu = unpad_2x( PLQLu , 1)
    PLQLv = unpad_2x( PLQLv , 1)
    PLQLw = unpad_2x( PLQLw , 1)


    main.Q[0::9,0::9,0::9] = unpad_2x(PLu,1) + main.w0_u
    main.Q[1::9,1::9,1::9] = unpad_2x(PLv,1) + main.w0_v
    main.Q[2::9,2::9,2::9] = unpad_2x(PLw,1) + main.w0_w
    main.Q[3::9,3::9,3::9] = -2./main.dt0*main.w0_u + 2.*PLQLu + main.w1_u  #+main.t*PLQLQLu 
    main.Q[4::9,4::9,4::9] = -2./main.dt0*main.w0_v + 2.*PLQLv + main.w1_v  #+main.t*PLQLQLv
    main.Q[5::9,5::9,5::9] = -2./main.dt0*main.w0_w + 2.*PLQLw + main.w1_w  #+main.t*PLQLQLw
    main.Q[6::9,6::9,6::9] = -2./main.dt1*main.w1_u + 2.*PLQLQLu  
    main.Q[7::9,7::9,7::9] = -2./main.dt1*main.w1_v + 2.*PLQLQLv 
    main.Q[8::9,8::9,8::9] = -2./main.dt1*main.w1_w + 2.*PLQLQLw
#    main.Q[3::9,3::9,3::9] = PLQLu + 0.0001*main.t*PLQLQLu 
#    main.Q[4::9,4::9,4::9] = PLQLv + 0.0001*main.t*PLQLQLv
#    main.Q[5::9,5::9,5::9] = PLQLw + 0.0001*main.t*PLQLQLw



def computeRHS_FM1_2term(main,grid,myFFT):
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

    main.uhat = unpad(pad(main.uhat,1),1)
    main.vhat = unpad(pad(main.vhat,1),1)
    main.what = unpad(pad(main.what,1),1)


    uhat_pad = pad_2x(main.uhat,1)
    vhat_pad = pad_2x(main.vhat,1)
    what_pad = pad_2x(main.what,1)
    ureal[:,:,:] = myFFT.ifftT_obj2(uhat_pad*scale)
    vreal[:,:,:] = myFFT.ifftT_obj2(vhat_pad*scale)
    wreal[:,:,:] = myFFT.ifftT_obj2(what_pad*scale)

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

    PLu_qreal[:,:,:] = myFFT.ifftT_obj2(PLu_q*scale)
    PLv_qreal[:,:,:] = myFFT.ifftT_obj2(PLv_q*scale)
    PLw_qreal[:,:,:] = myFFT.ifftT_obj2(PLw_q*scale)

    up_PLuq = unpad_2x( myFFT.fft_obj2(ureal*PLu_qreal),1)
    vp_PLuq = unpad_2x( myFFT.fft_obj2(vreal*PLu_qreal),1)
    wp_PLuq = unpad_2x( myFFT.fft_obj2(wreal*PLu_qreal),1)

    up_PLvq = unpad_2x( myFFT.fft_obj2(ureal*PLv_qreal),1)
    vp_PLvq = unpad_2x( myFFT.fft_obj2(vreal*PLv_qreal),1)
    wp_PLvq = unpad_2x( myFFT.fft_obj2(wreal*PLv_qreal),1)

    up_PLwq = unpad_2x( myFFT.fft_obj2(ureal*PLw_qreal),1)
    vp_PLwq = unpad_2x( myFFT.fft_obj2(vreal*PLw_qreal),1)
    wp_PLwq = unpad_2x( myFFT.fft_obj2(wreal*PLw_qreal),1)

    pterm = 2.*grid.ksqr_i*( grid.k1*grid.k1*up_PLuq + grid.k2*grid.k2*vp_PLvq + grid.k3*grid.k3*wp_PLwq + \
                          grid.k1*grid.k2*(up_PLvq + vp_PLuq) + grid.k1*grid.k3*(up_PLwq + wp_PLuq) + \
                          grid.k2*grid.k3*(vp_PLwq + wp_PLvq) )

    main.PLQLu = -1j*grid.k1*up_PLuq - 1j*grid.k2*vp_PLuq - 1j*grid.k3*wp_PLuq - \
            1j*grid.k1*up_PLuq - 1j*grid.k2*up_PLvq - 1j*grid.k3*up_PLwq + \
            1j*grid.k1*pterm

    main.PLQLv = -1j*grid.k1*up_PLvq - 1j*grid.k2*vp_PLvq - 1j*grid.k3*wp_PLvq - \
            1j*grid.k1*vp_PLuq - 1j*grid.k2*vp_PLvq - 1j*grid.k3*vp_PLwq + \
            1j*grid.k2*pterm

    main.PLQLw = -1j*grid.k1*up_PLwq - 1j*grid.k2*vp_PLwq - 1j*grid.k3*wp_PLwq -\
            1j*grid.k1*wp_PLuq - 1j*grid.k2*wp_PLvq - 1j*grid.k3*wp_PLwq + \
            1j*grid.k3*pterm

    main.Q[0::9,0::9,0::9] = unpad_2x(PLu,1) + main.w0_u + main.w01_u
    main.Q[1::9,1::9,1::9] = unpad_2x(PLv,1) + main.w0_v + main.w01_v
    main.Q[2::9,2::9,2::9] = unpad_2x(PLw,1) + main.w0_w + main.w01_w
    main.Q[3::9,3::9,3::9] = -2./main.dt0*main.w0_u + 2.*main.PLQLu 
    main.Q[4::9,4::9,4::9] = -2./main.dt0*main.w0_v + 2.*main.PLQLv
    main.Q[5::9,5::9,5::9] = -2./main.dt0*main.w0_w + 2.*main.PLQLw
    main.Q[6::9,6::9,6::9] = -2./main.dt0*main.w01_u - 2.*main.PLQLu + 4./main.dt0*main.w0_u 
    main.Q[7::9,7::9,7::9] = -2./main.dt0*main.w01_v - 2.*main.PLQLv + 4./main.dt0*main.w0_v
    main.Q[8::9,8::9,8::9] = -2./main.dt0*main.w01_w - 2.*main.PLQLw + 4./main.dt0*main.w0_w

