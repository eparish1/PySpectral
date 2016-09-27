import numpy as np
from padding import *
def computeRHS_NOSGS(main,grid,myFFT,utilities):
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

    #del ureal
    #del vreal
    #del wreal

    phat  = grid.ksqr_i*( -grid.k1*grid.k1*uuhat - grid.k2*grid.k2*vvhat - \
             grid.k3*grid.k3*wwhat - 2.*grid.k1*grid.k2*uvhat - \
             2.*grid.k1*grid.k3*uwhat - 2.*grid.k2*grid.k3*vwhat )

    if (main.rotate == 1):
      phat[:,:,:] = phat[:,:,:] - 2.*grid.ksqr_i*1j*( grid.k1*(main.vhat*main.Om3 - main.what*main.Om2) + 
                    grid.k2*(main.what*main.Om1 - main.uhat*main.Om3) + \
                    grid.k3*(main.uhat*main.Om2 - main.vhat*main.Om1))

    main.Q[0::3,0::3,0::3] = -1j*grid.k1*uuhat - 1j*grid.k2*uvhat - 1j*grid.k3*uwhat - \
                                         1j*grid.k1*phat - main.nu*grid.ksqr*main.uhat

    main.Q[1::3,1::3,1::3] = -1j*grid.k1*uvhat - 1j*grid.k2*vvhat - 1j*grid.k3*vwhat - \
                                         1j*grid.k2*phat - main.nu*grid.ksqr*main.vhat

    main.Q[2::3,2::3,2::3] = -1j*grid.k1*uwhat - 1j*grid.k2*vwhat - 1j*grid.k3*wwhat - \
                                         1j*grid.k3*phat - main.nu*grid.ksqr*main.what

    if (main.rotate == 1):
      main.Q[0::3,0::3,0::3] = main.Q[0::3,0::3,0::3] + 2.*(main.vhat*main.Om3 - main.what*main.Om2)
      main.Q[1::3,1::3,1::3] = main.Q[1::3,1::3,1::3] + 2.*(main.what*main.Om1 - main.uhat*main.Om3)
      main.Q[2::3,2::3,2::3] = main.Q[2::3,2::3,2::3] + 2.*(main.uhat*main.Om2 - main.vhat*main.Om1)


def computeRHS_Orthogonal(main,grid,myFFT,utilities):
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

    #del ureal
    #del vreal
    #del wreal

    phat  = grid.ksqr_i*( -grid.k1*grid.k1*uuhat - grid.k2*grid.k2*vvhat - \
             grid.k3*grid.k3*wwhat - 2.*grid.k1*grid.k2*uvhat - \
             2.*grid.k1*grid.k3*uwhat - 2.*grid.k2*grid.k3*vwhat )

    if (main.rotate == 1):
      phat[:,:,:] = phat[:,:,:] - 2.*grid.ksqr_i*1j*( grid.k1*(main.vhat*main.Om3 - main.what*main.Om2) + 
                    grid.k2*(main.what*main.Om1 - main.uhat*main.Om3) + \
                    grid.k3*(main.uhat*main.Om2 - main.vhat*main.Om1))

    main.Q[0::3,0::3,0::3] = -1j*grid.k1*uuhat - 1j*grid.k2*uvhat - 1j*grid.k3*uwhat - \
                                         1j*grid.k1*phat - main.nu*grid.ksqr*main.uhat

    main.Q[1::3,1::3,1::3] = -1j*grid.k1*uvhat - 1j*grid.k2*vvhat - 1j*grid.k3*vwhat - \
                                         1j*grid.k2*phat - main.nu*grid.ksqr*main.vhat

    main.Q[2::3,2::3,2::3] = -1j*grid.k1*uwhat - 1j*grid.k2*vwhat - 1j*grid.k3*wwhat - \
                                         1j*grid.k3*phat - main.nu*grid.ksqr*main.what

    if (main.rotate == 1):
      main.Q[0::3,0::3,0::3] = main.Q[0::3,0::3,0::3] + 2.*(main.vhat*main.Om3 - main.what*main.Om2)
      main.Q[1::3,1::3,1::3] = main.Q[1::3,1::3,1::3] + 2.*(main.what*main.Om1 - main.uhat*main.Om3)
      main.Q[2::3,2::3,2::3] = main.Q[2::3,2::3,2::3] + 2.*(main.uhat*main.Om2 - main.vhat*main.Om1)



def computeRHS_SMAG(main,grid,myFFT,utilities):
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

    if (main.rotate == 1):
      phat[:,:,:] = phat[:,:,:] - 2.*grid.ksqr_i*1j*( grid.k1*(main.vhat*main.Om3 - main.what*main.Om2) + 
                    grid.k2*(main.what*main.Om1 - main.uhat*main.Om3) + \
                    grid.k3*(main.uhat*main.Om2 - main.vhat*main.Om1))


    S11hat = 1j*grid.k1*main.uhat
    S22hat = 1j*grid.k2*main.vhat
    S33hat = 1j*grid.k3*main.what
    S12hat = 0.5*(1j*grid.k2*main.uhat + 1j*grid.k1*main.vhat)
    S13hat = 0.5*(1j*grid.k3*main.uhat + 1j*grid.k1*main.what)
    S23hat = 0.5*(1j*grid.k3*main.vhat + 1j*grid.k2*main.what)

    S11real = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    S22real = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    S33real = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    S12real = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    S13real = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    S23real = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )

    S11real[:,:,:] = myFFT.ifftT_obj(pad(S11hat,1)*scale)
    S22real[:,:,:] = myFFT.ifftT_obj(pad(S22hat,1)*scale)
    S33real[:,:,:] = myFFT.ifftT_obj(pad(S33hat,1)*scale)
    S12real[:,:,:] = myFFT.ifftT_obj(pad(S12hat,1)*scale)
    S13real[:,:,:] = myFFT.ifftT_obj(pad(S13hat,1)*scale)
    S23real[:,:,:] = myFFT.ifftT_obj(pad(S23hat,1)*scale)
 
    S_magreal = np.sqrt( 2.*(S11real*S11real + S22real*S22real + S33real*S33real + \
              2.*S12real*S12real + 2.*S13real*S13real + 2.*S23real*S23real ) )
    nutreal = 0.16*grid.Delta*0.16*grid.Delta*np.abs(S_magreal)
    
    tau11real = -2.*nutreal*S11real
    tau22real = -2.*nutreal*S22real
    tau33real = -2.*nutreal*S33real
    tau12real = -2.*nutreal*S12real
    tau13real = -2.*nutreal*S13real
    tau23real = -2.*nutreal*S23real
   
    tauhat = np.zeros((grid.N1,grid.N2,grid.N3/2+1,6),dtype='complex') 
    tauhat[:,:,:,0] = unpad( myFFT.fft_obj( -2.*nutreal*S11real ),1)  #11
    tauhat[:,:,:,1] = unpad( myFFT.fft_obj( -2.*nutreal*S22real ),1)  #22
    tauhat[:,:,:,2] = unpad( myFFT.fft_obj( -2.*nutreal*S33real ),1)  #33
    tauhat[:,:,:,3] = unpad( myFFT.fft_obj( -2.*nutreal*S12real ),1)  #12
    tauhat[:,:,:,4] = unpad( myFFT.fft_obj( -2.*nutreal*S13real ),1)  #13
    tauhat[:,:,:,5] = unpad( myFFT.fft_obj( -2.*nutreal*S23real ),1)  #23

    ## contribution of projection to RHS sgs (k_m k_j)/k^2 \tau_{jm}
    tau_projection  = 1j*grid.ksqr_i*( grid.k1*grid.k1*tauhat[:,:,:,0] + grid.k2*grid.k2*tauhat[:,:,:,1] + \
             grid.k3*grid.k3*tauhat[:,:,:,2] + 2.*grid.k1*grid.k2*tauhat[:,:,:,3] + \
             2.*grid.k1*grid.k3*tauhat[:,:,:,4] + 2.*grid.k2*grid.k3*tauhat[:,:,:,5] )

    #Now SGS contributions. w = -\delta_{im}\tau_{jm}  + k_i k_m / k^2 i k_j \tau_{jm}
    main.w0_u[:,:,:,0] = -1j*grid.k1*tauhat[:,:,:,0] - 1j*grid.k2*tauhat[:,:,:,3] - 1j*grid.k3*tauhat[:,:,:,4] + 1j*grid.k1*tau_projection
    main.w0_v[:,:,:,0] = -1j*grid.k1*tauhat[:,:,:,3] - 1j*grid.k2*tauhat[:,:,:,1] - 1j*grid.k3*tauhat[:,:,:,5] + 1j*grid.k2*tau_projection
    main.w0_w[:,:,:,0] = -1j*grid.k1*tauhat[:,:,:,4] - 1j*grid.k2*tauhat[:,:,:,5] - 1j*grid.k3*tauhat[:,:,:,2] + 1j*grid.k3*tau_projection

    main.Q[0::3,0::3,0::3] = -1j*grid.k1*uuhat - 1j*grid.k2*uvhat - 1j*grid.k3*uwhat - \
                           1j*grid.k1*phat - main.nu*grid.ksqr*main.uhat + main.w0_u[:,:,:,0]

    main.Q[1::3,1::3,1::3] = -1j*grid.k1*uvhat - 1j*grid.k2*vvhat - 1j*grid.k3*vwhat - \
                           1j*grid.k2*phat - main.nu*grid.ksqr*main.vhat + main.w0_v[:,:,:,0] 
 

    main.Q[2::3,2::3,2::3] = -1j*grid.k1*uwhat - 1j*grid.k2*vwhat - 1j*grid.k3*wwhat - \
                           1j*grid.k3*phat - main.nu*grid.ksqr*main.what + main.w0_w[:,:,:,0]

    if (main.rotate == 1):
      main.Q[0::3,0::3,0::3] = main.Q[0::3,0::3,0::3] + 2.*(main.vhat*main.Om3 - main.what*main.Om2)
      main.Q[1::3,1::3,1::3] = main.Q[1::3,1::3,1::3] + 2.*(main.what*main.Om1 - main.uhat*main.Om3)
      main.Q[2::3,2::3,2::3] = main.Q[2::3,2::3,2::3] + 2.*(main.uhat*main.Om2 - main.vhat*main.Om1)


def computeRHS_tmodel(main,grid,myFFT,utilities):
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

    if (main.rotate == 1):
      phat[:,:,:] = phat[:,:,:] - 2.*grid.ksqrf_i*1j*( grid.k1f*(vhat_pad*main.Om3 - what_pad*main.Om2) + 
                    grid.k2f*(what_pad*main.Om1 - uhat_pad*main.Om3) + \
                    grid.k3f*(uhat_pad*main.Om2 - vhat_pad*main.Om1))



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

    main.w0_u[:,:,:,0] = main.Ct*main.t*( -1j*grid.k1*up_PLuq - 1j*grid.k2*vp_PLuq - 1j*grid.k3*wp_PLuq - \
            1j*grid.k1*up_PLuq - 1j*grid.k2*up_PLvq - 1j*grid.k3*up_PLwq + \
            1j*grid.k1*pterm )

    main.w0_v[:,:,:,0] = main.Ct*main.t*( -1j*grid.k1*up_PLvq - 1j*grid.k2*vp_PLvq - 1j*grid.k3*wp_PLvq - \
            1j*grid.k1*vp_PLuq - 1j*grid.k2*vp_PLvq - 1j*grid.k3*vp_PLwq + \
            1j*grid.k2*pterm )

    main.w0_w[:,:,:,0] = main.Ct*main.t*( -1j*grid.k1*up_PLwq - 1j*grid.k2*vp_PLwq - 1j*grid.k3*wp_PLwq -\
            1j*grid.k1*wp_PLuq - 1j*grid.k2*wp_PLvq - 1j*grid.k3*wp_PLwq + \
            1j*grid.k3*pterm )

    main.Q[0::3,0::3,0::3] = unpad_2x(PLu,1) + main.w0_u[:,:,:,0]

    main.Q[1::3,1::3,1::3] = unpad_2x(PLv,1) + main.w0_v[:,:,:,0]

    main.Q[2::3,2::3,2::3] = unpad_2x(PLw,1) + main.w0_w[:,:,:,0]

    if (main.rotate == 1):
      main.Q[0::3,0::3,0::3] = main.Q[0::3,0::3,0::3] + 2.*(main.vhat*main.Om3 - main.what*main.Om2)
      main.Q[1::3,1::3,1::3] = main.Q[1::3,1::3,1::3] + 2.*(main.what*main.Om1 - main.uhat*main.Om3)
      main.Q[2::3,2::3,2::3] = main.Q[2::3,2::3,2::3] + 2.*(main.uhat*main.Om2 - main.vhat*main.Om1)


def computeRHS_staumodel(main,grid,myFFT,utilities):
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

    if (main.rotate == 1):
      phat[:,:,:] = phat[:,:,:] - 2.*grid.ksqrf_i*1j*( grid.k1f*(vhat_pad*main.Om3 - what_pad*main.Om2) + 
                    grid.k2f*(what_pad*main.Om1 - uhat_pad*main.Om3) + \
                    grid.k3f*(uhat_pad*main.Om2 - vhat_pad*main.Om1))



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

    PLQLu = -1j*grid.k1*up_PLuq - 1j*grid.k2*vp_PLuq - 1j*grid.k3*wp_PLuq - \
             1j*grid.k1*up_PLuq - 1j*grid.k2*up_PLvq - 1j*grid.k3*up_PLwq + \
             1j*grid.k1*pterm

    PLQLv = -1j*grid.k1*up_PLvq - 1j*grid.k2*vp_PLvq - 1j*grid.k3*wp_PLvq - \
             1j*grid.k1*vp_PLuq - 1j*grid.k2*vp_PLvq - 1j*grid.k3*vp_PLwq + \
             1j*grid.k2*pterm 

    PLQLw = -1j*grid.k1*up_PLwq - 1j*grid.k2*vp_PLwq - 1j*grid.k3*wp_PLwq -\
             1j*grid.k1*wp_PLuq - 1j*grid.k2*wp_PLvq - 1j*grid.k3*wp_PLwq + \
             1j*grid.k3*pterm 

    main.tau = main.tau_a + main.tau_b*main.t
    tau = max(0.,main.tau)
    #tau = 0.1
    main.w0_u[:,:,:,0] =tau*PLQLu
    main.w0_v[:,:,:,0] =tau*PLQLv
    main.w0_w[:,:,:,0] =tau*PLQLw

    main.Q[0::3,0::3,0::3] = unpad_2x(PLu,1) + main.w0_u[:,:,:,0]

    main.Q[1::3,1::3,1::3] = unpad_2x(PLv,1) + main.w0_v[:,:,:,0]

    main.Q[2::3,2::3,2::3] = unpad_2x(PLw,1) + main.w0_w[:,:,:,0]

    if (main.rotate == 1):
      main.Q[0::3,0::3,0::3] = main.Q[0::3,0::3,0::3] + 2.*(main.vhat*main.Om3 - main.what*main.Om2)
      main.Q[1::3,1::3,1::3] = main.Q[1::3,1::3,1::3] + 2.*(main.what*main.Om1 - main.uhat*main.Om3)
      main.Q[2::3,2::3,2::3] = main.Q[2::3,2::3,2::3] + 2.*(main.uhat*main.Om2 - main.vhat*main.Om1)


def computeRHS_Dtaumodel(main,grid,myFFT,utilities):
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

    if (main.rotate == 1):
      phat[:,:,:] = phat[:,:,:] - 2.*grid.ksqrf_i*1j*( grid.k1f*(vhat_pad*main.Om3 - what_pad*main.Om2) + 
                    grid.k2f*(what_pad*main.Om1 - uhat_pad*main.Om3) + \
                    grid.k3f*(uhat_pad*main.Om2 - vhat_pad*main.Om1))



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

    PLQLu = -1j*grid.k1*up_PLuq - 1j*grid.k2*vp_PLuq - 1j*grid.k3*wp_PLuq - \
             1j*grid.k1*up_PLuq - 1j*grid.k2*up_PLvq - 1j*grid.k3*up_PLwq + \
             1j*grid.k1*pterm

    PLQLv = -1j*grid.k1*up_PLvq - 1j*grid.k2*vp_PLvq - 1j*grid.k3*wp_PLvq - \
             1j*grid.k1*vp_PLuq - 1j*grid.k2*vp_PLvq - 1j*grid.k3*vp_PLwq + \
             1j*grid.k2*pterm 

    PLQLw = -1j*grid.k1*up_PLwq - 1j*grid.k2*vp_PLwq - 1j*grid.k3*wp_PLwq -\
             1j*grid.k1*wp_PLuq - 1j*grid.k2*wp_PLvq - 1j*grid.k3*wp_PLwq + \
             1j*grid.k3*pterm 

    ## Now do dynamic procedure to figure out tau 
    ureal_filt = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(2*grid.N3)) )
    vreal_filt = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(2*grid.N3)) )
    wreal_filt = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(2*grid.N3)) )

    PLu_qreal_filt = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(2*grid.N3)) )
    PLv_qreal_filt = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(2*grid.N3)) )
    PLw_qreal_filt = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(2*grid.N3)) )

    PLu_p_filt = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(grid.N3+1)) ,dtype = 'complex')
    PLv_p_filt = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(grid.N3+1)) ,dtype = 'complex')
    PLw_p_filt = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(grid.N3+1)) ,dtype = 'complex')
    PLu_q_filt = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(grid.N3+1)) ,dtype = 'complex')
    PLv_q_filt = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(grid.N3+1)) ,dtype = 'complex')
    PLw_q_filt = np.zeros( (int(2.*grid.N1),int(2.*grid.N2),int(grid.N3+1)) ,dtype = 'complex')


    uhat_filt = main.test_filter(main.uhat)
    vhat_filt = main.test_filter(main.vhat)
    what_filt = main.test_filter(main.what)


    uhat_pad_filt = pad_2x(uhat_filt,1)
    vhat_pad_filt = pad_2x(vhat_filt,1)
    what_pad_filt = pad_2x(what_filt,1)
    ureal_filt[:] = myFFT.ifftT_obj2(uhat_pad_filt*scale)
    vreal_filt[:] = myFFT.ifftT_obj2(vhat_pad_filt*scale)
    wreal_filt[:] = myFFT.ifftT_obj2(what_pad_filt*scale)

    uuhat_filt = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    vvhat_filt = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    wwhat_filt = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    uvhat_filt = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    uwhat_filt = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')
    vwhat_filt = np.zeros((2*grid.N1,2*grid.N2,grid.N3+1),dtype = 'complex')

    uuhat_filt[:,:,:] = myFFT.fft_obj2(ureal_filt[:,:,:]*ureal_filt[:,:,:])
    vvhat_filt[:,:,:] = myFFT.fft_obj2(vreal_filt[:,:,:]*vreal_filt[:,:,:])
    wwhat_filt[:,:,:] = myFFT.fft_obj2(wreal_filt[:,:,:]*wreal_filt[:,:,:])
    uvhat_filt[:,:,:] = myFFT.fft_obj2(ureal_filt[:,:,:]*vreal_filt[:,:,:])
    uwhat_filt[:,:,:] = myFFT.fft_obj2(ureal_filt[:,:,:]*wreal_filt[:,:,:])
    vwhat_filt[:,:,:] = myFFT.fft_obj2(vreal_filt[:,:,:]*wreal_filt[:,:,:])

    uuhat2_filt = unpad_2x(uuhat_filt,1)
    vvhat2_filt = unpad_2x(vvhat_filt,1)
    wwhat2_filt = unpad_2x(wwhat_filt,1)
    uvhat2_filt = unpad_2x(uvhat_filt,1)
    uwhat2_filt = unpad_2x(uwhat_filt,1)
    vwhat2_filt = unpad_2x(vwhat_filt,1)

    phat_filt  = -grid.k1f*grid.k1f*grid.ksqrf_i*uuhat_filt - grid.k2f*grid.k2f*grid.ksqrf_i*vvhat_filt - \
             grid.k3f*grid.k3f*grid.ksqrf_i*wwhat_filt - 2.*grid.k1f*grid.k2f*grid.ksqrf_i*uvhat_filt - \
             2.*grid.k1f*grid.k3f*grid.ksqrf_i*uwhat_filt - 2.*grid.k2f*grid.k3f*grid.ksqrf_i*vwhat_filt

    if (main.rotate == 1):
      phat_filt[:,:,:] = phat_filt[:,:,:] - 2.*grid.ksqrf_i*1j*( grid.k1f*(vhat_pad_filt*main.Om3 - what_pad_filt*main.Om2) + 
                    grid.k2f*(what_pad_filt*main.Om1 - uhat_pad_filt*main.Om3) + \
                    grid.k3f*(uhat_pad_filt*main.Om2 - vhat_pad_filt*main.Om1))



    PLu_filt = -1j*grid.k1f*uuhat_filt - 1j*grid.k2f*uvhat_filt - 1j*grid.k3f*uwhat_filt - \
                                         1j*grid.k1f*phat_filt - main.nu*grid.ksqrf*pad_2x(uhat_filt,1)

    PLv_filt = -1j*grid.k1f*uvhat_filt - 1j*grid.k2f*vvhat_filt - 1j*grid.k3f*vwhat_filt - \
                                         1j*grid.k2f*phat_filt - main.nu*grid.ksqrf*pad_2x(vhat_filt,1)

    PLw_filt = -1j*grid.k1f*uwhat_filt - 1j*grid.k2f*vwhat_filt - 1j*grid.k3f*wwhat_filt - \
                                         1j*grid.k3f*phat_filt - main.nu*grid.ksqrf*pad_2x(what_filt,1)

    PLu_p_filt[:],PLu_q_filt[:] = seperateModes_testFilter(PLu_filt,main)
    PLv_p_filt[:],PLv_q_filt[:] = seperateModes_testFilter(PLv_filt,main)
    PLw_p_filt[:],PLw_q_filt[:] = seperateModes_testFilter(PLw_filt,main)

    PLu_qreal_filt[:] = myFFT.ifftT_obj2(PLu_q_filt*scale)
    PLv_qreal_filt[:] = myFFT.ifftT_obj2(PLv_q_filt*scale)
    PLw_qreal_filt[:] = myFFT.ifftT_obj2(PLw_q_filt*scale)

    up_PLuq_filt = unpad_2x( myFFT.fft_obj2(ureal_filt*PLu_qreal_filt),1)
    vp_PLuq_filt = unpad_2x( myFFT.fft_obj2(vreal_filt*PLu_qreal_filt),1)
    wp_PLuq_filt = unpad_2x( myFFT.fft_obj2(wreal_filt*PLu_qreal_filt),1)

    up_PLvq_filt = unpad_2x( myFFT.fft_obj2(ureal_filt*PLv_qreal_filt),1)
    vp_PLvq_filt = unpad_2x( myFFT.fft_obj2(vreal_filt*PLv_qreal_filt),1)
    wp_PLvq_filt = unpad_2x( myFFT.fft_obj2(wreal_filt*PLv_qreal_filt),1)

    up_PLwq_filt = unpad_2x( myFFT.fft_obj2(ureal_filt*PLw_qreal_filt),1)
    vp_PLwq_filt = unpad_2x( myFFT.fft_obj2(vreal_filt*PLw_qreal_filt),1)
    wp_PLwq_filt = unpad_2x( myFFT.fft_obj2(wreal_filt*PLw_qreal_filt),1)


    pterm_filt = 2.*grid.ksqr_i*( grid.k1*grid.k1*up_PLuq_filt + grid.k2*grid.k2*vp_PLvq_filt + grid.k3*grid.k3*wp_PLwq_filt + \
                          grid.k1*grid.k2*(up_PLvq_filt + vp_PLuq_filt) + grid.k1*grid.k3*(up_PLwq_filt + wp_PLuq_filt) + \
                          grid.k2*grid.k3*(vp_PLwq_filt + wp_PLvq_filt) )

    PLQLu_filt = -1j*grid.k1*up_PLuq_filt - 1j*grid.k2*vp_PLuq_filt - 1j*grid.k3*wp_PLuq_filt - \
             1j*grid.k1*up_PLuq_filt - 1j*grid.k2*up_PLvq_filt - 1j*grid.k3*up_PLwq_filt + \
             1j*grid.k1*pterm_filt

    PLQLv_filt = -1j*grid.k1*up_PLvq_filt - 1j*grid.k2*vp_PLvq_filt - 1j*grid.k3*wp_PLvq_filt - \
             1j*grid.k1*vp_PLuq_filt - 1j*grid.k2*vp_PLvq_filt - 1j*grid.k3*vp_PLwq_filt + \
             1j*grid.k2*pterm_filt 

    PLQLw_filt = -1j*grid.k1*up_PLwq_filt - 1j*grid.k2*vp_PLwq_filt - 1j*grid.k3*wp_PLwq_filt -\
             1j*grid.k1*wp_PLuq_filt - 1j*grid.k2*wp_PLvq_filt - 1j*grid.k3*wp_PLwq_filt + \
             1j*grid.k3*pterm_filt 

    ## Now compute Leonard stress
    Lu = main.test_filter_2x( -1j*grid.k1f*uuhat      - 1j*grid.k2f*uvhat      - 1j*grid.k3f*uwhat      + 1j*grid.k1f*phat) - \
                           main.test_filter_2x ( -1j*grid.k1f*uuhat_filt - 1j*grid.k2f*uvhat_filt - 1j*grid.k3f*uwhat_filt + 1j*grid.k1f*phat_filt)
    Lv = main.test_filter_2x( -1j*grid.k1f*uvhat      - 1j*grid.k2f*vvhat      - 1j*grid.k3f*vwhat      + 1j*grid.k2f*phat) - \
                           main.test_filter_2x ( -1j*grid.k1f*uvhat_filt - 1j*grid.k2f*vvhat_filt - 1j*grid.k3f*vwhat_filt + 1j*grid.k2f*phat_filt)
    Lw = main.test_filter_2x( -1j*grid.k1f*uwhat      - 1j*grid.k2f*vwhat      - 1j*grid.k3f*wwhat      + 1j*grid.k3f*phat) - \
                           main.test_filter_2x ( -1j*grid.k1f*uwhat_filt - 1j*grid.k2f*vwhat_filt - 1j*grid.k3f*wwhat_filt + 1j*grid.k3f*phat_filt)

    ## Now compute energy up to test filter
    LE =(np.sum(Lu[:,:,1:grid.N3/2]*np.conj(uhat_pad_filt[:,:,1:grid.N3/2]*2) ) + \
         np.sum(Lu[:,:,0]*np.conj(uhat_pad_filt[:,:,0])) + \
         np.sum(Lv[:,:,1:grid.N3/2]*np.conj(vhat_pad_filt[:,:,1:grid.N3/2]*2) ) + \
         np.sum(Lv[:,:,0]*np.conj(vhat_pad_filt[:,:,0])) + \
         np.sum(Lw[:,:,1:grid.N3/2]*np.conj(what_pad_filt[:,:,1:grid.N3/2]*2) ) + \
         np.sum(Lw[:,:,0]*np.conj(what_pad_filt[:,:,0])) )/(grid.N1*grid.N2*grid.N3)

    filt_PLQLu_filt = main.test_filter(PLQLu_filt)
    filt_PLQLv_filt = main.test_filter(PLQLv_filt)
    filt_PLQLw_filt = main.test_filter(PLQLw_filt)


    PLQLE_filt = (\
         np.sum(filt_PLQLu_filt[:,:,1:grid.N3/2]*np.conj(uhat_filt[:,:,1:grid.N3/2]*2) ) + \
         np.sum(filt_PLQLu_filt[:,:,0]*np.conj(uhat_filt[:,:,0])) + \
         np.sum(filt_PLQLv_filt[:,:,1:grid.N3/2]*np.conj(vhat_filt[:,:,1:grid.N3/2]*2) ) + \
         np.sum(filt_PLQLv_filt[:,:,0]*np.conj(vhat_filt[:,:,0])) + \
         np.sum(filt_PLQLw_filt[:,:,1:grid.N3/2]*np.conj(what_filt[:,:,1:grid.N3/2]*2) ) + \
         np.sum(filt_PLQLw_filt[:,:,0]*np.conj(what_filt[:,:,0])) ) / (grid.N1*grid.N2*grid.N3)

    filt_PLQLu = main.test_filter(PLQLu)
    filt_PLQLv = main.test_filter(PLQLv)
    filt_PLQLw = main.test_filter(PLQLw)

    filt_PLQLE = (\
         np.sum(PLQLu[:,:,1:grid.N3/2]*np.conj(uhat_filt[:,:,1:grid.N3/2]*2) ) + \
         np.sum(PLQLu[:,:,0]*np.conj(uhat_filt[:,:,0])) + \
         np.sum(PLQLv[:,:,1:grid.N3/2]*np.conj(vhat_filt[:,:,1:grid.N3/2]*2) ) + \
         np.sum(PLQLv[:,:,0]*np.conj(vhat_filt[:,:,0])) + \
         np.sum(PLQLw[:,:,1:grid.N3/2]*np.conj(what_filt[:,:,1:grid.N3/2]*2) ) + \
         np.sum(PLQLw[:,:,0]*np.conj(what_filt[:,:,0])) ) / (grid.N1*grid.N2*grid.N3)
    print('LE = ' + str(np.linalg.norm(LE)))
#    tau =  np.real( LE ) / (np.real(PLQLE_filt)  - np.real(filt_PLQLE) )
#    tau =  np.real( LE ) / (np.real((grid.kc**2/main.kf**2)*PLQLE_filt)  - np.real(filt_PLQLE) + 1.e-5 )
#    tau =  np.abs( LE ) / (np.abs((grid.kc**2/main.kf**2)*PLQLE_filt)  - np.abs(filt_PLQLE) + 1.e-5 )
    tau =  np.abs( LE ) / (np.abs(PLQLE_filt)  - np.abs(filt_PLQLE) + 1.e-5 )

    main.tau = tau
    print('tau =' + str(tau))
    tau = max(0.,tau)
    #tau = 0.1
    main.w0_u[:,:,:,0] =tau*PLQLu
    main.w0_v[:,:,:,0] =tau*PLQLv
    main.w0_w[:,:,:,0] =tau*PLQLw

    #print(np.linalg.norm(filt_PLQLE))

    main.Q[0::3,0::3,0::3] = unpad_2x(PLu,1) + main.w0_u[:,:,:,0]

    main.Q[1::3,1::3,1::3] = unpad_2x(PLv,1) + main.w0_v[:,:,:,0]

    main.Q[2::3,2::3,2::3] = unpad_2x(PLw,1) + main.w0_w[:,:,:,0]

    if (main.rotate == 1):
      main.Q[0::3,0::3,0::3] = main.Q[0::3,0::3,0::3] + 2.*(main.vhat*main.Om3 - main.what*main.Om2)
      main.Q[1::3,1::3,1::3] = main.Q[1::3,1::3,1::3] + 2.*(main.what*main.Om1 - main.uhat*main.Om3)
      main.Q[2::3,2::3,2::3] = main.Q[2::3,2::3,2::3] + 2.*(main.uhat*main.Om2 - main.vhat*main.Om1)



def computeRHS_FM1(main,grid,myFFT,utilities):
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


    uhat_pad = pad_2x_var(main.uhat,1)
    vhat_pad = pad_2x_var(main.vhat,1)
    what_pad = pad_2x_var(main.what,1)
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


    if (main.rotate == 1):
      phat[:,:,:] = phat[:,:,:] - 2.*grid.ksqrf_i*1j*( grid.k1f*(vhat_pad*main.Om3 - what_pad*main.Om2) + 
                    grid.k2f*(what_pad*main.Om1 - uhat_pad*main.Om3) + \
                    grid.k3f*(uhat_pad*main.Om2 - vhat_pad*main.Om1))


    PLu = -1j*grid.k1f*uuhat - 1j*grid.k2f*uvhat - 1j*grid.k3f*uwhat - \
                                         1j*grid.k1f*phat - main.nu*grid.ksqrf*pad_2x_var(main.uhat,1)

    PLv = -1j*grid.k1f*uvhat - 1j*grid.k2f*vvhat - 1j*grid.k3f*vwhat - \
                                         1j*grid.k2f*phat - main.nu*grid.ksqrf*pad_2x_var(main.vhat,1)

    PLw = -1j*grid.k1f*uwhat - 1j*grid.k2f*vwhat - 1j*grid.k3f*wwhat - \
                                         1j*grid.k3f*phat - main.nu*grid.ksqrf*pad_2x_var(main.what,1)


    if (main.rotate == 1):
      PLu[:] = PLu[:] + 2.*(vhat_pad*main.Om3 - what_pad*main.Om2)
      PLv[:] = PLv[:] + 2.*(what_pad*main.Om1 - uhat_pad*main.Om3)
      PLw[:] = PLw[:] + 2.*(uhat_pad*main.Om2 - vhat_pad*main.Om1)


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



    main.Q[0::main.nvars,0::main.nvars,0::main.nvars] = unpad_2x(PLu,1) + np.sum(main.w0_u,axis=3)
    main.Q[1::main.nvars,1::main.nvars,1::main.nvars] = unpad_2x(PLv,1) + np.sum(main.w0_v,axis=3)
    main.Q[2::main.nvars,2::main.nvars,2::main.nvars] = unpad_2x(PLw,1) + np.sum(main.w0_w,axis=3)

    sum_u = np.zeros(np.shape(main.uhat),dtype='complex') 
    sum_v = np.zeros(np.shape(main.vhat),dtype='complex') 
    sum_w = np.zeros(np.shape(main.what),dtype='complex') 
    term = 3 
    dt0 = main.dt0/main.dt0_subintegrations
    for i in range(1,main.dt0_subintegrations+1):
      for j in range(1,i):
        sum_u += 4./dt0*(-1.)**(i+j+1)*main.w0_u[:,:,:,j-1]
        sum_v += 4./dt0*(-1.)**(i+j+1)*main.w0_v[:,:,:,j-1]
        sum_w += 4./dt0*(-1.)**(i+j+1)*main.w0_w[:,:,:,j-1]
      main.Q[term+0::main.nvars,term+0::main.nvars,term+0::main.nvars]  = -2./dt0*main.w0_u[:,:,:,i-1] + (-1.)**(i+1)*2.*main.PLQLu + sum_u
      main.Q[term+1::main.nvars,term+1::main.nvars,term+1::main.nvars]  = -2./dt0*main.w0_v[:,:,:,i-1] + (-1.)**(i+1)*2.*main.PLQLv + sum_v
      main.Q[term+2::main.nvars,term+2::main.nvars,term+2::main.nvars]  = -2./dt0*main.w0_w[:,:,:,i-1] + (-1.)**(i+1)*2.*main.PLQLw + sum_w
      term += 3



def computeRHS_FM2(main,grid,myFFT,utilities):
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

    if (main.rotate == 1):
      phat[:,:,:] = phat[:,:,:] - 2.*grid.ksqr_i*1j*( grid.k1*(main.vhat*main.Om3 - main.what*main.Om2) + 
                    grid.k2*(main.what*main.Om1 - main.uhat*main.Om3) + \
                    grid.k3*(main.uhat*main.Om2 - main.vhat*main.Om1))


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

    ## We need to pad for the convolutions with p \in F U G 
    scale2= np.sqrt( (3.)**3*np.sqrt(grid.N1*grid.N2*grid.N3) ) #2*3/2=3

    PLu_real  = np.zeros((3*grid.N1,3*grid.N2,3*grid.N3))
    PLv_real  = np.zeros((3*grid.N1,3*grid.N2,3*grid.N3))
    PLw_real  = np.zeros((3*grid.N1,3*grid.N2,3*grid.N3))
    PLu_qreal = np.zeros((3*grid.N1,3*grid.N2,3*grid.N1))
    PLv_qreal = np.zeros((3*grid.N1,3*grid.N2,3*grid.N3))
    PLw_qreal = np.zeros((3*grid.N1,3*grid.N2,3*grid.N3))

    PLu_real[:,:,:]  = myFFT.ifftT_obj3(pad(PLu[:,:,:],1)*scale2)
    PLv_real[:,:,:]  = myFFT.ifftT_obj3(pad(PLv[:,:,:],1)*scale2)
    PLw_real[:,:,:]  = myFFT.ifftT_obj3(pad(PLw[:,:,:],1)*scale2)

    PLu_qreal[:,:,:] = myFFT.ifftT_obj3(pad(PLu_q,1)*scale2)
    PLv_qreal[:,:,:] = myFFT.ifftT_obj3(pad(PLv_q,1)*scale2)
    PLw_qreal[:,:,:] = myFFT.ifftT_obj3(pad(PLw_q,1)*scale2)

    PLu_PLuq = unpad_2x( unpad( myFFT.fft_obj3(PLu_real*PLu_qreal) , 1 ) , 1)
    PLv_PLuq = unpad_2x( unpad( myFFT.fft_obj3(PLv_real*PLu_qreal) , 1 ) , 1)
    PLw_PLuq = unpad_2x( unpad( myFFT.fft_obj3(PLw_real*PLu_qreal) , 1 ) , 1)
    PLu_PLvq = unpad_2x( unpad( myFFT.fft_obj3(PLu_real*PLv_qreal) , 1 ) , 1)
    PLv_PLvq = unpad_2x( unpad( myFFT.fft_obj3(PLv_real*PLv_qreal) , 1 ) , 1)
    PLw_PLvq = unpad_2x( unpad( myFFT.fft_obj3(PLw_real*PLv_qreal) , 1 ) , 1)
    PLu_PLwq = unpad_2x( unpad( myFFT.fft_obj3(PLu_real*PLw_qreal) , 1 ) , 1)
    PLv_PLwq = unpad_2x( unpad( myFFT.fft_obj3(PLv_real*PLw_qreal) , 1 ) , 1)
    PLw_PLwq = unpad_2x( unpad( myFFT.fft_obj3(PLw_real*PLw_qreal) , 1 ) , 1)

    PLQLu_qreal = np.zeros((2*grid.N1,2*grid.N2,2*grid.N3))
    PLQLv_qreal = np.zeros((2*grid.N1,2*grid.N2,2*grid.N3))
    PLQLw_qreal = np.zeros((2*grid.N1,2*grid.N2,2*grid.N3))
    PLQLu_qreal[:,:,:] = myFFT.ifftT_obj2(PLQLu_q*scale)
    PLQLv_qreal[:,:,:] = myFFT.ifftT_obj2(PLQLv_q*scale)
    PLQLw_qreal[:,:,:] = myFFT.ifftT_obj2(PLQLw_q*scale)

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


    #main.Q[0::9,0::9,0::9] = unpad_2x(PLu,1) + main.w0_u
    #main.Q[1::9,1::9,1::9] = unpad_2x(PLv,1) + main.w0_v
    #main.Q[2::9,2::9,2::9] = unpad_2x(PLw,1) + main.w0_w
    #main.Q[3::9,3::9,3::9] = -2./main.dt0*main.w0_u + 2.*PLQLu + main.w1_u  #+main.t*PLQLQLu 
    #main.Q[4::9,4::9,4::9] = -2./main.dt0*main.w0_v + 2.*PLQLv + main.w1_v  #+main.t*PLQLQLv
    #main.Q[5::9,5::9,5::9] = -2./main.dt0*main.w0_w + 2.*PLQLw + main.w1_w  #+main.t*PLQLQLw
    #main.Q[6::9,6::9,6::9] = -2./main.dt1*main.w1_u + 2.*PLQLQLu  
    #main.Q[7::9,7::9,7::9] = -2./main.dt1*main.w1_v + 2.*PLQLQLv 
    #main.Q[8::9,8::9,8::9] = -2./main.dt1*main.w1_w + 2.*PLQLQLw

    main.Q[0::main.nvars,0::main.nvars,0::main.nvars] = unpad_2x(PLu,1) + np.sum(main.w0_u,axis=3)
    main.Q[1::main.nvars,1::main.nvars,1::main.nvars] = unpad_2x(PLv,1) + np.sum(main.w0_v,axis=3)
    main.Q[2::main.nvars,2::main.nvars,2::main.nvars] = unpad_2x(PLw,1) + np.sum(main.w0_w,axis=3)

    ## Add contribution to RHS of FM1 models
    sum_u = np.zeros(np.shape(main.uhat),dtype='complex') 
    sum_v = np.zeros(np.shape(main.vhat),dtype='complex') 
    sum_w = np.zeros(np.shape(main.what),dtype='complex') 
    term = 3 
    dt0 = main.dt0/main.dt0_subintegrations
    for i in range(1,main.dt0_subintegrations+1):
      for j in range(1,i):
        sum_u += 4./dt0*(-1.)**(i+j+1)*main.w0_u[:,:,:,j-1]
        sum_v += 4./dt0*(-1.)**(i+j+1)*main.w0_v[:,:,:,j-1]
        sum_w += 4./dt0*(-1.)**(i+j+1)*main.w0_w[:,:,:,j-1]
      main.Q[term+0::main.nvars,term+0::main.nvars,term+0::main.nvars]  = -2./dt0*main.w0_u[:,:,:,i-1] + \
                             (-1.)**(i+1)*2.*PLQLu + sum_u + main.w1_u[:,:,:,i-1]
      main.Q[term+1::main.nvars,term+1::main.nvars,term+1::main.nvars]  = -2./dt0*main.w0_v[:,:,:,i-1] + \
                             (-1.)**(i+1)*2.*PLQLv + sum_v + main.w1_v[:,:,:,i-1]
      main.Q[term+2::main.nvars,term+2::main.nvars,term+2::main.nvars]  = -2./dt0*main.w0_w[:,:,:,i-1] + \
                             (-1.)**(i+1)*2.*PLQLw + sum_w + main.w1_w[:,:,:,i-1]
      ## Increment the array location. Add 6 if there are still FM2 terms, just add 3 if there are not
      if (i < main.dt1_subintegrations):
        term += 6
      else:
        term += 3
    ## Add contribution to RHS of FM2 models
    sum_u = np.zeros(np.shape(main.uhat),dtype='complex') 
    sum_v = np.zeros(np.shape(main.vhat),dtype='complex') 
    sum_w = np.zeros(np.shape(main.what),dtype='complex') 
    term = 6 
    dt1 = main.dt1/main.dt1_subintegrations
    for i in range(1,main.dt1_subintegrations+1):
      for j in range(1,i):
        sum_u += 4./dt1*(-1.)**(i+j+1)*main.w1_u[:,:,:,j-1]
        sum_v += 4./dt1*(-1.)**(i+j+1)*main.w1_v[:,:,:,j-1]
        sum_w += 4./dt1*(-1.)**(i+j+1)*main.w1_w[:,:,:,j-1]
      main.Q[term+0::main.nvars,term+0::main.nvars,term+0::main.nvars]  = -2./dt1*main.w1_u[:,:,:,i-1] + \
                                                                       (-1.)**(i+1)*2.*PLQLQLu + sum_u
      main.Q[term+1::main.nvars,term+1::main.nvars,term+1::main.nvars]  = -2./dt1*main.w1_v[:,:,:,i-1] + \
                                                                       (-1.)**(i+1)*2.*PLQLQLv + sum_v
      main.Q[term+2::main.nvars,term+2::main.nvars,term+2::main.nvars]  = -2./dt1*main.w1_w[:,:,:,i-1] + \
                                                                       (-1.)**(i+1)*2.*PLQLQLw + sum_w
      ## Increment the array location. Add 6 if there are still FM1 terms, just add 3 if there are not
      if (i < main.dt0_subintegrations):
        term += 6
      else:
        term += 3



def computeRHS_FM1_2term(main,grid,myFFT,utilities):
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

    if (main.rotate == 1):
      phat[:,:,:] = phat[:,:,:] - 2.*grid.ksqr_i*1j*( grid.k1*(main.vhat*main.Om3 - main.what*main.Om2) + 
                    grid.k2*(main.what*main.Om1 - main.uhat*main.Om3) + \
                    grid.k3*(main.uhat*main.Om2 - main.vhat*main.Om1))


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


def computeRHS_CM1(main,grid,myFFT,utilities):
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

    if (main.rotate == 1):
      phat[:,:,:] = phat[:,:,:] - 2.*grid.ksqr_i*1j*( grid.k1*(main.vhat*main.Om3 - main.what*main.Om2) + 
                    grid.k2*(main.what*main.Om1 - main.uhat*main.Om3) + \
                    grid.k3*(main.uhat*main.Om2 - main.vhat*main.Om1))


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



    main.Q[0::main.nvars,0::main.nvars,0::main.nvars] = unpad_2x(PLu,1) + main.w0_u[:,:,:,0]
    main.Q[1::main.nvars,1::main.nvars,1::main.nvars] = unpad_2x(PLv,1) + main.w0_v[:,:,:,0]
    main.Q[2::main.nvars,2::main.nvars,2::main.nvars] = unpad_2x(PLw,1) + main.w0_w[:,:,:,0]
    main.Q[3::main.nvars,3::main.nvars,3::main.nvars]  = main.PLQLu - 250*main.nu*grid.ksqr*main.w0_u[:,:,:,0]
    main.Q[4::main.nvars,4::main.nvars,4::main.nvars]  = main.PLQLv - 250*main.nu*grid.ksqr*main.w0_v[:,:,:,0]
    main.Q[5::main.nvars,5::main.nvars,5::main.nvars]  = main.PLQLw - 250*main.nu*grid.ksqr*main.w0_w[:,:,:,0]












def computeRHS_BUDGETS(main,grid,myFFT,utilities):
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

    if (main.rotate == 1):
      phat[:,:,:] = phat[:,:,:] - 2.*grid.ksqr_i*1j*( grid.k1*(main.vhat*main.Om3 - main.what*main.Om2) + 
                    grid.k2*(main.what*main.Om1 - main.uhat*main.Om3) + \
                    grid.k3*(main.uhat*main.Om2 - main.vhat*main.Om1))


    PLu = -1j*grid.k1f*uuhat - 1j*grid.k2f*uvhat - 1j*grid.k3f*uwhat - \
                                         1j*grid.k1f*phat - main.nu*grid.ksqrf*pad_2x(main.uhat,1)

    PLv = -1j*grid.k1f*uvhat - 1j*grid.k2f*vvhat - 1j*grid.k3f*vwhat - \
                                         1j*grid.k2f*phat - main.nu*grid.ksqrf*pad_2x(main.vhat,1)

    PLw = -1j*grid.k1f*uwhat - 1j*grid.k2f*vwhat - 1j*grid.k3f*wwhat - \
                                         1j*grid.k3f*phat - main.nu*grid.ksqrf*pad_2x(main.what,1)

    PLu_p[:,:,:],PLu_q[:,:,:] = seperateModesBudgets(PLu,1,main.kc)
    PLv_p[:,:,:],PLv_q[:,:,:] = seperateModesBudgets(PLv,1,main.kc)
    PLw_p[:,:,:],PLw_q[:,:,:] = seperateModesBudgets(PLw,1,main.kc)

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


    main.PLu[:,:,:] = unpad_2x(PLu,1)
    main.PLv[:,:,:] = unpad_2x(PLv,1)
    main.PLw[:,:,:] = unpad_2x(PLw,1)
    main.Q[0::main.nvars,0::main.nvars,0::main.nvars] = main.PLu[:,:,:]
    main.Q[1::main.nvars,1::main.nvars,1::main.nvars] = main.PLv[:,:,:]
    main.Q[2::main.nvars,2::main.nvars,2::main.nvars] = main.PLw[:,:,:]

def computeRHS_DSMAG(main,grid,myFFT,utilities):
    main.Q2U()
    DSmag_alpha = 2.
    scale = np.sqrt( (3./2.)**3*np.sqrt(grid.N1*grid.N2*grid.N3) )
    ureal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    vreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    wreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    urealF = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    vrealF = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    wrealF = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )

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

    if (main.rotate == 1):
      phat[:,:,:] = phat[:,:,:] - 2.*grid.ksqr_i*1j*( grid.k1*(main.vhat*main.Om3 - main.what*main.Om2) + 
                    grid.k2*(main.what*main.Om1 - main.uhat*main.Om3) + \
                    grid.k3*(main.uhat*main.Om2 - main.vhat*main.Om1))


    #### Dynamic Smagorinsky Contribution
    ## First Need to compute Leonard Stress. Apply test filter at scale k_L
    ## k_c = DSmag_alpha*k_L
    kL = (grid.kc)/DSmag_alpha
    uhatF = utilities.myFilter(main.uhat,kL)
    vhatF = utilities.myFilter(main.vhat,kL)
    whatF = utilities.myFilter(main.what,kL)
    urealF[:,:,:] = myFFT.ifftT_obj(pad(uhatF,1)*scale)
    vrealF[:,:,:] = myFFT.ifftT_obj(pad(vhatF,1)*scale)
    wrealF[:,:,:] = myFFT.ifftT_obj(pad(whatF,1)*scale)
    uuhatF = unpad( myFFT.fft_obj(urealF*urealF),1)
    vvhatF = unpad( myFFT.fft_obj(vrealF*vrealF),1)
    wwhatF = unpad( myFFT.fft_obj(wrealF*wrealF),1)
    uvhatF = unpad( myFFT.fft_obj(urealF*vrealF),1)
    uwhatF = unpad( myFFT.fft_obj(urealF*wrealF),1)
    vwhatF = unpad( myFFT.fft_obj(vrealF*wrealF),1)
    ## Make Leonard Stress Tensor
    Lhat = np.zeros((grid.N1,grid.N2,(grid.N3/2+1),6),dtype='complex')
    Lhat[:,:,:,0] = utilities.myFilter(uuhat,kL) - uuhatF
    Lhat[:,:,:,1] = utilities.myFilter(vvhat,kL) - vvhatF
    Lhat[:,:,:,2] = utilities.myFilter(wwhat,kL) - wwhatF
    Lhat[:,:,:,3] = utilities.myFilter(uvhat,kL) - uvhatF
    Lhat[:,:,:,4] = utilities.myFilter(uwhat,kL) - uwhatF
    Lhat[:,:,:,5] = utilities.myFilter(vwhat,kL) - vwhatF
    ## Now compute the resolved stress tensor and the filtered stress tensor   
    #resolved
    S11hat = 1j*grid.k1*main.uhat
    S22hat = 1j*grid.k2*main.vhat
    S33hat = 1j*grid.k3*main.what
    S12hat = 0.5*(1j*grid.k2*main.uhat + 1j*grid.k1*main.vhat)
    S13hat = 0.5*(1j*grid.k3*main.uhat + 1j*grid.k1*main.what)
    S23hat = 0.5*(1j*grid.k3*main.vhat + 1j*grid.k2*main.what)
    S11real = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    S22real = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    S33real = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    S12real = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    S13real = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    S23real = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    S11real[:,:,:] = myFFT.ifftT_obj(pad(S11hat,1)*scale)
    S22real[:,:,:] = myFFT.ifftT_obj(pad(S22hat,1)*scale)
    S33real[:,:,:] = myFFT.ifftT_obj(pad(S33hat,1)*scale)
    S12real[:,:,:] = myFFT.ifftT_obj(pad(S12hat,1)*scale)
    S13real[:,:,:] = myFFT.ifftT_obj(pad(S13hat,1)*scale)
    S23real[:,:,:] = myFFT.ifftT_obj(pad(S23hat,1)*scale)
    S_magreal = np.sqrt( 2.*(S11real*S11real + S22real*S22real + S33real*S33real + \
              2.*S12real*S12real + 2.*S13real*S13real + 2.*S23real*S23real ) )
    ## Filtered Stress Tensor
    S11hatF = 1j*grid.k1*uhatF
    S22hatF = 1j*grid.k2*vhatF
    S33hatF = 1j*grid.k3*whatF
    S12hatF = 0.5*(1j*grid.k2*uhatF + 1j*grid.k1*vhatF)
    S13hatF = 0.5*(1j*grid.k3*uhatF + 1j*grid.k1*whatF)
    S23hatF = 0.5*(1j*grid.k3*vhatF + 1j*grid.k2*whatF)
    S11realF = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    S22realF = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    S33realF = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    S12realF = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    S13realF = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    S23realF = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    S11realF[:,:,:] = myFFT.ifftT_obj(pad(S11hatF,1)*scale)
    S22realF[:,:,:] = myFFT.ifftT_obj(pad(S22hatF,1)*scale)
    S33realF[:,:,:] = myFFT.ifftT_obj(pad(S33hatF,1)*scale)
    S12realF[:,:,:] = myFFT.ifftT_obj(pad(S12hatF,1)*scale)
    S13realF[:,:,:] = myFFT.ifftT_obj(pad(S13hatF,1)*scale)
    S23realF[:,:,:] = myFFT.ifftT_obj(pad(S23hatF,1)*scale)
    S_magrealF = np.sqrt( 2.*(S11realF*S11realF + S22realF*S22realF + S33realF*S33realF + \
              2.*S12realF*S12realF + 2.*S13realF*S13realF + 2.*S23realF*S23realF ) )


    ## Now compute terms needed for M. Do pseudo-spectral for |S|Sij
    # First do for S at the test filter
    SFS11F = S_magrealF*S11realF[:,:,:]
    SFS22F = S_magrealF*S22realF[:,:,:]
    SFS33F = S_magrealF*S33realF[:,:,:]
    SFS12F = S_magrealF*S12realF[:,:,:]
    SFS13F = S_magrealF*S13realF[:,:,:]
    SFS23F = S_magrealF*S23realF[:,:,:]
    SFS11Fhat = unpad( myFFT.fft_obj( SFS11F ),1)  
    SFS22Fhat = unpad( myFFT.fft_obj( SFS22F ),1)  
    SFS33Fhat = unpad( myFFT.fft_obj( SFS33F ),1)  
    SFS12Fhat = unpad( myFFT.fft_obj( SFS12F ),1)  
    SFS13Fhat = unpad( myFFT.fft_obj( SFS13F ),1)  
    SFS23Fhat = unpad( myFFT.fft_obj( SFS23F ),1)  
    # Now do for resolved S. Apply test filter after transforming back to freq space
    SS11  = S_magreal*S11real[:,:,:]
    SS22  = S_magreal*S22real[:,:,:]
    SS33  = S_magreal*S33real[:,:,:]
    SS12  = S_magreal*S12real[:,:,:]
    SS13  = S_magreal*S13real[:,:,:]
    SS23  = S_magreal*S23real[:,:,:]
    SS11hatF = utilities.myFilter( unpad( myFFT.fft_obj( SS11 ),1) ,kL)
    SS22hatF = utilities.myFilter( unpad( myFFT.fft_obj( SS22 ),1) ,kL)
    SS33hatF = utilities.myFilter( unpad( myFFT.fft_obj( SS33 ),1) ,kL)
    SS12hatF = utilities.myFilter( unpad( myFFT.fft_obj( SS12 ),1) ,kL)
    SS13hatF = utilities.myFilter( unpad( myFFT.fft_obj( SS13 ),1) ,kL)
    SS23hatF = utilities.myFilter( unpad( myFFT.fft_obj( SS23 ),1) ,kL) 
    
    ## Now create Mhat tensor
    Mhat = np.zeros((grid.N1,grid.N2,(grid.N3/2+1),6),dtype='complex')
    Mhat[:,:,:,0] = -2.*grid.Delta**2*(DSmag_alpha**2*SFS11Fhat - SS11hatF)
    Mhat[:,:,:,1] = -2.*grid.Delta**2*(DSmag_alpha**2*SFS22Fhat - SS22hatF)
    Mhat[:,:,:,2] = -2.*grid.Delta**2*(DSmag_alpha**2*SFS33Fhat - SS33hatF)
    Mhat[:,:,:,3] = -2.*grid.Delta**2*(DSmag_alpha**2*SFS12Fhat - SS12hatF)
    Mhat[:,:,:,4] = -2.*grid.Delta**2*(DSmag_alpha**2*SFS13Fhat - SS13hatF)
    Mhat[:,:,:,5] = -2.*grid.Delta**2*(DSmag_alpha**2*SFS23Fhat - SS23hatF)

    ## Now find Cs^2 = <Lij Mij>/<Mij Mij> Need to transform back to real space to get this
    Mreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3),6) )
    Lreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3),6) )
    Mreal[:,:,:,0] = myFFT.ifftT_obj(pad(Mhat[:,:,:,0],1))*scale
    Mreal[:,:,:,1] = myFFT.ifftT_obj(pad(Mhat[:,:,:,1],1))*scale
    Mreal[:,:,:,2] = myFFT.ifftT_obj(pad(Mhat[:,:,:,2],1))*scale
    Mreal[:,:,:,3] = myFFT.ifftT_obj(pad(Mhat[:,:,:,3],1))*scale
    Mreal[:,:,:,4] = myFFT.ifftT_obj(pad(Mhat[:,:,:,4],1))*scale
    Mreal[:,:,:,5] = myFFT.ifftT_obj(pad(Mhat[:,:,:,5],1))*scale
    Lreal[:,:,:,0] = myFFT.ifftT_obj(pad(Lhat[:,:,:,0],1))*scale
    Lreal[:,:,:,1] = myFFT.ifftT_obj(pad(Lhat[:,:,:,1],1))*scale
    Lreal[:,:,:,2] = myFFT.ifftT_obj(pad(Lhat[:,:,:,2],1))*scale
    Lreal[:,:,:,3] = myFFT.ifftT_obj(pad(Lhat[:,:,:,3],1))*scale
    Lreal[:,:,:,4] = myFFT.ifftT_obj(pad(Lhat[:,:,:,4],1))*scale
    Lreal[:,:,:,5] = myFFT.ifftT_obj(pad(Lhat[:,:,:,5],1))*scale
    num = Mreal[:,:,:,0]*Lreal[:,:,:,0]  + Mreal[:,:,:,1]*Lreal[:,:,:,1] + Mreal[:,:,:,2]*Lreal[:,:,:,2] + \
          Mreal[:,:,:,3]*Lreal[:,:,:,3]  + Mreal[:,:,:,4]*Lreal[:,:,:,4] + Mreal[:,:,:,5]*Lreal[:,:,:,5]
    den = Mreal[:,:,:,0]*Mreal[:,:,:,0]  + Mreal[:,:,:,1]*Mreal[:,:,:,1] + Mreal[:,:,:,2]*Mreal[:,:,:,2] + \
          Mreal[:,:,:,3]*Mreal[:,:,:,3]  + Mreal[:,:,:,4]*Mreal[:,:,:,4] + Mreal[:,:,:,5]*Mreal[:,:,:,5]
    Cs_sqr = np.mean(num)/np.mean(den)
    nutreal = Cs_sqr*grid.Delta*grid.Delta*np.abs(S_magreal)
    
    tau11real = -2.*nutreal*S11real
    tau22real = -2.*nutreal*S22real
    tau33real = -2.*nutreal*S33real
    tau12real = -2.*nutreal*S12real
    tau13real = -2.*nutreal*S13real
    tau23real = -2.*nutreal*S23real
   
    tauhat = np.zeros((grid.N1,grid.N2,grid.N3/2+1,6),dtype='complex') 
    tauhat[:,:,:,0] = unpad( myFFT.fft_obj( -2.*nutreal*S11real ),1)  #11
    tauhat[:,:,:,1] = unpad( myFFT.fft_obj( -2.*nutreal*S22real ),1)  #22
    tauhat[:,:,:,2] = unpad( myFFT.fft_obj( -2.*nutreal*S33real ),1)  #33
    tauhat[:,:,:,3] = unpad( myFFT.fft_obj( -2.*nutreal*S12real ),1)  #12
    tauhat[:,:,:,4] = unpad( myFFT.fft_obj( -2.*nutreal*S13real ),1)  #13
    tauhat[:,:,:,5] = unpad( myFFT.fft_obj( -2.*nutreal*S23real ),1)  #23

    ## contribution of projection to RHS sgs (k_m k_j)/k^2 \tau_{jm}
    tau_projection  = 1j*grid.ksqr_i*( grid.k1*grid.k1*tauhat[:,:,:,0] + grid.k2*grid.k2*tauhat[:,:,:,1] + \
             grid.k3*grid.k3*tauhat[:,:,:,2] + 2.*grid.k1*grid.k2*tauhat[:,:,:,3] + \
             2.*grid.k1*grid.k3*tauhat[:,:,:,4] + 2.*grid.k2*grid.k3*tauhat[:,:,:,5] )

    #Now SGS contributions. w = -\delta_{im}\tau_{jm}  + k_i k_m / k^2 i k_j \tau_{jm}
    main.w0_u[:,:,:,0] = -1j*grid.k1*tauhat[:,:,:,0] - 1j*grid.k2*tauhat[:,:,:,3] - 1j*grid.k3*tauhat[:,:,:,4] + 1j*grid.k1*tau_projection
    main.w0_v[:,:,:,0] = -1j*grid.k1*tauhat[:,:,:,3] - 1j*grid.k2*tauhat[:,:,:,1] - 1j*grid.k3*tauhat[:,:,:,5] + 1j*grid.k2*tau_projection
    main.w0_w[:,:,:,0] = -1j*grid.k1*tauhat[:,:,:,4] - 1j*grid.k2*tauhat[:,:,:,5] - 1j*grid.k3*tauhat[:,:,:,2] + 1j*grid.k3*tau_projection

    main.Q[0::3,0::3,0::3] = -1j*grid.k1*uuhat - 1j*grid.k2*uvhat - 1j*grid.k3*uwhat - \
                           1j*grid.k1*phat - main.nu*grid.ksqr*main.uhat + main.w0_u[:,:,:,0]

    main.Q[1::3,1::3,1::3] = -1j*grid.k1*uvhat - 1j*grid.k2*vvhat - 1j*grid.k3*vwhat - \
                           1j*grid.k2*phat - main.nu*grid.ksqr*main.vhat + main.w0_v[:,:,:,0] 
 

    main.Q[2::3,2::3,2::3] = -1j*grid.k1*uwhat - 1j*grid.k2*vwhat - 1j*grid.k3*wwhat - \
                           1j*grid.k3*phat - main.nu*grid.ksqr*main.what + main.w0_w[:,:,:,0]

    if (main.rotate == 1):
      main.Q[0::3,0::3,0::3] = main.Q[0::3,0::3,0::3] + 2.*(main.vhat*main.Om3 - main.what*main.Om2)
      main.Q[1::3,1::3,1::3] = main.Q[1::3,1::3,1::3] + 2.*(main.what*main.Om1 - main.uhat*main.Om3)
      main.Q[2::3,2::3,2::3] = main.Q[2::3,2::3,2::3] + 2.*(main.uhat*main.Om2 - main.vhat*main.Om1)

