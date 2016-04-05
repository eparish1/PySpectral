import numpy as np
from scipy import weave
from scipy.weave import converters
from padding import pad,unpad,pad_2x,unpad_2x,seperateModes
def computeRHS_NOSGS_WEAVE(main,grid,myFFT,utilities):
    main.Q2U()
    scale = np.sqrt( (3./2.)**3*np.sqrt(grid.N1*grid.N2*grid.N3) )
    ureal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    vreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    wreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )

    main.uhat = unpad(pad(main.uhat,1),1)
    main.vhat = unpad(pad(main.vhat,1),1)
    main.what = unpad(pad(main.what,1),1)

    ureal = myFFT.ifftT_obj(pad(main.uhat,1))*scale
    vreal = myFFT.ifftT_obj(pad(main.vhat,1))*scale
    wreal = myFFT.ifftT_obj(pad(main.what,1))*scale

    uuhat = unpad( myFFT.fft_obj(ureal*ureal),1)
    vvhat = unpad( myFFT.fft_obj(vreal*vreal),1)
    wwhat = unpad( myFFT.fft_obj(wreal*wreal),1)
    uvhat = unpad( myFFT.fft_obj(ureal*vreal),1)
    uwhat = unpad( myFFT.fft_obj(ureal*wreal),1)
    vwhat = unpad( myFFT.fft_obj(vreal*wreal),1)

    ### Setup pointers for weave
    k1 = grid.k1
    k2 = grid.k2
    k3 = grid.k3
    ksqr_i = grid.ksqr_i
    ksqr = grid.ksqr
    Q = main.Q
    uhat = main.uhat
    vhat = main.vhat
    what = main.what
    phat = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    nu = main.nu
    N1 = grid.N1
    N2 = grid.N2
    N3 = grid.N3
    Zj = 1j
    nthreads = myFFT.nthreads
    code="""
    int i,j,k;
    omp_set_num_threads(nthreads);
    #pragma omp parallel for private(i,j,k) shared(uhat,vhat,what,phat,k1,k2,k3,ksqr,ksqr_i,uuhat,vvhat,wwhat,uvhat,uwhat,vwhat,Q,nu)
    for(i=0;i<N1;i++){
      for (j=0;j<N2;j++){
        for (k=0;k<N3/2+1;k++){
          phat(i,j,k)  = ksqr_i(i,j,k)*( -k1(i,j,k)*k1(i,j,k)*uuhat(i,j,k) - k2(i,j,k)*k2(i,j,k)*vvhat(i,j,k) - 
                         k3(i,j,k)*k3(i,j,k)*wwhat(i,j,k) - 2.*k1(i,j,k)*k2(i,j,k)*uvhat(i,j,k) - 
                         2.*k1(i,j,k)*k3(i,j,k)*uwhat(i,j,k) - 2.*k2(i,j,k)*k3(i,j,k)*vwhat(i,j,k) );

          Q(3*i,3*j,3*k) = -Zj*k1(i,j,k)*uuhat(i,j,k) - Zj*k2(i,j,k)*uvhat(i,j,k) - Zj*k3(i,j,k)*uwhat(i,j,k) - 
                                         Zj*k1(i,j,k)*phat(i,j,k) - nu*ksqr(i,j,k)*uhat(i,j,k);

          Q(3*i+1,3*j+1,3*k+1) = -Zj*k1(i,j,k)*uvhat(i,j,k) - Zj*k2(i,j,k)*vvhat(i,j,k) - Zj*k3(i,j,k)*vwhat(i,j,k) - 
                                         Zj*k2(i,j,k)*phat(i,j,k) - nu*ksqr(i,j,k)*vhat(i,j,k);

          Q(3*i+2,3*j+2,3*k+2) = -Zj*k1(i,j,k)*uwhat(i,j,k) - Zj*k2(i,j,k)*vwhat(i,j,k) - Zj*k3(i,j,k)*wwhat(i,j,k) - 
                                         Zj*k3(i,j,k)*phat(i,j,k) - nu*ksqr(i,j,k)*what(i,j,k);
          }
      }
    }
    """
    weave.inline(code,['N1','N2','N3','uhat','vhat','what','phat','k1','k2','k3','ksqr','ksqr_i','uuhat','vvhat','wwhat','uvhat','uwhat','vwhat','Q','nu','Zj','nthreads'],\
                 type_converters=converters.blitz,compiler='gcc',extra_compile_args=\
                 ['-march=native -O3 -fopenmp'],
                 support_code = \
                 r"""
                 #include <iostream>
                 #include <complex>
                 #include <cmath> 
                 #include <omp.h>
                 """,
                 libraries=['gomp']  )
  
