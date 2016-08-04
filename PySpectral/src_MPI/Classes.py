import numpy as np
import sys
from RHSfunctions import *

class variables:
  def __init__(self,turb_model,rotate,Om1,Om2,Om3,grid,u,v,w,uhat,vhat,what,t,dt,nu,myFFT,mpi_rank,initDomain):
    self.turb_model = turb_model
    self.rotate = rotate
    self.Om1 = Om1
    self.Om2 = Om2
    self.Om3 = Om3
    self.t = t
    self.kc = np.amax(grid.k1)
    self.dt = dt
    self.nu = nu
    self.NL = np.empty((6,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    self.u = u
    self.v = v
    self.w = w
    self.uhat = uhat
    self.vhat = vhat
    self.what = what

    #self.uhat = np.zeros((grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    #self.vhat = np.zeros((grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    #self.what = np.zeros((grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    if (initDomain == 'Physical'):
      if (mpi_rank == 0):
        print('Initializing off physical fields u,v,w')
      myFFT.myfft3D(self.u,self.uhat)
      myFFT.myfft3D(self.v,self.vhat)
      myFFT.myfft3D(self.w,self.what)
    if (initDomain == 'Fourier'):
      if (mpi_rank == 0):
        print('Initializing off Fourier fields uhat,vhat,what')
      myFFT.myifft3D(self.uhat,self.u)
      myFFT.myifft3D(self.vhat,self.v)
      myFFT.myifft3D(self.what,self.w)
      ## retake the fft/ifft to make sure that the field is symmetric
      myFFT.myfft3D(self.u,self.uhat)
      myFFT.myfft3D(self.v,self.vhat)
      myFFT.myfft3D(self.w,self.what)
      myFFT.myifft3D(self.uhat,self.u)
      myFFT.myifft3D(self.vhat,self.v)
      myFFT.myifft3D(self.what,self.w)

    self.work_spectral = np.zeros((grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    #self.cfl = cfl
    ##============ DNS MODE ========================
    if (turb_model == 0):
      if (mpi_rank == 0):
        sys.stdout.write('Not using any SGS \n')
        sys.stdout.flush()
      self.Q = np.zeros( (3,grid.Npx,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.Q0 = np.zeros( (3,grid.Npx,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.nvars = 3
      self.Q[0] = self.uhat[:,:,:]
      self.Q[1] = self.vhat[:,:,:]
      self.Q[2] = self.what[:,:,:]
      def U2Q():
        self.Q[0] = self.uhat[:,:,:]
        self.Q[1] = self.vhat[:,:,:]
        self.Q[2] = self.what[:,:,:]
      def Q2U():
        self.uhat[:,:,:] = self.Q[0]
        self.vhat[:,:,:] = self.Q[1]
        self.what[:,:,:] = self.Q[2]
      self.computeRHS = computeRHS_NOSGS
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================


class gridclass:
  def __init__(self,N1,N2,N3,x,y,z,kc,num_processes,L1,L2,L3,mpi_rank,comm,turb_model):
    self.Npx = int(float(N1 / num_processes))
    self.Npy = int(float(N2 / num_processes))
    self.num_processes = num_processes
    self.mpi_rank = mpi_rank
    self.comm = comm
    self.N1 = N1
    self.N2 = N2
    self.N3 = N3
    self.x = x
    self.y = y
    self.z = z
    self.xG = allGather_physical(self.x,comm,mpi_rank,self.N1,self.N2,self.N3,num_processes,self.Npy)
    self.yG = allGather_physical(self.y,comm,mpi_rank,self.N1,self.N2,self.N3,num_processes,self.Npy)
    self.zG = allGather_physical(self.z,comm,mpi_rank,self.N1,self.N2,self.N3,num_processes,self.Npy)
    self.dx = x[1,0,0] - x[0,0,0]
    self.dy = y[0,1,0] - y[0,0,0]
    self.dz = z[0,0,1] - z[0,0,0]
    self.k1 = np.fft.fftshift( np.linspace(-N1/2,N1/2-1,N1) )[mpi_rank*self.Npx:(mpi_rank+1)*self.Npx] *2.*np.pi/L1
    self.k2 = np.fft.fftshift( np.linspace(-N2/2,N2/2-1,N2) ) *2.*np.pi/L2
    self.k3 = np.linspace( 0,N3/2,N3/2+1 ) *2.*np.pi/L3
    self.ksqr = self.k1[:,None,None]*self.k1[:,None,None] + self.k2[None,:,None]*self.k2[None,:,None] + self.k3[None,None,:]*self.k3[None,None,:] + 1.e-50
    self.ksqr_i = 1./self.ksqr
    self.kc = kc
    self.Delta = np.pi/self.kc
    self.dealias_x = np.ones(self.Npx)
    self.dealias_y = np.ones(self.N2)
    self.dealias_z = np.ones(self.N3/2+1)
    for i in range(0,self.Npx):
      if (abs(self.k1[i]) >= (self.N1/2)*2./3.*2.*np.pi/L1):
        self.dealias_x[i] = 0.
    for i in range(0,self.N2):
      if (abs(self.k2[i]) >= (self.N2/2)*2./3.*2.*np.pi/L2):
        self.dealias_y[i] = 0.
    if (self.N3 == 2):
      pass
    else:
      self.dealias_z[int( (self.N3/2)*2./3. )::] = 0.

    self.filter_x = np.ones(self.Npx)
    self.filter_y = np.ones(self.N2)
    self.filter_z = np.ones(self.N3/2+1)
    for i in range(0,self.Npx):
      if (abs(self.k1[i]) >= (self.kc)*2.*np.pi/L1):
        self.filter_x[i] = 0.
    for i in range(0,self.N2):
      if (abs(self.k2[i]) >= (self.kc)*2.*np.pi/L2):
        self.filter_y[i] = 0.
    if (self.N3 == 2):
      pass
    else:
      self.filter_z[self.kc::] = 0.
    def filter(uhat):
      uhat_filter = self.filter_x[:,None,None]*self.filter_y[None,:,None]*self.filter_z[None,None,:]*uhat[:]
      return uhat_filter
    self.filter = filter

class FFTclass:
  def __init__(self,N1,N2,N3,nthreads,fft_type,Npx,Npy,num_processes,comm,mpi_rank):
    self.nthreads = nthreads
    self.Uc_hat = np.zeros((Npx,N2,N3/2+1),dtype='complex')
    self.Uc_hatT = np.zeros((N1,Npy,N3/2+1) ,dtype='complex')
    self.U_mpi = np.zeros((num_processes,Npx,Npy,N3/2+1),dtype='complex')

    def myifft3D(uhat,u):
      self.Uc_hat[:,:,:] = np.fft.ifft(uhat,axis=1) 
      self.U_mpi[:] = np.rollaxis(self.Uc_hat.reshape(Npx, num_processes, Npy, N3/2+1) ,1)
      comm.Alltoall(self.U_mpi,self.Uc_hatT)
      u[:] = np.fft.irfft2(self.Uc_hatT,axes=(0,2) ) 
      return u

    def myfft3D(u,uhat):
      self.Uc_hatT[:,:,:] = np.fft.rfft2(u,axes=(0,2) ) 
      comm.Alltoall(self.Uc_hatT, self.U_mpi )
      self.Uc_hat[:,:,:] = np.rollaxis(self.U_mpi,1).reshape(self.Uc_hat.shape)
      uhat[:] = np.fft.fft(self.Uc_hat,axis=1) 
      return uhat
    self.myfft3D = myfft3D
    self.myifft3D = myifft3D

    def dealias(uhat,grid):
      uhat[:,:,:] = grid.dealias_x[:,None,None]*grid.dealias_y[None,:,None]*grid.dealias_z[None,None,:]*uhat
      return uhat 
    self.dealias = dealias

class utilitiesClass():
  def computeEnergy(self,main,grid):
      ## compute energy on each node
      uE = np.sum(main.uhat[:,:,1:grid.N3/2]*np.conj(main.uhat[:,:,1:grid.N3/2]*2) ) + \
           np.sum(main.uhat[:,:,0]*np.conj(main.uhat[:,:,0]))
      vE = np.sum(main.vhat[:,:,1:grid.N3/2]*np.conj(main.vhat[:,:,1:grid.N3/2]*2) ) + \
           np.sum(main.vhat[:,:,0]*np.conj(main.vhat[:,:,0]))
      wE = np.sum(main.what[:,:,1:grid.N3/2]*np.conj(main.what[:,:,1:grid.N3/2]*2) ) + \
           np.sum(main.what[:,:,0]*np.conj(main.what[:,:,0]))
      energy_local = np.real(0.5*(uE + vE + wE) )
      data = grid.comm.gather(energy_local,root = 0)
      if (grid.mpi_rank == 0):
        energy = 0.
        for j in range(0,grid.num_processes):
          energy = energy + data[j]
        return  energy/(grid.N1*grid.N2*grid.N3)**2
      else:
        return 0


  def computeEnergy_resolved(self,main,grid):
      uFilt = grid.filter(main.uhat)
      vFilt = grid.filter(main.vhat)
      wFilt = grid.filter(main.what)
      uE = np.sum(uFilt[:,:,1:grid.N3/2]*np.conj(uFilt[:,:,1:grid.N3/2]*2) ) + \
           np.sum(uFilt[:,:,0]*np.conj(uFilt[:,:,0]))
      vE = np.sum(vFilt[:,:,1:grid.N3/2]*np.conj(vFilt[:,:,1:grid.N3/2]*2) ) + \
           np.sum(vFilt[:,:,0]*np.conj(vFilt[:,:,0]))
      wE = np.sum(wFilt[:,:,1:grid.N3/2]*np.conj(wFilt[:,:,1:grid.N3/2]*2) ) + \
           np.sum(wFilt[:,:,0]*np.conj(wFilt[:,:,0]))
      energy_local = np.real(0.5*(uE + vE + wE) )
      data = grid.comm.gather(energy_local,root = 0)
      if (grid.mpi_rank == 0):
        energy = 0.
        for j in range(0,grid.num_processes):
          energy = energy + data[j]
        return  energy / (grid.N1*grid.N2*grid.N3)**2
      else:
        return 0

  def computeEnstrophy(self,main,grid):
      omega1 = 1j*grid.k2[None,:,None]*main.what - 1j*grid.k3[None,None,:]*main.vhat
      omega2 = 1j*grid.k3[None,None,:]*main.uhat - 1j*grid.k1[:,None,None]*main.what
      omega3 = 1j*grid.k1[:,None,None]*main.vhat - 1j*grid.k2[None,:,None]*main.uhat
      om1E = np.sum(omega1[:,:,1:grid.N3/2]*np.conj(omega1[:,:,1:grid.N3/2]*2) ) + \
           np.sum(omega1[:,:,0]*np.conj(omega1[:,:,0]))
      om2E = np.sum(omega2[:,:,1:grid.N3/2]*np.conj(omega2[:,:,1:grid.N3/2]*2) ) + \
           np.sum(omega2[:,:,0]*np.conj(omega2[:,:,0]))
      om3E = np.sum(omega3[:,:,1:grid.N3/2]*np.conj(omega3[:,:,1:grid.N3/2]*2) ) + \
           np.sum(omega3[:,:,0]*np.conj(omega3[:,:,0]))
      enstrophy_local = np.real(0.5*(om1E + om2E + om3E) )
      data = grid.comm.gather(enstrophy_local,root = 0)
      if (grid.mpi_rank == 0):
        enstrophy = 0.
        for j in range(0,grid.num_processes):
          enstrophy = enstrophy + data[j]
        return enstrophy / (grid.N1*grid.N2*grid.N3)**2
      else:
        return 0

  def computeSpectrum(self,main,grid):

      ##====== Compute Spectra locally ===============
      kmag = np.sqrt(grid.k1[:,None,None]**2 + grid.k2[None,:,None]**2 + grid.k3[None,None,1::]**2)
      E =  (2.*main.uhat[:,:,1::]*np.conj(main.uhat[:,:,1::]) ).flatten() + \
           (2.*main.vhat[:,:,1::]*np.conj(main.vhat[:,:,1::]) ).flatten() + \
           (2.*main.what[:,:,1::]*np.conj(main.what[:,:,1::]) ).flatten()
 
      ksort = np.int16(np.round(np.sort(kmag.flatten())))
      kargsort = np.argsort(kmag.flatten())
      E = np.bincount(ksort,np.real(E)[kargsort])
      
      kmag2 = np.sqrt(grid.k1[:,None,None]**2 + grid.k2[None,:,None]**2 + grid.k3[None,None,0]**2)
      E2 = (main.uhat[:,:,0]*np.conj(main.uhat[:,:,0])).flatten() + \
           (main.vhat[:,:,0]*np.conj(main.vhat[:,:,0])).flatten() + \
           (main.what[:,:,0]*np.conj(main.what[:,:,0])).flatten()

      ksort2 = np.int16(np.round(np.sort(kmag2.flatten())))
      kargsort2 = np.argsort(kmag2.flatten())
      E2 = np.bincount(ksort2,np.real(E2)[kargsort2])

      k_all = grid.comm.gather(np.amax(ksort)+1,root = 0) 
      E_all = grid.comm.gather(E,root = 0)
      k_all2 = grid.comm.gather(np.amax(ksort2)+1,root = 0)
      E_all2 = grid.comm.gather(E2,root = 0)

      ##====== send to proc zero to add them all up
      if (grid.mpi_rank == 0):
        kmax = np.round(np.sqrt((grid.N1/2)**2 + (grid.N2/2)**2 + (grid.N3/2)**2) )
        kspec = np.linspace(0 , kmax, kmax + 1)
        spectrum = np.zeros(kmax+1)
        for j in range(0,grid.num_processes):
          spectrum[0:k_all[j]] = spectrum[0:k_all[j]] + E_all[j][:]
          spectrum[0:k_all2[j]] = spectrum[0:k_all2[j]] + E_all2[j][:]

        return kspec,spectrum / (grid.N1*grid.N2*grid.N3)**2
      else:
        return 0,0


  def computeSpectrum_resolved(self,main,grid):
      ##====== Compute Spectra locally ===============
      kmag = np.sqrt(grid.k1[:,None,None]**2 + grid.k2[None,:,None]**2 + grid.k3[None,None,1::]**2)
      uFilt = grid.filter(main.uhat)
      vFilt = grid.filter(main.vhat)
      wFilt = grid.filter(main.what)

      E =  (2.*uFilt[:,:,1::]*np.conj(uFilt[:,:,1::]) ).flatten() + \
           (2.*vFilt[:,:,1::]*np.conj(vFilt[:,:,1::]) ).flatten() + \
           (2.*wFilt[:,:,1::]*np.conj(wFilt[:,:,1::]) ).flatten()
 
      ksort = np.int16(np.round(np.sort(kmag.flatten())))
      kargsort = np.argsort(kmag.flatten())
      E = np.bincount(ksort,np.real(E)[kargsort])
      
      kmag2 = np.sqrt(grid.k1[:,None,None]**2 + grid.k2[None,:,None]**2 + grid.k3[None,None,0]**2)
      E2 = (uFilt[:,:,0]*np.conj(uFilt[:,:,0])).flatten() + \
           (vFilt[:,:,0]*np.conj(vFilt[:,:,0])).flatten() + \
           (wFilt[:,:,0]*np.conj(wFilt[:,:,0])).flatten()

      ksort2 = np.int16(np.round(np.sort(kmag2.flatten())))
      kargsort2 = np.argsort(kmag2.flatten())
      E2 = np.bincount(ksort2,np.real(E2)[kargsort2])

      k_all = grid.comm.gather(np.amax(ksort)+1,root = 0) 
      E_all = grid.comm.gather(E,root = 0)
      k_all2 = grid.comm.gather(np.amax(ksort2)+1,root = 0)
      E_all2 = grid.comm.gather(E2,root = 0)

      ##====== send to proc zero to add them all up
      if (grid.mpi_rank == 0):
        kmax = np.round(np.sqrt((grid.N1/2)**2 + (grid.N2/2)**2 + (grid.N3/2)**2) )
        kspec = np.linspace(0 , kmax, kmax + 1)
        spectrum = np.zeros(kmax+1)
        for j in range(0,grid.num_processes):
          spectrum[0:k_all[j]] = spectrum[0:k_all[j]] + E_all[j][:]
          spectrum[0:k_all2[j]] = spectrum[0:k_all2[j]] + E_all2[j][:]

        return kspec,spectrum / (grid.N1*grid.N2*grid.N3)**2
      else:
        return 0,0


  def computeAllStats(self,main,grid,myFFT):
      enstrophy = self.computeEnstrophy(main,grid)
      energy = self.computeEnergy(main,grid)
      kdata,spectrum = self.computeSpectrum(main,grid)
      dissipation = 2.*enstrophy*main.nu
      lambda_k = (main.nu**3/(dissipation+1.e-30))**0.25
      tau_k = (main.nu/(dissipation+1.e-30))**0.5
      Re_lambda = energy*np.sqrt(20./(3.*main.nu*dissipation+1.e-30))
      return enstrophy,energy,dissipation,lambda_k,tau_k,Re_lambda,kdata,spectrum


  def computeTransfer(self,main,grid,myFFT):

    main.uhat = myFFT.dealias(main.uhat,grid)
    main.vhat = myFFT.dealias(main.vhat,grid)
    main.what = myFFT.dealias(main.what,grid)

    myFFT.myifft3D(main.uhat,main.u)
    myFFT.myifft3D(main.vhat,main.v)
    myFFT.myifft3D(main.what,main.w)

    myFFT.myfft3D(main.u*main.u,main.NL[0])
    myFFT.myfft3D(main.v*main.v,main.NL[1])
    myFFT.myfft3D(main.w*main.w,main.NL[2])
    myFFT.myfft3D(main.u*main.v,main.NL[3])
    myFFT.myfft3D(main.u*main.w,main.NL[4])
    myFFT.myfft3D(main.v*main.w,main.NL[5])


    phat  = grid.ksqr_i*( -grid.k1[:,None,None]*grid.k1[:,None,None]*main.NL[0] - grid.k2[None,:,None]*grid.k2[None,:,None]*main.NL[1] - \
             grid.k3[None,None,:]*grid.k3[None,None,:]*main.NL[2] - 2.*grid.k1[:,None,None]*grid.k2[None,:,None]*main.NL[3] - \
             2.*grid.k1[:,None,None]*grid.k3[None,None,:]*main.NL[4] - 2.*grid.k2[None,:,None]*grid.k3[None,None,:]*main.NL[5] )

    RHSu = np.conj(main.uhat)*(myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[0] - 1j*grid.k2[None,:,None]*main.NL[3] - 1j*grid.k3[None,None,:]*main.NL[4] - \
                                         1j*grid.k1[:,None,None]*phat , grid) )

    RHSv = np.conj(main.vhat)*(myFFT.dealias(-1j*grid.k1[:,None,None]*main.NL[3] - 1j*grid.k2[None,:,None]*main.NL[1] - 1j*grid.k3[None,None,:]*main.NL[5] - \
                                         1j*grid.k2[None,:,None]*phat , grid) )

    RHSw = np.conj(main.what)*(myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[4] - 1j*grid.k2[None,:,None]*main.NL[5] - 1j*grid.k3[None,None,:]*main.NL[2] - \
                                         1j*grid.k3[None,None,:]*phat , grid))

    kmag = np.sqrt(grid.k1[:,None,None]**2 + grid.k2[None,:,None]**2 + grid.k3[None,None,1::]**2)
    E =  (2.*RHSu[:,:,1::]).flatten() + \
         (2.*RHSv[:,:,1::]).flatten() + \
         (2.*RHSw[:,:,1::]).flatten()
    ksort = np.int16(np.round(np.sort(kmag.flatten())))
    kargsort = np.argsort(kmag.flatten())
    E = np.bincount(ksort,weights=np.real(E)[kargsort])

    kmag2 = np.sqrt(grid.k1[:,None,None]**2 + grid.k2[None,:,None]**2 + grid.k3[None,None,0]**2)
    E2 = (RHSu[:,:,0]).flatten() + \
         (RHSv[:,:,0]).flatten() + \
         (RHSw[:,:,0]).flatten()

    ksort2 = np.int16(np.round(np.sort(kmag2.flatten())))
    kargsort2 = np.argsort(kmag2.flatten())
    E2 = np.bincount(ksort2,weights=np.real(E2)[kargsort2])

    k_all = grid.comm.gather(np.amax(ksort)+1,root = 0)
    E_all = grid.comm.gather(E,root = 0)
    k_all2 = grid.comm.gather(np.amax(ksort2)+1,root = 0)
    E_all2 = grid.comm.gather(E2,root = 0)

    ##====== send to proc zero to add them all up
    if (grid.mpi_rank == 0):
      kmax = np.round(np.sqrt((grid.N1/2)**2 + (grid.N2/2)**2 + (grid.N3/2)**2) )
      kspec = np.linspace(0 , kmax, kmax + 1)
      transfer = np.zeros(kmax+1)
      for j in range(0,grid.num_processes):
        transfer[0:k_all[j]] = transfer[0:k_all[j]] + E_all[j][:]
        transfer[0:k_all2[j]] = transfer[0:k_all2[j]] + E_all2[j][:]

      return kspec,transfer / (grid.N1*grid.N2*grid.N3)**2
    else:
      return 0,0


  def computeTransfer_resolved(self,main,grid,myFFT):

    main.uhat = myFFT.dealias(main.uhat,grid)
    main.vhat = myFFT.dealias(main.vhat,grid)
    main.what = myFFT.dealias(main.what,grid)

    myFFT.myifft3D(grid.filter(main.uhat),main.u)
    myFFT.myifft3D(grid.filter(main.vhat),main.v)
    myFFT.myifft3D(grid.filter(main.what),main.w)

    myFFT.myfft3D(main.u*main.u,main.NL[0])
    myFFT.myfft3D(main.v*main.v,main.NL[1])
    myFFT.myfft3D(main.w*main.w,main.NL[2])
    myFFT.myfft3D(main.u*main.v,main.NL[3])
    myFFT.myfft3D(main.u*main.w,main.NL[4])
    myFFT.myfft3D(main.v*main.w,main.NL[5])


    phat  = grid.ksqr_i*( -grid.k1[:,None,None]*grid.k1[:,None,None]*main.NL[0] - grid.k2[None,:,None]*grid.k2[None,:,None]*main.NL[1] - \
             grid.k3[None,None,:]*grid.k3[None,None,:]*main.NL[2] - 2.*grid.k1[:,None,None]*grid.k2[None,:,None]*main.NL[3] - \
             2.*grid.k1[:,None,None]*grid.k3[None,None,:]*main.NL[4] - 2.*grid.k2[None,:,None]*grid.k3[None,None,:]*main.NL[5] )

    RHSu = np.conj(grid.filter(main.uhat))*(myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[0] - 1j*grid.k2[None,:,None]*main.NL[3] - 1j*grid.k3[None,None,:]*main.NL[4] - \
                                         1j*grid.k1[:,None,None]*phat , grid) )

    RHSv = np.conj(grid.filter(main.vhat))*(myFFT.dealias(-1j*grid.k1[:,None,None]*main.NL[3] - 1j*grid.k2[None,:,None]*main.NL[1] - 1j*grid.k3[None,None,:]*main.NL[5] - \
                                         1j*grid.k2[None,:,None]*phat , grid) )

    RHSw = np.conj(grid.filter(main.what))*(myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[4] - 1j*grid.k2[None,:,None]*main.NL[5] - 1j*grid.k3[None,None,:]*main.NL[2] - \
                                         1j*grid.k3[None,None,:]*phat , grid))

    kmag = np.sqrt(grid.k1[:,None,None]**2 + grid.k2[None,:,None]**2 + grid.k3[None,None,1::]**2)
    E =  (2.*RHSu[:,:,1::]).flatten() + \
         (2.*RHSv[:,:,1::]).flatten() + \
         (2.*RHSw[:,:,1::]).flatten()
    ksort = np.int16(np.round(np.sort(kmag.flatten())))
    kargsort = np.argsort(kmag.flatten())
    E = np.bincount(ksort,weights=np.real(E)[kargsort])

    kmag2 = np.sqrt(grid.k1[:,None,None]**2 + grid.k2[None,:,None]**2 + grid.k3[None,None,0]**2)
    E2 = (RHSu[:,:,0]).flatten() + \
         (RHSv[:,:,0]).flatten() + \
         (RHSw[:,:,0]).flatten()

    ksort2 = np.int16(np.round(np.sort(kmag2.flatten())))
    kargsort2 = np.argsort(kmag2.flatten())
    E2 = np.bincount(ksort2,weights=np.real(E2)[kargsort2])

    k_all = grid.comm.gather(np.amax(ksort)+1,root = 0)
    E_all = grid.comm.gather(E,root = 0)
    k_all2 = grid.comm.gather(np.amax(ksort2)+1,root = 0)
    E_all2 = grid.comm.gather(E2,root = 0)

    ##====== send to proc zero to add them all up
    if (grid.mpi_rank == 0):
      kmax = np.round(np.sqrt((grid.N1/2)**2 + (grid.N2/2)**2 + (grid.N3/2)**2) )
      kspec = np.linspace(0 , kmax, kmax + 1)
      transfer = np.zeros(kmax+1)
      for j in range(0,grid.num_processes):
        transfer[0:k_all[j]] = transfer[0:k_all[j]] + E_all[j][:]
        transfer[0:k_all2[j]] = transfer[0:k_all2[j]] + E_all2[j][:]

      return kspec,transfer / (grid.N1*grid.N2*grid.N3)**2
    else:
      return 0,0

