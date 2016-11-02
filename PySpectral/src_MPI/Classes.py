import numpy as np
import sys
from RHSfunctions import *
from time_schemes import *
class variables:
  def __init__(self,turb_model,rotate,Om1,Om2,Om3,grid,u,v,w,uhat,vhat,what,t,dt,nu,myFFT,mpi_rank,initDomain,time_scheme):
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
    self.time_scheme = time_scheme
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


    if (time_scheme == 'RK4'):
      self.advanceQ = advanceQ_RK4
    if (time_scheme == 'Semi-Implicit'):
      self.advanceQ = advanceQ_SI
      self.H = np.empty((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
      self.H_old = np.empty((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
      self.viscous_term = np.empty((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')

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
    ##============ ORTHOGONAL DYNAMICS MODE ========================
    if (turb_model == 'Orthogonal Dynamics'):
      if (mpi_rank == 0):
        sys.stdout.write('Solving the orthogonal dynamics equation \n')
        sys.stdout.flush()
      self.Q = np.zeros( (3,grid.Npx,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.Q0 = np.zeros( (3,grid.Npx,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.F = np.zeros( (3,grid.Npx,grid.N2,(grid.N3/2+1)),dtype='complex')
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
      self.computeRHS = computeRHS_Ortho
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
    self.L1 = L1
    self.L2 = L2
    self.L3 = L3
    #self.xG = allGather_physical(self.x,comm,mpi_rank,self.N1,self.N2,self.N3,num_processes,self.Npy)
    #self.yG = allGather_physical(self.y,comm,mpi_rank,self.N1,self.N2,self.N3,num_processes,self.Npy)
    #self.zG = allGather_physical(self.z,comm,mpi_rank,self.N1,self.N2,self.N3,num_processes,self.Npy)
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

      kmag = np.sqrt((grid.k1[:,None,None]*grid.L1/(2.*np.pi))**2 + (grid.k2[None,:,None]*grid.L2/(2.*np.pi))**2 + (grid.k3[None,None,1::]*grid.L3/(2.*np.pi))**2)
      E =  (2.*main.uhat[:,:,1::]*np.conj(main.uhat[:,:,1::]) ).flatten() + \
           (2.*main.vhat[:,:,1::]*np.conj(main.vhat[:,:,1::]) ).flatten() + \
           (2.*main.what[:,:,1::]*np.conj(main.what[:,:,1::]) ).flatten()
 
      ksort = np.int16(np.round(np.sort(kmag.flatten())))
      kargsort = np.argsort(kmag.flatten())
      E = np.bincount(ksort,np.real(E)[kargsort])
      
      kmag2 = np.sqrt((grid.k1[:,None,None]*grid.L1/(2.*np.pi))**2 + (grid.k2[None,:,None]*grid.L2/(2.*np.pi))**2 + (grid.k3[None,None,0]*grid.L3/(2.*np.pi))**2)
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
        spectrum = np.zeros(int(kmax+1))
        for j in range(0,grid.num_processes):
          spectrum[0:k_all[j]] = spectrum[0:k_all[j]] + E_all[j][:]
          spectrum[0:k_all2[j]] = spectrum[0:k_all2[j]] + E_all2[j][:]

        return kspec,spectrum / (grid.N1*grid.N2*grid.N3)**2
      else:
        return 0,0


  def computeSpectrum_resolved(self,main,grid):
      ##====== Compute Spectra locally ===============
      kmag = np.sqrt((grid.k1[:,None,None]*grid.L1/(2.*np.pi))**2 + (grid.k2[None,:,None]*grid.L2/(2.*np.pi))**2 + (grid.k3[None,None,1::]*grid.L3/(2.*np.pi))**2)
      uFilt = grid.filter(main.uhat)
      vFilt = grid.filter(main.vhat)
      wFilt = grid.filter(main.what)

      E =  (2.*uFilt[:,:,1::]*np.conj(uFilt[:,:,1::]) ).flatten() + \
           (2.*vFilt[:,:,1::]*np.conj(vFilt[:,:,1::]) ).flatten() + \
           (2.*wFilt[:,:,1::]*np.conj(wFilt[:,:,1::]) ).flatten()
 
      ksort = np.int16(np.round(np.sort(kmag.flatten())))
      kargsort = np.argsort(kmag.flatten())
      E = np.bincount(ksort,np.real(E)[kargsort])
      
      kmag2 = np.sqrt((grid.k1[:,None,None]*grid.L1/(2.*np.pi))**2 + (grid.k2[None,:,None]*grid.L2/(2.*np.pi))**2 + (grid.k3[None,None,0]*grid.L3/(2.*np.pi))**2)
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
        spectrum = np.zeros(int(kmax+1))
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
      transfer = np.zeros(int(kmax+1))
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

    kmag = np.sqrt((grid.k1[:,None,None]*grid.L1/(2.*np.pi))**2 + (grid.k2[None,:,None]*grid.L2/(2.*np.pi))**2 + (grid.k3[None,None,1::]*grid.L3/(2.*np.pi))**2)

    E =  (2.*RHSu[:,:,1::]).flatten() + \
         (2.*RHSv[:,:,1::]).flatten() + \
         (2.*RHSw[:,:,1::]).flatten()
    ksort = np.int16(np.round(np.sort(kmag.flatten())))
    kargsort = np.argsort(kmag.flatten())
    E = np.bincount(ksort,weights=np.real(E)[kargsort])

    kmag2 = np.sqrt((grid.k1[:,None,None]*grid.L1/(2.*np.pi))**2 + (grid.k2[None,:,None]*grid.L2/(2.*np.pi))**2 + (grid.k3[None,None,0]*grid.L3/(2.*np.pi))**2)
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
      transfer = np.zeros(int(kmax+1))
      for j in range(0,grid.num_processes):
        transfer[0:k_all[j]] = transfer[0:k_all[j]] + E_all[j][:]
        transfer[0:k_all2[j]] = transfer[0:k_all2[j]] + E_all2[j][:]

      return kspec,transfer / (grid.N1*grid.N2*grid.N3)**2
    else:
      return 0,0


  def computeSGS_DNS(self,main,grid,myFFT):
    uhat_f = grid.filter(main.uhat)
    vhat_f = grid.filter(main.vhat)
    what_f = grid.filter(main.what)
    ureal = np.zeros((grid.N1,grid.Npy,grid.N3))
    vreal = np.zeros((grid.N1,grid.Npy,grid.N3))
    wreal = np.zeros((grid.N1,grid.Npy,grid.N3))
    ureal_f = np.zeros((grid.N1,grid.Npy,grid.N3))
    vreal_f = np.zeros((grid.N1,grid.Npy,grid.N3))
    wreal_f = np.zeros((grid.N1,grid.Npy,grid.N3))
    NL = np.zeros((6,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    NL_f = np.zeros((6,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')

    myFFT.myifft3D(main.uhat,ureal)
    myFFT.myifft3D(main.vhat,vreal)
    myFFT.myifft3D(main.what,wreal)
    myFFT.myifft3D(uhat_f,ureal_f)
    myFFT.myifft3D(vhat_f,vreal_f)
    myFFT.myifft3D(what_f,wreal_f)

    myFFT.myfft3D(ureal*ureal,NL[0])
    myFFT.myfft3D(vreal*vreal,NL[1])
    myFFT.myfft3D(wreal*wreal,NL[2])
    myFFT.myfft3D(ureal*vreal,NL[3])
    myFFT.myfft3D(ureal*wreal,NL[4])
    myFFT.myfft3D(vreal*wreal,NL[5])

    myFFT.myfft3D(ureal_f*ureal_f,NL_f[0])
    myFFT.myfft3D(vreal_f*vreal_f,NL_f[1])
    myFFT.myfft3D(wreal_f*wreal_f,NL_f[2])
    myFFT.myfft3D(ureal_f*vreal_f,NL_f[3])
    myFFT.myfft3D(ureal_f*wreal_f,NL_f[4])
    myFFT.myfft3D(vreal_f*wreal_f,NL_f[5])


    ## Compute SGS tensor. 
    # tau_ij = [d/dx 
    tauhat = np.zeros((6,grid.Npx,grid.N2,(grid.N3/2+1)),dtype='complex')
    tauhat[0] = grid.filter(NL[0]) - NL_f[0]
    tauhat[1] = grid.filter(NL[1]) - NL_f[1]
    tauhat[2] = grid.filter(NL[2]) - NL_f[2]
    tauhat[3] = grid.filter(NL[3]) - NL_f[3]
    tauhat[4] = grid.filter(NL[4]) - NL_f[4]
    tauhat[5] = grid.filter(NL[5]) - NL_f[5]

    ## contribution of projection to RHS sgs (k_m k_j)/k^2 \tau_{jm}
    tau_projection  = 1j*grid.ksqr_i*( grid.k1[:,None,None]*grid.k1[:,None,None]*tauhat[0] + grid.k2[None,:,None]*grid.k2[None,:,None]*tauhat[1] + \
             grid.k3[None,None,:]*grid.k3[None,None,:]*tauhat[2] + 2.*grid.k1[:,None,None]*grid.k2[None,:,None]*tauhat[3] + \
             2.*grid.k1[:,None,None]*grid.k3[None,None,:]*tauhat[4] + 2.*grid.k2[None,:,None]*grid.k3[None,None,:]*tauhat[5] )

    w0_u = np.zeros((grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    w0_v = np.zeros((grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    w0_w = np.zeros((grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')

    w0_u[:,:,:] = -(1j*grid.k1[:,None,None]*tauhat[0] + 1j*grid.k2[None,:,None]*tauhat[3] + 1j*grid.k3[None,None,:]*tauhat[4]) + 1j*grid.k1[:,None,None]*tau_projection
    w0_v[:,:,:] = -(1j*grid.k1[:,None,None]*tauhat[3] + 1j*grid.k2[None,:,None]*tauhat[1] + 1j*grid.k3[None,None,:]*tauhat[5]) + 1j*grid.k2[None,:,None]*tau_projection
    w0_w[:,:,:] = -(1j*grid.k1[:,None,None]*tauhat[4] + 1j*grid.k2[None,:,None]*tauhat[5] + 1j*grid.k3[None,None,:]*tauhat[2]) + 1j*grid.k3[None,None,:]*tau_projection
    return w0_u,w0_v,w0_w


  def computeF(self,main,grid,myFFT):
    #function to compute F (it is just the right hand side computation)
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    mpi_rank = comm.Get_rank()


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

    if (main.rotate == 1):
      phat[:,:,:] = phat[:,:,:] - 2.*grid.ksqr_i*1j*( grid.k1[:,None,None]*(main.vhat*main.Om3 - main.what*main.Om2) +
                    grid.k2[None,:,None]*(main.what*main.Om1 - main.uhat*main.Om3) + \
                    grid.k3[None,None,:]*(main.uhat*main.Om2 - main.vhat*main.Om1))

    main.F[0] = myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[0] - 1j*grid.k2[None,:,None]*main.NL[3] - 1j*grid.k3[None,None,:]*main.NL[4] - \
                                         1j*grid.k1[:,None,None]*phat - main.nu*grid.ksqr*main.uhat ,grid)

    main.F[1] = myFFT.dealias(-1j*grid.k1[:,None,None]*main.NL[3] - 1j*grid.k2[None,:,None]*main.NL[1] - 1j*grid.k3[None,None,:]*main.NL[5] - \
                                         1j*grid.k2[None,:,None]*phat - main.nu*grid.ksqr*main.vhat ,grid)

    main.F[2] = myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[4] - 1j*grid.k2[None,:,None]*main.NL[5] - 1j*grid.k3[None,None,:]*main.NL[2] - \
                                         1j*grid.k3[None,None,:]*phat - main.nu*grid.ksqr*main.what ,grid)

    if (main.rotate == 1):
      main.F[0] = main.F[0] + 2.*(main.vhat*main.Om3 - main.what*main.Om2)
      main.F[1] = main.F[1] + 2.*(main.what*main.Om1 - main.uhat*main.Om3)
      main.F[2] = main.F[2] + 2.*(main.uhat*main.Om2 - main.vhat*main.Om1)
    #=========================================================================

    uhat_f = grid.filter(main.uhat)
    vhat_f = grid.filter(main.vhat)
    what_f = grid.filter(main.what)
    u_f = np.zeros(np.shape(main.u))
    v_f = np.zeros(np.shape(main.v))
    w_f = np.zeros(np.shape(main.w))
    myFFT.myifft3D(uhat_f,u_f)
    myFFT.myifft3D(vhat_f,v_f)
    myFFT.myifft3D(what_f,w_f)

    myFFT.myfft3D(u_f*u_f,main.NL[0])
    myFFT.myfft3D(v_f*v_f,main.NL[1])
    myFFT.myfft3D(w_f*w_f,main.NL[2])
    myFFT.myfft3D(u_f*v_f,main.NL[3])
    myFFT.myfft3D(u_f*w_f,main.NL[4])
    myFFT.myfft3D(v_f*w_f,main.NL[5])

    phat_f  = grid.ksqr_i*( -grid.k1[:,None,None]*grid.k1[:,None,None]*main.NL[0] - grid.k2[None,:,None]*grid.k2[None,:,None]*main.NL[1] - \
             grid.k3[None,None,:]*grid.k3[None,None,:]*main.NL[2] - 2.*grid.k1[:,None,None]*grid.k2[None,:,None]*main.NL[3] - \
             2.*grid.k1[:,None,None]*grid.k3[None,None,:]*main.NL[4] - 2.*grid.k2[None,:,None]*grid.k3[None,None,:]*main.NL[5] )

    if (main.rotate == 1):
      phat_f[:,:,:] = phat_f[:,:,:] - 2.*grid.ksqr_i*1j*( grid.k1[:,None,None]*(vhat_f*main.Om3 - what_f*main.Om2) +
                    grid.k2[None,:,None]*(what_f*main.Om1 - uhat_f*main.Om3) + \
                    grid.k3[None,None,:]*(uhat_f*main.Om2 - vhat_f*main.Om1))

    main.F[0] = main.F[0] - myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[0] - 1j*grid.k2[None,:,None]*main.NL[3] - 1j*grid.k3[None,None,:]*main.NL[4] - \
                                         1j*grid.k1[:,None,None]*phat_f - main.nu*grid.ksqr*uhat_f ,grid)

    main.F[1] = main.F[1] - myFFT.dealias(-1j*grid.k1[:,None,None]*main.NL[3] - 1j*grid.k2[None,:,None]*main.NL[1] - 1j*grid.k3[None,None,:]*main.NL[5] - \
                                         1j*grid.k2[None,:,None]*phat_f - main.nu*grid.ksqr*vhat_f ,grid)

    main.F[2] = main.F[2] - myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[4] - 1j*grid.k2[None,:,None]*main.NL[5] - 1j*grid.k3[None,None,:]*main.NL[2] - \
                                         1j*grid.k3[None,None,:]*phat_f - main.nu*grid.ksqr*what_f ,grid)

    if (main.rotate == 1):
      main.F[0] = main.F[0] - 2.*(vhat_f*main.Om3 - what_f*main.Om2)
      main.F[1] = main.F[1] - 2.*(what_f*main.Om1 - uhat_f*main.Om3)
      main.F[2] = main.F[2] - 2.*(uhat_f*main.Om2 - vhat_f*main.Om1)

  def computeQcriterion(self,main,grid,myFFT):
    ## get strain rate tensor
    Shat= np.zeros((6,grid.Npx,grid.N2,grid.N3),dtype='complex')
    S = np.zeros((6,grid.N1,grid.Npy,grid.N3),dtype='complex')

    Shat[0] = 1j*grid.k1[:,None,None]*main.uhat
    Shat[1] = 1j*grid.k2[None,:,None]*main.vhat
    Shat[2] = 1j*grid.k3[None,None,:]*main.what
    Shat[3] = 0.5*1j*(grid.k2[None,:,None]*main.uhat + grid.k1[:,None,None]*main.vhat)
    Shat[4] = 0.5*1j*(grid.k3[None,None,:]*main.uhat + grid.k1[:,None,None]*main.what)
    Shat[5] = 0.5*1j*(grid.k3[None,None,:]*main.vhat + grid.k2[None,:,None]*main.what)
    for i in range(0,6):
      myFFT.myifft3D(Shat[i],S[i])
    ## get vorticity rate tensor (diagonals are zero)
    Omhat = np.zeros((3,grid.Npx,grid.N2,grid.N3),dtype='complex')
    Om = np.zeros((3,grid.N1,grid.Npy,grid.N3),dtype='complex')
    Omhat[0] = 0.5*1j*(grid.k2[None,:,None]*main.uhat - grid.k1[:,None,None]*main.vhat) #12
    Omhat[1] = 0.5*1j*(grid.k3[None,None,:]*main.uhat - grid.k1[:,None,None]*main.what) #13
    Omhat[2] = 0.5*1j*(grid.k3[None,None,:]*main.vhat - grid.k2[None,:,None]*main.what) #23
    for i in range(0,3):
      myFFT.myifft3D(Omhat[i],Om[i] )
    S_mag = (S[0]*S[0] + S[1]*S[1] + S[2]*S[2] + S[3]*S[3] + S[4]*S[4] + S[5]*S[5])
    Om_mag= (Om[0]*Om[0]*2 + Om[1]*Om[1]*2 + Om[2]*Om[2]*2)
    Q = S_mag - Om_mag
    return Q
 
  def computePLQLU(self,main,grid,myFFT):
    #function to compute PLQLu
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    mpi_rank = comm.Get_rank()

    uhat_f = grid.filter(main.uhat)
    vhat_f = grid.filter(main.vhat)
    what_f = grid.filter(main.what)
    u_f = np.zeros((grid.N1,grid.Npy,grid.N3))
    v_f = np.zeros((grid.N1,grid.Npy,grid.N3))
    w_f = np.zeros((grid.N1,grid.Npy,grid.N3))
    myFFT.myifft3D(uhat_f,u_f)
    myFFT.myifft3D(vhat_f,v_f)
    myFFT.myifft3D(what_f,w_f)

    myFFT.myfft3D(u_f*u_f,main.NL[0])
    myFFT.myfft3D(v_f*v_f,main.NL[1])
    myFFT.myfft3D(w_f*w_f,main.NL[2])
    myFFT.myfft3D(u_f*v_f,main.NL[3])
    myFFT.myfft3D(u_f*w_f,main.NL[4])
    myFFT.myfft3D(v_f*w_f,main.NL[5])

    phat  = grid.ksqr_i*( -grid.k1[:,None,None]*grid.k1[:,None,None]*main.NL[0] - grid.k2[None,:,None]*grid.k2[None,:,None]*main.NL[1] - \
             grid.k3[None,None,:]*grid.k3[None,None,:]*main.NL[2] - 2.*grid.k1[:,None,None]*grid.k2[None,:,None]*main.NL[3] - \
             2.*grid.k1[:,None,None]*grid.k3[None,None,:]*main.NL[4] - 2.*grid.k2[None,:,None]*grid.k3[None,None,:]*main.NL[5] )

    if (main.rotate == 1):
      phat[:,:,:] = phat[:,:,:] - 2.*grid.ksqr_i*1j*( grid.k1[:,None,None]*(vhat_f*main.Om3 - what_f*main.Om2) +
                    grid.k2[None,:,None]*(what_f*main.Om1 - uhat_f*main.Om3) + \
                    grid.k3[None,None,:]*(uhat_f*main.Om2 - vhat_f*main.Om1))

    PLU = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    PLU[0] = myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[0] - 1j*grid.k2[None,:,None]*main.NL[3] - 1j*grid.k3[None,None,:]*main.NL[4] - \
                                         1j*grid.k1[:,None,None]*phat - main.nu*grid.ksqr*uhat_f ,grid)

    PLU[1] = myFFT.dealias(-1j*grid.k1[:,None,None]*main.NL[3] - 1j*grid.k2[None,:,None]*main.NL[1] - 1j*grid.k3[None,None,:]*main.NL[5] - \
                                         1j*grid.k2[None,:,None]*phat - main.nu*grid.ksqr*vhat_f ,grid)

    PLU[2] = myFFT.dealias( -1j*grid.k1[:,None,None]*main.NL[4] - 1j*grid.k2[None,:,None]*main.NL[5] - 1j*grid.k3[None,None,:]*main.NL[2] - \
                                         1j*grid.k3[None,None,:]*phat - main.nu*grid.ksqr*what_f ,grid)

    if (main.rotate == 1):
      PLU[0] = PLU[0] + 2.*(vhat_f*main.Om3 - what_f*main.Om2)
      PLU[1] = PLU[1] + 2.*(what_f*main.Om1 - uhat_f*main.Om3)
      PLU[2] = PLU[2] + 2.*(uhat_f*main.Om2 - vhat_f*main.Om1)
    #=========================================================================

    PLU_p = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    PLU_q = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')

    PLU_p[0],PLU_q[0] = seperateModes(PLU[0],grid)
    PLU_p[1],PLU_q[1] = seperateModes(PLU[1],grid)
    PLU_p[2],PLU_q[2] = seperateModes(PLU[2],grid)
    PLU_qreal = np.zeros((3,grid.N1,grid.Npy,grid.N3))
    myFFT.myifft3D(PLU_q[0],PLU_qreal[0])
    myFFT.myifft3D(PLU_q[1],PLU_qreal[1])
    myFFT.myifft3D(PLU_q[2],PLU_qreal[2])

    up_PLUq = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    vp_PLUq = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    wp_PLUq = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')

    myFFT.myfft3D(u_f*PLU_qreal[0],up_PLUq[0])
    myFFT.myfft3D(v_f*PLU_qreal[0],vp_PLUq[0])
    myFFT.myfft3D(w_f*PLU_qreal[0],wp_PLUq[0])

    myFFT.myfft3D(u_f*PLU_qreal[1],up_PLUq[1])
    myFFT.myfft3D(v_f*PLU_qreal[1],vp_PLUq[1])
    myFFT.myfft3D(w_f*PLU_qreal[1],wp_PLUq[1])

    myFFT.myfft3D(u_f*PLU_qreal[2],up_PLUq[2])
    myFFT.myfft3D(v_f*PLU_qreal[2],vp_PLUq[2])
    myFFT.myfft3D(w_f*PLU_qreal[2],wp_PLUq[2])


    pterm = 2.*grid.ksqr_i*( grid.k1[:,None,None]*grid.k1[:,None,None]*up_PLUq[0] + grid.k2[None,:,None]*grid.k2[None,:,None]*vp_PLUq[1] + grid.k3[None,None,:]*grid.k3[None,None,:]*wp_PLUq[2] + \
                          grid.k1[:,None,None]*grid.k2[None,:,None]*(up_PLUq[1] + vp_PLUq[0]) + grid.k1[:,None,None]*grid.k3[None,None,:]*(up_PLUq[2] + wp_PLUq[0]) + \
                          grid.k2[None,:,None]*grid.k3[None,None,:]*(vp_PLUq[2] + wp_PLUq[1]) )


    PLQLU = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    PLQLU[0] =  -1j*grid.k1[:,None,None]*up_PLUq[0] - 1j*grid.k2[None,:,None]*vp_PLUq[0] - 1j*grid.k3[None,None,:]*wp_PLUq[0] - \
            1j*grid.k1[:,None,None]*up_PLUq[0] - 1j*grid.k2[None,:,None]*up_PLUq[1] - 1j*grid.k3[None,None,:]*up_PLUq[2] + \
            1j*grid.k1[:,None,None]*pterm

    PLQLU[1] = -1j*grid.k1[:,None,None]*up_PLUq[1] - 1j*grid.k2[None,:,None]*vp_PLUq[1] - 1j*grid.k3[None,None,:]*wp_PLUq[1] - \
            1j*grid.k1[:,None,None]*vp_PLUq[0] - 1j*grid.k2[None,:,None]*vp_PLUq[1] - 1j*grid.k3[None,None,:]*vp_PLUq[2] + \
            1j*grid.k2[None,:,None]*pterm

    PLQLU[2] =  -1j*grid.k1[:,None,None]*up_PLUq[2] - 1j*grid.k2[None,:,None]*vp_PLUq[2] - 1j*grid.k3[None,None,:]*wp_PLUq[2] -\
            1j*grid.k1[:,None,None]*wp_PLUq[0] - 1j*grid.k2[None,:,None]*wp_PLUq[1] - 1j*grid.k3[None,None,:]*wp_PLUq[2] + \
            1j*grid.k3[None,None,:]*pterm

    return PLQLU
