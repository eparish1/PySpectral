import numpy as np
import sys
import pyfftw
from RHSfunctions import *
from RHSfunctionsWEAVE import *

class variables:
  def __init__(self,weave,turb_model,rotate,Om1,Om2,Om3,grid,uhat,vhat,what,t,dt,nu,Ct,dt0,\
                dt0_subintegrations,dt1,dt1_subintegrations,cfl,dim):
    self.turb_model = turb_model
    self.rotate = rotate
    self.Om1 = Om1
    self.Om2 = Om2
    self.Om3 = Om3
    self.t = t
    self.kc = np.amax(grid.k1)
    self.dt = dt
    self.nu = nu
    self.tauhat = np.zeros((grid.N1,grid.N2,grid.N3/2+1,6),dtype='complex')
    self.uhat = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    self.uhat[:,:,:] = uhat[:,:,:]
    self.vhat = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    self.vhat[:,:,:] = vhat[:,:,:]
    self.what = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    self.what[:,:,:] = what[:,:,:]
    self.cfl = cfl
    ##============ DNS MODE ========================
    if (turb_model == 0):
      sys.stdout.write('Not using any SGS \n')
      self.Q = np.zeros( (3*grid.N1,3*grid.N2,3*(grid.N3/2+1)),dtype='complex')
      self.nvars = 3
      self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
      self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
      self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def U2Q():
        self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
        self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
        self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def Q2U():
        self.uhat[:,:,:] = self.Q[0::3,0::3,0::3]
        self.vhat[:,:,:] = self.Q[1::3,1::3,1::3]
        self.what[:,:,:] = self.Q[2::3,2::3,2::3]
      if (weave == 0):
        self.computeRHS = computeRHS_NOSGS

      else:
        sys.stdout.write('Using WEAVE for inline C++ \n')
        if (dim == 2):
          sys.stdout.write('2D Version \n')
          self.computeRHS = computeRHS_NOSGS_WEAVE_2D
        else:
          self.computeRHS = computeRHS_NOSGS_WEAVE
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================
    if (turb_model == 'Orthogonal Dynamics'):
      sys.stdout.write('Solving the orthogonal dynamics \n')
      self.Q = np.zeros( (3*grid.N1,3*grid.N2,3*(grid.N3/2+1)),dtype='complex')
      self.nvars = 3
      self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
      self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
      self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def U2Q():
        self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
        self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
        self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def Q2U():
        self.uhat[:,:,:] = self.Q[0::3,0::3,0::3]
        self.vhat[:,:,:] = self.Q[1::3,1::3,1::3]
        self.what[:,:,:] = self.Q[2::3,2::3,2::3]
      self.computeRHS = computeRHS_Orthogonal
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================

    ##============ SMAGORINSKY ====================
    if (turb_model == 1):
      print('Using Smagorinsky SGS')
      self.Q = np.zeros( (3*grid.N1,3*grid.N2,3*(grid.N3/2+1)),dtype='complex')
      self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
      self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
      self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      self.w0_u = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.w0_v = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.w0_w = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')

      def U2Q():
        self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
        self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
        self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def Q2U():
        self.uhat[:,:,:] = self.Q[0::3,0::3,0::3]
        self.vhat[:,:,:] = self.Q[1::3,1::3,1::3]
        self.what[:,:,:] = self.Q[2::3,2::3,2::3]
      self.computeRHS = computeRHS_SMAG
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================

    ##============ t-model ========================
    if (turb_model == 2):
      print('Using the t-model')
      if Ct == -10:
        print('Did not assign Ct for t-model, using default Ct=0.1')
        self.Ct = 0.1
      else:
        print('Assigning Ct = ' + str(Ct))
        self.Ct = Ct
      self.w0_u = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.w0_v = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.w0_w = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.Q = np.zeros( (3*grid.N1,3*grid.N2,3*(grid.N3/2+1)),dtype='complex')
      self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
      self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
      self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def U2Q():
        self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
        self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
        self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def Q2U():
        self.uhat[:,:,:] = self.Q[0::3,0::3,0::3]
        self.vhat[:,:,:] = self.Q[1::3,1::3,1::3]
        self.what[:,:,:] = self.Q[2::3,2::3,2::3]
      self.computeRHS = computeRHS_tmodel
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================
    ##============ tau-model ========================
    if (turb_model == 'Dynamic tau-model'):
      print('Using the dynamic tau-model')
      self.kf = int(grid.kc*2./3.)
      self.testfilt_k1 = np.ones(grid.N1)
      self.testfilt_k2 = np.ones(grid.N2)
      self.testfilt_k3 = np.ones(grid.N3/2+1)
      for i in range(0,grid.N1):
        if (abs(grid.k1[i,0,0]) >= self.kf):
          self.testfilt_k1[i] = 0.
 
      for i in range(0,grid.N2):
        if (abs(grid.k2[0,i,0]) >= self.kf):
          self.testfilt_k2[i] = 0.

      for i in range(0,grid.N3/2+1):
        if (abs(grid.k3[0,0,i]) >= self.kf):
          self.testfilt_k3[i] = 0.

      def test_filter(uhat):
        ufilter = self.testfilt_k1[:,None,None]*self.testfilt_k2[None,:,None]*self.testfilt_k3[None,None,:]*uhat
        return ufilter
      self.test_filter = test_filter

      self.testfilt_k1_2x = np.ones(grid.N1*2)
      self.testfilt_k2_2x = np.ones(grid.N2*2)
      self.testfilt_k3_2x = np.ones(grid.N3+1)
      for i in range(0,grid.N1*2):
        if (abs(grid.k1f[i,0,0]) >= self.kf):
          self.testfilt_k1_2x[i] = 0.
 
      for i in range(0,grid.N2*2):
        if (abs(grid.k2f[0,i,0]) >= self.kf):
          self.testfilt_k2_2x[i] = 0.

      for i in range(0,grid.N3+1):
        if (abs(grid.k3f[0,0,i]) >= self.kf):
          self.testfilt_k3_2x[i] = 0.

      def test_filter_2x(uhat_pad):
        ufilter = self.testfilt_k1_2x[:,None,None]*self.testfilt_k2_2x[None,:,None]*self.testfilt_k3_2x[None,None,:]*uhat_pad
        return ufilter
      self.test_filter_2x = test_filter_2x
      self.w0_u = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.w0_v = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.w0_w = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.Q = np.zeros( (3*grid.N1,3*grid.N2,3*(grid.N3/2+1)),dtype='complex')
      self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
      self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
      self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def U2Q():
        self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
        self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
        self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def Q2U():
        self.uhat[:,:,:] = self.Q[0::3,0::3,0::3]
        self.vhat[:,:,:] = self.Q[1::3,1::3,1::3]
        self.what[:,:,:] = self.Q[2::3,2::3,2::3]
      self.computeRHS = computeRHS_Dtaumodel
      self.Q2U = Q2U
      self.U2Q = U2Q 


    ##============ tau-model ========================
    if (turb_model == 'Static tau-model'):
      print('Using the static tau-model')
      self.tau_a = dt0
      self.tau_b = dt1
      print('tau = ' + str(self.tau_a) + ' t * ' + str(self.tau_b) )
      self.kf = int(grid.kc/2.)
      self.testfilt_k1 = np.ones(grid.N1)
      self.testfilt_k2 = np.ones(grid.N2)
      self.testfilt_k3 = np.ones(grid.N3/2+1)
      for i in range(0,grid.N1):
        if (abs(grid.k1[i,0,0]) >= self.kf):
          self.testfilt_k1[i] = 0.
 
      for i in range(0,grid.N2):
        if (abs(grid.k2[0,i,0]) >= self.kf):
          self.testfilt_k2[i] = 0.

      for i in range(0,grid.N3/2+1):
        if (abs(grid.k3[0,0,i]) >= self.kf):
          self.testfilt_k3[i] = 0.
      self.w0_u = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.w0_v = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.w0_w = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.Q = np.zeros( (3*grid.N1,3*grid.N2,3*(grid.N3/2+1)),dtype='complex')
      self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
      self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
      self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def U2Q():
        self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
        self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
        self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def Q2U():
        self.uhat[:,:,:] = self.Q[0::3,0::3,0::3]
        self.vhat[:,:,:] = self.Q[1::3,1::3,1::3]
        self.what[:,:,:] = self.Q[2::3,2::3,2::3]
      self.computeRHS = computeRHS_staumodel
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================

    ##============ FM1 model ========================
    if (turb_model == 3):

      print('Using the First Order Finite Memory Model')
      if dt0 == -10:
        print('Did not assign dt0 for FM1 Model, using default dt0=0.1')
        self.dt0 = 0.1
      else:
        print('Assigning dt0 = ' + str(dt0))
        self.dt0 = dt0

      if dt0_subintegrations == -10:
        print('Did not assign dt0_subintegrations for FM1 Model, using default dt_subintegrations=1')
        self.dt0_subintegrations = 1
      else:
        print('Assigning dt0_subintegrations = ' + str(dt0_subintegrations))
        self.dt0_subintegrations = dt0_subintegrations

      self.PLQLu = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.PLQLv = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.PLQLw = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.w0_u = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.w0_v = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.w0_w = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.nvars = 3 + 3*self.dt0_subintegrations
      self.Q = np.zeros( (self.nvars*grid.N1,self.nvars*grid.N2,self.nvars*(grid.N3/2+1)),dtype='complex')
      self.Q[0::self.nvars,0::self.nvars,0::self.nvars] = self.uhat[:,:,:]
      self.Q[1::self.nvars,1::self.nvars,1::self.nvars] = self.vhat[:,:,:]
      self.Q[2::self.nvars,2::self.nvars,2::self.nvars] = self.what[:,:,:]
      j = 3
      for i in range(0,self.dt0_subintegrations):
        self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars] = self.w0_u[:,:,:,i]
        self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars] = self.w0_v[:,:,:,i]
        self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars] = self.w0_w[:,:,:,i]
        j += 3

      def U2Q():
        self.Q[0::self.nvars,0::self.nvars,0::self.nvars] = self.uhat[:,:,:]
        self.Q[1::self.nvars,1::self.nvars,1::self.nvars] = self.vhat[:,:,:]
        self.Q[2::self.nvars,2::self.nvars,2::self.nvars] = self.what[:,:,:]
        j = 3
        for i in range(0,self.dt0_subintegrations):
          self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars] = self.w0_u[:,:,:,i]
          self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars] = self.w0_v[:,:,:,i]
          self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars] = self.w0_w[:,:,:,i]
          j += 3
      def Q2U():
        self.uhat[:,:,:] = self.Q[0::self.nvars,0::self.nvars,0::self.nvars]
        self.vhat[:,:,:] = self.Q[1::self.nvars,1::self.nvars,1::self.nvars]
        self.what[:,:,:] = self.Q[2::self.nvars,2::self.nvars,2::self.nvars]
        j = 3
        for i in range(0,self.dt0_subintegrations):
          self.w0_u[:,:,:,i] = self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars]
          self.w0_v[:,:,:,i] = self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars]
          self.w0_w[:,:,:,i] = self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars]
          j += 3
      self.computeRHS = computeRHS_FM1
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================

    ##============ FM2 model ========================
    if (turb_model == 4):
      print('Using the Second Order Finite Memory Model')
      if dt0 == -10:
        print('Did not assign dt0 for FM2 Model, using default dt0=0.1')
        self.dt0 = 0.1
      else:
        print('Assigning dt0 = ' + str(dt0))
        self.dt0 = dt0
      if dt1 == -10:
        print('Did not assign dt1 for FM2 Model, using default dt1=0.05')
        self.dt1 = 0.05
      else:
        print('Assigning dt1 = ' + str(dt1))
        self.dt1 = dt1
      if dt0_subintegrations == -10:
        print('Did not assign dt0_subintegrations for FM2 Model, using default dt_subintegrations=1')
        self.dt0_subintegrations = 1
      else:
        print('Assigning dt0_subintegrations = ' + str(dt0_subintegrations))
        self.dt0_subintegrations = dt0_subintegrations
      if dt1_subintegrations == -10:
        print('Did not assign dt1_subintegrations for FM2 Model, using default dt_subintegrations=1')
        self.dt1_subintegrations = 1
      else:
        print('Assigning dt1_subintegrations = ' + str(dt1_subintegrations))
        self.dt1_subintegrations = dt1_subintegrations


      self.nvars = 3 + 3*self.dt0_subintegrations + 3*self.dt1_subintegrations

      self.w0_v = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.w0_u = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.w0_w = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt0_subintegrations),dtype='complex')
      self.w1_v = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt1_subintegrations),dtype='complex')
      self.w1_u = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt1_subintegrations),dtype='complex')
      self.w1_w = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),self.dt1_subintegrations),dtype='complex')
      self.Q = np.zeros( (self.nvars*grid.N1,self.nvars*grid.N2,self.nvars*(grid.N3/2+1)),dtype='complex')
      self.Q[0::self.nvars,0::self.nvars,0::self.nvars] = self.uhat[:,:,:]
      self.Q[1::self.nvars,1::self.nvars,1::self.nvars] = self.vhat[:,:,:]
      self.Q[2::self.nvars,2::self.nvars,2::self.nvars] = self.what[:,:,:]
      j = 3
      for i in range(0,self.dt0_subintegrations):
        self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars] = self.w0_u[:,:,:,i]
        self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars] = self.w0_v[:,:,:,i]
        self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars] = self.w0_w[:,:,:,i]
        if (i < self.dt1_subintegrations-1):
          j += 6
        else: 
          j += 3
      j = 6
      for i in range(0,self.dt1_subintegrations):
        self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars] = self.w1_u[:,:,:,i]
        self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars] = self.w1_v[:,:,:,i]
        self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars] = self.w1_w[:,:,:,i]
        if (i < self.dt0_subintegrations-1):
          j += 6
        else:
          j += 3

      def U2Q():
        self.Q[0::self.nvars,0::self.nvars,0::self.nvars] = self.uhat[:,:,:]
        self.Q[1::self.nvars,1::self.nvars,1::self.nvars] = self.vhat[:,:,:]
        self.Q[2::self.nvars,2::self.nvars,2::self.nvars] = self.what[:,:,:]
        j = 3
        for i in range(0,self.dt0_subintegrations):
          self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars] = self.w0_u[:,:,:,i]
          self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars] = self.w0_v[:,:,:,i]
          self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars] = self.w0_w[:,:,:,i]
          if (i < self.dt1_subintegrations - 1):
            j += 6
          else: 
            j += 3
        j = 6
        for i in range(0,self.dt1_subintegrations):
          self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars] = self.w1_u[:,:,:,i]
          self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars] = self.w1_v[:,:,:,i]
          self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars] = self.w1_w[:,:,:,i]
          if (i < self.dt0_subintegrations - 1):
            j += 6
          else:
            j += 3

      def Q2U():
        self.uhat[:,:,:] = self.Q[0::self.nvars,0::self.nvars,0::self.nvars]
        self.vhat[:,:,:] = self.Q[1::self.nvars,1::self.nvars,1::self.nvars]
        self.what[:,:,:] = self.Q[2::self.nvars,2::self.nvars,2::self.nvars]
        j = 3
        for i in range(0,self.dt0_subintegrations):
          self.w0_u[:,:,:,i] = self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars] 
          self.w0_v[:,:,:,i] = self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars] 
          self.w0_w[:,:,:,i] = self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars]
          if (i < self.dt1_subintegrations - 1):
            j += 6
          else: 
            j += 3
        j = 6
        for i in range(0,self.dt1_subintegrations):
          self.w1_u[:,:,:,i] = self.Q[j  ::self.nvars,j  ::self.nvars,j  ::self.nvars] 
          self.w1_v[:,:,:,i] = self.Q[j+1::self.nvars,j+1::self.nvars,j+1::self.nvars] 
          self.w1_w[:,:,:,i] = self.Q[j+2::self.nvars,j+2::self.nvars,j+2::self.nvars] 
          if (i < self.dt0_subintegrations - 1):
            j += 6
          else:
            j += 3

      self.computeRHS = computeRHS_FM2
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================



    ##============ FM1 model -two term trapezoidal ========================
    ## ONLY NEED FOR VERIFICATION NOW. TURB MODELS 3,4 SHOULD HAVE AN N TERM TRAP RULE BUILT IN
    if (turb_model == 5):
      print('Using the Second Order Finite Memory Model')
      if dt0 == -10:
        print('Did not assign dt0 for FM1 Model, using default dt0=0.1')
        self.dt0 = 0.1
      else:
        print('Assigning dt0 = ' + str(dt0))
        self.dt0 = dt0
      self.w0_v = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.w0_u = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.w0_w = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.w01_v = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.w01_u = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.w01_w = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')

      self.Q = np.zeros( (9*grid.N1,9*grid.N2,9*(grid.N3/2+1)),dtype='complex')
      self.Q[0::9,0::9,0::9] = self.uhat[:,:,:]
      self.Q[1::9,1::9,1::9] = self.vhat[:,:,:]
      self.Q[2::9,2::9,2::9] = self.what[:,:,:]
      self.Q[3::9,3::9,3::9] = self.w0_u[:,:,:]
      self.Q[4::9,4::9,4::9] = self.w0_v[:,:,:]
      self.Q[5::9,5::9,5::9] = self.w0_w[:,:,:]
      self.Q[6::9,6::9,6::9] = self.w01_u[:,:,:]
      self.Q[7::9,7::9,7::9] = self.w01_v[:,:,:]
      self.Q[8::9,8::9,8::9] = self.w01_w[:,:,:]
      def U2Q():
        self.Q[0::9,0::9,0::9] = self.uhat[:,:,:]
        self.Q[1::9,1::9,1::9] = self.vhat[:,:,:]
        self.Q[2::9,2::9,2::9] = self.what[:,:,:]
        self.Q[3::9,3::9,3::9] = self.w0_u[:,:,:]
        self.Q[4::9,4::9,4::9] = self.w0_v[:,:,:]
        self.Q[5::9,5::9,5::9] = self.w0_w[:,:,:]
        self.Q[6::9,6::9,6::9] = self.w01_u[:,:,:]
        self.Q[7::9,7::9,7::9] = self.w01_v[:,:,:]
        self.Q[8::9,8::9,8::9] = self.w01_w[:,:,:]
      def Q2U():
        self.uhat[:,:,:] = self.Q[0::9,0::9,0::9]
        self.vhat[:,:,:] = self.Q[1::9,1::9,1::9]
        self.what[:,:,:] = self.Q[2::9,2::9,2::9]
        self.w0_u[:,:,:] = self.Q[3::9,3::9,3::9]
        self.w0_v[:,:,:] = self.Q[4::9,4::9,4::9]
        self.w0_w[:,:,:] = self.Q[5::9,5::9,5::9]
        self.w01_u[:,:,:] = self.Q[6::9,6::9,6::9]
        self.w01_v[:,:,:] = self.Q[7::9,7::9,7::9]
        self.w01_w[:,:,:] = self.Q[8::9,8::9,8::9]

      self.computeRHS = computeRHS_FM1_2term
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================

    ##============ CM1 model ========================
    if (turb_model == 6):

      print('Using the First Order Complete Memory Model')
      self.PLQLu = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.PLQLv = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.PLQLw = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.w0_u = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.w0_v = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.w0_w = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.nvars = 6 
      self.Q = np.zeros( (self.nvars*grid.N1,self.nvars*grid.N2,self.nvars*(grid.N3/2+1)),dtype='complex')
      self.Q[0::self.nvars,0::self.nvars,0::self.nvars] = self.uhat[:,:,:]
      self.Q[1::self.nvars,1::self.nvars,1::self.nvars] = self.vhat[:,:,:]
      self.Q[2::self.nvars,2::self.nvars,2::self.nvars] = self.what[:,:,:]
      self.Q[3::self.nvars,3::self.nvars,3::self.nvars] = self.w0_u[:,:,:,0]
      self.Q[4::self.nvars,4::self.nvars,4::self.nvars] = self.w0_v[:,:,:,0]
      self.Q[5::self.nvars,5::self.nvars,5::self.nvars] = self.w0_w[:,:,:,0]
      def U2Q():
        self.Q[0::self.nvars,0::self.nvars,0::self.nvars] = self.uhat[:,:,:]
        self.Q[1::self.nvars,1::self.nvars,1::self.nvars] = self.vhat[:,:,:]
        self.Q[2::self.nvars,2::self.nvars,2::self.nvars] = self.what[:,:,:]
        self.Q[3::self.nvars,3::self.nvars,3::self.nvars] = self.w0_u[:,:,:,0]
        self.Q[4::self.nvars,4::self.nvars,4::self.nvars] = self.w0_v[:,:,:,0]
        self.Q[5::self.nvars,5::self.nvars,5::self.nvars] = self.w0_w[:,:,:,0]
      def Q2U():
        self.uhat[:,:,:] = self.Q[0::self.nvars,0::self.nvars,0::self.nvars]
        self.vhat[:,:,:] = self.Q[1::self.nvars,1::self.nvars,1::self.nvars]
        self.what[:,:,:] = self.Q[2::self.nvars,2::self.nvars,2::self.nvars]
        self.w0_u[:,:,:,0] = self.Q[3::self.nvars,3::self.nvars,3::self.nvars]
        self.w0_v[:,:,:,0] = self.Q[3::self.nvars,4::self.nvars,4::self.nvars]
        self.w0_w[:,:,:,0] = self.Q[5::self.nvars,5::self.nvars,5::self.nvars]
      self.computeRHS = computeRHS_CM1
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================
    ##============ Dynamic Smag========================
    if (turb_model == 7):
      print('Using Dynamic Smagorinsky SGS')
      self.Q = np.zeros( (3*grid.N1,3*grid.N2,3*(grid.N3/2+1)),dtype='complex')
      self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
      self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
      self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      self.w0_u = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.w0_v = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')
      self.w0_w = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1),1),dtype='complex')

      def U2Q():
        self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
        self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
        self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def Q2U():
        self.uhat[:,:,:] = self.Q[0::3,0::3,0::3]
        self.vhat[:,:,:] = self.Q[1::3,1::3,1::3]
        self.what[:,:,:] = self.Q[2::3,2::3,2::3]
      self.computeRHS = computeRHS_DSMAG
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================


    ##============ DNS Budgets ========================
    if (turb_model == 99):
      print('Running with no SGS and computing budgets')
      self.nvars = 3
      self.PLu = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.PLv = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.PLw = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.PLQLu = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.PLQLv = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.PLQLw = np.zeros( (grid.N1,grid.N2,(grid.N3/2+1)),dtype='complex')
      self.Q = np.zeros( (3*grid.N1,3*grid.N2,3*(grid.N3/2+1)),dtype='complex')
      self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
      self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
      self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def U2Q():
        self.Q[0::3,0::3,0::3] = self.uhat[:,:,:]
        self.Q[1::3,1::3,1::3] = self.vhat[:,:,:]
        self.Q[2::3,2::3,2::3] = self.what[:,:,:]
      def Q2U():
        self.uhat[:,:,:] = self.Q[0::3,0::3,0::3]
        self.vhat[:,:,:] = self.Q[1::3,1::3,1::3]
        self.what[:,:,:] = self.Q[2::3,2::3,2::3]
      self.computeRHS = computeRHS_BUDGETS
      self.Q2U = Q2U
      self.U2Q = U2Q 
    ##=============================================


class gridclass:
  def __init__(self,N1,N2,N3,x,y,z,kc,L1,L2,L3):
    self.N1 = N1
    self.N2 = N2
    self.N3 = N3
    self.L1 = L1
    self.L2 = L2
    self.L3 = L3
    self.x = np.zeros(np.shape(x))
    self.x[:,:,:] = x[:,:,:]
    self.y = np.zeros(np.shape(y))
    self.y[:,:,:] = y[:,:,:]
    self.z = np.zeros(np.shape(z))
    self.z[:,:,:] = z[:,:,:]
    self.dx = x[1,0,0] - x[0,0,0]
    self.dy = y[0,1,0] - y[0,0,0]
    self.dz = z[0,0,1] - z[0,0,0]
    k1 = np.fft.fftshift( np.linspace(-N1/2,N1/2-1,N1) ) * 2.*np.pi / L1
    k2 = np.fft.fftshift( np.linspace(-N2/2,N2/2-1,N2) ) * 2.*np.pi / L2
    k3 = np.linspace( 0,N3/2,N3/2+1 ) * 2. * np.pi / L3
    k1f = np.fft.fftshift( np.linspace(-N1,N1-1,2.*N1) ) * 2.*np.pi / L1
    k2f = np.fft.fftshift( np.linspace(-N2,N2-1,2.*N2) ) * 2.*np.pi / L2
    k3f = np.linspace( 0,N3,N3+1 ) * 2. * np.pi / L3


    self.k2,self.k1,self.k3 = np.meshgrid(k2,k1,k3)
    self.k2f,self.k1f,self.k3f = np.meshgrid(k2f,k1f,k3f)
  

    self.ksqr = self.k1*self.k1 + self.k2*self.k2 + self.k3*self.k3 + 1.e-50
    self.ksqr_i = 1./self.ksqr
    self.ksqrf = self.k1f*self.k1f + self.k2f*self.k2f + self.k3f*self.k3f + 1.e-50
    self.ksqrf_i = 1./self.ksqrf
    self.kc = kc
    vol = (self.dx*self.dy*self.dz)**(1./3.)
    self.Delta = vol
    self.Gf = np.zeros(np.shape(self.k1)) #Sharp Spectral cutoff (we cutoff the oddball frequency)
    self.Gf[0:self.kc,0:self.kc,0:self.kc] = 1 # get first quardants
    self.Gf[0:self.kc,self.N2-self.kc+1::,0:self.kc] = 1 #0:kc in k1 and -kc:0 in k2
    self.Gf[self.N1-self.kc+1::,0:self.kc,0:self.kc] = 1 #-kc:0 in k1 and 0:kc in k2
    self.Gf[self.N1-self.kc+1::,self.N2-self.kc+1::,0:self.kc] = 1 #-kc:0 in k1 and k2

class FFTclass:
  def __init__(self,N1,N2,N3,nthreads):
    self.nthreads = nthreads
    self.scale = np.sqrt( (3./2.)**3*np.sqrt(N1*N2*N3) ) #scaling for FFTS
    ## Inverse transforms of uhat,vhat,what are of the truncated padded variable. 
    ## Input is complex truncate,output is real untruncated
    self.invalT =    pyfftw.n_byte_align_empty((int(3./2.*N1),int(3./2.*N2),int(3./4.*N3+1)), 16, 'complex128')
    self.outvalT=    pyfftw.n_byte_align_empty((int(3./2.*N1),int(3./2.*N2),int(3./2*N3   )), 16, 'float64')
    self.ifftT_obj = pyfftw.FFTW(self.invalT,self.outvalT,axes=(0,1,2,),\
                     direction='FFTW_BACKWARD',threads=nthreads)
    ## Fourier transforms of padded vars like u*u.
    ## Input is real full, output is imag truncated 
    self.inval =   pyfftw.n_byte_align_empty((int(3./2.*N1),int(3./2.*N2),int(3./2.*N3) ), 16, 'float64')
    self.outval=   pyfftw.n_byte_align_empty((int(3./2.*N1),int(3./2.*N2),int(3./4*N3+1)), 16, 'complex128')
    self.fft_obj = pyfftw.FFTW(self.inval,self.outval,axes=(0,1,2,),\
                    direction='FFTW_FORWARD', threads=nthreads)

    self.invalT2 =    pyfftw.n_byte_align_empty((int(2.*N1),int(2.*N2),int(N3+1)), 16, 'complex128')
    self.outvalT2 =   pyfftw.n_byte_align_empty((int(2.*N1),int(2.*N2),int(2*N3)), 16, 'float64')
    self.ifftT_obj2 = pyfftw.FFTW(self.invalT2,self.outvalT2,axes=(0,1,2,),\
                     direction='FFTW_BACKWARD',threads=nthreads)

    self.inval2 =   pyfftw.n_byte_align_empty((int(2.*N1),int(2.*N2),int(2.*N3) ), 16, 'float64')
    self.outval2=   pyfftw.n_byte_align_empty((int(2.*N1),int(2.*N2),int(N3+1)), 16, 'complex128')
    self.fft_obj2 = pyfftw.FFTW(self.inval2,self.outval2,axes=(0,1,2,),\
                    direction='FFTW_FORWARD', threads=nthreads)

    self.invalT3 =    pyfftw.n_byte_align_empty((int(3.*N1),int(3.*N2),int(3*N3/2+1)), 16, 'complex128')
    self.outvalT3 =   pyfftw.n_byte_align_empty((int(3.*N1),int(3.*N2),int(3*N3)), 16, 'float64')
    self.ifftT_obj3 = pyfftw.FFTW(self.invalT3,self.outvalT3,axes=(0,1,2,),\
                     direction='FFTW_BACKWARD',threads=nthreads)

    self.inval3 =   pyfftw.n_byte_align_empty((int(3.*N1),int(3.*N2),int(3.*N3) ), 16, 'float64')
    self.outval3=   pyfftw.n_byte_align_empty((int(3.*N1),int(3.*N2),int(3*N3/2+1)), 16, 'complex128')
    self.fft_obj3 = pyfftw.FFTW(self.inval3,self.outval3,axes=(0,1,2,),\
                    direction='FFTW_FORWARD', threads=nthreads)





class utilitiesClass():
  def preAdvanceQ_hook(self,main,grid,myFFT):
    if (main.iteration%20 == 0 and main.turb_model == 99):
      main.u0 = np.zeros(np.shape(main.uhat),dtype='complex')
      main.u0[:,:,:] = main.uhat[:,:,:]
      main.w0_u0,main.w0_v0,main.w0_w0 = self.computeSGS_DNS(main,grid,myFFT)
    else:
      pass
  def postAdvanceQ_hook(self,main,grid,myFFT):
    if (main.iteration%20 == 0 and main.turb_model == 99):
      w0_u,w0_v,w0_w = self.computeSGS_DNS(main,grid,myFFT)
      string = '3DSolution/budget' + str(main.iteration)
      np.savez_compressed(string,w0_udot = (w0_u-main.w0_u0)/main.dt,\
                               w0_vdot = (w0_v-main.w0_v0)/main.dt,\
                               w0_wdot = (w0_w-main.w0_w0)/main.dt,\
                               u_dot = (main.uhat - main.u0)/main.dt,\
             PLQLu=main.PLQLu,PLQLv=main.PLQLv,PLQLw=main.PLQLw,PLu = main.PLu)
    else:
      pass 


  def computeEnergy(self,main,grid):
      uE = np.sum(main.uhat[:,:,1:grid.N3/2]*np.conj(main.uhat[:,:,1:grid.N3/2]*2) ) + \
           np.sum(main.uhat[:,:,0]*np.conj(main.uhat[:,:,0])) 
      vE = np.sum(main.vhat[:,:,1:grid.N3/2]*np.conj(main.vhat[:,:,1:grid.N3/2]*2) ) + \
           np.sum(main.vhat[:,:,0]*np.conj(main.vhat[:,:,0])) 
      wE = np.sum(main.what[:,:,1:grid.N3/2]*np.conj(main.what[:,:,1:grid.N3/2]*2) ) + \
           np.sum(main.what[:,:,0]*np.conj(main.what[:,:,0]))
      return np.real(0.5*(uE + vE + wE)/(grid.N1*grid.N2*grid.N3))

  def computeEnergy_2D(self,main,grid):
      uE = np.sum(main.uhat[:,:,0]*np.conj(main.uhat[:,:,0]) ) 
      vE = np.sum(main.vhat[:,:,0]*np.conj(main.vhat[:,:,0]) ) 
      return np.real(0.5*(uE + vE)/(grid.N1*grid.N2*grid.N3))

  def computeEnergy_resolved(self,main,grid):
      uFilt = grid.Gf*main.uhat
      vFilt = grid.Gf*main.vhat
      wFilt = grid.Gf*main.what
      uE = np.sum(uFilt[:,:,1:grid.N3/2]*np.conj(uFilt[:,:,1:grid.N3/2]*2) ) + \
           np.sum(uFilt[:,:,0]*np.conj(uFilt[:,:,0])) 
      vE = np.sum(vFilt[:,:,1:grid.N3/2]*np.conj(vFilt[:,:,1:grid.N3/2]*2) ) + \
           np.sum(vFilt[:,:,0]*np.conj(vFilt[:,:,0])) 
      wE = np.sum(wFilt[:,:,1:grid.N3/2]*np.conj(wFilt[:,:,1:grid.N3/2]*2) ) + \
           np.sum(wFilt[:,:,0]*np.conj(wFilt[:,:,0]))
      return np.real(0.5*(uE + vE + wE)/(grid.N1*grid.N2*grid.N3))

  def computeEnergy_resolved_2D(self,main,grid):
      uFilt = grid.Gf*main.uhat
      vFilt = grid.Gf*main.vhat
      wFilt = grid.Gf*main.what
      uE = np.sum(uFilt[:,:,0]*np.conj(uFilt[:,:,0]) ) 
      vE = np.sum(vFilt[:,:,0]*np.conj(vFilt[:,:,0]) )
      return np.real(0.5*(uE + vE )/(grid.N1*grid.N2*grid.N3))


  def compute_dt(self,main,grid):
    if (main.cfl > 0):
      u = np.fft.irfftn(main.uhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
      v = np.fft.irfftn(main.vhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
      w = np.fft.irfftn(main.what)*np.sqrt(grid.N1*grid.N2*grid.N3)
      max_vel = np.amax( abs(u)/grid.dx + abs(v)/grid.dy + abs(w)/grid.dz)
      main.dt = main.cfl/(max_vel + 1e-10)
      #CFL = c * dt / dx  -> dt = CFL*dx/c
      if (main.nu > 0):
        main.dt=np.minimum(main.dt,0.634/(1./grid.dx**2+1./grid.dy**2+1./grid.dz**2)/main.nu*main.cfl/1.35)
    else:
      main.dt = -main.cfl


  def computeAllStats(self,main,grid):
      enstrophy = self.computeEnstrophy(main,grid)
      energy = self.computeEnergy(main,grid)
      dissipation = 2*enstrophy*main.nu
      lambda_k = (main.nu**3/dissipation)**0.25
      tau_k = (main.nu/dissipation)**0.5
#      Re_lambda = energy*np.sqrt(20./(3.*main.nu*dissipation))
#      uprime = main.uhat - np.mean(main.uhat)
#      vprime = main.vhat - np.mean(main.vhat)
#      wprime = main.what - np.mean(main.what)
#      Vprime_RMS = np.mean(np.sqrt(uprime*uprime + vprime*vprime + wprime*wprime))
#      lam = np.sqrt(15.*main.nu/dissipation)*Vprime_RMS
#      ux = 1j*grid.k1*main.uhat
#      uxreal = np.fft.irfftn(ux)*np.sqrt(grid.N1*grid.N2*grid.N3)
#      ureal = np.fft.irfftn(main.uhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
#      vreal = np.fft.irfftn(main.vhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
#      wreal = np.fft.irfftn(main.what)*np.sqrt(grid.N1*grid.N2*grid.N3)
#      ureal = np.fft.irfftn(main.uhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
#      ureal = np.fft.irfftn(main.uhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
#      lam = (np.mean(ureal*ureal) / np.mean(uxreal*uxreal) )**0.5
#      Vprime_RMS = np.mean( np.sqrt( (ureal - np.mean(ureal) )**2 + \
#                                     (vreal - np.mean(vreal) )**2 + \
#                                     (wreal - np.mean(wreal) )**2 ) )
 #     uxruxr = uxreal*uxreal
 #     ux2hat = np.mean(np.fft.rfftn(uxruxr)/np.sqrt(grid.N1*grid.N2*grid.N3))
 #     lam = dissipation/(15.*main.nu*ux2hat)
      #k = 0.5*Vprime_RMS
      lam = np.sqrt(10.*main.nu*np.sqrt(grid.ksqr)/dissipation) 
      #Re_lambda = Vprime_RMS*lam/main.nu#*grid.N1*grid.N2*grid.N3
      Re_lambda = energy*np.sqrt(20./(3.*main.nu*dissipation))
      return enstrophy,energy,dissipation,lambda_k,tau_k,Re_lambda

  def computeEnstrophy(self,main,grid):
      omega1 = 1j*grid.k2*main.what - 1j*grid.k3*main.vhat
      omega2 = 1j*grid.k3*main.uhat - 1j*grid.k1*main.what
      omega3 = 1j*grid.k1*main.vhat - 1j*grid.k2*main.uhat
      om1E = np.sum(omega1[:,:,1:grid.N3/2]*np.conj(omega1[:,:,1:grid.N3/2]*2) ) + \
           np.sum(omega1[:,:,0]*np.conj(omega1[:,:,0])) 
      om2E = np.sum(omega2[:,:,1:grid.N3/2]*np.conj(omega2[:,:,1:grid.N3/2]*2) ) + \
           np.sum(omega2[:,:,0]*np.conj(omega2[:,:,0])) 
      om3E = np.sum(omega3[:,:,1:grid.N3/2]*np.conj(omega3[:,:,1:grid.N3/2]*2) ) + \
           np.sum(omega3[:,:,0]*np.conj(omega3[:,:,0]))
      enstrophy = np.real(0.5*(om1E + om2E + om3E)/(grid.N1*grid.N2*grid.N3))
      return enstrophy 

  def computeEnstrophy_2D(self,main,grid):
      omega = 1j*grid.k1*main.vhat - 1j*grid.k2*main.uhat
      om1E = np.sum(omega[:,:,0]*np.conj(omega[:,:,0]) )  
      enstrophy = np.real(om1E)/(grid.N1*grid.N2*grid.N3)
      return enstrophy 

  def computeEnstrophy_resolved_2D(self,main,grid):
      uFilt = grid.Gf*main.uhat
      vFilt = grid.Gf*main.vhat
      omega = 1j*grid.k1*vFilt - 1j*grid.k2*uFilt
      om1E = np.sum(omega[:,:,0]*np.conj(omega[:,:,0]) )  
      enstrophy = np.real(om1E)/(grid.N1*grid.N2*grid.N3)
      return enstrophy



  def computeSpectrum(self,main,grid):

      ksqr = (grid.k1*grid.L1/(2.*np.pi))**2 + (grid.k2*grid.L2/(2.*np.pi))**2 + (grid.k3*grid.L3/(2.*np.pi))**2
      k_m, indices1 = np.unique((np.rint(np.sqrt(ksqr[:,:,1:grid.N3/2].flatten()))), return_inverse=True)
      k_0, indices2 = np.unique((np.rint(np.sqrt(ksqr[:,:,0].flatten()))), return_inverse=True)
#      k_m, indices1 = np.unique(np.rint(np.sqrt(grid.ksqr[:,:,:].flatten())), return_inverse=True)
      kmax = np.int(np.round(np.amax(k_m)))
      kdata = np.linspace(0,kmax,kmax+1)
      spectrum = np.zeros((kmax+1,3),dtype='complex')
      spectrum2 = np.zeros((kmax+1,3),dtype='complex')
      np.add.at( spectrum[:,0],np.int8(k_m[indices1]),2*main.uhat[:,:,1:grid.N3/2].flatten()*np.conj(main.uhat[:,:,1:grid.N3/2].flatten()))
      np.add.at( spectrum[:,0],np.int8(k_0[indices2]),  main.uhat[:,:,0].flatten()*          np.conj(main.uhat[:,:,0].flatten()))
      np.add.at( spectrum[:,1],np.int8(k_m[indices1]),2*main.vhat[:,:,1:grid.N3/2].flatten()*np.conj(main.vhat[:,:,1:grid.N3/2].flatten()))
      np.add.at( spectrum[:,1],np.int8(k_0[indices2]),main.vhat[:,:,0].flatten()*np.conj(main.vhat[:,:,0].flatten()))
      np.add.at( spectrum[:,2],np.int8(k_m[indices1]),2*main.what[:,:,1:grid.N3/2].flatten()*np.conj(main.what[:,:,1:grid.N3/2].flatten()))
      np.add.at( spectrum[:,2],np.int8(k_0[indices2]),main.what[:,:,0].flatten()*np.conj(main.what[:,:,0].flatten()))
      #for i in range(0,grid.N1):
      #  for j in range(0,grid.N2):
      #    for k in range(1,grid.N3/2):
      #      kmag = np.sqrt(grid.k1[i,j,k]**2 + grid.k2[i,j,k]**2 + grid.k3[i,j,k]**2)
      #      kint = int(np.round(kmag))
      #      spectrum[kint,0] += 2.*main.uhat[i,j,k]*np.conj(main.uhat[i,j,k])
      #      spectrum[kint,1] += 2.*main.vhat[i,j,k]*np.conj(main.vhat[i,j,k])
      #      spectrum[kint,2] += 2.*main.what[i,j,k]*np.conj(main.what[i,j,k])
      #k = 0
      #for i in range(0,grid.N1):
      #    for j in range(0,grid.N2):
      #      kmag = np.sqrt(grid.k1[i,j,k]**2 + grid.k2[i,j,k]**2 + grid.k3[i,j,k]**2)
      #      kint = int(np.round(kmag))
      #      spectrum[kint,0] += main.uhat[i,j,k]*np.conj(main.uhat[i,j,k])
      #      spectrum[kint,1] += main.vhat[i,j,k]*np.conj(main.vhat[i,j,k])
      #      spectrum[kint,2] += main.what[i,j,k]*np.conj(main.what[i,j,k])

      spectrum = spectrum/(grid.N1*grid.N2*grid.N3)
     
      return kdata,spectrum

  def computeEnstrophySpectrum_2D(self,main,grid):
      kmag = np.sqrt( grid.k1**2 + grid.k2**2 )
      k_m, indices1 = np.unique((np.rint(kmag[:,:,0].flatten())), return_inverse=True)
      kmax = np.int(np.round(np.amax(k_m)))
      kdata = np.linspace(0,kmax,kmax+1)
      spectrum = np.zeros((kmax+1),dtype='complex')
      omega = 1j*grid.k1*main.vhat - 1j*grid.k2*main.uhat
      np.add.at( spectrum[:],np.int8(k_m[indices1]),omega[:,:,0].flatten()*np.conj(omega[:,:,0].flatten()))
      spectrum = spectrum/(grid.N1*grid.N2*grid.N3)
      return kdata,spectrum



  def computeSpectrum_2D(self,main,grid):
      kmag = np.sqrt( grid.k1**2 + grid.k2**2 )
      k_m, indices1 = np.unique((np.rint(kmag[:,:,0].flatten())), return_inverse=True)
      kmax = np.int(np.round(np.amax(k_m)))
      kdata = np.linspace(0,kmax,kmax+1)
      spectrum = np.zeros((kmax+1,3),dtype='complex')
      spectrum2 = np.zeros((kmax+1,3),dtype='complex')
      np.add.at( spectrum[:,0],np.int8(k_m[indices1]),main.uhat[:,:,0].flatten()*np.conj(main.uhat[:,:,0].flatten()))
      np.add.at( spectrum[:,1],np.int8(k_m[indices1]),main.vhat[:,:,0].flatten()*np.conj(main.vhat[:,:,0].flatten()))
      spectrum = spectrum/(grid.N1*grid.N2*grid.N3)
      return kdata,spectrum


  def computeSpectrum_resolved(self,main,grid):
      ksqr = (grid.k1*grid.L1/(2.*np.pi))**2 + (grid.k2*grid.L2/(2.*np.pi))**2 + (grid.k3*grid.L3/(2.*np.pi))**2
      k_m, indices1 = np.unique((np.rint(np.sqrt(ksqr[:,:,1:grid.N3/2].flatten()))), return_inverse=True)
      k_0, indices2 = np.unique((np.rint(np.sqrt(ksqr[:,:,0].flatten()))), return_inverse=True)
#      k_m, indices1 = np.unique(np.rint(np.sqrt(grid.ksqr[:,:,:].flatten())), return_inverse=True)
      kmax = np.int(np.round(np.amax(k_m)))
      kdata = np.linspace(0,kmax,kmax+1)
      spectrum = np.zeros((kmax+1,3),dtype='complex')
      spectrum2 = np.zeros((kmax+1,3),dtype='complex')
      uFilt = grid.Gf*main.uhat
      vFilt = grid.Gf*main.vhat
      wFilt = grid.Gf*main.what
      np.add.at( spectrum[:,0],np.int8(k_m[indices1]),2*uFilt[:,:,1:grid.N3/2].flatten()*np.conj(uFilt[:,:,1:grid.N3/2].flatten()))
      np.add.at( spectrum[:,0],np.int8(k_0[indices2]),  uFilt[:,:,0].flatten()*          np.conj(uFilt[:,:,0].flatten()))
      np.add.at( spectrum[:,1],np.int8(k_m[indices1]),2*vFilt[:,:,1:grid.N3/2].flatten()*np.conj(vFilt[:,:,1:grid.N3/2].flatten()))
      np.add.at( spectrum[:,1],np.int8(k_0[indices2]),  vFilt[:,:,0].flatten()*          np.conj(vFilt[:,:,0].flatten()))
      np.add.at( spectrum[:,2],np.int8(k_m[indices1]),2*wFilt[:,:,1:grid.N3/2].flatten()*np.conj(wFilt[:,:,1:grid.N3/2].flatten()))
      np.add.at( spectrum[:,2],np.int8(k_0[indices2]),wFilt[:,:,0].flatten()*np.conj(wFilt[:,:,0].flatten()))
      spectrum = spectrum/(grid.N1*grid.N2*grid.N3)
      return k_m,spectrum 

  def computeSpectrum_resolved_2D(self,main,grid):
      kmag = np.sqrt( grid.k1**2 + grid.k2**2 )
      k_m, indices1 = np.unique((np.rint(kmag[:,:,0].flatten())), return_inverse=True)
      kmax = np.int(np.round(np.amax(k_m)))
      kdata = np.linspace(0,kmax,kmax+1)
      spectrum = np.zeros((kmax+1,3),dtype='complex')
      spectrum2 = np.zeros((kmax+1,3),dtype='complex')
      uFilt = grid.Gf*main.uhat
      vFilt = grid.Gf*main.vhat
      np.add.at( spectrum[:,0],np.int8(k_m[indices1]),uFilt[:,:,0].flatten()*np.conj(uFilt[:,:,0].flatten()))
      np.add.at( spectrum[:,1],np.int8(k_m[indices1]),vFilt[:,:,0].flatten()*np.conj(vFilt[:,:,0].flatten()))
      spectrum = spectrum/(grid.N1*grid.N2*grid.N3)
      return k_m,spectrum 


  def computeSGS_DNS(self,main,grid,myFFT):
    N1,N2,N3 = grid.N1,grid.N2,grid.N3
    ufilt = grid.Gf*main.uhat
    vfilt = grid.Gf*main.vhat
    wfilt = grid.Gf*main.what
    scale = np.sqrt( (3./2.)**3*np.sqrt(grid.N1*grid.N2*grid.N3) )

    ureal = np.zeros((N1*3/2,N2*3/2,N3*3/2))
    vreal = np.zeros((N1*3/2,N2*3/2,N3*3/2))
    wreal = np.zeros((N1*3/2,N2*3/2,N3*3/2))
    ureal_filt = np.zeros((N1*3/2,N2*3/2,N3*3/2))
    vreal_filt = np.zeros((N1*3/2,N2*3/2,N3*3/2))
    wreal_filt = np.zeros((N1*3/2,N2*3/2,N3*3/2))
    uuhat = np.zeros((N1,N2,(N3/2+1)),dtype='complex')
    vvhat = np.zeros((N1,N2,(N3/2+1)),dtype='complex')
    wwhat = np.zeros((N1,N2,(N3/2+1)),dtype='complex')
    uvhat = np.zeros((N1,N2,(N3/2+1)),dtype='complex')
    uwhat = np.zeros((N1,N2,(N3/2+1)),dtype='complex')
    vwhat = np.zeros((N1,N2,(N3/2+1)),dtype='complex')
    uuhat_filt = np.zeros((N1,N2,(N3/2+1)),dtype='complex')
    vvhat_filt = np.zeros((N1,N2,(N3/2+1)),dtype='complex')
    wwhat_filt = np.zeros((N1,N2,(N3/2+1)),dtype='complex')
    uvhat_filt = np.zeros((N1,N2,(N3/2+1)),dtype='complex')
    uwhat_filt = np.zeros((N1,N2,(N3/2+1)),dtype='complex')
    vwhat_filt = np.zeros((N1,N2,(N3/2+1)),dtype='complex')

    ureal[:,:,:] = myFFT.ifftT_obj(pad(main.uhat,1)*scale)
    vreal[:,:,:] = myFFT.ifftT_obj(pad(main.vhat,1)*scale)
    wreal[:,:,:]  = myFFT.ifftT_obj(pad(main.what,1)*scale)
    ureal_filt[:,:,:] = myFFT.ifftT_obj(pad(ufilt,1)*scale)
    vreal_filt[:,:,:] = myFFT.ifftT_obj(pad(vfilt,1)*scale)
    wreal_filt[:,:,:] = myFFT.ifftT_obj(pad(wfilt,1)*scale)
 
    uuhat[:,:,:] = unpad( myFFT.fft_obj(ureal*ureal),1)
    vvhat[:,:,:] = unpad( myFFT.fft_obj(vreal*vreal),1)
    wwhat[:,:,:] = unpad( myFFT.fft_obj(wreal*wreal),1)
    uvhat[:,:,:] = unpad( myFFT.fft_obj(ureal*vreal),1)
    uwhat[:,:,:] = unpad( myFFT.fft_obj(ureal*wreal),1)
    vwhat[:,:,:] = unpad( myFFT.fft_obj(vreal*wreal),1)
    uuhat_filt[:,:,:] = unpad( myFFT.fft_obj(ureal_filt*ureal_filt),1)
    vvhat_filt[:,:,:] = unpad( myFFT.fft_obj(vreal_filt*vreal_filt),1)
    wwhat_filt[:,:,:] = unpad( myFFT.fft_obj(wreal_filt*wreal_filt),1)
    uvhat_filt[:,:,:] = unpad( myFFT.fft_obj(ureal_filt*vreal_filt),1)
    uwhat_filt[:,:,:] = unpad( myFFT.fft_obj(ureal_filt*wreal_filt),1)
    vwhat_filt[:,:,:] = unpad( myFFT.fft_obj(vreal_filt*wreal_filt),1)
  
    ## Compute SGS tensor. 
    # tau_ij = [d/dx 
    tauhat = np.zeros((N1,N2,(N3/2+1),6),dtype='complex')
    tauhat[:,:,:,0] = grid.Gf*uuhat - uuhat_filt
    tauhat[:,:,:,1] = grid.Gf*vvhat - vvhat_filt
    tauhat[:,:,:,2] = grid.Gf*wwhat - wwhat_filt
    tauhat[:,:,:,3] = grid.Gf*uvhat - uvhat_filt
    tauhat[:,:,:,4] = grid.Gf*uwhat - uwhat_filt
    tauhat[:,:,:,5] = grid.Gf*vwhat - vwhat_filt
 
    ## contribution of projection to RHS sgs (k_m k_j)/k^2 \tau_{jm}
    tau_projection  = 1j*grid.ksqr_i*( grid.k1*grid.k1*tauhat[:,:,:,0] + grid.k2*grid.k2*tauhat[:,:,:,1] + \
             grid.k3*grid.k3*tauhat[:,:,:,2] + 2.*grid.k1*grid.k2*tauhat[:,:,:,3] + \
             2.*grid.k1*grid.k3*tauhat[:,:,:,4] + 2.*grid.k2*grid.k3*tauhat[:,:,:,5] )

    w0_u = np.zeros((N1,N2,N3/2+1,1),dtype='complex')
    w0_v = np.zeros((N1,N2,N3/2+1,1),dtype='complex')
    w0_w = np.zeros((N1,N2,N3/2+1,1),dtype='complex')

    w0_u[:,:,:,0] = -(1j*grid.k1*tauhat[:,:,:,0] + 1j*grid.k2*tauhat[:,:,:,3] + 1j*grid.k3*tauhat[:,:,:,4]) + 1j*grid.k1*tau_projection
    w0_v[:,:,:,0] = -(1j*grid.k1*tauhat[:,:,:,3] + 1j*grid.k2*tauhat[:,:,:,1] + 1j*grid.k3*tauhat[:,:,:,5]) + 1j*grid.k2*tau_projection
    w0_w[:,:,:,0] = -(1j*grid.k1*tauhat[:,:,:,4] + 1j*grid.k2*tauhat[:,:,:,5] + 1j*grid.k3*tauhat[:,:,:,2]) + 1j*grid.k3*tau_projection
    return w0_u,w0_v,w0_w
    #return tauhat 

  def computeSGS_DNS_2D(self,main,grid,myFFT):
    N1,N2,N3 = grid.N1,grid.N2,grid.N3
    ufilt = grid.Gf*main.uhat
    vfilt = grid.Gf*main.vhat
    scale = np.sqrt( (3./2.)**3*np.sqrt(grid.N1*grid.N2*grid.N3) )

    ureal = np.zeros((N1*3/2,N2*3/2,N3*3/2))
    vreal = np.zeros((N1*3/2,N2*3/2,N3*3/2))
    ureal_filt = np.zeros((N1*3/2,N2*3/2,N3*3/2))
    vreal_filt = np.zeros((N1*3/2,N2*3/2,N3*3/2))
    uuhat = np.zeros((N1,N2,(N3/2+1)),dtype='complex')
    vvhat = np.zeros((N1,N2,(N3/2+1)),dtype='complex')
    uvhat = np.zeros((N1,N2,(N3/2+1)),dtype='complex')
    uuhat_filt = np.zeros((N1,N2,(N3/2+1)),dtype='complex')
    vvhat_filt = np.zeros((N1,N2,(N3/2+1)),dtype='complex')
    uvhat_filt = np.zeros((N1,N2,(N3/2+1)),dtype='complex')

    ureal[:,:,:] = myFFT.ifftT_obj(pad(main.uhat,1)*scale)
    vreal[:,:,:] = myFFT.ifftT_obj(pad(main.vhat,1)*scale)
    ureal_filt[:,:,:] = myFFT.ifftT_obj(pad(ufilt,1)*scale)
    vreal_filt[:,:,:] = myFFT.ifftT_obj(pad(vfilt,1)*scale)
 
    uuhat[:,:,:] = unpad( myFFT.fft_obj(ureal*ureal),1)
    vvhat[:,:,:] = unpad( myFFT.fft_obj(vreal*vreal),1)
    uvhat[:,:,:] = unpad( myFFT.fft_obj(ureal*vreal),1)
    uuhat_filt[:,:,:] = unpad( myFFT.fft_obj(ureal_filt*ureal_filt),1)
    vvhat_filt[:,:,:] = unpad( myFFT.fft_obj(vreal_filt*vreal_filt),1)
    uvhat_filt[:,:,:] = unpad( myFFT.fft_obj(ureal_filt*vreal_filt),1)
  
    ## Compute SGS tensor. 
    # tau_ij = [d/dx 
    tauhat = np.zeros((N1,N2,(N3/2+1),6),dtype='complex')
    tauhat[:,:,:,0] = grid.Gf*uuhat - uuhat_filt
    tauhat[:,:,:,1] = grid.Gf*vvhat - vvhat_filt
    tauhat[:,:,:,2] = 0.
    tauhat[:,:,:,3] = grid.Gf*uvhat - uvhat_filt
    tauhat[:,:,:,4] = 0.
    tauhat[:,:,:,5] = 0.
 
    ## contribution of projection to RHS sgs (k_m k_j)/k^2 \tau_{jm}
    tau_projection  = 1j*grid.ksqr_i*( grid.k1*grid.k1*tauhat[:,:,:,0] + grid.k2*grid.k2*tauhat[:,:,:,1] + \
             grid.k3*grid.k3*tauhat[:,:,:,2] + 2.*grid.k1*grid.k2*tauhat[:,:,:,3] + \
             2.*grid.k1*grid.k3*tauhat[:,:,:,4] + 2.*grid.k2*grid.k3*tauhat[:,:,:,5] )

    w0_u = np.zeros((N1,N2,N3/2+1,1),dtype='complex')
    w0_v = np.zeros((N1,N2,N3/2+1,1),dtype='complex')
    w0_w = np.zeros((N1,N2,N3/2+1,1),dtype='complex')

    w0_u[:,:,:,0] = -(1j*grid.k1*tauhat[:,:,:,0] + 1j*grid.k2*tauhat[:,:,:,3] + 1j*grid.k3*tauhat[:,:,:,4]) + 1j*grid.k1*tau_projection
    w0_v[:,:,:,0] = -(1j*grid.k1*tauhat[:,:,:,3] + 1j*grid.k2*tauhat[:,:,:,1] + 1j*grid.k3*tauhat[:,:,:,5]) + 1j*grid.k2*tau_projection
    w0_w[:,:,:,0] = -(1j*grid.k1*tauhat[:,:,:,4] + 1j*grid.k2*tauhat[:,:,:,5] + 1j*grid.k3*tauhat[:,:,:,2]) + 1j*grid.k3*tau_projection
    return w0_u,w0_v,w0_w




  def computeTransfer(self,main,grid,myFFT):
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

    phat  = grid.ksqr_i*( -grid.k1*grid.k1*uuhat - grid.k2*grid.k2*vvhat - \
           grid.k3*grid.k3*wwhat - 2.*grid.k1*grid.k2*uvhat - \
           2.*grid.k1*grid.k3*uwhat - 2.*grid.k2*grid.k3*vwhat )

    RHSu = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    RHSv = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    RHSw = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')


    RHSu[:,:,:] = np.conj(main.uhat)*(-1j*grid.k1*uuhat - 1j*grid.k2*uvhat - 1j*grid.k3*uwhat - \
                                       1j*grid.k1*phat )

    RHSv[:,:,:] = np.conj(main.vhat)*(-1j*grid.k1*uvhat - 1j*grid.k2*vvhat - 1j*grid.k3*vwhat - \
                                       1j*grid.k2*phat)

    RHSw[:,:,:] = np.conj(main.what)*(-1j*grid.k1*uwhat - 1j*grid.k2*vwhat - 1j*grid.k3*wwhat - \
                                      1j*grid.k3*phat)

    ksqr = (grid.k1*grid.L1/(2.*np.pi))**2 + (grid.k2*grid.L2/(2.*np.pi))**2 + (grid.k3*grid.L3/(2.*np.pi))**2
    k_m, indices1 = np.unique((np.rint(np.sqrt(ksqr[:,:,1:grid.N3/2].flatten()))), return_inverse=True)
    k_0, indices2 = np.unique((np.rint(np.sqrt(ksqr[:,:,0].flatten()))), return_inverse=True)
    kmax = np.int(np.round(np.amax(k_m)))
    kdata = np.linspace(0,kmax,kmax+1)
    spectrum = np.zeros((kmax+1,3),dtype='complex')
    spectrum2 = np.zeros((kmax+1,3),dtype='complex')
    np.add.at( spectrum[:,0],np.int8(k_m[indices1]),2*RHSu[:,:,1:grid.N3/2].flatten())
    np.add.at( spectrum[:,0],np.int8(k_0[indices2]),  RHSu[:,:,0].flatten())
    np.add.at( spectrum[:,1],np.int8(k_m[indices1]),2*RHSv[:,:,1:grid.N3/2].flatten())
    np.add.at( spectrum[:,1],np.int8(k_0[indices2]),  RHSv[:,:,0].flatten())
    np.add.at( spectrum[:,2],np.int8(k_m[indices1]),2*RHSw[:,:,1:grid.N3/2].flatten())
    np.add.at( spectrum[:,2],np.int8(k_0[indices2]),  RHSw[:,:,0].flatten())
    Transfer = (spectrum[:,0] + spectrum[:,1] + spectrum[:,2] ) / (grid.N1*grid.N2*grid.N3)
    return kdata,Transfer 

  def computeTransfer_2D(self,main,grid,myFFT):
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

    phat  = grid.ksqr_i*( -grid.k1*grid.k1*uuhat - grid.k2*grid.k2*vvhat - \
           grid.k3*grid.k3*wwhat - 2.*grid.k1*grid.k2*uvhat - \
           2.*grid.k1*grid.k3*uwhat - 2.*grid.k2*grid.k3*vwhat )

    RHSu = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    RHSv = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    RHSw = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')


    RHSu[:,:,:] = np.conj(main.uhat)*(-1j*grid.k1*uuhat - 1j*grid.k2*uvhat - 1j*grid.k3*uwhat - \
                                       1j*grid.k1*phat )

    RHSv[:,:,:] = np.conj(main.vhat)*(-1j*grid.k1*uvhat - 1j*grid.k2*vvhat - 1j*grid.k3*vwhat - \
                                       1j*grid.k2*phat)

    RHSw[:,:,:] = np.conj(main.what)*(-1j*grid.k1*uwhat - 1j*grid.k2*vwhat - 1j*grid.k3*wwhat - \
                                      1j*grid.k3*phat)
    kmag = np.sqrt(grid.k1**2 + grid.k2**2)
    k_m, indices1 = np.unique((np.rint(kmag[:,:,0].flatten())), return_inverse=True)
    kmax = np.int(np.round(np.amax(k_m)))
    kdata = np.linspace(0,kmax,kmax+1)
    spectrum = np.zeros((kmax+1,3),dtype='complex')
    np.add.at( spectrum[:,0],np.int8(k_m[indices1]),RHSu[:,:,0].flatten())
    np.add.at( spectrum[:,1],np.int8(k_m[indices1]),RHSv[:,:,0].flatten())
    Transfer = (spectrum[:,0] + spectrum[:,1] + spectrum[:,2] ) / (grid.N1*grid.N2*grid.N3)
    return kdata,Transfer 



  def computeTransfer_resolved(self,main,grid,myFFT):
    scale = np.sqrt( (3./2.)**3*np.sqrt(grid.N1*grid.N2*grid.N3) )
    ureal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    vreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    wreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    
    uFilt = grid.Gf*main.uhat
    vFilt = grid.Gf*main.vhat
    wFilt = grid.Gf*main.what

    uFilt = unpad(pad(uFilt,1),1)
    vFilt = unpad(pad(vFilt,1),1)
    wFilt = unpad(pad(wFilt,1),1)

    ureal[:,:,:] = myFFT.ifftT_obj(pad(uFilt,1))*scale
    vreal[:,:,:] = myFFT.ifftT_obj(pad(vFilt,1))*scale
    wreal[:,:,:] = myFFT.ifftT_obj(pad(wFilt,1))*scale

    uuhat = unpad( myFFT.fft_obj(ureal*ureal),1)
    vvhat = unpad( myFFT.fft_obj(vreal*vreal),1)
    wwhat = unpad( myFFT.fft_obj(wreal*wreal),1)
    uvhat = unpad( myFFT.fft_obj(ureal*vreal),1)
    uwhat = unpad( myFFT.fft_obj(ureal*wreal),1)
    vwhat = unpad( myFFT.fft_obj(vreal*wreal),1)

    phat  = grid.ksqr_i*( -grid.k1*grid.k1*uuhat - grid.k2*grid.k2*vvhat - \
           grid.k3*grid.k3*wwhat - 2.*grid.k1*grid.k2*uvhat - \
           2.*grid.k1*grid.k3*uwhat - 2.*grid.k2*grid.k3*vwhat )

    RHSu = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    RHSv = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    RHSw = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')


    RHSu[:,:,:] = np.conj(uFilt)*(-1j*grid.k1*uuhat - 1j*grid.k2*uvhat - 1j*grid.k3*uwhat - \
                                       1j*grid.k1*phat )

    RHSv[:,:,:] = np.conj(vFilt)*(-1j*grid.k1*uvhat - 1j*grid.k2*vvhat - 1j*grid.k3*vwhat - \
                                       1j*grid.k2*phat)

    RHSw[:,:,:] = np.conj(wFilt)*(-1j*grid.k1*uwhat - 1j*grid.k2*vwhat - 1j*grid.k3*wwhat - \
                                      1j*grid.k3*phat)
    ksqr = (grid.k1*grid.L1/(2.*np.pi))**2 + (grid.k2*grid.L2/(2.*np.pi))**2 + (grid.k3*grid.L3/(2.*np.pi))**2
    k_m, indices1 = np.unique((np.rint(np.sqrt(ksqr[:,:,1:grid.N3/2].flatten()))), return_inverse=True)
    k_0, indices2 = np.unique((np.rint(np.sqrt(ksqr[:,:,0].flatten()))), return_inverse=True)
    kmax = np.int(np.round(np.amax(k_m)))
    kdata = np.linspace(0,kmax,kmax+1)
    spectrum = np.zeros((kmax+1,3),dtype='complex')
    spectrum2 = np.zeros((kmax+1,3),dtype='complex')
    np.add.at( spectrum[:,0],np.int8(k_m[indices1]),2*RHSu[:,:,1:grid.N3/2].flatten())
    np.add.at( spectrum[:,0],np.int8(k_0[indices2]),  RHSu[:,:,0].flatten())
    np.add.at( spectrum[:,1],np.int8(k_m[indices1]),2*RHSv[:,:,1:grid.N3/2].flatten())
    np.add.at( spectrum[:,1],np.int8(k_0[indices2]),  RHSv[:,:,0].flatten())
    np.add.at( spectrum[:,2],np.int8(k_m[indices1]),2*RHSw[:,:,1:grid.N3/2].flatten())
    np.add.at( spectrum[:,2],np.int8(k_0[indices2]),  RHSw[:,:,0].flatten())
    Transfer = (spectrum[:,0] + spectrum[:,1] + spectrum[:,2] ) / (grid.N1*grid.N2*grid.N3)
    return kdata,Transfer 


  def computeTransfer_resolved_2D(self,main,grid,myFFT):
    scale = np.sqrt( (3./2.)**3*np.sqrt(grid.N1*grid.N2*grid.N3) )
    ureal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    vreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    wreal = np.zeros( (int(3./2.*grid.N1),int(3./2.*grid.N2),int(3./2.*grid.N3)) )
    
    uFilt = grid.Gf*main.uhat
    vFilt = grid.Gf*main.vhat
    wFilt = grid.Gf*main.what

    uFilt = unpad(pad(uFilt,1),1)
    vFilt = unpad(pad(vFilt,1),1)
    wFilt = unpad(pad(wFilt,1),1)

    ureal[:,:,:] = myFFT.ifftT_obj(pad(uFilt,1))*scale
    vreal[:,:,:] = myFFT.ifftT_obj(pad(vFilt,1))*scale
    wreal[:,:,:] = myFFT.ifftT_obj(pad(wFilt,1))*scale

    uuhat = unpad( myFFT.fft_obj(ureal*ureal),1)
    vvhat = unpad( myFFT.fft_obj(vreal*vreal),1)
    wwhat = unpad( myFFT.fft_obj(wreal*wreal),1)
    uvhat = unpad( myFFT.fft_obj(ureal*vreal),1)
    uwhat = unpad( myFFT.fft_obj(ureal*wreal),1)
    vwhat = unpad( myFFT.fft_obj(vreal*wreal),1)

    phat  = grid.ksqr_i*( -grid.k1*grid.k1*uuhat - grid.k2*grid.k2*vvhat - \
           grid.k3*grid.k3*wwhat - 2.*grid.k1*grid.k2*uvhat - \
           2.*grid.k1*grid.k3*uwhat - 2.*grid.k2*grid.k3*vwhat )

    RHSu = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    RHSv = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    RHSw = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')


    RHSu[:,:,:] = np.conj(uFilt)*(-1j*grid.k1*uuhat - 1j*grid.k2*uvhat - 1j*grid.k3*uwhat - \
                                       1j*grid.k1*phat )

    RHSv[:,:,:] = np.conj(vFilt)*(-1j*grid.k1*uvhat - 1j*grid.k2*vvhat - 1j*grid.k3*vwhat - \
                                       1j*grid.k2*phat)

    RHSw[:,:,:] = np.conj(wFilt)*(-1j*grid.k1*uwhat - 1j*grid.k2*vwhat - 1j*grid.k3*wwhat - \
                                      1j*grid.k3*phat)

    ksqr2d = grid.k1**2 + grid.k2**2
    k_m, indices1 = np.unique((np.rint(np.sqrt(ksqr2d[:,:,0].flatten()))), return_inverse=True)
    kmax = np.int(np.round(np.amax(k_m)))
    kdata = np.linspace(0,kmax,kmax+1)
    spectrum = np.zeros((kmax+1,3),dtype='complex')
    np.add.at( spectrum[:,0],np.int8(k_m[indices1]),RHSu[:,:,0].flatten())
    np.add.at( spectrum[:,1],np.int8(k_m[indices1]),RHSv[:,:,0].flatten())
    Transfer = (spectrum[:,0] + spectrum[:,1] + spectrum[:,2] ) / (grid.N1*grid.N2*grid.N3)
    return kdata,Transfer 


  def computeTransfer_SGS(self,main,grid,myFFT):
    RHSu = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    RHSv = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    RHSw = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')

    uFilt = grid.Gf*main.uhat
    vFilt = grid.Gf*main.vhat
    wFilt = grid.Gf*main.what

    uFilt = unpad(pad(uFilt,1),1)
    vFilt = unpad(pad(vFilt,1),1)
    wFilt = unpad(pad(wFilt,1),1)

    RHSu[:,:,:] =  np.conj(uFilt)*main.w0_u[:,:,:,0] 

    RHSv[:,:,:] =  np.conj(vFilt)*main.w0_v[:,:,:,0]

    RHSw[:,:,:] =  np.conj(wFilt)*main.w0_w[:,:,:,0]

    ksqr = (grid.k1*grid.L1/(2.*np.pi))**2 + (grid.k2*grid.L2/(2.*np.pi))**2 + (grid.k3*grid.L3/(2.*np.pi))**2
    k_m, indices1 = np.unique((np.rint(np.sqrt(ksqr[:,:,1:grid.N3/2].flatten()))), return_inverse=True)
    k_0, indices2 = np.unique((np.rint(np.sqrt(ksqr[:,:,0].flatten()))), return_inverse=True)
    kmax = np.int(np.round(np.amax(k_m)))
    kdata = np.linspace(0,kmax,kmax+1)
    spectrum = np.zeros((kmax+1,3),dtype='complex')
    spectrum2 = np.zeros((kmax+1,3),dtype='complex')
    np.add.at( spectrum[:,0],np.int8(k_m[indices1]),np.real(2*RHSu[:,:,1:grid.N3/2].flatten()))
    np.add.at( spectrum[:,0],np.int8(k_0[indices2]),np.real(  RHSu[:,:,0].flatten()))
    np.add.at( spectrum[:,1],np.int8(k_m[indices1]),np.real(2*RHSv[:,:,1:grid.N3/2].flatten()))
    np.add.at( spectrum[:,1],np.int8(k_0[indices2]),np.real(  RHSv[:,:,0].flatten()))
    np.add.at( spectrum[:,2],np.int8(k_m[indices1]),np.real(2*RHSw[:,:,1:grid.N3/2].flatten()))
    np.add.at( spectrum[:,2],np.int8(k_0[indices2]),np.real(  RHSw[:,:,0].flatten()))
    Transfer = (spectrum[:,0] + spectrum[:,1] + spectrum[:,2] ) / (grid.N1*grid.N2*grid.N3)
    return kdata,Transfer

  def computeTransfer_SGS_2D(self,main,grid,myFFT):
    RHSu = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    RHSv = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')
    RHSw = np.zeros((grid.N1,grid.N2,grid.N3/2+1),dtype='complex')

    uFilt = grid.Gf*main.uhat
    vFilt = grid.Gf*main.vhat
    wFilt = grid.Gf*main.what

    uFilt = unpad(pad(uFilt,1),1)
    vFilt = unpad(pad(vFilt,1),1)
    wFilt = unpad(pad(wFilt,1),1)

    RHSu[:,:,:] =  np.conj(uFilt)*main.w0_u[:,:,:,0] 

    RHSv[:,:,:] =  np.conj(vFilt)*main.w0_v[:,:,:,0]

    RHSw[:,:,:] =  np.conj(wFilt)*main.w0_w[:,:,:,0]
    ksqr2d = grid.k1**2 + grid.k2**2
    k_m, indices1 = np.unique((np.rint(np.sqrt(ksqr2d[:,:,0].flatten()))), return_inverse=True)
    kmax = np.int(np.round(np.amax(k_m)))
    kdata = np.linspace(0,kmax,kmax+1)
    spectrum = np.zeros((kmax+1,3),dtype='complex')
    np.add.at( spectrum[:,0],np.int8(k_m[indices1]),np.real(RHSu[:,:,0].flatten()))
    np.add.at( spectrum[:,1],np.int8(k_m[indices1]),np.real(RHSv[:,:,0].flatten()))
    Transfer = (spectrum[:,0] + spectrum[:,1] + spectrum[:,2] ) / (grid.N1*grid.N2*grid.N3)
    return kdata,Transfer



  def myFilter(self,uhat,kc):
    uhatF = np.zeros(np.shape(uhat),dtype='complex')
    uhatF[0:kc,0:kc,0:kc] = uhat[0:kc,0:kc,0:kc]

    uhatF[0:kc,0:kc,0:kc] = uhat[0:kc,0:kc,0:kc] # get first quardants
    uhatF[0:kc,-kc+1::,0:kc] = uhat[0:kc,-kc+1::,0:kc] #0:kc in k1 and -kc:0 in k2
    uhatF[-kc+1::,0:kc,0:kc] = uhat[-kc+1::,0:kc,0:kc] #-kc:0 in k1 and 0:kc in k2
    uhatF[-kc+1::,-kc+1::,0:kc] = uhat[-kc+1::,-kc+1::,0:kc] #-kc:0 in k1 and k2
    return uhatF 




  def compute_PLQLu(self,main,grid,myFFT):
    ## in the t-model, do 2x padding because we want to have convolutions where 
    ## the modes in G support twice the modes in F.
    scale = np.sqrt( (2.)**3*np.sqrt(grid.N1*grid.N2*grid.N3) )
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

    PLQLu_u =  -1j*grid.k1*up_PLuq - 1j*grid.k2*vp_PLuq - 1j*grid.k3*wp_PLuq - \
            1j*grid.k1*up_PLuq - 1j*grid.k2*up_PLvq - 1j*grid.k3*up_PLwq + \
            1j*grid.k1*pterm 

    PLQLu_v = -1j*grid.k1*up_PLvq - 1j*grid.k2*vp_PLvq - 1j*grid.k3*wp_PLvq - \
            1j*grid.k1*vp_PLuq - 1j*grid.k2*vp_PLvq - 1j*grid.k3*vp_PLwq + \
            1j*grid.k2*pterm 

    PLQLu_w =  -1j*grid.k1*up_PLwq - 1j*grid.k2*vp_PLwq - 1j*grid.k3*wp_PLwq -\
            1j*grid.k1*wp_PLuq - 1j*grid.k2*wp_PLvq - 1j*grid.k3*wp_PLwq + \
            1j*grid.k3*pterm 
    return PLQLu_u,PLQLu_v,PLQLu_w 


