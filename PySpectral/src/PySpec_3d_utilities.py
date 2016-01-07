import numpy as np
def dummy_function():
  pass

class utilitiesClass():
  def computeEnergy(self,uhat,vhat,what,grid):
      uE = np.sum(uhat[:,:,1:grid.N3/2]*np.conj(uhat[:,:,1:grid.N3/2]*2) ) + \
           np.sum(uhat[:,:,grid.N3/2]*np.conj(uhat[:,:,grid.N3/2])) + \
           np.sum(uhat[:,:,0]*np.conj(uhat[:,:,0])) 
      vE = np.sum(vhat[:,:,1:grid.N3/2]*np.conj(vhat[:,:,1:grid.N3/2]*2) ) + \
           np.sum(vhat[:,:,grid.N3/2]*np.conj(vhat[:,:,grid.N3/2])) + \
           np.sum(vhat[:,:,0]*np.conj(vhat[:,:,0])) 
      wE = np.sum(what[:,:,1:grid.N3/2]*np.conj(what[:,:,1:grid.N3/2]*2) ) + \
           np.sum(what[:,:,grid.N3/2]*np.conj(what[:,:,grid.N3/2])) + \
           np.sum(what[:,:,0]*np.conj(what[:,:,0]))
      return np.real(0.5*(uE + vE + wE)/(grid.N1*grid.N2*grid.N3))

