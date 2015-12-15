import numpy as np
def TaylorGreenIC(grid):
  u =  np.cos(grid.x)*np.sin(grid.y)*np.cos(grid.z)
  v = -np.sin(grid.x)*np.cos(grid.y)*np.cos(grid.z)
  w =  np.zeros((grid.N1,grid.N2,grid.N3))
  uhat =  np.fft.rfftn(u) / np.sqrt(grid.N1*grid.N2*grid.N3) 
  vhat =  np.fft.rfftn(v) / np.sqrt(grid.N1*grid.N2*grid.N3)
  what =  np.fft.rfftn(w) / np.sqrt(grid.N1*grid.N2*grid.N3)
  return u,v,w,uhat,vhat,what
