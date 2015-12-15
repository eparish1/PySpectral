import numpy as np
from evtk.hl import gridToVTK
def savehook(uhat,vhat,what,grid,iteration):
    string = '3DSolution/PVsol' + str(iteration)
    string2 = '3DSolution/npsol' + str(iteration)
    u = np.fft.irfftn(uhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
    v = np.fft.irfftn(vhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
    w = np.fft.irfftn(what)*np.sqrt(grid.N1*grid.N2*grid.N3)
    gridToVTK(string, grid.x,grid.y,grid.z, pointData = {"u" : np.real(u.transpose()) , \
      "v" : np.real(v.transpose()), \
      "w" : np.real(w.transpose())} )
    np.savez_compressed(string2,uhat=uhat,vhat=vhat,what=what)

