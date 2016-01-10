import numpy as np
from evtk.hl import gridToVTK
def savehook(main,grid,iteration):
    string = '3DSolution/PVsol' + str(iteration)
    string2 = '3DSolution/npsol' + str(iteration)
    u = np.fft.irfftn(main.uhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
    v = np.fft.irfftn(main.vhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
    w = np.fft.irfftn(main.what)*np.sqrt(grid.N1*grid.N2*grid.N3)
    gridToVTK(string, grid.x,grid.y,grid.z, pointData = {"u" : np.real(main.u.transpose()) , \
      "v" : np.real(main.v.transpose()), \
      "w" : np.real(main.w.transpose())} )
    np.savez_compressed(string2,uhat=main.uhat,vhat=main.vhat,what=main.what,t=main.t)

