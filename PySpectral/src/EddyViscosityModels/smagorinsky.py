import numpy as np
def smag_getnut(grid,S11real,S22real,S33real,S12real,S13real,S23real):
  S_magreal = np.sqrt( 2.*(S11real*S11real + S22real*S22real + S33real*S33real + \
             2.*S12real*S12real + 2.*S13real*S13real + 2.*S23real*S23real ) )
  nutreal = 0.16*grid.Delta*0.16*grid.Delta*np.abs(S_magreal)
  #print(np.amin(nutreal))
  print('max eddy viscosity/ viscocity = ' + str(np.amax(nutreal)/(1./400.)))
  #print(np.mean(nutreal))
  return nutreal
