import numpy as np
def advanceQ_RK4(main,grid,myFFT):
  main.Q0[:] = main.Q[:]
  rk4const = np.array([1./4,1./3,1./2,1.])
  for i in range(0,4):
    main.computeRHS(main,grid,myFFT)
    main.Q[:] = main.Q0[:] + main.dt*rk4const[i]*main.Q[:]

def advanceQ_SI(main,grid,myFFT):
  main.computeRHS(main,grid,myFFT)
  main.Q[:] = (main.Q[:] + main.dt/2.*(3.*main.H[:] - main.H_old[:]) + main.dt/2.*main.viscous_term[:]) / (1. + main.nu*main.dt/2.*grid.ksqr[None,:,:,:])

