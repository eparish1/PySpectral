import numpy
import sys
from evtk.hl import gridToVTK
sys.path.append("../../src")

#-------- MESH ---------------------
N1 = 128
N2 = 128
N3 = 128 #this direction is halved for conjugate symmetry
#-------------------------------------

#----- Physical Properties --------
nu = 1./1600.
#----------------------------------

#----- Solver Settings -----------
t = 0                #start time
et = 10              #end time
save_freq =20        #frequency to call save_hook
dt = 0.02            #time step
#--------------------------------

#----- FFT properties -----------
nthreads = 20        #number of threads for fft/iffts
#-------------------------------

#---- Initialize Setup -------------------
from PySpec_3dClasses import gridclass, FFTclass
from PySpec_3dinitialconditions import *
from PySpec_3d_savehook import savehook
myFFT = FFTclass()
myFFT.initialize(N1,N2,N3,nthreads)
grid = gridclass()
grid.initialize(N1,N2,N3)
u,v,w,uhat,vhat,what = TaylorGreenIC(grid)
from PySpec_3d_importmodule import *

#execfile('../../src/PySpec_3d_initialize.py')
#________________________________________

#------ Run Solver -----------------------
execfile('../../src/PySpec_3d.py')
#----------------------------------------
                 
