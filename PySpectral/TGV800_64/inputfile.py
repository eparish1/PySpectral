import numpy
import sys
from evtk.hl import gridToVTK
sys.path.append("../src")

#-------- MESH ---------------------
N1 = 32
N2 = 32
N3 = 32 #this direction is halved for conjugate symmetry
#-------------------------------------

#----- Physical Properties --------
nu = 0.01
#----------------------------------

#----- Solver Settings -----------
t = 0                #start time
et = 20              #end time
save_freq =20        #frequency to call save_hook
dt = 0.01            #time step
#--------------------------------

#----- FFT properties -----------
nthreads = 20        #number of threads for fft/iffts
#-------------------------------

#---- Initialize Setup -------------------
execfile('../src/PySpec_3dinit_module.py')
#________________________________________

#------ Run Solver -----------------------
execfile('../src/PySpec_3d.py')
#----------------------------------------

