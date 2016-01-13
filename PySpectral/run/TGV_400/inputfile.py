import numpy as np
import sys
from evtk.hl import gridToVTK
sys.path.append("../../src")

#
turb_model = 0
#-------- MESH ---------------------
N1 = 64
N2 = 64
N3 = 64 #this direction is halved for conjugate symmetry
kc = 16 #cutoff frequency
#-------------------------------------

#----- Physical Properties --------
nu = 1./400.
#----------------------------------

#----- Solver Settings -----------
t = 0                #start time
et = 10              #end time
save_freq =20        #frequency to call save_hook
dt = 0.02            #time step
#--------------------------------

#----- FFT properties -----------
nthreads = 8        #number of threads for fft/iffts
#-------------------------------

#---- Initialize Setup -------------------
dx = 2*np.pi/(N1-1)
dy = 2*np.pi/(N2-1)
dz = 2*np.pi/(N3-1)
x = np.linspace(-np.pi,np.pi-dx,N1)
y = np.linspace(-np.pi,np.pi-dy,N2)
z = np.linspace(-np.pi,np.pi-dz,N3) 

y,x,z = np.meshgrid(y,x,z)
u =  np.cos(x)*np.sin(y)*np.cos(z)
v = -np.sin(x)*np.cos(y)*np.cos(z)
w =  np.zeros((N1,N2,N3))
uhat =  np.fft.rfftn(u) / np.sqrt(N1*N2*N3) 
vhat =  np.fft.rfftn(v) / np.sqrt(N1*N2*N3)
what =  np.fft.rfftn(w) / np.sqrt(N1*N2*N3)
#________________________________________
#-------- Save function called at each iteration ----
def savehook(main,grid,iteration):
    string = '3DSolution/PVsol' + str(iteration)
    string2 = '3DSolution/npsol' + str(iteration)
    u = np.fft.irfftn(main.uhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
    v = np.fft.irfftn(main.vhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
    w = np.fft.irfftn(main.what)*np.sqrt(grid.N1*grid.N2*grid.N3)
    #gridToVTK(string, grid.x,grid.y,grid.z, pointData = {"u" : np.real(u.transpose()) , \
    #  "v" : np.real(v.transpose()), \
    #  "w" : np.real(w.transpose())} )
    np.savez_compressed(string2,uhat=main.uhat,vhat=main.vhat,what=main.what,t=main.t)
#----------------------------------------------------------

#------ Run Solver -----------------------
execfile('../../src/PySpec_3d.py')
#----------------------------------------
