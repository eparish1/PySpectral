import numpy as np
import sys
from evtk.hl import gridToVTK
sys.path.append("../../../src")

#
#-------- MESH ---------------------
N1 = 32
N2 = 32
N3 = 32 #this direction is halved for conjugate symmetry
kc = 16 #cut off frequency
#-------------------------------------

#----- Physical Properties --------
nu = 1./1600.
#----------------------------------

#----- Solver Settings -----------
turb_model = 3       #turbulence model
t = 0                #start time
et = 20              #end time
save_freq =1        #frequency to call save_hook
dt = 0.05            #time step
#--------------------------------

#----- FFT properties -----------
nthreads = 8        #number of threads for fft/iffts
#-------------------------------

#---- Initialize Setup -------------------
dx = 2*np.pi/(N1-1)
dy = 2*np.pi/(N2-1)
dz = 2*np.pi/(N3-1)
x = np.linspace(0,2*np.pi-dx,N1)
y = np.linspace(0,2*np.pi-dy,N2)
z = np.linspace(0,2*np.pi-dz,N3) 
y,x,z = np.meshgrid(y,x,z)
u =  np.cos(x)*np.sin(y)*np.cos(z)
v = -np.sin(x)*np.cos(y)*np.cos(z)
w =  np.zeros((N1,N2,N3))
uhat =  np.fft.rfftn(u) / np.sqrt(N1*N2*N3) 
vhat =  np.fft.rfftn(v) / np.sqrt(N1*N2*N3)
what =  np.fft.rfftn(w) / np.sqrt(N1*N2*N3)
#________________________________________

#-------- Save function called at each iteration ----
def savehook(main,grid,utilities,iteration):
    string = '3DSolution/PVsol' + str(iteration)
    string2 = '3DSolution/npsol' + str(iteration)
    string3 = '3DSolution/npspec' + str(iteration)
    u = np.fft.irfftn(main.uhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
    v = np.fft.irfftn(main.vhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
    w = np.fft.irfftn(main.what)*np.sqrt(grid.N1*grid.N2*grid.N3)
    #gridToVTK(string, grid.x,grid.y,grid.z, pointData = {"u" : np.real(u.transpose()) , \
    #  "v" : np.real(v.transpose()), \
    #  "w" : np.real(w.transpose())} )
    k,E = utilities.computeSpectrum(main,grid)
    k_res,E_res = utilities.computeSpectrum_resolved(main,grid)
    np.savez_compressed(string2,uhat=main.uhat,vhat=main.vhat,what=main.what,w0_u=main.w0_u,w0_v=main.w0_v,w0_w=main.w0_w,t=main.t)
    np.savez_compressed(string3,k=k,k_res=k_res,E=E,E_res=E_res,t=main.t)
#----------------------------------------------------------

#------ Run Solver -----------------------
execfile('../../../src/PySpec_3d.py')
