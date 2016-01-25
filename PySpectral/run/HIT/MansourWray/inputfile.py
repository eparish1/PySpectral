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


### Mansour Wray initialization
#Decay of isotropic turbulence is computed using direct numerical simulations
#http://scitation.aip.org/content/aip/journal/pof2/6/2/10.1063/1.868319
sigma = 4.
kp = 3.
nu = 0.0007

k1 = np.fft.fftshift( np.linspace(-N1/2,N1/2-1,N1) )
k2 = np.fft.fftshift( np.linspace(-N2/2,N2/2-1,N2) )
k3 = np.linspace( 0,N3/2,N3/2+1 )
k2,k1,k3 = np.meshgrid(k2,k1,k3)
ksqr = k1**2 + k2**2 + k3**2
### First compute A integral: A = int k^sigma * exp( -sigma * k^2 / 2 ) dk
dv2 = (2.*np.pi)**3./(N1*N2*N3)
Ak = np.sqrt(ksqr)**sigma * np.exp(-sigma * ksqr / 2. ) * dv2
Aint = np.sum(Ak[:,:,1:-1])*2. + np.sum(Ak[:,:,0])

#Now get E(k) = q^2/(2*A) * 1./(kp**(sigma+1))*k**sigma * exp(-sigma/2 * (k/kp)**2 )
#k_m, indices1 = np.unique(np.rint(np.sqrt(grid.ksqr[:,:,0:grid.N3/2].flatten())), return_inverse=True)
q2 = 3
kp = 3
E = q2/(2.*Aint) * 1./(kp**(sigma+1)) * np.sqrt(ksqr) ** sigma * np.exp(-sigma/2 * ksqr/kp**2 )

## Now get velocity field
np.random.seed(1)
theta1 = np.rand((N1,N2,N3))*2.*pi
theta2 = np.rand((N1,N2,N3))*2.*pi
phi    = np.rand((N1,N2,N3))*2.*pi

alpha = E/(4.*pi*ksqr)*exp(1j*theta1)*cos(phi)
beta =  E/(4.*pi*ksqr)*exp(1j*theta2)*sin(phi)
u =  np.cos(x)*np.sin(y)*np.cos(z)
v = -np.sin(x)*np.cos(y)*np.cos(z)
w =  np.zeros((N1,N2,N3))
uhat =  np.fft.rfftn(u) / np.sqrt(N1*N2*N3) 
vhat =  np.fft.rfftn(v) / np.sqrt(N1*N2*N3)
what =  np.fft.rfftn(w) / np.sqrt(N1*N2*N3)
#________________________________________
#-------- Save function called at each iteration ----
#def savehook(main,grid,iteration):
#    string = '3DSolution/PVsol' + str(iteration)
#    string2 = '3DSolution/npsol' + str(iteration)
#    u = np.fft.irfftn(main.uhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
#    v = np.fft.irfftn(main.vhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
#    w = np.fft.irfftn(main.what)*np.sqrt(grid.N1*grid.N2*grid.N3)
#    #gridToVTK(string, grid.x,grid.y,grid.z, pointData = {"u" : np.real(u.transpose()) , \
#    #  "v" : np.real(v.transpose()), \
#    #  "w" : np.real(w.transpose())} )
#    np.savez_compressed(string2,uhat=main.uhat,vhat=main.vhat,what=main.what,t=main.t)
##----------------------------------------------------------
#
##------ Run Solver -----------------------
#execfile('../../src/PySpec_3d.py')
#----------------------------------------
