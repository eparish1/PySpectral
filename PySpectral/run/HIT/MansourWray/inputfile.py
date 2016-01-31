import numpy as np
import sys
from evtk.hl import gridToVTK
sys.path.append("../../../src")

#
turb_model = 0
#-------- MESH ---------------------
N1 = 128
N2 = 128
N3 = 128 #this direction is halved for conjugate symmetry
kc = 64 #cutoff frequency
cfl = 0.5
#-------------------------------------

#----- Physical Properties --------
nu = 0.0007
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
L1 = 2.*np.pi
L2 = 2.*np.pi
L3 = 2.*np.pi
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
sigma = 6.
kp = 15.
q2 = 3.

k1 = np.fft.fftshift( np.linspace(-N1/2,N1/2-1,N1) )
k2 = np.fft.fftshift( np.linspace(-N2/2,N2/2-1,N2) )
k3 = np.linspace( 0,N3/2,N3/2+1 )
k2,k1,k3 = np.meshgrid(k2,k1,k3)
ksqr = k1**2 + k2**2 + k3**2 
kmag = np.sqrt(ksqr) + 1e-30 
### First compute A integral: A = int k^sigma * exp( -sigma * k^2 / 2 ) dk
dv2 = (2.*np.pi)**3./(N1*N2*N3)
Ak = np.sqrt(ksqr)**sigma * np.exp(-sigma * ksqr / 2. ) * dv2
#Aint = np.sum(Ak[:,:,1:-1])*2. + np.sum(Ak[:,:,0])
kdummy = np.linspace(0,500,501)
Edummy = kdummy**sigma * np.exp(-sigma * kdummy**2 / 2. ) 
Aint = np.trapz(Edummy,kdummy)
#Now get E(k) = q^2/(2*A) * 1./(kp**(sigma+1))*k**sigma * exp(-sigma/2 * (k/kp)**2 )
#k_m, indices1 = np.unique(np.rint(np.sqrt(grid.ksqr[:,:,0:grid.N3/2].flatten())), return_inverse=True)
#E = q2/(2.*Aint) * 1./(kp**(sigma+1)) * np.sqrt(ksqr) ** sigma * np.exp(-sigma/2. * ksqr/kp**2 )
#Ekdummy = q2/(2.*Aint) * 1./(kp**(sigma+1)) * kdummy ** sigma * np.exp(-sigma/2. * kdummy**2/kp**2 )
E = q2/(2.*Aint)/kp*(kmag/kp)**sigma*np.exp(-sigma/2. * ksqr/kp**2 )

## Now get velocity field
np.random.seed(1)
theta1 = np.random.rand(N1,N2,N3/2+1)*2.*np.pi
theta2 = np.random.rand(N1,N2,N3/2+1)*2.*np.pi
phi    = np.random.rand(N1,N2,N3/2+1)*2.*np.pi
alpha = np.sqrt( E/(4.*np.pi*ksqr + 1e-30) )*np.exp(1j*theta1)*np.cos(phi)
beta  = np.sqrt( E/(4.*np.pi*ksqr + 1e-30) )*np.exp(1j*theta2)*np.sin(phi)
#e1_1 = k2/(kmag + 1e-30 )
#e1_2 = k3/(kmag + 1e-30 )
#e1_3 = k1/(kmag + 1e-30 ) 
#e2_1 = k3/(kmag + 1e-30 )
#e2_2 = k1/(kmag + 1e-30 )
#e2_3 = k2/(kmag + 1e-30 )
#uhat = (alpha*e1_1 + beta*e2_1)
#vhat = (alpha*e1_2 + beta*e2_2)
#what = (alpha*e1_3 + beta*e2_3)

## Karthiks
uhat = (alpha*kmag*k2*2.*np.pi/L1 + beta*k1*2.*np.pi/L2 * k3*2.*np.pi/L3)/(kmag*np.sqrt(k1**2 + k2**2) + 1e-30)
vhat = (-alpha*kmag*k1*2.*np.pi/L1 + beta*k2*2.*np.pi/L2* k3*2.*np.pi/L3)/(kmag*np.sqrt(k1**2 + k2**2) + 1e-30)
what = -beta*np.sqrt(k1**2 + k2**2)/kmag

uhat = uhat*np.sqrt(N1*N2*N3)
vhat = vhat*np.sqrt(N1*N2*N3)
what = what*np.sqrt(N1*N2*N3)


uE = np.sum(uhat[:,:,1:N3/2]*np.conj(uhat[:,:,1:N3/2]*2) ) + \
     np.sum(uhat[:,:,0]*np.conj(uhat[:,:,0]))
vE = np.sum(vhat[:,:,1:N3/2]*np.conj(vhat[:,:,1:N3/2]*2) ) + \
     np.sum(vhat[:,:,0]*np.conj(vhat[:,:,0]))
wE = np.sum(what[:,:,1:N3/2]*np.conj(what[:,:,1:N3/2]*2) ) + \
     np.sum(what[:,:,0]*np.conj(what[:,:,0]))

print(np.real(0.5*(uE + vE + wE)/(N1*N2*N3)))
u =  np.fft.irfftn(uhat) * np.sqrt(N1*N2*N3) 
v =  np.fft.irfftn(vhat) * np.sqrt(N1*N2*N3)
w =  np.fft.irfftn(what) * np.sqrt(N1*N2*N3)
#________________________________________
#-------- Save function called at each iteration ----
def savehook(main,grid,utilities,iteration):
    string = '3DSolution/PVsol' + str(iteration)
    string2 = '3DSolution/npsol' + str(iteration)
    u = np.fft.irfftn(main.uhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
    v = np.fft.irfftn(main.vhat)*np.sqrt(grid.N1*grid.N2*grid.N3)
    w = np.fft.irfftn(main.what)*np.sqrt(grid.N1*grid.N2*grid.N3)
    #gridToVTK(string, grid.x,grid.y,grid.z, pointData = {"u" : np.real(u.transpose()) , \
    #  "v" : np.real(v.transpose()), \
    #  "w" : np.real(w.transpose())} )
    k,E = utilities.computeSpectrum(main,grid)
    k_res,E_res = utilities.computeSpectrum_resolved(main,grid)
    np.savez_compressed(string2,uhat=main.uhat,vhat=main.vhat,what=main.what,t=main.t)
    np.savez_compressed(string2,uhat=main.uhat,vhat=main.vhat,what=main.what,t=main.t)
##----------------------------------------------------------
#
##------ Run Solver -----------------------
execfile('../../../src/PySpec_3d.py')
#----------------------------------------
