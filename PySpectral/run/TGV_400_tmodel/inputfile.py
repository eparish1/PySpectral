import numpy as np
import sys
from evtk.hl import gridToVTK
sys.path.append("../../src")

#
#-------- MESH ---------------------
N1 = 32
N2 = 32
N3 = 32 #this direction is halved for conjugate symmetry
kc = 16 #cut off frequency
#-------------------------------------

#----- Physical Properties --------
nu = 1./400.
#----------------------------------

#----- Solver Settings -----------
turb_model = 2       #turbulence model
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
    np.savez_compressed(string2,uhat=main.uhat,vhat=main.vhat,what=main.what,PLQLu=main.PLQLu,PLQLv=main.PLQLv,PLQLw=main.PLQLw,t=main.t)
#----------------------------------------------------------

#------ Run Solver -----------------------
execfile('../../src/PySpec_3d.py')
#----------------------------------------


########## Post Process############
from pylab import *
axis_font = {'size':'24','family':'serif','weight':'light'}
rundata = np.load('3DSolution/stats.npz')
ts = rundata['t']
Es = rundata['Energy']
Ds = -rundata['Dissipation']
Ds[0] = 0


mdata = np.load('PySpec_400_64.npz')
tm = mdata['t']
Em = mdata['Energy_resolved']
Dm = -mdata['Dissipation_resolved']
Dm[0] = 0

ta = linspace(0,t,size(Energy))
plot(tm[:],Em[0::],color='black',label='Resolved Energy')
plot(ts,Es,color='red',label='t-model')
xlim([0,20])
xlabel(r'$t$',**axis_font)
ylabel(r'Energy',**axis_font)
legend(loc = 1)
savefig('Validate_E.pdf')

figure(2)
plot(tm[1::],Dm[0::],color='black',label='Resolved Dissipation')
plot(ts[1::],Ds,color='red',label='t-model')
legend(loc = 1)
xlabel(r'$t$',**axis_font)
ylabel(r'Dissipation',**axis_font)
xlim([0,20])
savefig('Validate_eps.pdf')
show()
                 
