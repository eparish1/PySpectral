import numpy as np
from PySpec_3dClasses import gridclass, FFTclass
from PySpec_3dinitialconditions import *
from PySpec_3d_savehook import savehook
myFFT = FFTclass() 
myFFT.initialize(N1,N2,N3,nthreads)
grid = gridclass()
grid.initialize(N1,N2,N3)
u,v,w,uhat,vhat,what = TaylorGreenIC(grid) 
from PySpec_3d_importmodule import *
