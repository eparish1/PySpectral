import sys
from settings import *
from PySpec_3d_padding import pad,unpad,pad_2x,unpad_2x,seperateModes
from PySpec_3d_utilities import *

utilities = utilitiesClass()
if (SGS_MODEL == 'DNS'):
  from PySpec_3d_utilities import dummy_function as computeSGS
  from PySpec_3d_functionmaps import Q2U_NOSGS as Q2U
  from PySpec_3d_functionmaps import U2Q_NOSGS as U2Q
  from PySpec_3d_RHS import computeRHS_NOSGS as computeRHS

if (SGS_MODEL == 'Markovian'):
  sys.path.append("../../src/EddyViscosityModels/") 
  from PySpec_3d_functionmaps import Q2U_NOSGS as Q2U
  from PySpec_3d_functionmaps import U2Q_NOSGS as U2Q
  if (EDDY_VISCOSITY_MODEL == 'Smagorinsky'):
    print('Using Smagorinsky EV SGS Model')
    from smagorinsky import smag_getnut as getnut
    from PySpec_3d_SGSmodels import computeSGS_EV_REALDOMAIN as computeSGS
  from PySpec_3d_RHS import computeRHS_MARKOVIAN as computeRHS

if (SGS_MODEL == 't-model'):
  from PySpec_3d_utilities import dummy_function as computeSGS
  from PySpec_3d_functionmaps import Q2U_NOSGS as Q2U
  from PySpec_3d_functionmaps import U2Q_NOSGS as U2Q
  from PySpec_3d_RHS import computeRHS_tmodel as computeRHS

