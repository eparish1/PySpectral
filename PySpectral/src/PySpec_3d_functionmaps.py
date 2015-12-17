import numpy as np

def Q2U_NOSGS(Q,uhat,vhat,what):
    uhat[:,:] = Q[0::3,0::3,0::3]
    vhat[:,:] = Q[1::3,1::3,1::3]
    what[:,:] = Q[2::3,2::3,2::3]
    return uhat,vhat,what

def U2Q_NOSGS(Q,uhat,vhat,what):
    Q[0::3,0::3,0::3] = uhat[:,:]
    Q[1::3,1::3,1::3] = vhat[:,:]
    Q[2::3,2::3,2::3] = what[:,:]
    return Q
