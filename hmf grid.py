# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 20:19:55 2021

@author: bjenf
"""

import numpy as np
import matplotlib.pyplot as plt
from hmf import MassFunction
import pickle

plt.close('all')

Data = np.genfromtxt('SDSS_cluster_data.dat', skip_header=1,  delimiter=' ', names=['log_M_200','error_low', 'error_up', 'RA', 'DEC', 'redshift'])

Omega = np.linspace(0.1,1,100)
Sigma = np.linspace(0.1,1.3,100)
mass = np.linspace(np.min(Data['log_M_200'])-np.max(Data['error_low']),np.max(Data['log_M_200'])+np.max(Data['error_up']),100)


mf = MassFunction()
mf.update(mdef_params = {"overdensity": 200}, cosmo_params={"H0":100})
mf.update(Mmin=np.min(mass), Mmax=np.max(mass), dlog10m=np.round(mass[1]-mass[0],10)-0.00001)


values = []

for i in range(len(Omega)):
    temp_grid = []
     
    for j in range(len(Sigma)):
        mf.update(cosmo_params={"Om0": Omega[i]}, sigma_8=Sigma[j])
        
        temp_grid.append(np.log10(mf.dndlog10m))
        
    values.append(temp_grid)
    
values = np.array(values)

points = (Omega,Sigma,np.log10(mf.m))

pickle.dump( points, open( "save.points", "wb" ))
pickle.dump( values, open( "save.values", "wb" ))


