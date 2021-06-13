# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:10:51 2021

@author: bjenf
"""

import numpy as np
import matplotlib.pyplot as plt
import Bachelor_function_library as bfl
import pickle


plt.close('all')

Data = np.genfromtxt('SDSS_cluster_data.dat', skip_header=1,  delimiter=' ', names=['log_M_200','error_low', 'error_up', 'RA', 'DEC', 'redshift'])

points = pickle.load( open( "save.points", "rb" ))
values = pickle.load( open( "save.values", "rb" ))


models = np.array([[10, -3, 1], [10, -3, 0],[20, -6, 1], [20, -6, 0]])


area, total_area_of_the_sky = bfl.MC_sim(Data['RA'], Data['DEC'], n_points=100000, gamma=0.0006, nu=0.0013, plot=0)

x, y, y_err, delta_x, weights, counts, Volume, width = bfl.fitting_data(Data['RA'], Data['DEC'], Data['log_M_200'], Data['error_low'], Data['error_up'], Data['redshift'], selectionfunction=1, cut_off=-6, hist_bins=20, plot=1)

omega_m, comoving_dist = bfl.Comoving_distance_for_Omega_m(Data['redshift'], np.linspace(0.2,0.4,20), plot=1)

bfl.plot_HMF_dependencies(np.arange(0.1,0.9,0.1), np.arange(0.3,1.1,0.1),x)


for i in range(len(models)):

    x, y, y_err, delta_x, weights, counts, Volume, width = bfl.fitting_data(Data['RA'], Data['DEC'], Data['log_M_200'], Data['error_low'], Data['error_up'], Data['redshift'], selectionfunction=models[i,2], cut_off=models[i,1], hist_bins=models[i,0], plot=1)


    flat_samples, log_prob, Val_M1 = bfl.MCMC_run_interpolate(10000, x[abs(models[i,1]):], y[abs(models[i,1]):], y_err[abs(models[i,1]):], counts[abs(models[i,1]):], Volume, width[abs(models[i,1]):], 100, points, values, Omega_m_0=0.3, Sigma_8_0=0.81, gauss=0, poisson=1)

    flat_samples, log_prob, Val_M2 = bfl.MCMC_run_interpolate_gauss(10000, x, y, y_err, counts, Volume, width, 100, points, values, Omega_m_0=0.3, Sigma_8_0=0.81, mean=13.85, std=0.15, gauss=0, poisson=1)

    flat_samples, log_prob, Val_M3 = bfl.MCMC_run_interpolate_gauss_error(10000, x, y, y_err, counts, Volume, width, weights, delta_x, 100, points, values, Omega_m_0=0.34, Sigma_8_0=0.65, mean=13.9, std=0.2, gauss=0, poisson=1)

    x_plot = np.linspace(np.min(x),np.max(x),1000)

    bfl.MF_fit_estimates(x_plot, x, y, y_err, omega_m=Val_M1[0] , sigma_8=Val_M1[1], gauss=0, base=1)
    
    bfl.MF_fit_estimates(x_plot, x, y, y_err, omega_m=Val_M2[0], sigma_8=Val_M2[1], mu=Val_M2[2], sigma=Val_M2[3], gauss=1, base=1)
    
    bfl.MF_fit_estimates(x_plot, x, y, y_err, omega_m=Val_M3[0], sigma_8=Val_M3[1], mu=Val_M3[2], sigma=Val_M3[3], gauss=1, base=1)

