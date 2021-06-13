# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 10:22:11 2021

@author: bjenf
"""

import numpy as np
import matplotlib.pyplot as plt
from hmf import MassFunction, cosmo
from sklearn.svm import OneClassSVM
import emcee
import corner
from scipy.interpolate import RegularGridInterpolator
from scipy import stats


def plot_HMF_dependencies(Omega_m_list, Sigma_8_list, x):
    
    mf = MassFunction(sigma_8=0.8)
    mf.update(mdef_params = {"overdensity": 200}, cosmo_params={"H0":100})
    mf.update(Mmin=np.min(x), Mmax=np.max(x), dlog10m=np.round(x[1]-x[0],10)/10)
    
    plt.subplot(1,2,1)

    for i in range(len(Omega_m_list)):
        mf.update(cosmo_params={"Om0":round(Omega_m_list[i],2)})
        plt.plot(np.log10(mf.m), mf.dndlog10m, label=r"$\Omega_{m}$" + str(round(Omega_m_list[i],2)))
        

    plt.yscale('log')
    plt.legend()
    plt.xlabel(r"Log(Mass), $[h^{-1}M_\odot]$")
    plt.ylabel(r"$dn/dLog(m)$, $[h^{3}{\rm Mpc}^{-3}logM_\odot^{-1}]$")
    plt.title(r"Mass Function with $\sigma_{8}$ fixed 0.8")
    plt.grid()
    
    mf = MassFunction(cosmo_params={"Om0":0.3})
    mf.update(mdef_params = {"overdensity": 200}, cosmo_params={"H0":100})
    mf.update(Mmin=np.min(x), Mmax=np.max(x), dlog10m=np.round(x[1]-x[0],10)/10)
    
    
    plt.subplot(1,2,2)
    
    for i in range(len(Sigma_8_list)):
        mf.update(sigma_8 = Sigma_8_list[i])
        plt.plot(np.log10(mf.m), mf.dndlog10m, label=r"$\sigma_{8}$" + str(round(Sigma_8_list[i],2)))
        
    plt.yscale('log')
    plt.legend()
    plt.xlabel(r"Log(Mass), $[h^{-1}M_\odot]$")
    plt.ylabel(r"$dn/dLog(m)$, $[h^{3}{\rm Mpc}^{-3}logM_\odot^{-1}]$")
    plt.title(r"Mass Function with $\Omega_{m}$ fixed 0.3")
    plt.grid()
    plt.show()
    
def Comoving_distance_for_Omega_m(redshift, omega_m, plot=0):
    Comoving_distance=[]
    
    for i in range(len(omega_m)):
        my_cosmo = cosmo.Cosmology(cosmo_params={"H0":100, "Om0":omega_m[i]})
        Comoving_distance.append(my_cosmo.cosmo.comoving_distance(np.max(redshift)).value)
    
    if plot==1:
        plt.figure()
        plt.plot(omega_m,Comoving_distance)
        plt.ylabel('Comoving distance')
        plt.xlabel(r"$\Omega_m$")
        plt.grid()
    
    return np.array([np.mean(omega_m),np.min(omega_m)-np.mean(omega_m), np.max(omega_m)-np.mean(omega_m)]), np.array([np.mean(Comoving_distance),np.min(Comoving_distance)-np.mean(Comoving_distance), np.max(Comoving_distance)-np.mean(Comoving_distance)])
    
    

def MC_sim(RA, DEC, n_points=100000, gamma=0.0006, nu=0.0013, plot=0):
    
    corr_deg = np.column_stack((RA,DEC))
    
    RA_Gen = np.random.uniform(0,360,n_points)#create random data right acersion
    DEC_Gen = np.rad2deg(np.arcsin(np.random.uniform(-1,1,n_points)))#cretae random data Declination
    
    svm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)#create the Supporting Vector machine for a single cluster
    svm.fit(corr_deg) # fit the SVM to the know classified data
    
    X = np.column_stack((RA_Gen,DEC_Gen))
    h = .2  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    pred = svm.predict(X)

    anom_index, = np.where(pred==-1)
    inside_index, = np.where(pred==1)
    outside = X[anom_index]
    inside = X[inside_index]
    
    if plot==1:
        plt.subplot(1,2,1)
        plt.title('Data from dataset')
        plt.xlabel('Right Ascension')
        plt.ylabel('Declination')
        plt.plot(RA, DEC, 'r.', markersize = 1, label='data set')
        plt.xlim(0,360)
        plt.ylim(-90, 90)
        plt.contour(xx, yy, Z, colors = 'k', linewidths = 0.7)
        plt.legend()
        plt.grid()
        plt.show()
        
        plt.subplot(1,2,2)
        plt.title('Monte Carlo simulated data')
        plt.xlabel('Right Ascension')
        plt.ylim(-90,90)
        plt.xlim(0,360)
        plt.plot(outside[:,0], outside[:,1], 'b.', markersize = 0.3, label='data gen outside the delimiter')
        plt.plot(inside[:,0], inside[:,1], 'r.',  markersize = 0.3, label='data gen inside the delimiter')
        plt.contour(xx, yy, Z, colors = 'k', linewidths = 0.7)
        plt.show()
    
    total_area_of_the_sky = 41252.96 #deg^2
    area = len(inside)/len(X) * total_area_of_the_sky
    
    return area, total_area_of_the_sky


def fitting_data(RA, DEC, log_M_200, error_low, error_up, redshift, hist_bins=10, selectionfunction=1, cut_off=0, plot=0):
    
    mass =  log_M_200
    err_low = error_low
    err_up = error_up
        
    hist = np.histogram(mass, bins=hist_bins)
    
    counts, cuts = hist[0], hist[1]

    delta_x = []
    
    for i in range(len(counts)):
        index = np.where(np.logical_and(mass>=cuts[i],mass<(cuts[i+1]+0.0000001)))[0]
        
        delta_x.append(np.linspace(np.mean(-err_low[index]), np.mean(err_up[index]), 10))
    
    weights = []
    
    for i in range(len(counts)):
        delta_x[i][delta_x[i][:] < 0]
        delta_x[i][delta_x[i][:] >= 0]
        
        start = stats.norm(loc=0, scale=err_low[i]/3).pdf(delta_x[i][delta_x[i][:] < 0])
        end = stats.norm(loc=0, scale=err_up[i]/3).pdf(delta_x[i][delta_x[i][:] >= 0])
    
        weights.append(np.concatenate((start, end), axis=0))
    
    weights = np.array(weights)/np.reshape(np.repeat(np.sum(weights, axis=1),len(weights[0])), (len(weights),len(weights[0])))
    delta_x = np.array(delta_x)
    
    if plot==1:
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
        hist = ax0.hist(log_M_200, bins=hist_bins)
        ax0.set_ylabel('Counts')
        ax0.set_title('Binning of mass and consequential error')
        ax0.grid() 
        mass_x_axis = ((hist[1][1:]+hist[1][:-1])/2)
        ax1.errorbar(mass_x_axis, np.zeros_like(mass_x_axis), yerr=(-delta_x[:,0],delta_x[:,-1]), fmt='.k',  ecolor='k', elinewidth=1, capsize=1, capthick=1 )
        ax1.set_ylim([-0.3,0.3])
        ax1.grid()
        ax1.set_ylabel(r"Error on Log(Mass), $[h^{-1}M_\odot]$")
        ax1.set_xlabel(r"Log(Mass), $[h^{-1}M_\odot]$")
    
    
    #################################################################################################################
    
    if selectionfunction==1:
        
        my_cosmo = cosmo.Cosmology(cosmo_params={"H0":100})
        area, total_area_of_the_sky = MC_sim(RA, DEC)
        
        def selectionfunction(z, a=1.07, b=293.4, gamma=2.97):
            D_m = my_cosmo.cosmo.comoving_distance(z).value
            
            return(a*np.exp(-(D_m/b)**gamma))
        
        z_bins = []
        for i in range(len(hist[0])):
            z_bins.append(redshift[np.where(np.logical_and(log_M_200 >= hist[1][i],log_M_200 < hist[1][i+1]))[0]])
        
        S = []
        S_err = []
        V = []
        Volume = 4*np.pi/3 * my_cosmo.cosmo.comoving_distance(np.max(redshift))**3 * area/total_area_of_the_sky #not used but must be there for function return
        
        for i in range(len(z_bins)):
            S.append(np.sum(1/np.where(selectionfunction(z_bins[i], a=1.07, b=293.4, gamma=2.97)>1,1,selectionfunction(z_bins[i], a=1.07, b=293.4, gamma=2.97))))
            V.append(np.sum(4*np.pi/3*area/total_area_of_the_sky*(my_cosmo.cosmo.comoving_distance(np.max(z_bins[i])).value**3-my_cosmo.cosmo.comoving_distance(np.min(z_bins[i])).value**3)))
            S_err.append(np.sqrt(np.sum(1/np.where(selectionfunction(z_bins[i], a=1.07, b=293.4, gamma=2.97)>1,1,selectionfunction(z_bins[i], a=1.07, b=293.4, gamma=2.97)))))
                    
        S = np.array(S)
        S_err = np.array(S_err)
        V = np.array(V)
        
        cut_index = np.where(hist[0]==np.max(hist[0]))[0][0] + cut_off
        
        points = S[cut_index:] / (hist[1][cut_index+1:]-hist[1][cut_index:-1]) /  V[cut_index:]
        err = S_err[cut_index:] / (hist[1][cut_index+1:]-hist[1][cut_index:-1]) / V[cut_index:] 
    
        mass_center_bins = hist[1][cut_index+1:] - (hist[1][cut_index+1:]-hist[1][cut_index:-1])/2
        
        return mass_center_bins, points, err, delta_x, weights, S[cut_index:], Volume.value, (hist[1][cut_index+1:]-hist[1][cut_index:-1])
        
    elif selectionfunction==0:
        
        my_cosmo = cosmo.Cosmology(cosmo_params={"H0":100})
        area, total_area_of_the_sky = MC_sim(RA, DEC)
        D_m = my_cosmo.cosmo.comoving_distance(np.max(redshift))
        Volume = 4*np.pi/3 * D_m**3 * area/total_area_of_the_sky

        cut_index = np.where(hist[0]==np.max(hist[0]))[0][0] + cut_off
        
        points = hist[0][cut_index:] / (hist[1][cut_index+1:]-hist[1][cut_index:-1]) /  Volume.value
        err = np.sqrt(hist[0][cut_index:])/ (hist[1][cut_index+1:]-hist[1][cut_index:-1]) / Volume.value
    
        mass_center_bins = (hist[1][cut_index+1:]+hist[1][cut_index:-1])/2
            
        return mass_center_bins, points, err, delta_x, weights, hist[0][cut_index:], Volume.value, (hist[1][cut_index+1:]-hist[1][cut_index:-1])
    



             
def MCMC_run_interpolate(itt, x, y, yerr, counts, Volume, width, discard, points, values, Omega_m_0=0.34, Sigma_8_0=0.65, gauss=1, poisson=0):
    
    mf = MassFunction()
    mf.update(mdef_params = {"overdensity": 200}, cosmo_params={"H0":100})
    mf.update(Mmin=np.min(x), Mmax=np.max(x), dlog10m=np.round(x[1]-x[0],10)-0.000000001)

    
    interpol_func = RegularGridInterpolator(points, values, method='linear')
    
    def log_likelihood_gauss(theta, y, yerr):
        omega_m, sigma_8 = theta
        new_points = np.column_stack((np.full_like(mf.m,omega_m),np.full_like(mf.m,sigma_8),np.log10(mf.m)))
        model = 10**(interpol_func(new_points))
        
        return -0.5 * np.sum((y - model) ** 2 / yerr**2)
    
    def log_likelihood_poisson(theta, counts, Volume, width):
        
        omega_m, sigma_8 = theta #random scattered value of omega_m and sigma_8
        
        new_points = np.column_stack((np.full_like(mf.m,omega_m),np.full_like(mf.m,sigma_8),np.log10(mf.m))) #points to be evalueated by the linear interpolation
        
        Lambda = 10**(interpol_func(new_points))*Volume*width #Lambda in poisson distribution, interpolation converted so not in log10
        
        return np.sum(np.log(Lambda)*counts-Lambda) #poisson log-likelihood
    
    def log_prior(theta):
        omega_m, sigma_8 = theta
        if 0.1 < omega_m < 1 and 0.1 < sigma_8 < 1.3: #defined range for omega_m and sigma_8 making propbability -inf if not in range
            return 0.0
        return -np.inf
    
    
    
    def log_probability(theta, y, yerr, counts, Volume, width, gauss, poisson):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        if gauss==1:
            return lp + log_likelihood_gauss(theta, y, yerr)
        elif poisson==1:
            return lp + log_likelihood_poisson(theta, counts, Volume, width)
        
    
    pos = [Omega_m_0,Sigma_8_0] + 1e-3 * np.random.randn(32, 2)
    
    nwalkers, ndim = pos.shape
    
    if gauss==1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(y, yerr, counts, Volume, width, gauss, poisson))
        sampler.run_mcmc(pos, itt, progress=True);
    elif poisson==1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(y, yerr, counts, Volume, width, gauss, poisson))
        sampler.run_mcmc(pos, itt, progress=True);
        

    fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["Omega_m", "Sigma_8"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("step number");
    
    flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True)
    
    
    corner.corner(flat_samples, quantiles=(0.0015, 0.9985), levels=(0.997,), labels=labels, truths=np.mean(flat_samples, axis=0));
    
    Val = []

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [0.15, 50, 99.85])
        Val.append(mcmc[1])
        q = np.diff(mcmc)
        print(f"{labels[i]}= {mcmc[1]:.3f} with lower error = {-q[0]:.4f} and upper error = {q[1]:.4f}")
        
    
    return(flat_samples, sampler.get_log_prob(), Val)
                
            
def MCMC_run_interpolate_gauss(itt, x, y, yerr, counts, Volume, width, discard, points, values, Omega_m_0=0.34, Sigma_8_0=0.65, mean=13.9, std=0.2, gauss=1, poisson=0):
    
    interpol_func = RegularGridInterpolator(points, values, method='linear')
    
    mf = MassFunction()
    mf.update(mdef_params = {"overdensity": 200}, cosmo_params={"H0":100})
    mf.update(Mmin=np.min(x), Mmax=np.max(x), dlog10m=np.round(x[1]-x[0],10)-0.000000001)

    
    def log_likelihood_gauss(theta, y, yerr):
        omega_m, sigma_8, mu, sigma = theta
        
        new_points = np.column_stack((np.full_like(mf.m,omega_m),np.full_like(mf.m,sigma_8),np.log10(mf.m)))
        
        index_start = np.where(new_points[:,2]<mu)
        model_start = np.log10(stats.norm.pdf(new_points[index_start][:,2], mu, sigma)*np.sqrt(2*np.pi)*sigma) + interpol_func((omega_m ,sigma_8, mu))

        index_end =  np.where(new_points[:,2] >= mu)
        model_end = interpol_func(new_points[index_end])
        
        model = 10**(np.concatenate((model_start,model_end), axis=0))
        
        return -0.5 * np.sum((y - model) ** 2 / yerr**2)
    
    def log_likelihood_poisson(theta, counts, Volume, width):
        
        omega_m, sigma_8, mu, sigma = theta #random scattered value of omega_m, sigma_8, mu and sigma
        
        new_points = np.column_stack((np.full_like(mf.m,omega_m),np.full_like(mf.m,sigma_8),np.log10(mf.m)))#points to be evalueated by the linear interpolation
        
        index_start = np.where(new_points[:,2]<mu) #finding the index to apply the gaussian
        model_start = np.log10(stats.norm.pdf(new_points[index_start][:,2], mu, sigma)*np.sqrt(2*np.pi)*sigma) + interpol_func((omega_m ,sigma_8, mu)) #applying the gaussian

        index_end =  np.where(new_points[:,2] >= mu)#finding the index to apply the MF
        model_end = interpol_func(new_points[index_end]) #applying the MF
        
        
        model = 10**(np.concatenate((model_start,model_end), axis=0)) #concatenate the two functions,  model converted so not in log10
        
        Lambda = model*Volume*width #Lambda in poisson distribution

        return np.sum(np.log(Lambda)*counts-Lambda) #compute the log-likelihood
    
    def log_prior(theta):
        omega_m, sigma_8, mu, sigma = theta
        if 0.1 < omega_m < 1 and 0.1 < sigma_8 < 1.3 and 13.45 < mu < 14.5 and 0.01 < sigma < 0.5: #defined range for omega_m, sigma_8, mu and sigma making propbability -inf if not in range
            return 0.0
        return -np.inf
    
    def log_probability(theta, y, yerr, counts, Volume, width, gauss, poisson):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        if gauss==1:
            return lp + log_likelihood_gauss(theta, y, yerr)
        elif poisson==1:
            return lp + log_likelihood_poisson(theta, counts, Volume, width)
        
    
    pos = [Omega_m_0, Sigma_8_0, mean, std] + 1e-4 * np.random.randn(32, 4)
    
    nwalkers, ndim = pos.shape
    
    if gauss==1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(y, yerr, counts, Volume, width, gauss, poisson))
        sampler.run_mcmc(pos, itt, progress=True);
    elif poisson==1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(y, yerr, counts, Volume, width, gauss, poisson))
        sampler.run_mcmc(pos, itt, progress=True);
        

    fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["Omega_m", "Sigma_8", "Mu", "Sigma"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("step number");
    
    flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True)
    
    corner.corner(flat_samples, quantiles=(0.0015, 0.9985), levels=(0.997,), labels=labels, truths=np.mean(flat_samples, axis=0));
    
    Val = []

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [0.15, 50, 99.85])
        Val.append(mcmc[1])
        q = np.diff(mcmc)
        print(f"{labels[i]}= {mcmc[1]:.3f} with lower error = {-q[0]:.4f} and upper error = {q[1]:.4f}")
        
    
    return(flat_samples, sampler.get_log_prob(), Val)
    
    
def MCMC_run_interpolate_gauss_error(itt, x, y, yerr, counts, Volume, width, weights, delta_x, discard, points, values, Omega_m_0=0.34, Sigma_8_0=0.65, mean=13.9, std=0.2, gauss=1, poisson=0):
    
    interpol_func = RegularGridInterpolator(points, values, method='linear')
    
    mf = MassFunction()
    mf.update(mdef_params = {"overdensity": 200}, cosmo_params={"H0":100})
    mf.update(Mmin=np.min(x), Mmax=np.max(x), dlog10m=np.round(x[1]-x[0],10)-0.000000001)

    
    def log_likelihood_gauss(theta, y, yerr, weights, delta_x):
        omega_m, sigma_8, mu, sigma = theta
        
        new_points = np.column_stack((np.full_like(mf.m,omega_m),np.full_like(mf.m,sigma_8),np.log10(mf.m)))
        
        index_start = np.where(new_points[:,2]<mu)
        model_start = []
        for i in range(len(new_points[index_start])):
            intermediate_model_start = []
            for j in range(len(delta_x[i])):
                intermediate_model_start.append((np.log10(stats.norm.pdf(new_points[index_start][i,2]+delta_x[i][j], mu, sigma)*np.sqrt(2*np.pi)*sigma) + interpol_func((omega_m ,sigma_8, mu)))*weights[i][j])
            
            model_start.append(np.sum(intermediate_model_start))
 
    
        index_end = np.where(new_points[:,2] >= mu)
        model_end = []
        for i in range(len(new_points[index_end])):
            intermediate_model_end = []
            for j in range(len(delta_x[i])):
                intermediate_model_end.append(interpol_func([omega_m ,sigma_8, new_points[index_end][i,2]+delta_x[i][j]])*weights[i][j])
              
            model_end.append(np.sum(intermediate_model_end))
                
        
        model = 10**(np.concatenate((model_start,model_end), axis=0))
        
        
        return -0.5 * np.sum((y - model) ** 2 / yerr**2)
    
    
    def log_likelihood_poisson(theta, counts, Volume, width,  weights, delta_x):
        
        omega_m, sigma_8, mu, sigma = theta#random scattered value of omega_m, sigma_8, mu and sigma
        
        new_points = np.column_stack((np.full_like(mf.m,omega_m),np.full_like(mf.m,sigma_8),np.log10(mf.m)))#points to be evalueated by the linear interpolation
        
        index_start = np.where(new_points[:,2]<mu)#finding the index to apply the gaussian
        
        #------- Nested for loop calculating the gaussian function summed over the range of error with the correlating weights-----------------------------------
        model_start = []
        for i in range(len(new_points[index_start])):
            intermediate_model_start = []
            for j in range(len(delta_x[i])):
                intermediate_model_start.append((np.log10(stats.norm.pdf(new_points[index_start][i,2]+delta_x[i][j], mu, sigma)*np.sqrt(2*np.pi)*sigma) + interpol_func((omega_m ,sigma_8, mu)))*weights[i][j]) 
            
            model_start.append(np.sum(intermediate_model_start))
 
    
        index_end = np.where(new_points[:,2] >= mu)#finding the index to apply the MF
        
        #------- Nested for loop calculating the HMF function summed over the range of error with the correlating weights-----------------------------------
        model_end = []
        for i in range(len(new_points[index_end])):
            intermediate_model_end = []
            for j in range(len(delta_x[i])):
                intermediate_model_end.append(interpol_func([omega_m ,sigma_8, new_points[index_end][i,2]+delta_x[i][j]])*weights[i][j])
              
            model_end.append(np.sum(intermediate_model_end))
        
        model = 10**(np.concatenate((model_start,model_end), axis=0))#concatenate the two functions,  model converted so not in log10
        
        Lambda = model*Volume*width #Lambda in poisson distribution

        return np.sum(np.log(Lambda)*counts-Lambda)  #compute the log-likelihood
    
    
    def log_prior(theta):
        omega_m, sigma_8, mu, sigma = theta
        if 0.1 < omega_m < 1 and 0.1 < sigma_8 < 1.3 and 13.45 < mu < 14.5 and 0.01 < sigma < 0.5:  #defined range for omega_m, sigma_8, mu and sigma making propbability -inf if not in range
            return 0.0
        return -np.inf
    
    def log_probability(theta, y, yerr, counts, Volume, width, weights, delta_x, gauss, poisson):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        if gauss==1:
            return lp + log_likelihood_gauss(theta, y, yerr,  weights, delta_x)
        
        elif poisson==1:
            return lp + log_likelihood_poisson(theta, counts, Volume, width, weights, delta_x)
        
    
    pos = [Omega_m_0, Sigma_8_0, mean, std] + 1e-4 * np.random.randn(32, 4)
    
    nwalkers, ndim = pos.shape
    
    if gauss==1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(y, yerr, counts, Volume, width, weights, delta_x, gauss, poisson))
        sampler.run_mcmc(pos, itt, progress=True);
    elif poisson==1:
        print()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(y, yerr, counts, Volume, width, weights, delta_x, gauss, poisson))
        sampler.run_mcmc(pos, itt, progress=True);
        

    fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["Omega_m", "Sigma_8", "Mu", "Sigma"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.4)
    
    axes[-1].set_xlabel("step number");
    
    flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True)
    
    corner.corner(flat_samples, quantiles=(0.0015, 0.9985), levels=(0.997,), labels=labels, truths=np.mean(flat_samples, axis=0));
    
    Val = []

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [0.15, 50, 99.85])
        Val.append(mcmc[1])
        q = np.diff(mcmc)
        print(f"{labels[i]}= {mcmc[1]:.3f} with lower error = {-q[0]:.4f} and upper error = {q[1]:.4f}")
            
    return(flat_samples, sampler.get_log_prob(), Val)
                
    
    

def MF_fit_estimates(x_plots, x, y, y_err, omega_m, sigma_8, mu=0, sigma=0, gauss=0, base=1):
    
    mf = MassFunction()
    mf.update(mdef_params = {"overdensity": 200}, cosmo_params={"H0":100})
    mf.update(Mmin=np.min(x_plots[np.where(x_plots>mu)]), Mmax=np.max(x_plots[np.where(x_plots>mu)]), dlog10m=np.round(x_plots[1]-x_plots[0],10))
    mf.update(cosmo_params={"Om0": omega_m}, sigma_8=sigma_8)
    
    mf_base = MassFunction()
    mf_base.update(mdef_params = {"overdensity": 200}, cosmo_params={"H0":100})
    mf_base.update(Mmin=np.min(x_plots[np.where(x_plots>mu)]), Mmax=np.max(x_plots[np.where(x_plots>mu)]), dlog10m=np.round(x_plots[1]-x_plots[0],10))
    
    plt.figure()
    plt.title('Mass Function found from data')
    plt.xlabel(r"Log(Mass), $[h^{-1}M_\odot]$")
    plt.ylabel(r"$dn/dLog(m)$, $[h^{3}{\rm Mpc}^{-3}logM_\odot^{-1}]$")
    plt.yscale('log')
    
    if gauss==1:
        MF_g = MassFunction()
        MF_g.update(mdef_params = {"overdensity": 200}, cosmo_params={"H0":100})
        MF_g.update(Mmin=mu)
        MF_g.update(cosmo_params={"Om0": omega_m}, sigma_8=sigma_8)
    
        gg=stats.norm.pdf(x_plots[np.where(x_plots<=mu)], mu, sigma)*np.sqrt(2*np.pi)*sigma * MF_g.dndlog10m[0]
        
        plt.plot(x_plots[np.where(x_plots<=mu)],gg, 'r', label='Gaussian')
        

    plt.errorbar(x,y, yerr=y_err, label='Data from histogram', fmt='.k',  ecolor='k', elinewidth=1, capsize=1, capthick=1)
    plt.plot(np.log10(mf.m),mf.dndlog10m, 'b', label='Mass Function Fit')
    
    if base==1:
        plt.plot(np.log10(mf_base.m),mf_base.dndlog10m, 'g', label=r"Mass Function with default values of $\Omega_m$=0.307 and $\sigma_8$=0.816")
    
    plt.grid()
    plt.legend()
    
    
    













                
            