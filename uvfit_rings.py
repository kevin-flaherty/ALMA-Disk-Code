import numpy as np
from scipy.special import j0
from scipy.integrate import trapz
from astropy.io import fits
import matplotlib.pyplot as plt
import emcee
from scipy.optimize import curve_fit

def uvfit_rings(file,incl=45,PA=60.,file2=None,npeaks=1,show_mcmc=False,Nburn=1500):
    '''Fit a series of concentric rings to continuum deprojected uv data, following the method of Zhang et al. 2016. The underlying intensity distribution is assumed to be a series of concentric gaussian rings. The number of rings can be specified, as well as if there is a central depression.

    The code uses emcee (Foreman-Mackey et al.) to perform the fitting. It prints out the median, plus one-sigma noise, for each of the parameters. It also displays a figure showing the deprojected real part of the visibilities (black dots) along with the model fit in the top panel. The bottom panel shows the intensity profile, normalized to its peak, as a function of angular distance. In both panels the model defined by the median of the posteriors is shown with a red-dashed line, while 50 models randomly drawn from the posterior distribution are shown with the solid grey lines (demonstrating the range of acceptable models). 

    :param file:
    The name of the visibility fits file

    :param incl: (default = 45)
    Inclination of the disk, in degrees

    :param PA: (default = 60)
    Position angle, measured east of north, in degrees

    :param file2: (default = None)
    A second visibility fits file, to be included with the original file. This is useful if you have observations of e.g. short and long baselines taken on separate days.

    :param npeaks: (default=1)
    Specifies the complexity of the underlying model.
    npeaks = 0: a constant intensity (a0) out to a given radius (theta0, in radians)
    npeaks = 1: a Gaussian profile centered at the phase center, with a peak intensity (a0) and width of the Gaussian profile (sigma0, in radians)
    npeaks = 1.5: Same as npeaks=1 (a0, sigma0), but with a central depression. The depression is a constant intensity out to a cavity radius (theta_cav, in radians), depressed by a multiplicative factor (delta)
    npeaks = 2: Central peak (a0, sigma0), along with another Gaussian ring at a certain distance from the phase center (position rho1, in kilo-lambda, peak intensity a1, width sigma1 in radians)
    npeaks = 2.5: Same as npeaks=2 (a0, sigma0, rho1, a1, sigma1), but with a central depression (theta_cav, delta)
    npeaks = 3: Central Gaussian peak (a0, sigma0), and two concentric rings (rho1, a1, sigma1, rho2, a2, sigma2)
    npeaks = 3.5: Same as npeaks=3 (a0, sigma0, rho1, a1, sigma1, rho2, a2, sigma2), but with a central depression (theta_cav, delta)
    **In all of these models the amplitudes are always assumed to be positive**

    :param show_mcmc: (default = False)
    Show a figure with the movement of the walkers? Can be useful to check for covergance (ie. if the walkers are still moving substantially), if there are walkers that haven't fully converged (often the case, and will lead to an overestimate of the uncertainty), or to check for degeneracies.

    :param Nburn: (default = 1500)
    The number of steps to be thown out as burn-in. Convergance on a final solution is one of the largest challenges with this model fitting, and increasing the number of burn-in step can help with the convergance.

'''
    
    im = fits.open(file)
    hdr = im[0].header
    data = im[0].data
    u,v = data['UU'],data['VV']
    freq0 = hdr['crval4']
    klam = freq0/1e3
    vis=(data.data).squeeze()
    im.close()
    
    if len(vis.shape)==3:
        real = (vis[:,0,0]+vis[:,1,0])/2.
        imag = (vis[:,0,1]+vis[:,1,1])/2.
    if len(vis.shape)==2:
        real = vis[:,0]
        imag = vis[:,1]
        
    u *= klam #convert from units of seconds to units of kilo-lambda
    v *= klam #convert from units of seconds to units of kilo-lambda

    if file2 is not None:
        im2 = fits.open(file)
        hdr2 = im2[0].header
        data2 = im2[0].data
        u2,v2 = data2['UU'],data2['VV']
        freq0 = hdr2['crval4']
        klam2 = freq0/1e3
        vis2 = (data2.data).squeeze()
        im2.close()
        
        if len(vis.shape)==3:
            real2 = (vis2[:,0,0]+vis2[:,1,0])/2.
            imag2 = (vis2[:,0,1]+vis2[:,1,1])/2.
        if len(vis.shape)==2:
            real2 = vis2[:,0]
            imag2 = vis2[:,1]
        u2 *= klam2 #convert from units of seconds to units of kilo-lambda
        v2 *= klam2 #convert from units of seconds to units of kilo-lambda
        
        real = np.concatenate((real,real2))
        imag = np.concatenate((imag,imag2))
        weight_real = np.concatenate((weight_real,weight_real2))
        weight_imag = np.concatenate((weight_imag,weight_imag2))
        u = np.concatenate((u,u2))
        v = np.concatenate((v,v2))


    incl,PA = np.radians(incl),np.radians(PA)

    #deprojected u and v distance
    ud = (u*np.cos(PA)-v*np.sin(PA))*np.cos(incl)
    vd = u*np.sin(PA)+v*np.cos(PA)
    rho = np.sqrt(ud**2+vd**2.)

    #bin down the visibilities to increase S/N and decrease computation time
    nbins = 500
    uvmin = rho.min()
    uvmax = rho.max()
    uv_bin = np.arange(nbins)*(uvmax-uvmin)/nbins+uvmin #center of uv bins
    duv = (uvmax-uvmin)/nbins
    real_bin = np.zeros(nbins)
    imag_bin = np.zeros(nbins)
    weight_real_bin = np.zeros(nbins)
    weight_imag_bin = np.zeros(nbins)
    for i in range(int(nbins)):
        w = (rho > uv_bin[i]-duv/2.) & (rho < uv_bin[i]+duv/2.) & (real != 0) & (imag != 0)
        if w.sum() > 5.:
            real_bin[i] = real[w].mean()
            imag_bin[i] = imag[w].mean()
            weight_real_bin[i] = 1/((real[w].std()/np.sqrt(w.sum()))**2.)
            weight_imag_bin[i] = 1/((imag[w].std()/np.sqrt(w.sum()))**2.)
        else:
            real_bin[i] = 0
            imag_bin[i] = 0
            weight_real_bin[i] = 0.
            weight_imag_bin[i] = 0.

    #Set up the walkers
    nwalkers = 30
    if npeaks == 0:
        ndim = 2
        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.random.rand(nwalkers)*(10-6)+6
        p0[:,1] = np.random.rand(nwalkers)*(np.log10(1/(uv_bin.min()*1e3))-np.log10(1/(uv_bin.max()*1e3)))+np.log10(1/(uv_bin.max()*1e3))
        names = ['a0','theta0']
    if npeaks ==1:
        ndim = 2
        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.random.rand(nwalkers)*(5.5-4.5)+4.5 #amplitude of the feautre
        p0[:,1] = np.random.rand(nwalkers)*(-5.6+6.)-6. #angular size, in radians
        names=['log(a0)','log(sigma0)']
    if npeaks == 1.5:
        ndim = 4
        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.random.rand(nwalkers)*(5.5-4.5)+4.5 #amplitude of the feautre
        p0[:,1] = np.random.rand(nwalkers)*(-5.6+6.)-6. #angular size, in radians
        p0[:,2] = np.random.rand(nwalkers)*(np.log10(1/(uv_bin.min()*1e3))-np.log10(1/(uv_bin.max()*1e3)))+np.log10(1/(uv_bin.max()*1e3)) #cavity size
        p0[:,3] = np.random.rand(nwalkers) #cavity depletion
        names=['log(a0)','log(sigma0)','log(theta_cav)','delta']
    if npeaks == 2:
        ndim = 5
        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.random.rand(nwalkers)*(5.5-4.5)+4.5
        p0[:,1] = np.random.rand(nwalkers)*(-5.6+6)-6
        p0[:,2] = np.random.rand(nwalkers)*(5.6-5.3)+5.3
        p0[:,3] = np.random.rand(nwalkers)*(3.-2)+2
        p0[:,4] = np.random.rand(nwalkers)*(-5+6)-6
        names=['log(a0)','log(sigma0)','log(rho1)','log(a1)','log(sigma1)']
    if npeaks == 2.5:
        ndim = 7
        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.random.rand(nwalkers)*(5.5-4.5)+4.5
        p0[:,1] = np.random.rand(nwalkers)*(-5.6+6)-6
        p0[:,2] = np.random.rand(nwalkers)*(5.6-5.3)+5.3
        p0[:,3] = np.random.rand(nwalkers)*(3.-2)+2
        p0[:,4] = np.random.rand(nwalkers)*(-5+6)-6
        p0[:,5] = np.random.rand(nwalkers)*(np.log10(1/(uv_bin.min()*1e3))-np.log10(1/(uv_bin.max()*1e3)))+np.log10(1/(uv_bin.max()*1e3)) #cavity size
        p0[:,6] = np.random.rand(nwalkers) #cavity depletion
        names=['log(a0)','log(sigma0)','log(rho1)','log(a1)','log(sigma1)','log(theta_cav)','delta']
    if npeaks == 3:
        ndim = 8
        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.random.rand(nwalkers)*(5.5-4.5)+4.5 #-1.16e5
        p0[:,1] = np.random.rand(nwalkers)*(-5.6+6)-6 #1.06e-6
        p0[:,2] = np.random.rand(nwalkers)*(5.6-5.3)+5.3 #298e3
        p0[:,3] = np.random.rand(nwalkers)*(3-2.)+2 #-6.4e4
        p0[:,4] = np.random.rand(nwalkers)*(-5+6)-6 #1.89e-6
        p0[:,5] = np.random.rand(nwalkers)*(5.9-5.7)+5.7 #709e3
        p0[:,6] = np.random.rand(nwalkers)*(3-2)+2 #-3.82e4
        p0[:,7] = np.random.rand(nwalkers)*(-5+6)-6 #1.69e-6
        names=['log(a0)','log(sigma0)','log(rho1)','log(a1)','log(sigma1)','log(rho2)','log(a2)','log(sigma2)']
    if npeaks == 3.5:
        ndim = 10
        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.random.rand(nwalkers)*(5.5-4.5)+4.5 #-1.16e5
        p0[:,1] = np.random.rand(nwalkers)*(-5.6+6)-6 #1.06e-6
        p0[:,2] = np.random.rand(nwalkers)*(5.6-5.3)+5.3 #298e3
        p0[:,3] = np.random.rand(nwalkers)*(3-2.)+2 #-6.4e4
        p0[:,4] = np.random.rand(nwalkers)*(-5+6)-6 #1.89e-6
        p0[:,5] = np.random.rand(nwalkers)*(5.9-5.7)+5.7 #709e3
        p0[:,6] = np.random.rand(nwalkers)*(3-2)+2 #-3.82e4
        p0[:,7] = np.random.rand(nwalkers)*(-5+6)-6 #1.69e-6
        p0[:,8] = np.random.rand(nwalkers)*(np.log10(1/(uv_bin.min()*1e3))-np.log10(1/(uv_bin.max()*1e3)))+np.log10(1/(uv_bin.max()*1e3)) #cavity size
        p0[:,9] = np.random.rand(nwalkers) #cavity depletion
        names=['log(a0)','log(sigma0)','log(rho1)','log(a1)','log(sigma1)','log(rho2)','log(a2)','log(sigma2)','log(theta_cav)','delta']


    #Run EMCEE
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=[npeaks,uv_bin,real_bin,weight_real_bin])
    Nburn, Nrun = 1500,200
    pos,prob,state = sampler.run_mcmc(p0,Nburn)
    print 'Burn in complete'
    sampler.reset()
    sampler.run_mcmc(pos,Nrun)
    

    #Display the results
    params=[]
    Nrand = 50 #number of models randomly drawn from the posterior
    full_params = np.zeros((Nrand,ndim))
    indices = np.random.choice(Nrun*nwalkers,Nrand,replace=False)
    for i in range(ndim):
        print names[i],': ',np.median(sampler.flatchain[:,i]),' +- ',sampler.flatchain[:,i].std()
        params.append(np.median(sampler.flatchain[:,i]))
        full_params[:,i] = sampler.flatchain[indices,i]
    plt.figure()
    plt.subplot(211)
    plot_model(file,incl=np.degrees(incl),PA=np.degrees(PA),params=params,npeaks=npeaks,color='r',ls='--')
    for i in range(Nrand):
        plot_model2(uv_bin,full_params[i,:],npeaks=npeaks,alpha=.05,color='k')
    plot_model2(uv_bin,params,npeaks=npeaks,color='r',ls='--')
    plt.subplot(212)
    for i in range(Nrand):
        plot_intensity(full_params[i,:],npeaks=npeaks,color='k',alpha=.05)
    plot_intensity(params,npeaks=npeaks,color='r',ls='--')

    if show_mcmc:
        plt.figure()
        for j in range(ndim):
            plt.subplot(ndim,2,2*j+1)
            for i in range(nwalkers):
                plt.plot(sampler.chain[i,:,j],alpha=.5)
                plt.axhline(np.median(sampler.flatchain[:,j]),color='k',lw=3)
                plt.axhline(np.median(sampler.flatchain[:,j])+3*sampler.flatchain[:,j].std(),color='k',ls=':',lw=3)
                plt.axhline(np.median(sampler.flatchain[:,j])-3*sampler.flatchain[:,j].std(),color='k',ls=':',lw=3)
            if j<ndim-1:
                plt.subplot(ndim,2,2*j+2)
                for i in range(nwalkers):
                    plt.plot(sampler.chain[i,:,j],sampler.chain[i,:,j+1],alpha=.5)



def lnprob(p,npeaks,uv_bin,real_bin,weight_real_bin):
    '''Calculate the log-likelihood (=-.5*chi-squared) for a given set of model parameters and a given set of data.'''
    if npeaks==0:
        if 10**(p[1])<(1/(5*uv_bin.max()*1e3)) or 10**(p[1])>(5/(uv_bin.min()*1e3)):
            return -np.inf
        else:
            model_vis = calc_model_vis0(uv_bin,p[0],p[1])
    if npeaks==1:
        if 10.**(p[1]) < (1/(5*uv_bin.max()*1e3)) or 10.**(p[1])>(5/(uv_bin.min()*1e3)):
            return -np.inf
        else:
            model_vis = calc_model_vis1(uv_bin,p[0],p[1])
    if npeaks==1.5:
        if 10.**(p[1]) < (1/(2*uv_bin.max()*1e3)) or 10.**(p[1])>(2/(uv_bin.min()*1e3)) or 10.**(p[2])>(2/(uv_bin.min()*1e3)) or 10.**(p[2])<(1/(2*uv_bin.max()*1e3)) or p[3]<0 or p[3]>1:
            return -np.inf
        else:
            model_vis = calc_model_vis1b(uv_bin,p[0],p[1],p[2],p[3])
    if npeaks==2:
        if 10.**(p[2])>uv_bin.max()*1e3 or 10.**(p[1])<(1/(5*uv_bin.max()*1e3)) or 10.**(p[1])>(5/(uv_bin.min()*1e3)) or 10.**(p[4])<(1/(5*uv_bin.max()*1e3)) or 10.**(p[4])>(5/(uv_bin.min()*1e3)):
            return -np.inf
        else:
            model_vis = calc_model_vis2(uv_bin,p[0],p[1],p[2],p[3],p[4])
    if npeaks == 2.5:
        if 10.**(p[2])>uv_bin.max()*1e3 or 10.**(p[1])<(1/(5*uv_bin.max()*1e3)) or 10.**(p[1])>(5/(uv_bin.min()*1e3)) or 10.**(p[4])<(1/(5*uv_bin.max()*1e3)) or 10.**(p[4])>(5/(uv_bin.min()*1e3)) or 10.**(p[5])<(1/(5*uv_bin.max()*1e3)) or 10.**(p[5])>(5/(uv_bin.min()*1e3)) or p[6]<0 or p[6]>1:
            return -np.inf
        else:
            model_vis = calc_model_vis2b(uv_bin,p[0],p[1],p[2],p[3],p[4],p[5],p[6])
    if npeaks==3:
        if 10.**(p[2])>uv_bin.max()*1e3 or 10.**(p[2])<uv_bin.min()*1e3 or 10.**(p[5])>uv_bin.max()*1e3 or 10.**(p[1])<(1/(5*uv_bin.max()*1e3)) or 10.**(p[1])>(5/(uv_bin.min()*1e3)) or 10.**(p[4])<(1/(5*uv_bin.max()*1e3)) or 10.**(p[4])>(5/(uv_bin.min()*1e3)) or 10.**(p[7])<(1/(5*uv_bin.max()*1e3)) or 10.**(p[7])>(5/(uv_bin.min()*1e3)) or 10.**(p[5])<uv_bin.min()*1e3 or p[0]>10 or p[3]>10 or p[6]>10 or p[0]<0 or p[3]<0 or p[6]<0:
            return -np.inf
        else:
            model_vis = calc_model_vis3(uv_bin,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7])

    if npeaks==3.5:
        if 10.**(p[2])>uv_bin.max()*1e3 or 10.**(p[2])<uv_bin.min()*1e3 or 10.**(p[5])>uv_bin.max()*1e3 or 10.**(p[1])<(1/(5*uv_bin.max()*1e3)) or 10.**(p[1])>(5/(uv_bin.min()*1e3)) or 10.**(p[4])<(1/(5*uv_bin.max()*1e3)) or 10.**(p[4])>(5/(uv_bin.min()*1e3)) or 10.**(p[7])<(1/(5*uv_bin.max()*1e3)) or 10.**(p[7])>(5/(uv_bin.min()*1e3)) or 10.**(p[5])<uv_bin.min()*1e3 or p[0]>10 or p[3]>10 or p[6]>10 or p[0]<0 or p[3]<0 or p[6]<0 or 10.**(p[8])<(1/(5*uv_bin.max()*1e3)) or 10.**(p[8])>(5/(uv_bin.min()*1e3)) or p[9]<0 or p[9]>1:
            return -np.inf
        else:
            model_vis = calc_model_vis3b(uv_bin,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9])
        
    return  -.5*((model_vis-real_bin)**2*weight_real_bin).sum()

def calc_model_vis0(uv_bin,a0,theta0):
    '''Constant intensity out to a certain angular size'''
    theta = np.linspace(0,1.3e-3/12,1000)
    intensity = np.zeros((len(theta)))
    intensity[theta<(10.**(theta0))] = -10**(a0)
    mod_vis = np.zeros(len(uv_bin))
    for i in range(len(uv_bin)):
        mod_vis[i] = 2*np.pi*trapz(theta,intensity*theta*j0(2*np.pi*uv_bin[i]*1e3*theta))

    return mod_vis

def calc_model_vis1(uv_bin,a0,sig0):
    '''One main ring'''
    a0, sig0 = 10.**(a0), 10.**(sig0)
    theta = np.linspace(0,1.3e-3/12,1000)
    intensity = -(a0/(np.sqrt(2*np.pi)*sig0)*np.exp(-theta**2./(2*sig0**2.)))
    mod_vis = np.zeros(len(uv_bin))
    for i in range(len(uv_bin)):
        mod_vis[i] = 2*np.pi*trapz(theta,intensity*theta*j0(2*np.pi*uv_bin[i]*1e3*theta))

    return mod_vis

def calc_model_vis1b(uv_bin,a0,sig0,Rcav,delta):
    '''One main ring, with a central depression'''
    a0, sig0, Rcav = 10.**(a0), 10.**(sig0), 10**(Rcav)
    theta = np.linspace(0,1.3e-3/12,1000)
    intensity = -(a0/(np.sqrt(2*np.pi)*sig0)*np.exp(-theta**2./(2*sig0**2.)))
    intensity[theta<Rcav] = delta*intensity[theta<Rcav].min()
    mod_vis = np.zeros(len(uv_bin))
    for i in range(len(uv_bin)):
        mod_vis[i] = 2*np.pi*trapz(theta,intensity*theta*j0(2*np.pi*uv_bin[i]*1e3*theta))

    return mod_vis


def calc_model_vis2(uv_bin,a0,sig0,rho1,a1,sig1):
    '''Two rings'''
    a0,sig0,rho1,a1,sig1 = 10.**(a0),10.**(sig0),10.**(rho1),10**(a1),10.**(sig1)
    #theta = np.arange(0,1/(uv_bin.min()*1e3),1/(uv_bin.max()*1e3))
    theta = np.linspace(0,1.3e-3/12,1000)
    intensity = -(a0/(np.sqrt(2*np.pi)*sig0)*np.exp(-theta**2./(2*sig0**2.))+np.cos(2*np.pi*theta*rho1)*a1/(np.sqrt(2*np.pi)*sig1)*np.exp(-theta**2./(2*sig1**2.)))
    mod_vis = np.zeros(len(uv_bin))
    for i in range(len(uv_bin)):
        mod_vis[i] = 2*np.pi*trapz(theta,intensity*theta*j0(2*np.pi*uv_bin[i]*1e3*theta))

    return mod_vis

def calc_model_vis2b(uv_bin,a0,sig0,rho1,a1,sig1,Rcav,delta):
    '''Two rings, with central cavity'''
    a0,sig0,rho1,a1,sig1,Rcav = 10.**(a0),10.**(sig0),10.**(rho1),10**(a1),10.**(sig1),10.**(Rcav)
    theta = np.linspace(0,1.3e-3/12,1000)
    intensity = -(a0/(np.sqrt(2*np.pi)*sig0)*np.exp(-theta**2./(2*sig0**2.))+np.cos(2*np.pi*theta*rho1)*a1/(np.sqrt(2*np.pi)*sig1)*np.exp(-theta**2./(2*sig1**2.)))
    intensity[theta<Rcav] = delta*intensity[theta<Rcav][-1]
    mod_vis = np.zeros(len(uv_bin))
    for i in range(len(uv_bin)):
        mod_vis[i] = 2*np.pi*trapz(theta,intensity*theta*j0(2*np.pi*uv_bin[i]*1e3*theta))

    return mod_vis



def calc_model_vis3(uv_bin,a0,sig0,rho1,a1,sig1,rho2,a2,sig2):
    '''Three rings'''
    a0,sig0,rho1,a1,sig1,rho2,a2,sig2 = 10.**(a0),10.**(sig0),10.**(rho1),10.**(a1),10.**(sig1),10.**(rho2),10.**(a2),10.**(sig2)
    theta = np.linspace(0,1.3e-3/12,1000)
    intensity = -(a0/(np.sqrt(2*np.pi)*sig0)*np.exp(-theta**2./(2*sig0**2.))+np.cos(2*np.pi*theta*rho1)*a1/(np.sqrt(2*np.pi)*sig1)*np.exp(-theta**2./(2*sig1**2.))+np.cos(2*np.pi*theta*rho2)*a2/(np.sqrt(2*np.pi)*sig2)*np.exp(-theta**2./(2*sig2**2.)))
    mod_vis = np.zeros(len(uv_bin))
    for i in range(len(uv_bin)):
        mod_vis[i] = 2*np.pi*trapz(theta,intensity*theta*j0(2*np.pi*uv_bin[i]*1e3*theta))

    return mod_vis

def calc_model_vis3b(uv_bin,a0,sig0,rho1,a1,sig1,rho2,a2,sig2,Rcav,delta):
    '''Three rings with central cavity'''
    a0,sig0,rho1,a1,sig1,rho2,a2,sig2,Rcav = 10.**(a0),10.**(sig0),10.**(rho1),10.**(a1),10.**(sig1),10.**(rho2),10.**(a2),10.**(sig2),10.**(Rcav)
    theta = np.linspace(0,1.3e-3/12,1000)
    intensity = -(a0/(np.sqrt(2*np.pi)*sig0)*np.exp(-theta**2./(2*sig0**2.))+np.cos(2*np.pi*theta*rho1)*a1/(np.sqrt(2*np.pi)*sig1)*np.exp(-theta**2./(2*sig1**2.))+np.cos(2*np.pi*theta*rho2)*a2/(np.sqrt(2*np.pi)*sig2)*np.exp(-theta**2./(2*sig2**2.)))
    intensity[theta<Rcav] = delta*intensity[theta<Rcav][-1]
    mod_vis = np.zeros(len(uv_bin))
    for i in range(len(uv_bin)):
        mod_vis[i] = 2*np.pi*trapz(theta,intensity*theta*j0(2*np.pi*uv_bin[i]*1e3*theta))

    return mod_vis



def plot_intensity(params,npeaks,**kwargs):
    ''' Given a set of model parameters, compute the intensity as a function of angular distance and plot the results.'''

    import matplotlib.pyplot as plt
    theta = np.linspace(0,1e-5,1000)
    for i in range(len(params)):
        params[i] = 10.**(params[i])
    if npeaks==0:
        intensity = np.zeros(len(theta))
        intensity[theta<params[1]] = params[0]
    if npeaks==1:
        intensity = params[0]/(np.sqrt(2*np.pi)*params[1])*np.exp(-theta**2./(2*params[1]**2.))
    if npeaks == 1.5:
        intensity = params[0]/(np.sqrt(2*np.pi)*params[1])*np.exp(-theta**2./(2*params[1]**2.))
        intensity[theta<params[2]]=np.log10(params[3])*intensity[theta<params[2]].min()
    if npeaks ==2 :
        intensity = (params[0]/(np.sqrt(2*np.pi)*params[1])*np.exp(-theta**2./(2*params[1]**2.))+np.cos(2*np.pi*theta*params[2])*params[3]/(np.sqrt(2*np.pi)*params[4])*np.exp(-theta**2./(2*params[4]**2.)))
    if npeaks==2.5:
        intensity = (params[0]/(np.sqrt(2*np.pi)*params[1])*np.exp(-theta**2./(2*params[1]**2.))+np.cos(2*np.pi*theta*params[2])*params[3]/(np.sqrt(2*np.pi)*params[4])*np.exp(-theta**2./(2*params[4]**2.)))
        intensity[theta<params[-2]]=np.log10(params[-1])*intensity[theta<params[-2]].min()
    if npeaks == 3:
        intensity = (params[0]/(np.sqrt(2*np.pi)*params[1])*np.exp(-theta**2./(2*params[1]**2.))+np.cos(2*np.pi*theta*params[2])*params[3]/(np.sqrt(2*np.pi)*params[4])*np.exp(-theta**2./(2*params[4]**2.))+np.cos(2*np.pi*theta*params[5])*params[6]/(np.sqrt(2*np.pi)*params[7])*np.exp(-theta**2./(2*params[7]**2.)))
    if npeaks == 3.5:
        intensity = (params[0]/(np.sqrt(2*np.pi)*params[1])*np.exp(-theta**2./(2*params[1]**2.))+np.cos(2*np.pi*theta*params[2])*params[3]/(np.sqrt(2*np.pi)*params[4])*np.exp(-theta**2./(2*params[4]**2.))+np.cos(2*np.pi*theta*params[5])*params[6]/(np.sqrt(2*np.pi)*params[7])*np.exp(-theta**2./(2*params[7]**2.)))
        intensity[theta<params[-2]] = np.log10(params[-1])*intensity[theta<params[-2]][-1]

    plt.rc('axes',lw=2)
    plt.plot(3600*np.degrees(theta),intensity/intensity.max(),lw=3,**kwargs)
    plt.xlabel(r'$\theta$ (")',fontsize=16,fontweight='bold')
    plt.ylabel('Intensity',fontsize=16,fontweight='bold')
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontweight('bold')

def plot_model(file,params,incl=45.,PA=60.,file2=None,npeaks=1,**kwargs):
    ''' Plot the model along with the data'''
    im = fits.open(file)
    hdr = im[0].header
    data = im[0].data
    u,v = data['UU'],data['VV']
    freq0 = hdr['crval4']
    klam = freq0/1e3
    vis=(data.data).squeeze()
    
    if len(vis.shape)==3:
        real = (vis[:,0,0]+vis[:,1,0])/2.
        imag = (vis[:,0,1]+vis[:,1,1])/2.
        #weight_real = vis[:,0,2]
        #weight_imag = vis[:,1,2]
    if len(vis.shape)==2:
        real = vis[:,0]
        imag = vis[:,1]
    u *= klam #convert from units of seconds to units of kilo-lambda
    v *= klam #convert from units of seconds to units of kilo-lambda

    if file2 is not None:
        im2 = fits.open(file)
        hdr2 = im2[0].header
        data2 = im2[0].data
        u2,v2 = data2['UU'],data2['VV']
        freq0 = hdr2['crval4']
        klam2 = freq0/1e3
        vis2 = (data2.data).squeeze()

        real2 = (vis2[:,0,0]+vis2[:,1,0])/2.
        imag2 = (vis2[:,0,1]+vis2[:,1,1])/2.
        weight_real2 = vis2[:,0,2]
        weight_imag2 = vis2[:,1,2]
        u2 *= klam2 #convert from units of seconds to units of kilo-lambda
        v2 *= klam2 #convert from units of seconds to units of kilo-lambda
        
        real = np.concatenate((real,real2))
        imag = np.concatenate((imag,imag2))
        weight_real = np.concatenate((weight_real,weight_real2))
        weight_imag = np.concatenate((weight_imag,weight_imag2))
        u = np.concatenate((u,u2))
        v = np.concatenate((v,v2))


    incl,PA = np.radians(incl),np.radians(PA)

    #deprojected u and v distance
    ud = (u*np.cos(PA)-v*np.sin(PA))*np.cos(incl)
    vd = u*np.sin(PA)+v*np.cos(PA)
    rho = np.sqrt(ud**2+vd**2.)

    #bin down the visibilities to increase S/N and decrease computation time
    nbins = 500
    uvmin = rho.min()
    uvmax = rho.max()
    uv_bin = np.arange(nbins)*(uvmax-uvmin)/nbins+uvmin #center of uv bins
    duv = (uvmax-uvmin)/nbins
    real_bin = np.zeros(nbins)
    imag_bin = np.zeros(nbins)
    weight_real_bin = np.zeros(nbins)
    weight_imag_bin = np.zeros(nbins)
    for i in range(int(nbins)):
        w = (rho > uv_bin[i]-duv/2.) & (rho < uv_bin[i]+duv/2.) & (real != 0) & (imag != 0)
        if w.sum() > 5.:
            real_bin[i] = real[w].mean()
            imag_bin[i] = imag[w].mean()
            weight_real_bin[i] = 1/((real[w].std()/np.sqrt(w.sum()))**2.)
            weight_imag_bin[i] = 1/((imag[w].std()/np.sqrt(w.sum()))**2.)
        else:
            real_bin[i] = 0
            imag_bin[i] = 0
            weight_real_bin[i] = 0.
            weight_imag_bin[i] = 0.

    plt.rc('axes',lw=2)
    wuse = (weight_real_bin!=0) & (weight_imag_bin!=0)
    plt.plot(uv_bin[wuse],real_bin[wuse],'.k')
    #plt.plot(uv_bin,imag_bin,'.r')
    if npeaks==0:
        plt.plot(uv_bin,calc_model_vis0(uv_bin,params[0],params[1]),lw=2,**kwargs)
        print 'chi-squared: ',-2*(lnprob([params[0],params[1]],npeaks,uv_bin,real_bin,weight_real_bin))/(nbins-2.)

    if npeaks==1:
        plt.plot(uv_bin,calc_model_vis1(uv_bin,params[0],params[1]),lw=2,**kwargs)
        print 'chi-squared: ',-2*(lnprob([params[0],params[1]],npeaks,uv_bin,real_bin,weight_real_bin))/(nbins-2.)

    if npeaks==1.5:
        plt.plot(uv_bin,calc_model_vis1b(uv_bin,params[0],params[1],params[2],params[3]),lw=2,**kwargs)
        print 'chi-squared: ',-2*(lnprob([params[0],params[1],params[2],params[3]],npeaks,uv_bin,real_bin,weight_real_bin))/(nbins-5.)

    if npeaks ==2:
        plt.plot(uv_bin,calc_model_vis2(uv_bin,params[0],params[1],params[2],params[3],params[4]),lw=2,**kwargs)
        print 'chi-squared: ',-2*(lnprob([params[0],params[1],params[2],params[3],params[4]],npeaks,uv_bin,real_bin,weight_real_bin))/(nbins-5.)

    if npeaks==2.5:
        plt.plot(uv_bin,calc_model_vis2b(uv_bin,params[0],params[1],params[2],params[3],params[4],params[5],params[6]),lw=2,**kwargs)
        print 'chi-squared: ',-2*(lnprob([params[0],params[1],params[2],params[3],params[4],params[5],params[6]],npeaks,uv_bin,real_bin,weight_real_bin))/(nbins-7.)

    if npeaks==3.:
        plt.plot(uv_bin,calc_model_vis3(uv_bin,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]),lw=2,**kwargs)
        print 'chi-squared: ',-2*(lnprob([params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]],npeaks,uv_bin,real_bin,weight_real_bin))/(nbins-8.)
        
    if npeaks==3.5:
        plt.plot(uv_bin,calc_model_vis3b(uv_bin,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9]),lw=2,**kwargs)
        print 'chi-squared: ',-2*(lnprob([params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9]],npeaks,uv_bin,real_bin,weight_real_bin))/(nbins-10.)


    plt.xlabel('uv distance (k$\lambda$)',fontweight='bold',fontsize=16)
    plt.ylabel('Visibility (Jy)',fontweight='bold',fontsize=16)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontweight('bold')

def plot_model2(uv_bin,params,npeaks=1,**kwargs):
    '''Similar to plot_model, but only plots the model, based on the specified array of uv spacings. This is useful for plotting many models (e.g. showing the range of models allowed by the posterior distributions), whithout plotting the data every time.'''

    if npeaks==0:
        plt.plot(uv_bin,calc_model_vis0(uv_bin,params[0],params[1]),lw=2,**kwargs)

    if npeaks==1:
        plt.plot(uv_bin,calc_model_vis1(uv_bin,params[0],params[1]),lw=2,**kwargs)

    if npeaks==1.5:
        plt.plot(uv_bin,calc_model_vis1b(uv_bin,params[0],params[1],params[2],params[3]),lw=2,**kwargs)

    if npeaks ==2:
        plt.plot(uv_bin,calc_model_vis2(uv_bin,params[0],params[1],params[2],params[3],params[4]),lw=2,**kwargs)

    if npeaks==2.5:
        plt.plot(uv_bin,calc_model_vis2b(uv_bin,params[0],params[1],params[2],params[3],params[4],params[5],params[6]),lw=2,**kwargs)

    if npeaks==3.:
        plt.plot(uv_bin,calc_model_vis3(uv_bin,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]),lw=2,**kwargs)
        
    if npeaks==3.5:
        plt.plot(uv_bin,calc_model_vis3b(uv_bin,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9]),lw=2,**kwargs)


    plt.xlabel('uv distance (k$\lambda$)',fontweight='bold',fontsize=16)
    plt.ylabel('Visibility (Jy)',fontweight='bold',fontsize=16)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontweight('bold')



