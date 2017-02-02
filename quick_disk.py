#calculate chi-squared of best fit...

def quick_disk(file,size=10.,vsys=None,input_model=None,PA=312,incl=48,niter=5,mstar=2.0,distance=122,return_spectrum=False,offs=[0.,0.],runquick=False):
    ''' Based on Scoville et al. 1983, this code is a quick estimate of the spatial distribution of the flux emitted by a flat rotating disk, based on the ratio of modeled intensity to the observed image plane intensity. By including velocity information it can pull out more detailed structure than simple visual inspection.

    Any results should be taken with a small grain of salt since this code does not do the full radiative transfer calculation and does not fit to the visibilities. The results will be distorted by the beam, which is assumed to be circular even though it could be highly elliptical, and any match to the image plane will be distorted by cleaning artifacts in the data. Also, the results derived by this code have not been rigorously tested against more detailed radiative transfer models. Also, it may be missing small structures since it relies on the cleaned map than the full visibility data set. It is useful as a first pass through a data set to determine if you need to model e.g. a rising/falling surface density profile, or multiple narrow rings vs a single broad ring. 

    The code returns a plot showing the fit in three channels (central channel, 1/4th channel and 3/4ths channel). The left panel shows the data (with 3-sigma contour marked by dashed line). The central panel shows model (with 3-sigma contour marked by dashed line). The right panel shows the residuals with contours at 3,5,7,etc sigma (with 3-sigma flux contour from the data). In all three panels the grey-scale is the same and stretches over the dynamic range of the data. The bottom panel shows both intensity of the rings used in the model, as well as an estimate of the surface density profile, derived assuming Sigma~I/R^(-.5). The units on the intensity and surface density are arbitrary. 
    
    :param file:
    A fits file with your image. Assumes that the image follows the format of standard ALMA images.

    :param size:
    Size of the region in the image to fit, in arcseconds. The fit extends over -size/2 to size/2 in both RA and Dec.

    :param vsys:
    Velocity of the central star, in km/sec

    :param input_model:
    An input model to use as a starting point. This model can come from fitting the moment 0 map with quick_disk_cont. It can also come from something else, although it must be sampled at the spatial positions defined in this code.

    :param PA:
    Position angle of the major axis of the disk, measured east of north

    :param incl:
    Inclination of the disk

    :param niter:
    Number of iterations to perform

    :param mstar:
    Stellar mass in units of solar mass

    :param distance:
    Distance to the disk, in units of parsecs

    :param return_spectrum:
    If set to True then the code returns the spectrum rather than the model emissivities. Useful for comparing to the data. Both the velocities of each channel, plus the spectrum, are returned [e.g. velo,spec = quick_disk('myfile.fits',return_spectrum=True)]. Velocity is in units of km/sec and is defined relative to line center (ie. you will have to add any systemic velocity)

    :param runquick:
    If set to True, then every 4th pixel in the image is modeled. This greatly speeds up computation time without an enourmous sacrifice in quality of the fit (since a single beam is generally >4 pixels across, this sampling still measures the flux in every beam)

'''
    from astropy.io import fits
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.constants import G,pc,M_sun

    # - Read in the data
    im = fits.open(file)
    data = im[0].data.squeeze()
    hdr = im[0].header
    im.close()
    ra = 3600*hdr['cdelt1']*(np.arange(hdr['naxis1'])-hdr['naxis1']/2.-0.5)
    dec = 3600*hdr['cdelt2']*(np.arange(hdr['naxis2'])-hdr['naxis2']/2.-0.5)
    noise = calc_noise(data,np.round(size/(3600.*hdr['cdelt1'])))
    ira = np.abs(ra) < size/2.
    ide = np.abs(dec) < size/2.
    data = data[:,:,ira]
    data = data[:,ide,:]
    ra = ra[ira]-offs[0]
    dec = dec[ide]-offs[1]
    dra = np.abs(3600*hdr['cdelt1'])
    ddec = np.abs(3600*hdr['cdelt2'])
    xnpix = ra.shape[0]
    ynpix = dec.shape[0]

    if runquick:
        data = data[:,::4,::4]
        ra = ra[::4]
        dec = dec[::4]
        dra*=4
        ddec*=4
        xnpix = len(ra)#xnpix/4+1
        ynpix = len(dec)#ynpix/4+1

    PA,incl = np.radians(PA),np.radians(incl)

    if (hdr['ctype3'] == 'VELO-LSR') or (hdr['ctype3']=='VELO-OBS'):
        velo = ((np.arange(hdr['naxis3'])+1-hdr['crpix3'])*hdr['cdelt3']+hdr['crval3'])*1e2#-vsys*1e5
        if vsys is None:
            velo -= np.median(velo)
        else:
            velo -= vsys*1e5
        dv = np.abs(hdr['cdelt3'])*1e2 #channel width in cm/s
        nv = velo.shape[0]
    else:
        freq = (np.arange(hdr['naxis3'])+1-hdr['crpix3'])*hdr['cdelt3']+hdr['crval3']
        try:
            velo = (hdr['restfrq']-freq)/hdr['restfrq']*2.99e10
        except KeyError:
            velo = (np.median(freq)-freq)/np.median(freq)*2.99e10
        if vsys is not None:
            velo -= vsys*1e5
        dv = np.abs(velo[1]-velo[0])
        nv = len(velo)

    # - Beam size
    sigma = np.mean([3600*hdr['bmaj']/(2*np.sqrt(2*np.log(2))),3600*hdr['bmin']/(2*np.sqrt(2*np.log(2)))])


    # initial model
    #intrinsic model
    ntheta = 100
#    nr = (float(size)/2*np.sqrt(2.)/(sigma/5.)).astype(int)
    nr = (float(size)/2/(sigma/5.)).astype(int)
    #radius = np.linspace(0.,float(size)/2*np.sqrt(2.),nr)
    radius = np.linspace(0.,float(size)/2.,nr)
    dr = radius[1]-radius[0]#radius-np.roll(radius,1)
    #dr[0] = 0.
    theta = np.linspace(0,2*np.pi,ntheta,endpoint=False)
    dtheta = theta[1]-theta[0]
    radiusm,thetam = np.meshgrid(radius,theta)
    xsi = radiusm*np.cos(thetam)
    eta = radiusm*np.sin(thetam)*np.cos(incl)
    if input_model is None:
        model = np.ones(nr)*np.max(data)#*(2*np.pi*sigma**2)
    else:
        model = input_model#/((velo.max()-velo.min())/1e5)
    modelm = np.outer(np.ones(ntheta),model)
    mstar *= M_sun.cgs.value
    distance *= pc.cgs.value#122*pc.cgs.value
    vmodel = np.sqrt(G.cgs.value*mstar/(np.radians(radiusm/3600.)*distance))*np.cos(thetam)*np.sin(incl)
    dvt = dv/3 #thermal broadening


    ram,decm = np.meshgrid(ra,dec)
    x = -ram*np.sin(PA)-decm*np.cos(PA)
    y = -ram*np.cos(PA)+decm*np.sin(PA)
    vmin,vmax=0,nv

    # model convolved with beam
    print 'Initial estimate'
    intensity = np.zeros((nv,xnpix,ynpix))
    for iv in range(vmin,vmax):
        #Pv = np.exp(-(velo[iv]-vmodel)**2/(2*dvt**2))#/(np.sqrt(2*np.pi)*dv)
        Pv = .5*(np.abs(velo[iv]-vmodel)<1*dv/2.)+.25*((np.abs(velo[iv]-vmodel)>1*dv/2.) & (np.abs(velo[iv]-vmodel)<3*dv/2.))
        for ix in range(xnpix):
            for iy in range(ynpix):
                Ps = 1/(2*np.pi*sigma**2)*np.exp(-((x[ix,iy]-xsi)**2+(y[ix,iy]-eta)**2)/(2*sigma**2))
                Pi = Ps*Pv*dr*dtheta#*dv*np.sqrt(2*np.pi)
                intensity[iv,ix,iy] = (radiusm*modelm*Pi).sum()

    #Calculate chi-squared
    print 'chi-squared: {:0.3f}'.format(((intensity-data)**2/noise**2).sum()/(xnpix*ynpix*nv-nr))

            

    for j in range(niter):
        print 'Adjust model {:1.0f}...'.format(j+1)
        for ir in range(1,nr):
            num,denom=0,0
            for iv in range(vmin,vmax):
                w = data[iv,:,:]>-5#3*noise
                if w.sum()>0:
                    for itheta in range(ntheta):
                        Psa = np.exp(-((x-xsi[itheta,ir])**2+(y-eta[itheta,ir])**2)/(2*sigma**2))
                        Pva = .5*(np.abs(velo[iv]-vmodel[itheta,ir])<1*dv/2.)+.25*((np.abs(velo[iv]-vmodel[itheta,ir])>1*dv/2.) & (np.abs(velo[iv]-vmodel[itheta,ir])<3*dv/2.))
                        #Pva = np.exp(-(velo[iv]-vmodel[itheta,ir])**2/(2*dvt**2))
                        if Pva > 0:
                            num += ((data[iv,:,:][w]/intensity[iv,:,:][w])*Psa[w]*Pva).sum()
                            denom += (Psa[w]*Pva).sum()
                else:
                    num+=0
                    denom+=0
            model[ir] = model[ir]*num/denom
            
        model[model<0] = 0.
        model[np.isnan(model)] = 0.
        model[0] = 0.
        modelm = np.outer(np.ones(ntheta),model)

        print 'Convolve new model...'
        for iv in range(vmin,vmax):
            #Pv = np.exp(-(velo[iv]-vmodel)**2/(2*dvt**2))#/(np.sqrt(2*np.pi)*dv)
            Pv = .5*(np.abs(velo[iv]-vmodel)<1*dv/2.)+.25*((np.abs(velo[iv]-vmodel)>1*dv/2.) & (np.abs(velo[iv]-vmodel)<3*dv/2.))
            for ix in range(xnpix):
                for iy in range(ynpix):
                    Ps = 1/(2*np.pi*sigma**2)*np.exp(-((x[ix,iy]-xsi)**2+(y[ix,iy]-eta)**2)/(2*sigma**2))
                    Pi = Ps*Pv*dr*dtheta
                    intensity[iv,ix,iy] = (radiusm*modelm*Pi).sum()

        #Calculate chi-squared
        print 'chi-squared: {:0.3f}'.format(((intensity-data)**2/noise**2).sum()/(xnpix*ynpix*nv-nr))
        

    i = 1
    plt.figure()
    for iv in [nv/2,nv/4,3*nv/4]:
        nlevels = 10#(np.max(data[iv,:,:])-3*noise)/(2*noise)
        levels = np.arange(nlevels)*2*noise+3*noise
        glevels = np.arange(101)/100.*np.max(data[iv,:,:])
    # plot the image
        plt.subplot(4,3,i)
        i+=1
        cs = plt.contourf(ra,dec,data[iv,:,:],glevels,cmap=plt.cm.Greys)
        plt.contour(ra,dec,data[iv,:,:],[3*noise],linewidths=3,colors='k',linestyles=':')
        plt.xlabel(r'$\Delta\alpha$ (")',fontsize=14)
        plt.ylabel('$\Delta\delta$ (")',fontsize=14)
    #plt.colorbar(cs,label='Jy/beam',pad=0.,shrink=.8,format='%0.3f')
        plt.gca().invert_xaxis()
        plt.title('Data')

        plt.subplot(4,3,i)
        i+=1
        cs = plt.contourf(ra,dec,intensity[iv,:,:],glevels,cmap=plt.cm.Greys)
        plt.contour(ra,dec,intensity[iv,:,:],[3*noise],linewidths=3,colors='k',linestyles=':')
        plt.xlabel(r'$\Delta\alpha$ (")',fontsize=14)
        plt.ylabel('$\Delta\delta$ (")',fontsize=14)
    #plt.colorbar(cs,label='Jy/beam',pad=0.,shrink=.8,format='%0.3f')
        plt.gca().invert_xaxis()
        plt.title('Convolved model')

        plt.subplot(4,3,i)
        i+=1
        cs = plt.contourf(ra,dec,data[iv,:,:]-intensity[iv,:,:],glevels,cmap=plt.cm.Greys)
        plt.contour(ra,dec,data[iv,:,:]-intensity[iv,:,:],levels,linewidths=3,colors='k')
        plt.contour(ra,dec,data[iv,:,:]-intensity[iv,:,:],-levels,linewidths=3,colors='r')
        plt.contour(ra,dec,data[iv,:,:],[3*noise],linewidths=3,colors='k',linestyles=':')
        plt.xlabel(r'$\Delta\alpha$ (")',fontsize=14)
        plt.ylabel('$\Delta\delta$ (")',fontsize=14)
        plt.colorbar(cs,label='Jy/beam',pad=0.,shrink=.8,format='%0.3f')
        plt.gca().invert_xaxis()
        plt.title('Data-Model')
    
    plt.subplot(414)
    plt.plot(radius,model*radius,'sk')
    plt.plot(radius,model*radius**(3./2),'or')
    plt.xlabel('Deprojected Radius (")',fontweight='bold')
    plt.legend(('Intensity','Surface Density'),loc='upper right',frameon=False)
    
    if return_spectrum:
        ram,dem = np.meshgrid(ra,dec)
        sigmaj = 3600*hdr['bmaj']/(2*np.sqrt(2*np.log(2)))
        sigmin = 3600*hdr['bmin']/(2*np.sqrt(2*np.log(2)))
        area = np.exp(-(ram**2/(2*sigmaj**2)+dem**2/(2*sigmin**2))).sum()
        im_dec = intensity.sum(axis=2)/area
        spec = im_dec.sum(axis=1)
        return velo/1e5,spec
    else:
        return radius,model*radius

def quick_disk_cont(file,size=10.,PA=312.,incl=48.,niter=5,gas_column=False,line='co32',tgas=25.,distance=122.,offs=[0.,0.]):
    ''' Based on Scoville et al. 1983, this code is a quick estimate of the spatial distribution of the flux emitted by a flat disk. This is useful for determining the spatial distribution of the flux (e.g. rings vs continous disk). More detailed radiative transfer codes are needed to pull out temperature, or if the velocity profile is not given by a simple flat disk. As compared to quick_disk, this code does not model the velocity profile, and is useful for modeling continuum emission or moment 0 maps

    The code returns the model emissivities (rho(R), where I~2*pi*R*rho(R)) and creates a plot of the results. The left panel shows the data (with 3-sigma contour marked by dashed line). The central panel shows model (with 3-sigma contour marked by dashed line). The right panel shows the residuals with contours at 3,5,7,etc sigma (with 3-sigma flux contour from the data). In all three panels the grey-scale is the same and stretches over the dynamic range of the data. The bottom panel shows both intensity of the rings used in the model, as well as an estimate of the surface density profile, derived assuming Sigma~I/R^(-.5). The units on the intensity and surface density are arbitrary. 
    
    :param file:
    A fits file with your image. Assumes that the image follows the format of standard ALMA images.

    :param size:
    Size of the region in the image to fit, in arcseconds. The fit extends over -size/2 to size/2 in both RA and Dec.

    :param PA:
    Position angle of the major axis of the disk, measured east of north

    :param incl:
    Inclination of the disk

    :param niter:
    Number of iterations to perform. Typically 2 gives you a good rough estimate (to make sure PA, incl and size are correct), while 5-10 gets down to the noise. 

    :param gas_column:
    Set this parameter to true if you want to display a map of the column density of gas in the upper level, rather than the intensity map. It also calculates the total mass within the image (and calculates the total mass of gas assuming LTE and a gas temperature). Optional keywords used when gas_column is set to true are line (for specifying the particular transition being observed), distance, and tgas (gas temperature, used when estimating total gas mass)

    :param distance:
    Distance to the disc in parsecs. Only necessary when gas_column=True

    :param line:
    Specify the particular transition to use. For now, only accepts 'co32', 'co21' or 'co10'. Once the line has been specified, various molecule and transition specific parameters are set within the code

    :param tgas:
    Gas temperature, in kelvin. All of the gas is assumed to be at this temperature (ie. no radial temperature profile). Only used when gas_column=True, and only necessary for estimating the total mass assuming LTE. 

    '''
    from astropy.io import fits
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.constants import G,pc,M_sun,h,c,k_B,m_p
    
    # - Read in the data
    im = fits.open(file)
    data = im[0].data.squeeze()
    hdr = im[0].header
    im.close()
    ra = 3600*hdr['cdelt1']*(np.arange(hdr['naxis1'])-hdr['naxis1']/2.-0.5)
    dec = 3600*hdr['cdelt2']*(np.arange(hdr['naxis2'])-hdr['naxis2']/2.-0.5)
    noise = calc_noise(data,np.round(size/(3600.*hdr['cdelt1'])))
    ira = np.abs(ra) < size/2.
    ide = np.abs(dec) < size/2.
    data = data[:,ira]
    data = data[ide,:]
    ra = ra[ira]-offs[0]
    dec = dec[ide]-offs[1]
    dra = np.abs(3600*hdr['cdelt1'])
    ddec = np.abs(3600*hdr['cdelt2'])
    xnpix = ra.shape[0]
    ynpix = dec.shape[0]
    
    PA,incl = np.radians(PA),np.radians(incl)

    # - Beam size
    sigma = np.mean([3600*hdr['bmaj']/(2*np.sqrt(2*np.log(2))),3600*hdr['bmin']/(2*np.sqrt(2*np.log(2)))])
    a,b,phi_beam = hdr['bmaj']/2*3600.,hdr['bmin']/2*3600.,np.radians(90+hdr['bpa'])
    t=np.linspace(0,2*np.pi,100)
    xbeam = -size/2.+1.1*sigma+a*np.cos(t)*np.cos(phi_beam)-b*np.sin(t)*np.sin(phi_beam)
    ybeam = -size/2.+1.1*sigma+a*np.cos(t)*np.sin(phi_beam)+b*np.sin(t)*np.cos(phi_beam)
    ram,dem = np.meshgrid(ra,dec)
    area = np.exp(-(ram**2/(2*(a/np.sqrt(2*np.log(2)))**2)+dem**2/(2*(b/np.sqrt(2*np.log(2)))**2))).sum()
    #noise*=np.sqrt(area)
    
    # initial model
    #intrinsic model
    ntheta = 100
    #nr = (float(size)/2*np.sqrt(2.)/(sigma/5.)).astype(int)
    #radius = np.linspace(0.,float(size)/2*np.sqrt(2.),nr)
    nr = (float(size)/2/(sigma/5.)).astype(int)
    radius = np.linspace(0.,float(size)/2.,nr)
    dr = radius[1]-radius[0]#radius-np.roll(radius,1)
    #dr[0] = 0.
    theta = np.linspace(0,2*np.pi,ntheta,endpoint=False)
    dtheta = theta[1]-theta[0]
    radiusm,thetam = np.meshgrid(radius,theta)
    xsi = radiusm*np.cos(thetam)
    eta = radiusm*np.sin(thetam)*np.cos(incl)
    model = np.ones(nr)*np.max(data)*(2*np.pi*sigma**2)
    modelm = np.outer(np.ones(ntheta),model)
    model_tot = np.ones((len(model),niter+1))
    model_tot[:,0] = model
    chisq_arr = np.zeros(niter+1)
            

    ram,decm = np.meshgrid(ra,dec)
    x = -ram*np.sin(PA)-decm*np.cos(PA)
    y = -ram*np.cos(PA)+decm*np.sin(PA)
    
    # model convolved with beam
    print 'Initial estimate'
    intensity = np.zeros((xnpix,ynpix))
    for ix in range(xnpix):
        for iy in range(ynpix):
            Ps = 1/(2*np.pi*sigma**2)*np.exp(-((x[ix,iy]-xsi)**2+(y[ix,iy]-eta)**2)/(2*sigma**2))
            intensity[ix,iy] = (radiusm*modelm*Ps*dr).sum()*dtheta

    #Calculate chi-squared
    dof = 1#xnpix*ynpix/area-nr #xnpix*ynpix-nr
    print 'chi-squared: {:0.3f}'.format(((intensity-data)**2/noise**2).sum()/(dof))
    chisq_arr[0] = ((intensity-data)**2/noise**2).sum()/(dof)

            
    for j in range(niter):
        print 'Adjust model {:1.0f}...'.format(j+1)
        for ir in range(nr):
            num,denom = 0,0
            for itheta in range(ntheta):
                num += ((data/intensity)*np.exp(-((x-radius[ir]*np.cos(theta[itheta]))**2+(y-radius[ir]*np.sin(theta[itheta])*np.cos(incl))**2)/(2*sigma**2))).sum()
                denom += (np.exp(-((x-radius[ir]*np.cos(theta[itheta]))**2+(y-radius[ir]*np.sin(theta[itheta])*np.cos(incl))**2)/(2*sigma**2))).sum()
                
            model[ir] = model[ir]*num/denom
                    
        #model[model<0] = 0.
        modelm = np.outer(np.ones(ntheta),model)
        model_tot[:,j+1] = model

        print 'Convolve new model...'
        for ix in range(xnpix):
            for iy in range(ynpix):
                Ps = 1/(2*np.pi*sigma**2)*np.exp(-((x[ix,iy]-xsi)**2+(y[ix,iy]-eta)**2)/(2*sigma**2))
                intensity[ix,iy] = (radiusm*modelm*Ps*dr).sum()*dtheta

        #Calculate chi-squared
        dof = 1#xnpix*ynpix/area-nr #xnpix*ynpix-nr
        print 'chi-squared: {:0.3f}'.format(((intensity-data)**2/noise**2).sum()/(dof))
        chisq_arr[j+1] =  ((intensity-data)**2/noise**2).sum()/(dof)

    if gas_column:
        if line.lower() == 'co32':
            Aul = 2.497e-6
            nu = 345.79599e9
            Jl = 2
            El = 11.53492*h.cgs.value*c.cgs.value
            mol_mass = 28*m_p.cgs.value
        if line.lower() == 'co21':
            Aul = 6.910e-7
            nu = 230.538e9
            Jl = 1
            El = 3.84503*h.cgs.value*c.cgs.value
            mol_mass = 28*m_p.cgs.value
        if line.lower() == 'co10':
            Aul = 7.203e-8
            nu = 115.2712018e9
            Jl = 0
            #El = 5.53*h.cgs.value*c.cgs.value
            mol_mass = 28*m_p.cgs.value
        distance *= pc.cgs.value
        column = intensity*1e-23*nu/c.cgs.value*4*np.pi/(h.cgs.value*nu*Aul)*1e5*distance**2
        sig_column = noise*1e-23*nu/c.cgs.value*4*np.pi/(h.cgs.value*nu*Aul)*1e5*distance**2
        #ram,dem = np.meshgrid(ra,dec)
        area = np.exp(-(ram**2/(2*sigma**2)+decm**2/(2*sigma**2))).sum()
        mass = column.sum()/area
        sig_mass = np.sqrt((sig_column**2)*xnpix**2)/area
        print 'Total number of molecules in upper level: {:0.2e} +- {:0.2e}(stat) +- {:0.2e}(sys) molecules'.format(mass,sig_mass,.2*mass)
        print 'Total mass in upper level: {:0.2e} +- {:0.2e}(stat) +-{:0.2e}(sys) gm'.format(mass*mol_mass,sig_mass*mol_mass,.2*mass*mol_mass)
        if line.lower() == 'co10':
            Te = 5.53
        else:
            Te = 2*El/(Jl*(Jl+1)*k_B.cgs.value)
        parZ = np.sqrt(1.+(2./Te)**2*tgas**2)
        gu = 2*(Jl+1)+1
        x = gu*np.exp(-(h.cgs.value*nu)/(k_B.cgs.value*tgas))/parZ
        print 'Total molecule mass, assuming LTE: {:0.2e} +- {:0.2e}(stat) +-{:0.2e}(sys) Msun'.format(mass*mol_mass/x/M_sun.cgs.value,sig_mass*mol_mass/x/M_sun.cgs.value,.2*mass*mol_mass/x/M_sun.cgs.value)
        

    nlevels = (np.max(data)-10*noise)/(5*noise)
    levels = np.arange(nlevels)*10*noise+5*noise
    glevels = np.arange(101)/100.*np.max(data)
    
    if niter>2:
        unc = np.zeros(nr)
        for i in range(nr):
            h = (np.abs(model_tot[i,-1]-model_tot[i,-2])+np.abs(model_tot[i,-3]-model_tot[i,-2]))/2.
            unc[i] = np.sqrt(2*np.abs((chisq_arr[-2]-2*chisq_arr[-1]+chisq_arr[-3])/h)**(-1))


    # plot the image
    plt.figure()
    plt.subplot(231)
    cs = plt.contourf(ra,dec,data,glevels,cmap=plt.cm.Greys)
    plt.contour(ra,dec,data,[5*noise],linewidths=3,colors='k',linestyles=':')
    plt.xlabel(r'$\Delta\alpha$ (")',fontsize=14)
    plt.ylabel('$\Delta\delta$ (")',fontsize=14)
    #plt.colorbar(cs,label='Jy/beam',pad=0.,shrink=.8,format='%0.3f')
    plt.gca().invert_xaxis()
    plt.title('Data')
    plt.plot(xbeam,ybeam,lw=2)


    plt.subplot(232)
    if gas_column:
        cs = plt.contourf(ra,dec,column,100,cmap=plt.cm.Greys)
        plt.colorbar(cs,label='N$_u$/beam',pad=0.,shrink=.8,format='%0.2e')
    else:
        cs = plt.contourf(ra,dec,intensity,glevels,cmap=plt.cm.Greys)
    plt.contour(ra,dec,intensity,[5*noise],linewidths=3,colors='k',linestyles=':')
    plt.xlabel(r'$\Delta\alpha$ (")',fontsize=14)
    plt.ylabel('$\Delta\delta$ (")',fontsize=14)
    #plt.colorbar(cs,label='Jy/beam',pad=0.,shrink=.8,format='%0.3f')
    plt.gca().invert_xaxis()
    plt.title('Convolved model')

    print 'ra centroid:',(ram*data).sum()/data.sum()
    print 'dec centroid:',(decm*data).sum()/data.sum()


    plt.subplot(233)
    cs = plt.contourf(ra,dec,data-intensity,glevels,cmap=plt.cm.Greys)
    plt.contour(ra,dec,data-intensity,levels,linewidths=3,colors='k')
    plt.contour(ra,dec,data,[5*noise],linewidths=3,colors='k',linestyles=':')
    plt.xlabel(r'$\Delta\alpha$ (")',fontsize=14)
    plt.ylabel('$\Delta\delta$ (")',fontsize=14)
    plt.colorbar(cs,label=hdr['bunit'],pad=0.,shrink=.8,format='%0.3f')
    plt.gca().invert_xaxis()
    plt.title('Data-Model')


    ax=plt.subplot(212)
    #plt.plot(radius,radius*model,'sk') #Intensity
#    if niter>2:
#        plt.errorbar(radius,model,yerr=unc,fmt='sk')
#    else:
    plt.plot(radius,model,'sk') #Emissivity
    #plt.plot(radius,model*radius**(3./2),'or')
    plt.xlabel('Deprojected Radius (")')
    #plt.legend(('Intensity','Surface Density'),loc='upper right',frameon=False)


    plt.show()

    return model
    
def calc_noise(image,imx=10):
    '''Calculate the noise within an image. The noise is used in multiple functions (im_plot_spec, mk_chmaps, imdiff) and the calculation is moved here to make sure it is consistent between all of them'''
    #assuming we are dealing with full image and not cropped images
    #imx is width of box in pixels. This box is used to define noise.
    import numpy as np
    if len(image.shape)>2:
    #could also consider calculating noise as a function of channel
        imx=np.abs(imx)
        npix = image.shape[1]
        nfreq = image.shape[0]
        noise1 = np.zeros(nfreq)
        noise2 = np.zeros(nfreq)
        noise3 = np.zeros(nfreq)
        noise4 = np.zeros(nfreq)
        noise5 = np.zeros(nfreq)
        noise = np.zeros(nfreq)

        if npix/2-3*imx/2 <0:
            low = 0
        else:
            low = npix/2-3*imx/2
        if npix/2+3*imx/2>npix:
            high = -1
        else:
            high = npix/2+3*imx/2
    
        for i in range(nfreq):
            noise1[i] = np.std(image[i,low:npix/2-imx/2,low:npix/2-imx/2])
            noise2[i] = np.std(image[i,low:npix/2-imx/2,npix/2+imx/2:high])
            noise3[i] = np.std(image[i,npix/2+imx/2:high,low:npix/2-imx/2])
            noise4[i] = np.std(image[i,npix/2+imx/2:high,npix/2+imx/2:high])
            noise5[i] = np.std(image[i,low:high,low:high])
            noise[i] = np.mean([noise1[i],noise2[i],noise3[i],noise4[i]])
        #flux = np.array([image[i,low:npix/2-imx/2,low:npix/2-imx/2],image[i,low:npix/2-imx/2,npix/2+imx/2:high],image[i,npix/2+imx/2:high,low:npix/2-imx/2],image[i,npix/2+imx/2:high,npix/2+imx/2:high]])
        #print 'N>3sigma:',float((np.abs(flux.flatten())>3*noise[i]).sum())/flux.flatten().shape[0]
        #print 'N>1sigma:',float((np.abs(flux.flatten())>noise[i]).sum())/flux.flatten().shape[0] #The number of 3sigma and 1sigma peaks in all the lines is consistent with what we would expect from gaussian statistics
        

        return np.mean([noise1,noise2,noise3,noise4])
    else:
        imx=np.abs(imx)
        npix = image.shape[1]
        
        if npix/2-3*imx/2 <0:
            low = 0
        else:
            low = npix/2-3*imx/2
        if npix/2+3*imx/2>npix:
            high = -1
        else:
            high = npix/2+3*imx/2
    
        noise1 = np.std(image[low:npix/2-imx/2,low:npix/2-imx/2])
        noise2 = np.std(image[low:npix/2-imx/2,npix/2+imx/2:high])
        noise3 = np.std(image[npix/2+imx/2:high,low:npix/2-imx/2])
        noise4 = np.std(image[npix/2+imx/2:high,npix/2+imx/2:high])
        #noise1 = np.std(image[0:npix/2-imx/2,0:npix/2-imx/2])
        #noise2 = np.std(image[0:npix/2-imx/2,npix/2+imx/2:-1])
        #noise3 = np.std(image[npix/2+imx/2:-1,0:npix/2-imx/2])
        #noise4 = np.std(image[npix/2+imx/2:-1,npix/2+imx/2:-1])
        #noise5 = np.std(image[low:high,low:high])
        return np.mean([noise1,noise2,noise3,noise4])

#radius2,model2=quick_disk('DCOplus_lowres.cm.fits',vsys=5.74,PA=312.,incl=-47.,niter=10,distance=122.,runquick=True,size=6,mstar=2.3)
#fig=plt.figure()
#ax1=fig.add_subplot(111)
#ax1.plot(radius2,model2,'sk')
#ax1.set_xlabel('Deprojected Radius (")',fontweight='bold',fontsize=18)
#ax1.set_ylabel('Intensity',fontweight='bold',fontsize=18)
#for tick in ax1.xaxis.get_major_ticks():
#    tick.label.set_fontsize(16)
#    tick.label.set_fontweight('bold')
#
#for tick in ax1.yaxis.get_major_ticks():
#    tick.label.set_fontsize(16)
#    tick.label.set_fontweight('bold')
#
#ax2=ax1.twiny()
#ax2.set_xlim(ax1.get_xlim())
#ax2.set_xticks([0./122,50./122,100./122,150./122,200./122,250./122,300./122,350./122])
#ax2.set_xticklabels(['','50','100','150','200','250','300','350'],fontweight='bold',fontsize=16)
#ax2.set_xlabel('Deprojected Radius (au)',fontweight='bold',fontsize=18)
#plt.axvline(90./122,color='k',lw=3,ls='--')
#plt.axvline(260./122,color='k',lw=3)
#plt.axvline(140./122,color='k',lw=3)
#plt.axvline(60./122,color='k',lw=3)
#plt.plot([.44-.23/2,.44+.23/2],[.29,.29],color='k',lw=3)
#plt.plot([.81-.15/2,.81+.15/2],[.29,.29],color='k',lw=3)
#plt.plot([1.13-.28/2,1.13+.28/2],[.28,.28],color='k',lw=3)
