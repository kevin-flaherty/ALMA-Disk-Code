

def quick_disk(file,size=10.,vsys=None,input_model=None,PA=312,incl=48,niter=5,mstar=2.0,distance=122):
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
    ra = ra[ira]
    dec = dec[ide]
    dra = np.abs(3600*hdr['cdelt1'])
    ddec = np.abs(3600*hdr['cdelt2'])
    xnpix = ra.shape[0]
    ynpix = dec.shape[0]

    #PA,incl = np.radians((PA-90.)/2),np.radians(incl)
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
        velo = (np.median(freq)-freq)/np.median(freq)*2.99e10
        if vsys is None:
            velo -= np.median(velo)
        else:
            velo -= vsys*1e5
        dv = np.abs(velo[1]-velo[0])
        nv = len(velo)

    # - Beam size
    sigma = np.mean([3600*hdr['bmaj']/(2*np.sqrt(2*np.log(2))),3600*hdr['bmin']/(2*np.sqrt(2*np.log(2)))])


    # initial model
    #intrinsic model
    ntheta = 100
    nr = (float(size)/2*np.sqrt(2.)/(sigma/5.)).astype(int)
    radius = np.linspace(0.,float(size)/2*np.sqrt(2.),nr)
    #radius = np.logspace(-3,np.log10(size/2*np.sqrt(2.)),nr)
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
    dvt = 2*dv #thermal broadening


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
            

    for j in range(niter):
        print 'Adjust model {:1.0f}...'.format(j+1)
        for ir in range(1,nr):
            num,denom=0,0
            for iv in range(vmin,vmax):
                for itheta in range(ntheta):
                    Psa = np.exp(-((x-xsi[itheta,ir])**2+(y-eta[itheta,ir])**2)/(2*sigma**2))
                    Pva = .5*(np.abs(velo[iv]-vmodel[itheta,ir])<1*dv/2.)+.25*((np.abs(velo[iv]-vmodel[itheta,ir])>1*dv/2.) & (np.abs(velo[iv]-vmodel[itheta,ir])<3*dv/2.))
                    #Pva = np.exp(-(velo[iv]-vmodel[itheta,ir])**2/(2*dvt**2))
                    num += ((data[iv,:,:]/intensity[iv,:,:])*Psa*Pva).sum()
                    denom += (Psa*Pva).sum()
            model[ir] = model[ir]*num/denom

        model[model<0] = 0.
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
    plt.xlabel('Deprojected Radius (")')
    plt.legend(('Intensity','Surface Density'),loc='upper right',frameon=False)
    


def quick_disk_cont(file,size=10.,PA=312.,incl=48.,niter=5,gas_column=False,line='co32',tgas=25.,distance=122.):
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
    ra = ra[ira]
    dec = dec[ide]
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
    
    # initial model
    #intrinsic model
    ntheta = 100
    nr = (float(size)/2*np.sqrt(2.)/(sigma/5.)).astype(int)
    radius = np.linspace(0.,float(size)/2*np.sqrt(2.),nr)
    #radius = np.logspace(-3,np.log10(size/2*np.sqrt(2.)),nr)
    dr = radius[1]-radius[0]#radius-np.roll(radius,1)
    #dr[0] = 0.
    theta = np.linspace(0,2*np.pi,ntheta,endpoint=False)
    dtheta = theta[1]-theta[0]
    radiusm,thetam = np.meshgrid(radius,theta)
    xsi = radiusm*np.cos(thetam)
    eta = radiusm*np.sin(thetam)*np.cos(incl)
    model = np.ones(nr)*np.max(data)*(2*np.pi*sigma**2)
    modelm = np.outer(np.ones(ntheta),model)
        

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
            
    for j in range(niter):
        print 'Adjust model {:1.0f}...'.format(j+1)
        for ir in range(nr):
            num,denom = 0,0
            for itheta in range(ntheta):
                num += ((data/intensity)*np.exp(-((x-radius[ir]*np.cos(theta[itheta]))**2+(y-radius[ir]*np.sin(theta[itheta])*np.cos(incl))**2)/(2*sigma**2))).sum()
                denom += (np.exp(-((x-radius[ir]*np.cos(theta[itheta]))**2+(y-radius[ir]*np.sin(theta[itheta])*np.cos(incl))**2)/(2*sigma**2))).sum()
            model[ir] = model[ir]*num/denom
        
        model[model<0] = 0.
        modelm = np.outer(np.ones(ntheta),model)        
    
        print 'Convolve new model...'
        for ix in range(xnpix):
            for iy in range(ynpix):
                Ps = 1/(2*np.pi*sigma**2)*np.exp(-((x[ix,iy]-xsi)**2+(y[ix,iy]-eta)**2)/(2*sigma**2))
                intensity[ix,iy] = (radiusm*modelm*Ps*dr).sum()*dtheta


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
        ram,dem = np.meshgrid(ra,dec)
        area = np.exp(-(ram**2/(2*sigma**2)+dem**2/(2*sigma**2))).sum()
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
        

    nlevels = (np.max(data)-3*noise)/(2*noise)
    levels = np.arange(nlevels)*2*noise+3*noise
    glevels = np.arange(101)/100.*np.max(data)
    

    # plot the image
    plt.figure()
    plt.subplot(231)
    cs = plt.contourf(ra,dec,data,glevels,cmap=plt.cm.Greys)
    plt.contour(ra,dec,data,[3*noise],linewidths=3,colors='k',linestyles=':')
    plt.xlabel(r'$\Delta\alpha$ (")',fontsize=14)
    plt.ylabel('$\Delta\delta$ (")',fontsize=14)
    #plt.colorbar(cs,label='Jy/beam',pad=0.,shrink=.8,format='%0.3f')
    plt.gca().invert_xaxis()
    plt.title('Data')
    plt.plot(xbeam,ybeam,lw=2)

    ax=plt.subplot(212)
    plt.plot(radius,radius*model,'sk')
    #plt.loglog(radius*61,model*radius**(3./2),'or')
    plt.plot(radius,model*radius**(3./2),'or')
    plt.xlabel('Deprojected Radius (")')
    #plt.xlabel('Radius (AU)',fontsize=14)
    #plt.ylabel('Gas Surface Density \n(arbitrary units)',fontsize=14)
    #plt.xlim(20,400)
    #plt.ylim(1e-2,1e-1)
    #ax.set_xticks([20,30,40,50,60,70,80,90,100,200,300])
    #ax.set_xticklabels(['20','30','40','50',' ','70',' ',' ','100','200','300'],fontsize=14)
    plt.legend(('Intensity','Surface Density'),loc='upper right',frameon=False)

    plt.subplot(232)
    if gas_column:
        cs = plt.contourf(ra,dec,column,100,cmap=plt.cm.Greys)
        plt.colorbar(cs,label='N$_u$/beam',pad=0.,shrink=.8,format='%0.2e')
    else:
        cs = plt.contourf(ra,dec,intensity,glevels,cmap=plt.cm.Greys)
    plt.contour(ra,dec,intensity,[3*noise],linewidths=3,colors='k',linestyles=':')
    plt.xlabel(r'$\Delta\alpha$ (")',fontsize=14)
    plt.ylabel('$\Delta\delta$ (")',fontsize=14)
    #plt.colorbar(cs,label='Jy/beam',pad=0.,shrink=.8,format='%0.3f')
    plt.gca().invert_xaxis()
    plt.title('Convolved model')

    plt.subplot(233)
    cs = plt.contourf(ra,dec,data-intensity,glevels,cmap=plt.cm.Greys)
    plt.contour(ra,dec,data-intensity,levels,linewidths=3,colors='k')
    plt.contour(ra,dec,data,[3*noise],linewidths=3,colors='k',linestyles=':')
    plt.xlabel(r'$\Delta\alpha$ (")',fontsize=14)
    plt.ylabel('$\Delta\delta$ (")',fontsize=14)
    plt.colorbar(cs,label=hdr['bunit'],pad=0.,shrink=.8,format='%0.3f')
    plt.gca().invert_xaxis()
    plt.title('Data-Model')
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
    
        noise5 = np.std(image[low:high,low:high])
        return noise5
