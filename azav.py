def azav(file,PA,incl,size=10.,offs=[0.,0.],giveme=False):
    '''Create an azimuthally averaged radial profile of disk emission. 

    :param file:
    Name of the ALMA image file to use

    :param PA:
    Position angle, in degrees, east of north

    :param incl:
    Inclination, in degrees

    :param size:
    Size in arcseconds of the emitting area to map (ie. size=10 restricts to a 10x10" box)

    :params offs:
    Offset in RA and Dec, in units of arcseconds, for the center of the disk

    :param giveme:
    If set to true then the function returns the radius (in arcseconds), flux and uncertainty (in Jy/beam) of the radial profile.

'''
    from astropy.io import fits
    import matplotlib.pyplot as plt
    import numpy as np

    alma = fits.open(file)
    im = alma[0].data.squeeze()
    hdr = alma[0].header

    ra = 3600*hdr['cdelt1']*(np.arange(hdr['naxis1'])-hdr['naxis1']/2.-0.5)
    #imx = size #image size in arcseconds
    de = -1*ra
    ra -=offs[0]
    de -=offs[1]
    #ira = np.abs(ra) < imx/2.
    #ide = np.abs(de) < imx/2.
    #im_tmp = im[ira,:]
    #im_tmp = im_tmp[:,ide]
    #ra = ra[ira]
    #de = de[ide]

    abeam,bbeam,phi = hdr['bmaj']/2*3600.,hdr['bmin']/2*3600.,np.radians(90+hdr['bpa'])
    t=np.linspace(0,2*np.pi,100)
    xbeam = -(size/2.-1)+abeam*np.cos(t)*np.cos(phi)-bbeam*np.sin(t)*np.sin(phi)
    ybeam = -(size/2.-1)+abeam*np.cos(t)*np.sin(phi)+bbeam*np.sin(t)*np.cos(phi)


    a = np.arange(0.,np.sqrt(2)*size/2.,np.mean([abeam,bbeam]))
    b = a*np.sin(np.radians(90-incl))
    flux = np.zeros(len(a)-1)
    radius = np.zeros(len(a)-1)
    unc = np.zeros(len(a)-1)

    plt.figure()
    plt.contourf(ra,de,im,100,cmap=plt.cm.afmhot)
    plt.gca().invert_xaxis()
    plt.xlabel(r'$\Delta\alpha$ (")',fontsize=14)
    plt.ylabel('$\Delta\delta$ (")',fontsize=14)
    axis = plt.gca()
    axis.set_aspect('equal')
    for tick in axis.xaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontweight('bold')
    for tick in axis.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontweight('bold')
    plt.fill(xbeam,ybeam,color='w')
    plt.xlim(size/2.,-size/2.)
    plt.ylim(-size/2.,size/2.)


    ram,dem = np.meshgrid(ra,de)
    sigmaj = 3600*hdr['bmaj']/(2*np.sqrt(2*np.log(2)))
    sigmin = 3600*hdr['bmin']/(2*np.sqrt(2*np.log(2)))
    area = np.exp(-(ram**2/(2*sigmaj**2)+dem**2/(2*sigmin**2))).sum() #number of pixels per beam
    for i in range(1,len(a)):
        dist1 = (ram*np.cos(np.radians(90-PA))+dem*np.sin(np.radians(90-PA)))**2/a[i-1]**2 + (ram*np.sin(np.radians(90-PA))-dem*np.cos(np.radians(90-PA)))**2/b[i-1]**2.
        dist2 = (ram*np.cos(np.radians(90-PA))+dem*np.sin(np.radians(90-PA)))**2/a[i]**2 + (ram*np.sin(np.radians(90-PA))-dem*np.cos(np.radians(90-PA)))**2/b[i]**2.

        x2 = a[i]*np.cos(t)*np.cos(np.radians(90-PA))-b[i]*np.sin(t)*np.sin(np.radians(90-PA))
        y2 = a[i]*np.cos(t)*np.sin(np.radians(90-PA))+b[i]*np.sin(t)*np.cos(np.radians(90-PA))

        plt.plot(x2,y2,ls='--',lw=1,color='w')
        w = (dist1>1) & (dist2<1)
        flux[i-1] = np.mean(im[w])
        unc[i-1] = np.std(im[w])/np.sqrt(w.sum()/area)
        radius[i-1] = (a[i-1]+a[i])/2.
        
    plt.figure()
    plt.errorbar(radius,flux,yerr=unc,fmt='ok',capsize=0.)
    axis = plt.gca()
    plt.xlabel('Radius (")',fontweight='bold',fontsize=16)
    plt.ylabel('Flux (Jy/beam)',fontweight='bold',fontsize=16)
    plt.axhline(0.,color='k',ls=':',lw=2)
    for tick in axis.xaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontweight('bold')
    for tick in axis.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontweight('bold')

    if giveme:
        return radius,flux,unc
