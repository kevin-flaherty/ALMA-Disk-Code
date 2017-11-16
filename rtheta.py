def rtheta(file,PA,incl,size,offs=[0.,0.]):
    ''' Create a R vs theta map of an image. Theta is the azimuthal angle in the deprojected disk. This projection is useful when looking for azimuthal variations in flux.

    :param file:
    Name of the ALMA image file to use

    :param PA:
    Position angle, in degrees, east of north

    :param incl:
    Inclination, in degrees.

    :param size:
    Size, in arcseconds, of the emitting area to map (ie size=10 restricts to R=10")

    :param offs:
    Offset in RA and Dec, in units of arcseconds, for the center of the disk

'''

    from astropy.io import fits
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import griddata

    alma = fits.open(file)
    im = alma[0].data.squeeze()
    hdr = alma[0].header

    ra = 3600*hdr['cdelt1']*(np.arange(hdr['naxis1'])-hdr['naxis1']/2.-0.5)
    #noise = calc_noise(im,np.round(size/(3600.*hdr['cdelt1'])))
    de = -1*ra
    ra -=offs[0]
    de -=offs[1]

    abeam,bbeam,phi = hdr['bmaj']/2*3600.,hdr['bmin']/2*3600.,np.radians(90+hdr['bpa'])
    t=np.linspace(0,2*np.pi,100)
    xbeam = abeam*np.cos(t)*np.cos(phi)-bbeam*np.sin(t)*np.sin(phi)
    ybeam = abeam*np.cos(t)*np.sin(phi)+bbeam*np.sin(t)*np.cos(phi)
    xbeam -= size-1.1*xbeam.max()
    ybeam -= size-1.1*ybeam.max()

    ram,dem = np.meshgrid(ra,de)
    major = (ram*np.cos(np.radians(90-PA))+dem*np.sin(np.radians(90-PA)))
    minor = (ram*np.sin(np.radians(90-PA))-dem*np.cos(np.radians(90-PA)))/np.sin(np.radians(90-incl))
    dist = np.sqrt(major**2+minor**2)
    
    theta = np.degrees(np.arctan2(minor,major))
    theta[theta<0]+=360
    theta = 360-theta #make sure theta is measured the same as PA (east of north)
    
    plt.figure()
    cs = plt.contourf(ram,dem,im,100,cmap=plt.cm.jet)
    plt.contour(ram,dem,dist,np.linspace(0,size,10),colors='k',linewidths=3,linestyles='--',alpha=.5)
    plt.xlim(size,-size)
    plt.ylim(-size,size)
    plt.fill(xbeam,ybeam,color='w')
    ax = plt.gca()
    ax.set_aspect('equal')
    for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
            tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    plt.xlabel('$\Delta$RA (")',fontweight='bold',fontsize=16)
    plt.ylabel('$\Delta$Dec (")',fontweight='bold',fontsize=16)

    w = (dist<size)
    levels = np.arange(1501)/1500.*(im[w].max()-im[w].min())+im[w].min()

    plt.figure()
    cs=plt.contourf(theta,dist,im*1e3,levels*1e3,cmap=plt.cm.jet)#copper
    cb=plt.colorbar(cs,pad=0,shrink=.8)
    cb.set_label(label='mJy/beam',size=16,weight='bold')
    for tick in cb.ax.yaxis.get_majorticklabels():
        tick.set_fontsize(14)
        tick.set_fontweight('bold')
    plt.ylim(0,size)
    ax = plt.gca()
    plt.xlabel(r'$\theta$ ($^{o}$)',fontweight='bold',fontsize=16)
    plt.ylabel('R (")',fontweight='bold',fontsize=16)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontweight('bold')


