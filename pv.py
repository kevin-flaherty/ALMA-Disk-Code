def pv(file,PA=0.,size=10.,vsys=0.,mstar=2.3,distance=122.,incl=48.,add_kep=False):
    '''Create a position velocity diagram across an image. It makes one plot showing the central velocity channel, marking the line along with the PV diagram is defined, along with a PV diagram. Can optionally (with add_kep) show a Keplerian rotation profile on the PV diagram

    :param file:
    Name of the image file (ie the three-dimensional cleaned map)

    :param PA:
    Position angle, defined east of north, in degrees for the major axis of the disk

    :param size:
    Size, in arcseconds, of the region to consider (e.g. setting size=10. pulls out the central +-5" of the image, rather than using the entire image).

    :param vsys:
    Systemic velocity for the line center, in km/s. Only needed when plotting the Keplerian profile.

    :param mstar:
    Stellar mass, in solar masses, of the central object. Only needed when plotting the Keplerian profile.

    :param distance:
    Distance to the disk in parsecs. Only needed when plotting the Keplerian profile.

    :param incl:
    Inclination in degrees. Only needed when plotting the Keplerian profile. If the PV diagram is inverted relative to the Keplerian profile (ie. the model shows positive velocities when the data has negative velocities) add a negative sign in front of the inclination.

    :param add_kep:
    Set to True to add a Keplerian profile to the PV diagram. This profile is shown with two dashed lines.

'''
    from astropy.io import fits
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage

    image = fits.open(file)
    hdr = image[0].header
    im = image[0].data.squeeze()

    ra = 3600*hdr['cdelt1']*(np.arange(hdr['naxis1'])-hdr['naxis1']/2.-0.5)
    de = -1*ra

    ira = np.abs(ra) < size/2.
    ide = np.abs(de) < size/2.
    shape = im.shape
    if shape[1]==shape[2]:
        #frequency is the first axis
        im_tmp = im[:,:,ira]
        im_tmp = im_tmp[:,ide,:]
    else:
        #frequency is the last axis
        im_tmp = im[ira,:,:]
        im_tmp = im[:,ide,:]
    ra = ra[ira]
    de = de[ide]

    if (hdr['ctype3'] =='VELO-LSR') or (hdr['ctype3']=='VELO-OBS'):
        velo = ((np.arange(hdr['naxis3'])+1-hdr['crpix3'])*hdr['cdelt3']+hdr['crval3'])/1e3
    else:
        freq = (np.arange(hdr['naxis3'])+1-hdr['crpix3'])*hdr['cdelt3']+hdr['crval3']
        try:
            velo = (hdr['restfreq']-freq)/hdr['restfreq']*2.99e5
        except KeyError:
            velo = (np.median(freq)-freq)/np.median(freq)*2.99e5

    
    
    #find the values of ra and dec along the line defined by the PA
    ra_line = de*np.tan(np.radians(PA))
    w = (ra_line>np.sqrt(2)*np.min(ra)) & (ra_line<np.sqrt(2)*np.max(ra))
    ra_line = ra_line[w]
    de_line = de[w]
    if PA==90.:
        de_line = np.zeros(len(de))
        ra_line = -ra
    if PA ==270:
        de_line = np.zeros(len(de))
        ra_line = ra

    plt.figure()
    plt.rc('axes',lw=2)
    cs=plt.contourf(ra,de,im_tmp[len(velo)/2-1,:,:],100,cmap=plt.cm.afmhot)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.plot(ra_line,de_line,color='w',lw=3,ls='--')
    plt.xlim(size/2.,-size/2.)
    plt.ylim(-size/2.,size/2.)
    cb = plt.colorbar(cs,pad=0.,shrink=0.9,format='%0.2f')
    cb.set_label(label='Jy/beam',size=16,weight='bold')
    for tick in cb.ax.yaxis.get_majorticklabels():
        tick.set_fontsize(14)
        tick.set_fontweight('bold')
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    plt.xlabel(r'$\Delta\alpha$ (")',fontweight='bold',fontsize=16)
    plt.ylabel('$\Delta\delta$ (")',fontweight='bold',fontsize=16)

    sra = ra.argsort()
    sde = de.argsort()
    raind = np.interp(ra_line,ra[sra],sra)
    deind = np.interp(de_line,de[sde],sde)

    dist = np.sqrt((ra_line-ra_line[0])**2+(de_line-de_line[0])**2)
    dist -= dist.mean()

    diagram = np.zeros((len(velo),len(dist)))
    for i in range(len(velo)):
        diagram[i,:] = ndimage.map_coordinates(im_tmp[i,:,:],[[raind],[deind]],order=1).flatten()


    plt.figure()
    cs=plt.contourf(dist,velo,diagram,100,cmap=plt.cm.afmhot)
    cb = plt.colorbar(cs,pad=0.,shrink=0.9,format='%0.2f')
    cb.set_label(label='Jy/beam',size=16,weight='bold')
    ax=plt.gca()
    for tick in cb.ax.yaxis.get_majorticklabels():
        tick.set_fontsize(14)
        tick.set_fontweight('bold')
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    plt.xlabel('Offset (")',fontweight='bold',fontsize=16)
    plt.ylabel('v$_{LSRK}$ (km s$^{-1}$)',fontweight='bold',fontsize=16)

    if add_kep:
        #add keplerian profiles
        radius = np.abs(dist)*distance
        au = 1.496e13
        G = 6.67e-8
        msun = 1.99e33

        vkep = np.sqrt(G*mstar*msun/(radius*au))*np.sin(np.radians(incl))/1e5
        wpos = dist>0
        wneg = dist<0
        plt.plot(dist[wpos],-vkep[wpos]+vsys,color='w',ls='--',lw=3)
        plt.plot(dist[wneg],vkep[wneg]+vsys,color='w',ls='--',lw=3)
        plt.ylim(velo.min(),velo.max())
