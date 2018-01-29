def pv(file,PA=0.,size=10.,vsys=0.,mstar=2.3,distance=122.,incl=48.,add_kep=False,offs=[0.,0.]):
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

    :param offs:
    A two element array containing the offset in arcseconds along the RA and Dec directions of the center of the disk.

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

    ra-=offs[0]
    de-=offs[1]

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
            velo = (hdr['restfrq']-freq)/hdr['restfrq']*2.99e5
        except KeyError:
            velo = (np.median(freq)-freq)/np.median(freq)*2.99e5

    
    
    #find the values of ra and dec along the line defined by the PA
    if PA ==0 or PA==360:
        ra_line = np.zeros(len(de))
        de_line = de
    else:
        de_line = ra/np.tan(np.radians(PA))
        w = (de_line>np.sqrt(2)*np.min(de)) & (de_line<np.sqrt(2)*np.max(de))
        ra_line = ra[w]
        de_line = de_line[w]


    #Display the central channel, as well as the line along which the PV diagram is calculated. Since the central channel should show emission (nearly) along the minor axis, if the PA is set correctly then the white-dashed line will be perpendicular to the emission in this channel. 
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
        diagram[i,:] = ndimage.map_coordinates(im_tmp[i,:,:],[[deind],[raind]],order=1).flatten()


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



def pv_point(file,PA=0.,size=10.,vsys=0.,offs=[0.,0.]):
    '''Similar to pv, but reflects the pv about vsys, offset=0. This is useful in looking for any asymmetries in the emission due to e.g. eccentricity.

:param file:
    Name of the image file (ie the three-dimensional cleaned map)

    :param PA:
    Position angle, defined east of north, in degrees for the major axis of the disk

    :param size:
    Size, in arcseconds, of the region to consider (e.g. setting size=10. pulls out the central +-5" of the image, rather than using the entire image).

    :param vsys:
    Systemic velocity for the line center, in km/s. Only needed when plotting the Keplerian profile.

    :param offs:
    A two element array containing the offset in arcseconds along the RA and Dec directions of the center of the disk.


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

    ra-=offs[0]
    de-=offs[1]

    noise = calc_noise(im,np.round(size/(3600.*hdr['cdelt1'])))
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
            velo = (hdr['restfrq']-freq)/hdr['restfrq']*2.99e5
        except KeyError:
            velo = (np.median(freq)-freq)/np.median(freq)*2.99e5

    
    
    #find the values of ra and dec along the line defined by the PA
    if PA ==0 or PA==360:
        ra_line = np.zeros(len(de))
        de_line = de
    else:
        de_line = ra/np.tan(np.radians(PA))
        w = (de_line>np.sqrt(2)*np.min(de)) & (de_line<np.sqrt(2)*np.max(de))
        ra_line = ra[w]
        de_line = de_line[w]


    sra = ra.argsort()
    sde = de.argsort()
    raind = np.interp(ra_line,ra[sra],sra)
    deind = np.interp(de_line,de[sde],sde)

    dist = np.sqrt((ra_line-ra_line[0])**2+(de_line-de_line[0])**2)
    dist -= dist.mean()

    diagram = np.zeros((len(velo),len(dist)))
    for i in range(len(velo)):
        diagram[i,:] = ndimage.map_coordinates(im_tmp[i,:,:],[[deind],[raind]],order=1).flatten()

    

    plt.figure()
    plt.subplot(131)
    cs = plt.contourf(dist,velo-vsys,diagram,100,cmap=plt.cm.afmhot)
    plt.xlim(0,size/2)
    plt.ylim(0,np.max(velo-vsys))
    plt.title('v$_{LSRK}>v_{sys}$')
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    plt.xlabel('|Offset| (")',fontweight='bold',fontsize=16)
    plt.ylabel('|v$_{LSRK}$| (km s$^{-1}$)',fontweight='bold',fontsize=16)
    plt.subplot(132)
    cs = plt.contourf(-dist,-(velo-vsys),diagram,100,cmap=plt.cm.afmhot)
    plt.xlim(0,size/2)
    plt.ylim(0,np.max(velo-vsys))
    plt.title('v$_{LSRK}<v_{sys}$')
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    plt.xlabel('|Offset| (")',fontweight='bold',fontsize=16)

    nlevels = 7#(np.max(diagram)-10*noise)/(5*noise)
    dlevels = (np.max(diagram)-5*noise)/9.
    levels = np.arange(nlevels)*dlevels+5*noise
    plt.subplot(133)
    cs = plt.contour(dist,velo-vsys,diagram,levels,colors='b',linestyles='--',label='v$_{LSRK}$>v$_{sys}$')
    cs = plt.contour(-dist,-(velo-vsys),diagram,levels,colors='k',linestyles=':',label='v$_{LSRK}$<v$_{sys}$')
    plt.xlim(0,size/2)
    plt.ylim(0,np.max(velo-vsys))
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    plt.xlabel('|Offset| (")',fontweight='bold',fontsize=16)

    distm,velom = np.meshgrid(dist,velo)
    w = (distm>0) & (velom>vsys) & (diagram>5*noise)
    cmd=np.average(distm[w],weights=diagram[w])
    cmv=np.average(velom[w],weights=diagram[w])
    var_cmd = np.sqrt(np.average((distm[w]-cmd)**2,weights=diagram[w])/np.sum(diagram[w]))
    var_cmv = np.sqrt(np.average((velom[w]-cmv)**2,weights=diagram[w])/np.sum(diagram[w]))
    print 'v_LSRK > v_sys (offset,vel): {:0.2f} +- {:0.2f}", {:0.2f} +- {:0.2f} km/s'.format(cmd,var_cmd,cmv-vsys,var_cmv)
    plt.errorbar(cmd,cmv-vsys,xerr=var_cmd,yerr=var_cmv,fmt='ob',label='v$_{LSRK}$>v$_{sys}$',ls='--')

    w = (distm<0) & (velom<vsys) & (diagram>5*noise)
    cmd = np.average(distm[w],weights=diagram[w])
    cmv = np.average(velom[w],weights=diagram[w])
    var_cmd = np.sqrt(np.average((distm[w]-cmd)**2,weights=diagram[w])/np.sum(diagram[w]))
    var_cmv = np.sqrt(np.average((velom[w]-cmv)**2,weights=diagram[w])/np.sum(diagram[w]))
    print 'v_LSRK < v_sys (offset,vel): {:0.2f} +- {:0.2f}", {:0.2f} +- {:0.2f} km/s'.format(-cmd,var_cmd,-(cmv-vsys),var_cmv)
    plt.errorbar(-cmd,-(cmv-vsys),xerr=var_cmd,yerr=var_cmv,fmt='xk',label='v$_{LSRK}$<v$_{sys}$',ls=':')
    plt.legend(frameon=False,fontsize=14)


    
def calc_noise(image,imx=10):
    '''Calculate the noise within an image. The noise is used in multiple functions (im_plot_spec, mk_chmaps, imdiff) and the calculation is moved here to make sure it is consistent between all of them'''
    #assuming we are dealing with full image and not cropped images
    #imx is width of box in pixels. This box is used to define noise.
    import numpy as np
    if len(image.shape)>2:
    #could also consider calculating noise as a function of channel
        imx=int(np.abs(imx))
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
            noise1[i] = np.std(image[i,low:npix/2-imx/2,low:npix/2-imx/2],ddof=1)
            noise2[i] = np.std(image[i,low:npix/2-imx/2,npix/2+imx/2:high],ddof=1)
            noise3[i] = np.std(image[i,npix/2+imx/2:high,low:npix/2-imx/2],ddof=1)
            noise4[i] = np.std(image[i,npix/2+imx/2:high,npix/2+imx/2:high],ddof=1)
            noise5[i] = np.std(image[i,low:high,low:high],ddof=1)
            noise[i] = np.mean([noise1[i],noise2[i],noise3[i],noise4[i]])

        return np.mean([noise1,noise2,noise3,noise4])
    else:
        imx=int(np.abs(imx))
        npix = image.shape[1]
        
        if npix/2-3*imx/2 <0:
            low = 0
        else:
            low = npix/2-3*imx/2
        if npix/2+3*imx/2>npix:
            high = -1
        else:
            high = npix/2+3*imx/2
    
        noise1 = np.std(image[low:npix/2-imx/2,low:npix/2-imx/2],ddof=1)
        noise2 = np.std(image[low:npix/2-imx/2,npix/2+imx/2:high],ddof=1)
        noise3 = np.std(image[npix/2+imx/2:high,low:npix/2-imx/2],ddof=1)
        noise4 = np.std(image[npix/2+imx/2:high,npix/2+imx/2:high],ddof=1)
        return np.mean([noise1,noise2,noise3,noise4])




    
