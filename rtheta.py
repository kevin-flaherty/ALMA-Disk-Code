def rtheta(file,PA,incl,size,offs=[0.,0.]):
    ''' Create a R vs theta map of an image. Theta is the azimuthal angle in the deprojected disk. This projection is useful when looking for azimuthal variations in flux.
    ***Doesn't work***

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

    alma = fits.open(file)
    im = alma[0].data.squeeze()
    hdr = alma[0].header

    ra = 3600*hdr['cdelt1']*(np.arange(hdr['naxis1'])-hdr['naxis1']/2.-0.5)
    noise = calc_noise(im,np.round(size/(3600.*hdr['cdelt1'])))
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


    ram,dem = np.meshgrid(ra,de)
    major = (ram*np.cos(np.radians(90-PA))+dem*np.sin(np.radians(90-PA)))
    minor = (ram*np.sin(np.radians(90-PA))-dem*np.cos(np.radians(90-PA)))/np.sin(np.radians(90-incl))
    dist = np.sqrt(major**2+minor**2)
    
    theta = np.degrees(np.arctan2(minor,major))
    theta[theta<0]+=360
    
    plt.figure()
    cs = plt.contourf(ram,dem,im,100,cmap=plt.cm.afmhot)
    plt.contour(ram,dem,dist,np.arange(0,1,.1),colors='w',linewidths=3)
    plt.xlim(size,-size)
    plt.ylim(-size,size)
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
    levels = np.arange(101)/100.*(im[w].max()-im[w].min())+im[w].min()

    nclevels = (im[w].max()-10*noise)/(50*noise)
    clevels = (np.arange(nclevels)+1)*50*noise

    plt.figure()
    cs=plt.contourf(theta,dist,im*1e3,levels*1e3,cmap=plt.cm.copper)
    plt.contour(theta,dist,im*1e3,clevels*1e3,colors='w',linestyles='--',linewidths=3)
    plt.colorbar(cs)
    plt.ylim(0,size)
    ax = plt.gca()
    plt.xlabel('Theta (degrees)',fontweight='bold',fontsize=16)
    plt.ylabel('R (arcseconds)',fontweight='bold',fontsize=16)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontweight('bold')


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
        
    #print noise1,noise2,noise3,noise4
    #print noise5.mean(),noise5
    #noise from boxes around the disk has similar noise to boxes centered in image, but using line free channels (noise=6.84e-5,6.53e-5 Jy/pixel for two scenarios). No strong channel dependence when looking at the line-free channels

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
        noise5 = np.std(image[low:high,low:high])
        #print noise1,noise2,noise3,noise4,noise5
        return np.mean([noise1,noise2,noise3,noise4])
        #return noise5
