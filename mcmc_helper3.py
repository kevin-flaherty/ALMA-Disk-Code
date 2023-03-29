import numpy as np
import matplotlib.pylab as plt
from galario import double as gdouble
from astropy.io import fits
import os

def im_plot_spec(file='data/HD163296.CO32.regridded.cen15.cm.fits',size=10,fwhm=False,gain=1.,threshold=None,norm_peak=False,mirror=False,line='co21',show_cloud=False,**kwargs):
    '''Given an image, plot the spectrum based on the sum of the flux over a given area.

    file (string): Name of the file. This can be a cleaned image, or a model image. ex. 'imlup_co21.fits'

    size (float): The area of the image over which the flux is to be summed, e.g. a value of 10 will sum the flux in a 10 arc-second by 10 arc-second box centered on the center of the image. Default=10

    fwhm (boolean): Calculate the full-width at half-maximum of the line. Default = False

    gain (float): Apply a multiplicative gain factor to the spectrum. Useful if you want to scale up or down the total flux of the line. Default = 1.

    threshold (float): Pixels with fluxes above the threshold are summed to create the spectrum. Pixels with fluxes below the threshold are ignored. If threshold=None, then the threshold is set to 3 times the noise level, as defined by boxes at the edge of the image. Default=None.

    norm_peak (boolean): Normalized the line flux to its peak value. Default = False.

    mirror (boolean): Mirror the line about its center. Useful when looking for asymmetries in the line profile. Default = False.

    line (string): Not used

    show_cloud (boolean): Mark in grey the regions in the spectrum where cloud contamination dimishes the flux. Set to 4 - 6 km/s, for IM Lup.

    **kwargs: Accepts other line profile keywords (e.g., color, linewidth, linestyle) and passes them on to the line plotting function.

    '''



    # - Read in the data
    im = fits.open(file)
    data = im[0].data.squeeze()
    hdr = im[0].header
    #noise = np.std(data[:,:,:10])
    noise = calc_noise(data,np.round(size/(3600.*hdr['cdelt1'])))
    ra = 3600*hdr['cdelt1']*(np.arange(hdr['naxis1'])-hdr['naxis1']/2.-0.5)
    dec = 3600*hdr['cdelt2']*(np.arange(hdr['naxis2'])-hdr['naxis2']/2.-0.5)
    offs = [0.,0.]
    #offs=[3.,0.]
    ira = np.abs(ra-offs[0]) < size/2.
    ide = np.abs(dec-offs[1]) < size/2.
    data = data[:,:,ira]
    data = data[:,ide,:]
    npix = ira.sum()*ide.sum()
    data *= gain
    noise *= gain

    #If displaying a model, convert flux from Jy/beam to Jy/pixel
    if (hdr['object'])[:5] != 'model':
        try:
            bmaj,bmin = hdr['bmaj'],hdr['bmin']
        except KeyError:
            #multiple beams are present, and listed in a table
            #convert from arceconds to degrees
            bmaj,bmin = im[1].data[0][0]/3600.,im[1].data[0][1]/3600.
        beam = np.pi*bmaj*bmin/(4*np.log(2))
        pix = np.abs(hdr['cdelt1'])*np.abs(hdr['cdelt2'])
        #data *= (pix/beam)
        #noise *= (pix/beam)
        sigmaj = 3600*bmaj/(2*np.sqrt(2*np.log(2)))
        sigmin = 3600*bmin/(2*np.sqrt(2*np.log(2)))
        ram,dem = np.meshgrid(ra,dec)
        area = np.exp(-(ram**2/(2*sigmaj**2)+dem**2/(2*sigmin**2))).sum()
    else:
        area = 1.

    if (hdr['ctype3'] == 'VELO-LSR') or (hdr['ctype3']=='VELO-OBS') or (hdr['ctype3']=='VRAD'):
        xaxis = ((np.arange(hdr['naxis3'])+1-hdr['crpix3'])*hdr['cdelt3']+hdr['crval3'])/1e3
        #xaxis -= np.median(xaxis)
    else:
        nfreq = hdr['naxis3']
        freq = (np.arange(hdr['naxis3'])+1-hdr['crpix3'])*hdr['cdelt3']+hdr['crval3']
        try:
            xaxis = (hdr['restfrq']-freq)/hdr['restfrq']*2.99e5
        except KeyError:
            xaxis = (np.median(freq)-freq)/np.median(freq)*2.99e5


    #im_dec = data.sum(axis=2)
    #spec = im_dec.sum(axis=1)/area
    spec = np.zeros(hdr['naxis3'])
    if threshold is None:
        threshold = 3*noise
    print('threshold: ',threshold)
    for i in range(hdr['naxis3']):
        bright = data[i,:,:]>threshold
        spec[i] = data[i,:,:][bright].sum()/area

    if norm_peak:
        spec = spec/spec.max()
    if mirror:
        xaxis = xaxis[::-1]-.25#-.1

    plt.rc('axes',lw=2)
    plt.plot(xaxis,spec,lw=4,**kwargs)
    ax = plt.gca() #apply_aspect,set_adjustable,set_aspect,get_adjustable,get_aspect
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontweight('bold')
    plt.xlabel('Velocity (km/sec)',fontweight='bold',fontsize=14)
    if norm_peak:
        plt.ylabel('Normalized Flux',fontweight='bold',fontsize=14)
    else:
        plt.ylabel('Flux (Jy)',fontweight='bold',fontsize=14)
    if show_cloud:
        #Highlight the region affected by foreground absorption
        plt.fill_between([4,6],[plt.ylim()[0],plt.ylim()[0]],[plt.ylim()[1],plt.ylim()[1]],alpha=.2,color='k')
    im.close()

    #print xaxis[:14],xaxis[27:]

    if fwhm:
        #Calculate the fwhm of the line
        m=spec.max()
        dv1 = np.interp(m/2,(spec[xaxis<0].squeeze())[::-1],(xaxis[xaxis<0].squeeze())[::-1])
        dv2 = np.interp(m/2,(spec[xaxis>0].squeeze()),(xaxis[xaxis>0].squeeze()))
        return np.abs(dv1)+np.abs(dv2)

    print('dv,v_centroid:',xaxis[1]-xaxis[0],(spec*xaxis).sum()/spec.sum())
    return spec.sum()*(xaxis[1]-xaxis[0])
    #return xaxis,spec
#plt.legend(('v$_z$=0','v$_z$=0.5c$_s$','v$_z$=c$_s$'),frameon=False,mode='expand',ncol=3,prop={'weight':'bold','size':20})

def mk_chmaps(datfile='imlup_co21.cm.fits',offs=[.0,0.],PA=144.6,incl=51.6,channels=[3,4,5,6,7,8,9,10,11],axis_label=True,scale_label=True,plot_data=True,gain=1.,mirror=False,altlevels=False,line='co21',size=10.,distance=145.):
    '''Given a model fits image file, make channel maps.

    datfile (string): Name of the cleaned image for the data. ex. 'imlup_co21.fits'

    offs (2-element list): X and Y offset, in arc-seconds, for the center of the emission

    PA (float): Position angle of the disk, measured east of north, in degrees. Default = 144.6

    incl (float): Inclination of the disk, in degrees. Default = 51.6

    channels (list of integers): The channels to be displayed. The code will not work if one of the channels requested is not in the images. Can be one channel, or multiple channels, but must be a list. If there are more than seven channels, it will plot 7 columns, and multiple rows.

    axis_label (boolean): If True, then adds labels to the axes (Delta RA, Delta Dec). Only applies when there is one channel. Default = True

    scale_label (boolean): At a colorbar showing the flux scale. Only applies when there is one channel. Default = True.

    plot_data: (boolean): Not used

    gain (float): Multiplicative gain factor to be applied uniformly to all of the data. Default=1

    mirror (boolean): Use contours to plot a mirrored version of the data, along with the difference between the data and its mirrored version. This assumes that the central channel corresponds to the line center. Default = False

    altlevels (boolean): By default, the levels for the filled contours are 5 times the noise level, or 10 times the noise level if the peak flux is >50 times the noise level. Setting altlevels=True sets the levels at 20 evenly spaced levels from the min to the max in the images. Useful since filled contours are used. Default = False.

    line (string): Not used. Default = 'co21'

    size (float): Size of the postage stamp cutout from the full images, in units of arc-seconds. Default = 10.

    distance (float): Distance to the object, in parsecs. Useful when plotting lines of constant radius, and for calculating the size of the 100 au scale bar. Default = 145.
    '''


    # - Read in the files
    cube = fits.open(datfile)
    hdr = cube[0].header
    image = cube[0].data.squeeze()
    #noise = np.std(image[:,:,:10])
    ra = 3600*hdr['cdelt1']*(np.arange(hdr['naxis1'])-hdr['naxis1']/2.-0.5)
    #ra2 = 3600*hdr2['cdelt1']*(np.arange(hdr2['naxis1'])-hdr2['naxis1']/2.-0.5)

    # - Crop image
    imx = size    # image size in arcseconds
    noise = calc_noise(image,np.round(imx/(3600.*hdr['cdelt1'])))#.004486
    de = -1*ra
    ra = ra-offs[0]
    de = de-offs[1]
    ira = np.abs(ra) < imx/2.
    ide = np.abs(de) < imx/2.
    image = image[:,:,ira]
    image = image[:,ide,:]
    ra = ra[ira]
    de = de[ide]
    #image *= gain
    #noise *= gain

    # - Crop image
    #imx = size    # image size in arcseconds
    #de2 = -1*ra2
    #ra2 = ra2-offs[0]
    #e2 = de2-offs[1]
    #ira2 = np.abs(ra2) < imx/2.
    #ide2 = np.abs(de2) < imx/2.
    #image2 = image2[:,:,ira2]
    #image2 = image2[:,ide2,:]
    #a2 = ra2[ira2]
    #de2 = de2[ide2]
    #mage2 = image2/gain

    #if mirror:
    #    ra2=-ra
    #    de2=-de
    #    image2 = np.transpose(image[::-1,:,:])

    try:
        bmaj,bmin,bpa = hdr['bmaj'],hdr['bmin'],hdr['bpa']
    except KeyError:
        bmaj,bmin,bpa = cube[1].data[0][0]/3600.,cube[1].data[0][1]/3600.,cube[1].data[0][1]
    if hdr['object'][:5] == 'model':
    # Model fluxes are in units of flux per pixel. Convert this to flux per beam
        beam=np.pi*bmaj*bmin/(4*np.log(2))
        pix = np.abs(hdr['cdelt1'])*np.abs(hdr['cdelt2'])
        image *= beam/pix

    #Beam size
    a,b,phi = bmaj/2*3600.,bmin/2*3600.,np.radians(bpa)
    t=np.linspace(0,2*np.pi,100)
    x = -(size/2.-1.)+a*np.cos(t)*np.cos(phi)-b*np.sin(t)*np.sin(phi)
    y = -(size/2.-1.)+a*np.cos(t)*np.sin(phi)+b*np.sin(t)*np.cos(phi)
    print('Peak flux: ',image.max())

    #CO ice line
    a_ice,b_ice,phi_ice = 250./140,250./140*np.sin(np.radians(incl)),np.radians(PA)
    x_ice = a_ice*np.cos(t)*np.cos(phi_ice)-b_ice*np.sin(t)*np.sin(phi_ice)
    y_ice = a_ice*np.cos(t)*np.sin(phi_ice)+b_ice*np.sin(t)*np.cos(phi_ice)

    channels=np.array(channels)
    nchans = channels.size
    nfreq = hdr['naxis3']
    if (hdr['ctype3'] == 'VELO-LSR') or (hdr['ctype3'] == 'VELO-OBS'):
        velo = ((np.arange(hdr['naxis3'])+1-hdr['crpix3'])*hdr['cdelt3']+hdr['crval3'])/1e3
        #velo -= 5.79#np.median(velo)
    else:
        nfreq = hdr['naxis3']
        freq = (np.arange(hdr['naxis3'])+1-hdr['crpix3'])*hdr['cdelt3']+hdr['crval3']
        try:
            velo = (hdr['restfrq']-freq)/hdr['restfrq']*2.99e5
        except KeyError:
            velo = (np.median(freq)-freq)/np.median(freq)*2.99e5

    # - Make figure
    nlevels = (np.max(image)-10*noise)/(10*noise)
    print('Initial number of levels: ',nlevels)
    if nlevels > 5:
        levels = (np.arange(nlevels)+1)*10*noise
    else:
        nlevels = (np.max(image)-5*noise)/(5*noise)
        levels = (np.arange(nlevels)+1)*5*noise
        #nlevels = (np.max(image)-5*noise)/(noise/2.)
        #levels = np.arange(nlevels)*noise/2+5*noise
    if altlevels:
        #levels as a percentage of max flux (10,25,40%,...)
        levels = (np.arange(7)*1.5+1)/10.*np.max(image)
        levels = (np.arange(100)+10)/100.*np.max(image)
        levels = (np.arange(20))/20.*(np.max(image)-np.min(image))+np.min(image)
        #nlevels = (np.max(image)-5*noise)/(noise/3)
        #levels= (np.arange(nlevels)+1)*noise/3+5*noise
    print('Min level, noise:',levels.min(),noise)
    #levels = (np.arange(5)+1)*2*noise
#    dl = (np.max(image)-5*noise)/100
#    levels = (np.arange(100)*dl)+5*noise
    r1 = 0.9*imx*np.array([-1,1])/2.
    theta1 = np.radians(-np.array([PA,PA]))
    r2 = 0.9*imx*np.array([-1,1])/2.*np.cos(np.radians(incl))
    theta2 = np.radians(90.-np.array([PA,PA]))


    if nchans > 1:
        Ncols = 7
        fig=plt.figure()
        nrows = (nchans-1)/Ncols+1
        if nchans >Ncols:
            ncols = Ncols
        else:
            ncols = nchans
        for i in range(nchans):
            plt.subplots_adjust(wspace=.001)
            if i ==0:
                ax1 = plt.subplot(nrows,ncols,1)
                ax1.set(aspect='equal',adjustable='box')
            else:
                ax = plt.subplot(nrows,ncols,i+1,sharex=ax1,sharey=ax1)
                ax.set(aspect='equal',adjustable='box')
            cs = plt.contourf(ra,de,image[channels[i],:,:],levels,cmap=plt.cm.afmhot)#plt.cm.YlOrRd Greys afmhot (ie ALMA color-scheme)
            if altlevels:
                plt.contour(ra,de,image[channels[i],:,:],[3*noise,],colors='w',linewidths=2,linestyles='--')
            if i==0:
                plt.gca().invert_xaxis()
            #plt.plot(x_ice,y_ice,'k',ls='--',alpha=.5,lw=2)
            if mirror:
                plt.contour(ra2,de2,image[channels[i],::-1,::-1]-image2[:,:,channels[i]],levels,colors='k',linewidths=2,alpha=0.7)
                plt.contour(ra2,de2,image[channels[i],::-1,::-1]-image2[:,:,channels[i]],-levels[::-1],colors='r',linewidths=2,alpha=0.7)
            #else:
            #    plt.contour(ra2,de2,image2[channels[i],:,:],levels,colors='k',linewidths=2,alpha=0.7,linestyles=':')
            plt.subplots_adjust(wspace=0.)
            plt.subplots_adjust(hspace=0.)
            plt.text(plt.xlim()[0]*.85,plt.ylim()[1]*.72,'{:.2f}'.format(velo[channels[i]]),fontsize=13,color='w')
            if (nrows ==1 and i == 0) or (nrows >1 and i % ncols ==0 and nchans-i <= ncols):
                plt.xlabel(r'$\Delta\alpha$ (")',fontsize=18)
                plt.ylabel('$\Delta\delta$ (")',fontsize=18)
                plt.fill(x-.5,y-.5,lw=2,color='w')
                axis=plt.gca()
                for tick in axis.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(16)
                    #tick.label1.set_fontweight('bold')
                for tick in axis.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(16)
                    #tick.label1.set_fontweight('bold')
                plt.xticks(rotation=45)
            else:
                if i>0:
                    plt.setp(ax.get_xticklabels(),visible=False)
                    plt.setp(ax.get_yticklabels(),visible=False)
                else:
                    plt.setp(ax1.get_xticklabels(),visible=False)
                    plt.setp(ax1.get_yticklabels(),visible=False)
            if i==nchans-1:
                fig.subplots_adjust(top=.8)
                cbar_ax = fig.add_axes([.125,.805,.775,.03])
                cb=fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',format='%0.3f')
                cb.ax.xaxis.set_ticks_position('top')
                cb.ax.xaxis.set_label_position('top')
                #cb.ax.invert_xaxis()
                cb.set_label(label='Jy/beam',size=16)#,weight='bold')
                for tick in cb.ax.xaxis.get_majorticklabels():
                    tick.set_fontsize(14)
                    #tick.set_fontweight('bold')
            if i==0:
                #add 100AU scale bar
                plt.plot((-(size-1.5)/2.+1,-(size-1.5)/2.+1-100./distance),(size/2.-1,size/2.-1),lw=3,color='w')
                #plt.text(-1.5,-3.8,'100 AU',fontsize=12,color='k')
    else:
        plt.rc('axes',lw=4)
        levels = (np.arange(40))/40.*(np.max(image)-np.min(image))+np.min(image)
        #levels = (np.arange(91)+10)/100.*np.max(image2[channels,:,:])
        #cs=plt.contour(ra2,de2,image2[channels,:,:],levels,colors='k',linewidths=2)
        cs=plt.contourf(ra,de2,image[channels,:,:],levels,cmap=plt.cm.afmhot)
        plt.contour(ra,de,image[channels,:,:],[3*noise,],colors='w',linewidths=2)
        #if plot_data:
        #    cs = plt.contourf(ra,de,image[channels,:,:],levels)

        ax = plt.gca()
        ax.set_aspect('equal')
        ax.invert_xaxis()
        plt.xticks(rotation=45)
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
            #tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
            #tick.label1.set_fontweight('bold')
        #plt.plot(r1*np.cos(theta1),r1*np.sin(theta1),color='k',lw=2)
        #plt.plot(r2*np.cos(theta2),r2*np.sin(theta2),color='k',lw=2)
        print('Velocity {:.3f}'.format(velo[channels]))
        if axis_label:
            plt.xlabel('$\Delta$RA (")',fontsize=16)
            plt.ylabel('$\Delta$Dec (")',fontsize=16)
        plt.fill(x,y,lw=4,color='w')
        if scale_label:
            #plt.rc('text',usetex=True)
            cb = plt.colorbar(cs,pad=0.,shrink=0.9,format='%0.3f')
            cb.set_label(label='Jy/beam',size=16)
            for l in cb.ax.yaxis.get_majorticklabels():
                #l.set_fontweight('bold')
                l.set_fontsize(14)
    cube.close()
    #cube2.close()


def imdiff(datfile='imlup_co21.fits',modfile='alma.diff.fits',channels=[10,20,30,40,50,60,70,80,90],sma=False,redo=True,gain=1.,altlevels=False,resid_moment=False,resid_spec=False,line='co21',size=10.,triplot=False,**kwargs):
    '''Plots channels maps for the data, model, and residuls.
    datfile (string): The name of the cleaned image of the data. ex. 'imlup_co21.fits'

    modfile (string): The name of the cleaned image of the residuals. When triplot=True, the code assumes that the model image file has the same base name, with '.model.fits' replacing '.diff.fits'. ex. If modfile is 'alma.diff.fits', the code will look for a cleaned image of the model with the name 'alma.model.fits'.

    channels (list of integers): A list of the channels to display. If the requested channels are not in the data, residual, or model images, then the code will crash.

    sma (boolean): Not used.

    redo (boolean): Not used.

    gain (float): Divide the residuals by this value. Default = 1.

    altlevels (boolean): In the normal mode, the levels are multiples of 5 sigma. When altlevels =True in the normal mode, the levels are a fraction of the maximum flux in the cleaned data image. In triplot=True, levels are 40 evenly spaced segments between the maximum in the cleaned data image, and the minimum in the residual image.

    resid_moment (boolean): Show a moment map of the residuals. Assumes that there is a moment 0 and moment 1 map, with the same based name as modfile, e.g. if modfile='alma.diff.fits' it will look for 'alma.diff.mom0.fits' and 'alma.diff.mom1.fits'. Default = False

    resid_spec (boolean): Show a spectrum of the residuals. Default = False.

    line (string): Specifies the emission line being used. When triplot=True, this specifies whether the line contour is at 5 sigma ('13co21sblb','13co21','co21sblb','co21') or 3 sigma (anything else). Default = 'co21'.

    size (float): Size, in arc-seconds, of the postage stamps that are cutout and displayed from the original image.

    triplot (boolean): If set to True, then instead of showing residuals on top of the data, it shows three separate rows; the top row is the data, the middle row is the model, and the bottom row shows the residuals. Default = False

    '''

    #modfile = modfile+'.diff.fits'
    #datfile = datfile+'.cm.fits'

    alma, model = fits.open(datfile),fits.open(modfile)
    alma_im, model_im = alma[0].data.squeeze(), model[0].data.squeeze()
    hdr, hdr2 = alma[0].header, model[0].header
    #noise = np.std(alma_im[:,:,:10])
    ra = 3600*hdr['cdelt1']*(np.arange(hdr['naxis1'])-hdr['naxis1']/2.-0.5)
    ra2 = 3600*hdr2['cdelt1']*(np.arange(hdr2['naxis1'])-hdr2['naxis1']/2.-0.5)
    nfreq = alma_im.shape[0]
    PA,incl=156.,30.

    imx = size #image size in arcseconds
    noise = calc_noise(alma_im,np.round(imx/(3600.*hdr['cdelt1'])))
    de = -1*ra
    ra = ra#-offs[0]
    de = de#-offs[1]
    ira = np.abs(ra) < imx/2.
    ide = np.abs(de) < imx/2.
    alma_cm_tmp = alma_im[:,:,ira]
    alma_cm_tmp = alma_cm_tmp[:,ide,:]
    ra = ra[ira]
    de = de[ide]
    #alma_cm_tmp *= gain
    #noise *= gain

    de2 = -1*ra2
    ra2 = ra2#-offs[0]
    de2 = de2#-offs[1]
    ira2 = np.abs(ra2) < imx/2.
    ide2 = np.abs(de2) < imx/2.
    model_cm_tmp = model_im[:,:,ira2]
    model_cm_tmp = model_cm_tmp[:,ide2,:]
    ra2 = ra2[ira2]
    de2 = de2[ide2]
    model_cm_tmp = model_cm_tmp/gain

    channels=np.array(channels)
    nchans = channels.size
    nfreq = hdr['naxis3']

    if (hdr['ctype3'] == 'VELO-LSR') or (hdr['ctype3'] == 'VELO-OBS'):
        velo = ((np.arange(hdr['naxis3'])+1-hdr['crpix3'])*hdr['cdelt3']+hdr['crval3'])/1e3
        #velo -= np.median(velo)
    else:
        freq = (np.arange(hdr['naxis3'])+1-hdr['crpix3'])*hdr['cdelt3']+hdr['crval3']
        velo = (np.median(freq)-freq)/np.median(freq)*2.99e5
        #freq2=(np.arange(hdr2['naxis3'])+1-hdr2['crpix3'])*hdr2['cdelt3']+hdr2['crval3']
        #print (hdr2['restfreq']-freq)/hdr2['restfreq']*2.99e5


    #Beam size
    try:
        bmaj,bmin,bpa = hdr['bmaj'],hdr['bmin'],hdr['bpa']
    except KeyError:
        #multiple beams are present, and listed in a table
        #convert from arceconds to degrees
        bmaj,bmin,bpa = alma[1].data[0][0]/3600.,alma[1].data[0][1]/3600.,alma[1].data[0][2]
    a,b,phi = bmaj/2*3600.,bmin/2*3600.,np.radians(90+bpa)
    t=np.linspace(0,2*np.pi,100)
    x = -(size/2.-1)+a*np.cos(t)*np.cos(phi)-b*np.sin(t)*np.sin(phi)
    y = -(size/2.-1)+a*np.cos(t)*np.sin(phi)+b*np.sin(t)*np.cos(phi)

    #CO ice line
    a_ice,b_ice,phi_ice = 300./140,300./140*np.cos(np.radians(incl)),np.radians(PA)
    x_ice = a_ice*np.cos(t)*np.cos(phi_ice)-b_ice*np.sin(t)*np.sin(phi_ice)
    y_ice = a_ice*np.cos(t)*np.sin(phi_ice)+b_ice*np.sin(t)*np.cos(phi_ice)



    # - Make figure
    #nlevels = (np.max(alma_cm_tmp)-3*noise)/(3*noise)
    #if nlevels > 5:
    #    levels = (np.arange(nlevels)+1)*10*noise
    #else:
    #if nlevels>5:
    nlevels = (np.max(alma_cm_tmp)-5*noise)/(5*noise)
    levels = (np.arange(nlevels))*5*noise+5*noise
    #else:
    #    nlevels+=1
    #    levels = (np.arange(nlevels))*3*noise+3*noise
    if altlevels:
        levels = (np.arange(7)*1.5+1)/10.*np.max(alma_cm_tmp)
        #levels = np.arange(50)*noise+3*noise
        #nlevels = (np.max(alma_cm_tmp)-5*noise)/(5*noise)
        #levels = (np.arange(nlevels)+1)*5*noise
    print('Min level, noise:',levels.min(),noise)
    r1 = 0.9*imx*np.array([-1,1])/2.
    theta1 = np.radians(-np.array([PA,PA]))
    r2 = 0.9*imx*np.array([-1,1])/2.*np.cos(np.radians(incl))
    theta2 = np.radians(90.-np.array([PA,PA]))

    if not resid_moment and not resid_spec and not triplot:
        if nchans > 1:
            Ncols = 6
            fig = plt.figure()
            nrows = (nchans-1)/Ncols+1
            if nchans > Ncols:
                ncols = Ncols
            else:
                ncols = nchans
            for i in range(nchans):
                plt.subplots_adjust(wspace=.001)
                plt.subplots_adjust(hspace=.001)
                if i ==0:
                    ax1 = plt.subplot(nrows,ncols,1)
                    ax1.set(aspect='equal',adjustable='box')
                else:
                    ax = plt.subplot(nrows,ncols,i+1,sharex=ax1,sharey=ax1)
                    ax.set(aspect='equal',adjustable='box')
                channel = channels[i]
                cs = plt.contourf(ra,de,alma_cm_tmp[channel,:,:].squeeze(),levels,cmap=plt.cm.Blues)#BuGn
                plt.contour(ra2,de2,model_cm_tmp[channel,:,:].squeeze(),levels,colors='k',linewidths=2)
                plt.contour(ra2,de2,model_cm_tmp[channel,:,:].squeeze(),-levels[::-1],colors='r',linewidths=2,linestyles=':')
                if i==0:
                    plt.gca().invert_xaxis()
                sig = alma_cm_tmp[channel,:,:].squeeze()>noise
                #plt.plot(x_ice,y_ice,'k',ls='--',alpha=.5,lw=2)
                #plt.plot(x_ice2,y_ice2,'k',ls='--',alpha=.5,lw=2)
                plt.text(plt.xlim()[0]*.8,plt.ylim()[1]*.7,'{:.2f}'.format(velo[channel]),fontsize=12)
                if (nrows ==1 and i == 0) or (nrows >1 and i % ncols == 0 and nchans-i <= ncols):
                    plt.xlabel(r'$\Delta\alpha$ (")',fontsize=14)
                    plt.ylabel('$\Delta\delta$ (")',fontsize=14)
                    plt.xticks(rotation=45)
                    plt.fill(x,y,lw=2)
                    axis = plt.gca()
                    for tick in axis.xaxis.get_major_ticks():
                        tick.label1.set_fontsize(14)
                        tick.label1.set_fontweight('bold')
                    for tick in axis.yaxis.get_major_ticks():
                        tick.label1.set_fontsize(14)
                        tick.label1.set_fontweight('bold')
                else:
                    if i>0:
                        plt.setp(ax.get_xticklabels(),visible=False)
                        plt.setp(ax.get_yticklabels(),visible=False)
                    else:
                        plt.setp(ax1.get_xticklabels(),visible=False)
                        plt.setp(ax1.get_yticklabels(),visible=False)
                if i==nchans-1:
                    fig.subplots_adjust(top=.8)
                    cbar_ax = fig.add_axes([.125,.805,.775,.03])
                    cb = fig.colorbar(cs,cax=cbar_ax,label='Jy/beam',orientation='horizontal',format='%0.3f')
                    cb.ax.xaxis.set_ticks_position('top')
                    cb.ax.xaxis.set_label_position('top')
                    #cb.ax.invert_xaxis()
                    cb.set_label(label='Jy/beam',size=18,weight='bold')
                    for tick in cb.ax.xaxis.get_majorticklabels():
                        tick.set_fontsize(16)
                        tick.set_fontweight('bold')

                if i==0:
                #add 100AU scale bar
                    plt.plot((-(size-2)/2+1,-(size-2)/2+1+100./140),(-(size/2-1),-(size/2-1)),lw=3,color='k')
                    #plt.text(-1.5,-3.8,'100 AU',fontsize=12,color='k')
                #plt.gca().invert_xaxis()
            if nchans < 1:
                plt.tight_layout()
        else:
            cs = plt.contourf(ra,de,alma_cm_tmp[channels,:,:].squeeze(),levels,cmap=plt.cm.Blues)
            plt.contour(ra2,de2,model_cm_tmp[channels,:,:].squeeze(),levels,colors='k',linewidths=2)
            plt.contour(ra2,de2,model_cm_tmp[channels,:,:].squeeze(),-levels[::-1],colors='r',linewidths=2,linestyles='-')
            ax = plt.gca()
            ax.set_aspect('equal')
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(20)
                tick.label1.set_fontweight('bold')
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(20)
                tick.label1.set_fontweight('bold')
            plt.xlabel(r'$\Delta\alpha$ (")',fontweight='bold',fontsize=16)
            plt.ylabel('$\Delta\delta$ (")',fontweight='bold',fontsize=16)
            plt.fill(x,y,lw=4,color='k')
            cb=plt.colorbar(cs,pad=0.,shrink=0.8,format='%0.2f')
            cb.set_label(label='Jy/beam',size=18,weight='bold')
            for l in cb.ax.yaxis.get_majorticklabels():
                l.set_fontweight('bold')
                l.set_fontsize(14)
            plt.gca().invert_xaxis()
    elif triplot:
        #Plot the data, the model and the residuals in separate rows
        resid_cm_tmp = model_cm_tmp

        modfile = modfile[:-10]+'.model.fits'
        #read in model image and crop
        model = fits.open(modfile)
        model_im = model[0].data.squeeze()
        hdr = model[0].header
        ra3 = 3600*hdr['cdelt1']*(np.arange(hdr['naxis1'])-hdr['naxis1']/2.-0.5)

        imx = size
        de3 = -1*ra3
        ira = np.abs(ra3) < imx/2.
        ide = np.abs(de3) < imx/2.
        model_cm_tmp = model_im[:,:,ira]
        model_cm_tmp = model_cm_tmp[:,ide,:]
        ra3 = ra3[ira]
        de3 = de3[ide]

        levels = (np.arange(40)/39.)*(np.max(alma_cm_tmp)-np.min(resid_cm_tmp))+np.min(resid_cm_tmp)
        if line =='13co21sblb' or line=='13co21' or line=='co21sblb' or line=='co21':
            nsig_levels = (np.max(alma_cm_tmp)-5*noise)/(5*noise)
            sig_levels = (np.arange(nlevels))*5*noise+5*noise
        else:
            nsig_levels = (np.max(alma_cm_tmp)-3*noise)/(3*noise)
            sig_levels = (np.arange(nlevels))*3*noise+3*noise
        ncols = nchans
        fig = plt.figure()
        for i in range(nchans):
            plt.subplots_adjust(wspace=.001)
            plt.subplots_adjust(hspace=.001)
            channel = channels[i]
            if i==0:
                ax1 = plt.subplot(3,ncols,1)
                ax1.set(aspect='equal',adjustable='box-forced')

                cs_data = plt.contourf(ra,de,alma_cm_tmp[channel,:,:].squeeze(),levels,cmap=plt.cm.Blues)
                plt.contour(ra,de,alma_cm_tmp[channel,:,:].squeeze(),[sig_levels[0],],colors='k')
                plt.gca().invert_xaxis()
                plt.text(-plt.xlim()[0]*.1,plt.ylim()[1]*.7,'{:.2f}'.format(velo[channel]),fontsize=12)
                plt.setp(ax1.get_xticklabels(),visible=False)
                plt.setp(ax1.get_yticklabels(),visible=False)
                plt.text((size-2)/2+.5,(size/2-1.5),'Data')

                ax = plt.subplot(3,ncols,ncols+1,sharex=ax1,sharey=ax1)
                ax.set(aspect='equal',adjustable='box-forced')
                cs_resid = plt.contourf(ra3,de3,model_cm_tmp[channel,:,:].squeeze(),levels,cmap=plt.cm.Blues)
                plt.contour(ra3,de3,model_cm_tmp[channel,:,:].squeeze(),[sig_levels[0],],colors='k')
                plt.setp(ax.get_xticklabels(),visible=False)
                plt.setp(ax.get_yticklabels(),visible=False)
                plt.text((size-2)/2+.5,(size/2-1.5),'Model')

                ax = plt.subplot(3,ncols,2*ncols+1,sharex=ax1,sharey=ax1)
                ax.set(aspect='equal',adjustable='box')
                cs_resid = plt.contourf(ra2,de2,resid_cm_tmp[channel,:,:].squeeze(),levels,cmap=plt.cm.Blues)
                plt.contour(ra2,de2,resid_cm_tmp[channel,:,:].squeeze(),sig_levels,colors='k',linewidths=2)
                plt.contour(ra2,de2,resid_cm_tmp[channel,:,:].squeeze(),-sig_levels[::-1],colors='r',linewidths=2,linestyles=':')
                plt.xlabel(r'$\Delta\alpha$ (")',fontsize=14)
                plt.ylabel('$\Delta\delta$ (")',fontsize=14)
                plt.xticks(rotation=45)
                plt.fill(x,y,lw=2,color='r')
                plt.plot(((size)/2-1,(size)/2-1-100./140),(-(size/2-1),-(size/2-1)),lw=3,color='k')
                plt.text((size-2)/2+.5,(size/2-1.5),'Residual')
            else:
                channel = channels[i]
                ax = plt.subplot(3,ncols,1+i,sharex=ax1,sharey=ax1)
                ax.set(aspect='equal',adjustable='box')
                cs_data = plt.contourf(ra,de,alma_cm_tmp[channel,:,:].squeeze(),levels,cmap=plt.cm.Blues)
                plt.contour(ra,de,alma_cm_tmp[channel,:,:].squeeze(),[sig_levels[0],],colors='k')
                plt.setp(ax.get_xticklabels(),visible=False)
                plt.setp(ax.get_yticklabels(),visible=False)
                plt.text(-plt.xlim()[0]*.1,plt.ylim()[1]*.7,'{:.2f}'.format(velo[channel]),fontsize=12)

                ax = plt.subplot(3,ncols,ncols+1+i,sharex=ax1,sharey=ax1)
                ax.set(aspect='equal',adjustable='box')
                cs_model = plt.contourf(ra3,de3,model_cm_tmp[channel,:,:].squeeze(),levels,cmap=plt.cm.Blues)
                plt.contour(ra3,de3,model_cm_tmp[channel,:,:].squeeze(),[sig_levels[0],],colors='k')
                plt.setp(ax.get_xticklabels(),visible=False)
                plt.setp(ax.get_yticklabels(),visible=False)

                ax = plt.subplot(3,ncols,2*ncols+1+i,sharex=ax1,sharey=ax1)
                ax.set(aspect='equal',adjustable='box')
                cs_resid = plt.contourf(ra2,de2,resid_cm_tmp[channel,:,:].squeeze(),levels,cmap=plt.cm.Blues)
                cs_resid2 = plt.contour(ra2,de2,resid_cm_tmp[channel,:,:].squeeze(),sig_levels,colors='k',linewidths=2)
                cs_resid3 = plt.contour(ra2,de2,resid_cm_tmp[channel,:,:].squeeze(),-sig_levels[::-1],colors='r',linewidths=2,linestyles=':')
                plt.setp(ax.get_xticklabels(),visible=False)
                plt.setp(ax.get_yticklabels(),visible=False)

                if i==nchans-1:
                    #add the scale bar
                    fig.subplots_adjust(top=.8)
                    cbar_ax = fig.add_axes([.125,.805,.775,.03])
                    cb = fig.colorbar(cs_resid,cax=cbar_ax,label='Jy/beam',orientation='horizontal',format='%0.3f')
                    cb.ax.xaxis.set_ticks_position('top')
                    cb.ax.xaxis.set_label_position('top')
                    cb.set_label(label='Jy/beam',size=18)#,weight='bold')
                    for tick in cb.ax.xaxis.get_majorticklabels():
                        tick.set_fontsize(14)
                        #tick.set_fontweight('bold')
                    #cb.add_lines(cs_resid2)
                    #cb.add_lines(cs_resid3,erase=False)

    elif resid_moment:
        #Moment 0 map (total intensity in residuals)
        #Use MIRIAD to generate moment 0 and 1 map from residuals
        #os.system('fits in='+modfile+' out='+modfile[:-5]+'.cm op=xyin')
        #os.system('moment in='+modfile[:-5]+'.cm out='+modfile[:-5]+'.mom0 mom=0')
        #os.system('moment in='+modfile[:-5]+'.cm out='+modfile[:-5]+'.mom1 mom=1')
        #os.system('fits in='+modfile[:-5]+'.mom0 out='+modfile[:-5]+'.mom0.fits op=xyout')
        #os.system('fits in='+modfile[:-5]+'.mom1 out='+modfile[:-5]+'.mom1.fits op=xyout')
        #os.system('rm -rf '+modfile[:-5]+'.cm '+modfile[:-5]+'.mom0 '+modfile[:-5]+'.mom1')
        mom0,mom1 = fits.open(modfile[:-5]+'.mom0.fits'),fits.open(modfile[:-5]+'.mom1.fits')
        mom0_im = mom0[0].data.squeeze()
        mom1_im = mom1[0].data.squeeze()
        noise_mom0 = calc_noise(mom0_im,np.round(imx/(3600.*hdr['cdelt1'])))
        mom0_im_tmp = mom0_im[:,ira]
        mom0_im_tmp = mom0_im_tmp[ide,:]
        mom1_im_tmp = mom1_im[:,ira]
        mom1_im_tmp = mom1_im_tmp[ide,:]
        diff_nlevels = np.ceil((np.max(np.abs(mom0_im_tmp))-5*noise_mom0)/(5*noise_mom0))
        if diff_nlevels<1:
            diff_nlevels=1
        diff_levels = (np.arange(diff_nlevels)+1)*5*noise_mom0
        mom0.close()
        mom1.close()

        plt.figure()
        plt.rc('axes',lw=2)
        map0 = alma_cm_tmp.sum(axis=0)*np.abs((velo[1]-velo[0]))
        levels = (np.arange(41))/40.*(np.max(map0)-np.min(map0))+np.min(map0)#+.1*np.max(map0)
        #plt.subplot(223)
        print('Noise in moment 0 map, min level: ',noise_mom0,diff_levels[0])
        cs = plt.contour(ra,de,mom0_im_tmp,diff_levels,linewidths=2,colors='k')
        plt.contour(ra,de,mom0_im_tmp,-diff_levels[::-1],colors='r',linestyles='--',linewidths=2)
        #plt.contour(ra,de,map0,levels,linewidths=2,colors='k',linestyles=':')
        #cs = plt.contour(ra,de,mom0_im_tmp,levels*np.sqrt(velo.shape[0])*np.abs((velo[1]-velo[0])),linewidths=2,colors='k')
        #plt.contour(ra,de,mom0_im_tmp,-levels*np.sqrt(velo.shape[0])*np.abs((velo[1]-velo[0])),colors='r',linestyles='--',linewidths=2)
        #plt.contourf(ra,de,map0,levels*np.sqrt(velo.shape[0])*np.abs((velo[1]-velo[0])),linewidths=2,cmap=plt.cm.Greys)
        cs=plt.contourf(ra,de,map0,levels,cmap=plt.cm.YlOrRd)
        cb=plt.colorbar(cs,label='Jy/beam km/s',pad=0.,shrink=.9,format='%0.3f')
        cb.set_label(label='Jy/beam km/s',size=16,weight='bold')
        for tick in cb.ax.yaxis.get_majorticklabels():
            tick.set_fontsize(14)
            tick.set_fontweight('bold')
        plt.contour(ra,de,map0,[3*noise_mom0,],linewidths=2,linestyles=':',colors='k')
        #plt.plot(r2*np.cos(theta1),r2*np.sin(theta1),color='k',lw=2,alpha=.2)
        #plt.plot(r1*np.cos(theta2),r1*np.sin(theta2),color='k',lw=2,alpha=.2)
        plt.xlabel(r'$\Delta\alpha$ (")',fontsize=18,fontweight='bold')
        plt.ylabel('$\Delta\delta$ (")',fontsize=18,fontweight='bold')
        ax = plt.gca()
        ax.set_aspect('equal')
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
            tick.label1.set_fontweight('bold')
        #plt.plot(x_ice,y_ice,'k',ls='--',alpha=.5,lw=2)
        plt.fill(x,y,lw=2)
        plt.plot(((size/2-1.5),(size/2-1.5)-100./140),(-(size/2-.5),-(size/2-.5)),lw=3,color='k')
        #if diff_nlevels>2:
        #    plt.colorbar(cs,label='Jy/beam km/s',pad=0.,shrink=.9,format='%0.2f')
        plt.gca().invert_xaxis()

        #Moment 1 map (intensity-weighted velocity)
        wremove = np.abs(mom0_im_tmp)<diff_levels[0]
        if line.lower()=='co32' or line.lower()=='co32hr':
            velo = velo[::-1]
            mom1_im_tmp -= np.median(mom1_im_tmp[~wremove])
        mom1_im_tmp[wremove]=-10
        #plt.subplot(224)
        #cs = plt.contourf(ra,de,mom1_im_tmp,velo[::-1])
        #plt.plot(r2*np.cos(theta1),r2*np.sin(theta1),color='k',lw=2,alpha=.2)
        #plt.plot(r1*np.cos(theta2),r1*np.sin(theta2),color='k',lw=2,alpha=.2)
        #plt.xlabel(r'$\Delta\alpha$ (")',fontsize=14)
        #plt.ylabel('$\Delta\delta$ (")',fontsize=14)
        #plt.plot(x,y,lw=2)
        #plt.plot((3,3-100./108),(-4,-4),lw=3,color='k')
        #plt.colorbar(cs,label='km/s',pad=0.,shrink=.9,format='%0.2f')
        #plt.gca().invert_xaxis()

    if resid_spec:
        #Start with the spectrum of the residuals
        #Convert from Jy/beam to Jy/pixel
        beam = np.pi*hdr2['bmaj']*hdr2['bmin']/(4*np.log(2))
        pix = np.abs(hdr2['cdelt1'])*np.abs(hdr2['cdelt2'])
        sigmaj = 3600*hdr['bmaj']/(2*np.sqrt(2*np.log(2)))
        sigmin = 3600*hdr['bmin']/(2*np.sqrt(2*np.log(2)))
        ram,dem = np.meshgrid(ra,de)
        area = np.exp(-(ram**2/(2*sigmaj**2)+dem**2/(2*sigmin**2))).sum()
        #model_cm_tmp *= (pix/beam)

        #im_dec = model_cm_tmp.sum(axis=2)
        #spec = im_dec.sum(axis=1)
        spec = np.zeros(hdr2['naxis3'])
        err = np.zeros(hdr2['naxis3'])
        threshold = 3*noise#*pix/beam
        for i in range(hdr2['naxis3']):
            bright = np.abs(model_cm_tmp[i,:,:])>threshold
            #print bright.sum()
            if bright.sum()>10:
                spec[i] = model_cm_tmp[i,:,:][bright].sum()/area
                err[i] = (noise*pix/beam)*np.sqrt(bright.sum())/area #Shouldn't this be sqrt(Npix)???

        plt.figure()
        plt.rc('axes',lw=2)
        plt.errorbar(velo,spec,yerr=err,capsize=0.,lw=4,**kwargs)
        plt.axhline(0.,color='k',ls='--',lw=3)
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
            tick.label1.set_fontweight('bold')
        plt.xlabel('Velocity (km/sec)',fontweight='bold',fontsize=14)
        plt.ylabel('Flux (Jy)',fontweight='bold',fontsize=14)


    alma.close()
    model.close()


def read_chain(dir='',flat=False):
    ''' Read in the files containing the parameters for each of the walkers. Assumes that you have a series of files called chain_qq.dat, chain_Tatm.dat, etc (the output from mpi_run_models_local.py) and that each one is formated such that each line contains a list of parameter values for each walker'''
    from astropy.io import ascii
    import os.path

    files = [dir+'chain_'+x+'.dat' for x in ['q','Rc','vturb','Tatm','Tmid','incl','Rin','vsys','xcen','ycen','PA','dep','abund','qvturb']]
    ndim = 0
    indeces =[]
    for i in range(len(files)):
        if os.path.exists(files[i]):
            indeces.append(i)
    ndim = len(indeces)

    for index in range(len(indeces)):
        data = open(files[indeces[index]],'r')
        chain = ascii.read(data)
        data.close()
        if index == 0:
            nwalkers = len(chain.dtype)-1
            nsteps = len(chain)
            chain_master = np.zeros((nwalkers,nsteps,ndim))
        for i in range(nwalkers):
            for j in range(nsteps):
                chain_master[i,j,index] = chain[j][i+1]
    if flat:
        chain_master = chain_master.reshape(nwalkers*nsteps,ndim)

    #chain_master[0,:,0] = 0th parameter for first walker at all steps
    #chain_master[1,:,3] = 4th parameter for second walkers at all steps

    #The result is an array with dimensions [Nwalkers x Nsteps x Nparameters]
    return chain_master

def read_chain_steps(file,nwalkers=20):
    '''Read the data file generated by mpi_run_models_steps.py. This is one file that contains all of the steps for all of the parameters in one large file. This code separates the different parameters and the different walkers. You need to specify the number of walkers correctly, or else the code will crash. '''

    from astropy.io import ascii

    chain=ascii.read(file)
    nparams = len(chain.dtype)-1
    params = chain.colnames
    nsteps = len(chain) // nwalkers
    print('# of steps: {:0.0f}'.format(nsteps))
    print('# of parameters: {:0.0f}'.format(nparams))
    print('# of walkers: {:0.0f}'.format(nwalkers))
    #chain_master = np.zeros((nwalkers,nsteps,nparams))
    chain_master = {}
    for i in range(nparams):
        chain_master[params[i]] = chain[params[i]].reshape(nsteps,nwalkers)

    #The result is a dictionary
    return chain_master

def corner_plot(chain,start=0,wuse=[]):
    '''Using the dictionary of steps, make a corner plot using the corner package. It only uses the walkers beyond the step number specified by 'start'
    wuse helps pick out only some of the walkers.

    To exclude 'stragglers', or that one walker that hasn't converged with the rest of them:
    >> wuse = chain['Tatm'][400:,:]<100
    >> corner_plot(chain,start=400,wuse)

    Contours at 1 and 3 sigma.
    '''

    import corner
    ndim = len(chain)
    names = list(chain.keys())
    nwalkers = chain[names[0]].shape[1]
    nsteps = chain[names[0]].shape[0]
    if len(wuse)>0:
        chain_new = np.zeros((wuse.sum(),ndim))
    else:
        chain_new = np.zeros((nwalkers*(nsteps-start),ndim))
    for i,n in enumerate(names):
        if len(wuse)>0:
            #print('-'*10)
            #print(chain_new[:,i].shape)
            #print(chain[n].shape)
            #print(chain[n][start:,:].shape)
            #print(chain[n][start:,:].flatten().shape)
            #print(wuse.shape)
            #print('-'*10)
            chain_new[:,i] = (chain[n][start:,:][wuse]).flatten()
        else:
            chain_new[:,i] = chain[n][start:,:].flatten()
        #print(i,n, chain[n][start:,:].flatten().shape,chain_new[:,i].shape)

    figure = corner.corner(chain_new,labels=names,quantiles=[.0015,.5,.9985],verbose=True,levels=(1-np.exp(-0.5),1-np.exp(-9./2)))#,1-np.exp(-25./2)))

def plot_chains(chain,values=None,names=['q','Rc','vturb/cCO','Tatm','Tmid','incl','gain']):
    '''Plot the positions of the walkers as a function of step number. Used with the output from mpi_run_models_local.py, as read in by read_chain'''
    # plot the positions of the walkers as a function of step number
#    names=['q','log(Mdisk)','log(Rc)','log(vturb/cCO)','Zq0','Tatm0','$\gamma$','Tmid0','incl']
#    names=['q','Rc','vturb/cCO','Tatm','Tmid','incl','gain']
    from math import ceil
    if values is not None:
        med = values[0]
        lunc = values[1]
        uunc = values[2]
    rows = ceil(len(names)/3.)
    nwalkers = chain.shape[0]
    for i in range(len(names)):
        plt.subplot(rows,3,1+i)
        for index in range(nwalkers):
            plt.plot(chain[index,:,i],alpha=.1)
        mp = np.median(chain[:,:,i],axis=0)
        plt.plot(mp,lw=2,color='r')
        plt.ylabel(names[i])
        plt.xlabel('step')
        if values is not None:
            plt.axhline(med[i],ls='--',lw=2,color='b')
            plt.plot((300,300),(med[i]-lunc[i],med[i]+uunc[i]),color='b',lw=2)
            plt.plot((400,400),(med[i]-lunc[i],med[i]+uunc[i]),color='b',lw=2)
            plt.plot((500,500),(med[i]-lunc[i],med[i]+uunc[i]),color='b',lw=2)

def plot_chains2(chain,labels=None):
    '''Plot the progression of the walkers. Uses the chain that is formated as a dictionary (from mpi_run_models_steps.py and read_chain_steps).'''
    #to change the names of the keys: chain[new_key]=chain.pop(old_key)
    from math import ceil

    rows = ceil(len(chain)/3.)
    keys = list(chain.keys())
    nwalkers = chain[keys[0]].shape[1]
    if (labels is None) or (len(labels) != len(keys)):
        labels = keys
    for i,key in enumerate(chain.keys()):
        plt.subplot(rows,3,1+i)
        for index in range(nwalkers):
            plt.plot(chain[key][:,index],alpha=.1)
        mp = np.median(chain[key],axis=1)
        plt.plot(mp,lw=2,color='r')
        plt.ylabel(labels[i])
        plt.xlabel('step')


def acceptance_fraction(chain):
    '''Return acceptance fraction for the motion of nwalkers for one parameters. Array should have dimensions nwalkers*nsteps.
    Adjusted to work with the new format of the chain array (read in by read_chain_steps)
    To call this function
    >> np.mean(acceptance_fraction(chain['q'][n:,:]))
    n is the starting step number. If you want to examine the acceptance fraction after step 400, then set n=400. Otherwise n can be left out. This will calculate the average acceptance fraction over all walkers.
    '''

    nsteps = chain.shape[0]
    nwalkers = chain.shape[1]

    naccepted=np.zeros(nwalkers)
    for i in range(nwalkers):
        dchain = chain[:,i]-np.roll(chain[:,i],-1)
        naccepted[i] = (dchain != 0.).sum() -1
        #subtract 1 because rolling the array will loop around, always leading the last element of dchain to be non-zero
    return naccepted/nsteps


def acor(chain):
    '''Calculate the average integrated autocorrelation time as a function of step number. Assumes the chain format as read in when the results are written out in steps

    The result is a figure where the colored lines are the integrated auto-correlation time as a function of step number for each walker. The dashed line is the average integrated auto-correlation time as a function of step number. The dotted line is the average integrated auto-correlation time at the very last step. The chains should be much longer than the auto-correlation time. If not then the chains have not properly converged.

    To call this function:
    >> acor(chain)
    where chain is the variable returned by read_chain_steps.
    '''
    from emcee.autocorr import integrated_time

    names = list(chain.keys())
    nwalkers = chain[names[0]].shape[1]
    nsteps = chain[names[0]].shape[0]
    tau = np.empty((nsteps-10,nwalkers))
    for j in range(nwalkers):
        for i in range(10,nsteps):
            tau[i-10,j]=integrated_time(chain[names[0]][:i,j],quiet=True,tol=1)
        plt.plot(tau[:,j],alpha=.5)
    plt.plot(tau.mean(axis=1),color='k',ls='--',lw=2)
    plt.axhline(np.mean(tau[-1,:]),color='k',ls=':',lw=2)

def write_diff_vis(file,modfile,diff_file='alma.diff.vis.fits'):
    '''Difference the visibilities using Galario and save the difference (for generating imaged residuals). The code reads in the data visibility file and the model image, creates model visibilities at the baselines specified in the data file, and then differences the two (data - model) and puts the differenced visibilities in a new file.

    file (string): Data visibility fits file, ex. 'imlup_co21.vis.fits'

    modfile (string): Model image file, ex. 'alma.fits'

    diff_file (string): Name of the file containing the differenced visibilities. Default = 'alma.diff.vis.fits'

    '''
    gdouble.threads(1)

    im = fits.open(file)
    hdr = im[0].header
    data = im[0].data
    u,v = data['UU'].astype(np.float64),data['VV'].astype(np.float64)
    freq0 = hdr['crval4']
    vis = (data.data).squeeze()

    #Assume that we are dealing with spectral lines.
    real_obj = (vis[:,:,0,0]+vis[:,:,1,0])/2.
    imag_obj = (vis[:,:,0,1]+vis[:,:,1,1])/2.

    u*=freq0
    v*=freq0

    #Generate model visibilities
    model_fits = fits.open(modfile)
    model = model_fits[0].data.squeeze()
    nxy,dxy = model_fits[0].header['naxis1'],np.radians(np.abs(model_fits[0].header['cdelt1']))
    model_fits.close()
    real_model = np.zeros(real_obj.shape)
    imag_model = np.zeros(imag_obj.shape)
    for i in range(real_obj.shape[1]):
        vis = gdouble.sampleImage(np.flipud(model[i,:,:]).byteswap().newbyteorder(),dxy,u,v)
        real_model[:,i] = vis.real
        imag_model[:,i] = vis.imag

    real_diff = real_obj - real_model
    imag_diff = imag_obj - imag_model

    data_diff = data
    data_diff['data'][:,0,0,0,:,0,0] = real_diff
    data_diff['data'][:,0,0,0,:,1,0] = real_diff
    data_diff['data'][:,0,0,0,:,0,1] = imag_diff
    data_diff['data'][:,0,0,0,:,1,1] = imag_diff
    im[0].data = data_diff
    im.writeto(diff_file,overwrite=True)
    im.close()


def write_model_vis(file,modfile,model_vis_file = 'alma.model.vis.fits'):
    '''Use galario and the model image to generate model visibilities, which are then written to a file. The code reads in the data visibility file and the model image, creates model visibilities at the baselines specified in the data file, and puts the model visibilities in a new file.

    file (string): Data visibility fits file, ex. 'imlup_co21.vis.fits'

    modfile (string): Model image file, ex. 'alma.fits'

    model_vis_file (string): Name of the file containing the model visibilities. Default = 'alma.model.vis.fits''''

    gdouble.threads(1)

    im=fits.open(file)
    hdr = im[0].header
    data = im[0].data
    u,v = data['UU'].astype(np.float64),data['VV'].astype(np.float64)
    freq0 = hdr['crval4']
    vis = (data.data).squeeze()
    u*=freq0
    v*=freq0

    real_obj = (vis[:,:,0,0]+vis[:,:,1,0])/2.
    imag_obj = (vis[:,:,0,1]+vis[:,:,1,0])/2.

    #Generate model visibilities
    model_fits = fits.open(modfile)
    model = model_fits[0].data.squeeze()
    hdr_model = model_fits[0].header
    nxy,dxy = model_fits[0].header['naxis1'],np.radians(np.abs(model_fits[0].header['cdelt1']))
    model_fits.close()
    real_model = np.zeros(real_obj.shape)
    imag_model = np.zeros(imag_obj.shape)
    for i in range(real_obj.shape[1]):
        vis = gdouble.sampleImage(np.flipud(model[i,:,:]).byteswap().newbyteorder(),dxy,u,v)
        real_model[:,i] = vis.real
        imag_model[:,i] = vis.imag

    model_vis = data
    model_vis['data'][:,0,0,0,:,0,0] = real_model
    model_vis['data'][:,0,0,0,:,1,0] = real_model
    model_vis['data'][:,0,0,0,:,0,1] = imag_model
    model_vis['data'][:,0,0,0,:,1,1] = imag_model
    im[0].data=model_vis
    im.writeto(model_vis_file,overwrite=True)
    im.close()


def calc_noise(image,imx=10):
    '''Calculate the noise within an image. The noise is used in multiple functions (im_plot_spec, mk_chmaps, imdiff) and the calculation is moved here to make sure it is consistent between all of them'''
    #assuming we are dealing with full image and not cropped images
    #imx is width of box in pixels. This box is used to define noise.

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
            noise1[i] = np.std(image[i,low:npix//2-imx//2,low:npix//2-imx//2])
            noise2[i] = np.std(image[i,low:npix//2-imx//2,npix//2+imx//2:high])
            noise3[i] = np.std(image[i,npix//2+imx//2:high,low:npix//2-imx//2])
            noise4[i] = np.std(image[i,npix//2+imx//2:high,npix//2+imx//2:high])
            noise5[i] = np.std(image[i,low:high,low:high])
            noise[i] = np.mean([noise1[i],noise2[i],noise3[i],noise4[i]])
        #flux = np.array([image[i,low:npix/2-imx/2,low:npix/2-imx/2],image[i,low:npix/2-imx/2,npix/2+imx/2:high],image[i,npix/2+imx/2:high,low:npix/2-imx/2],image[i,npix/2+imx/2:high,npix/2+imx/2:high]])
        #print('N>3sigma:',float((np.abs(flux.flatten())>3*noise[i]).sum())/flux.flatten().shape[0])
        #print('N>1sigma:',float((np.abs(flux.flatten())>noise[i]).sum())/flux.flatten().shape[0]) #The number of 3sigma and 1sigma peaks in all the lines is consistent with what we would expect from gaussian statistics

    #print(noise1,noise2,noise3,noise4)
    #print(noise5.mean(),noise5)
    #noise from boxes around the disk has similar noise to boxes centered in image, but using line free channels (noise=6.84e-5,6.53e-5 Jy/pixel for two scenarios). No strong channel dependence when looking at the line-free channels

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

        noise1 = np.std(image[low:npix//2-imx//2,low:npix//2-imx//2])
        noise2 = np.std(image[low:npix//2-imx//2,npix/2+imx//2:high])
        noise3 = np.std(image[npix//2+imx//2:high,low:npix//2-imx//2])
        noise4 = np.std(image[npix//2+imx//2:high,npix//2+imx//2:high])
        noise5 = np.std(image[low:high,low:high])
        #print(noise1,noise2,noise3,noise4,noise5)
        return np.mean([noise1,noise2,noise3,noise4])
        #return noise5
