## Code to create an image mask that highlights regions where we expect Keplerian emission from a rotating, inclined disk. This is useful for creating masks for cleaning, and for searching for faint emission.

### To use as a mask in MIRIAD, execute the following commands:
# fits in=mask.fits out=mask.cm op=xyin
# maths exp="<mask.cm>" mask="<mask.cm>.gt.0.5" out=mask.mask
##
## the 'mask.mask' file can then be used in cleaning with the region keyword ( region='mask(mask.mask)')

### To use as a mask in CASA, first set imcen to RA and Dec, in decimal degrees, of the phase center of the image and then execute the following commands within CASA
# importfits(fitsimage='mask.fits',imagename='mask.im',defaultaxes=True,defaultaxesvalues=['','','','I'])
# makemask(mode='copy',inpimage='mask.im',inpmask='mask.im',output='mask.im:mask0')
##
## This should then be used in clean with the mask keyword (mask='mask.im') in theory, but in practice it doesn't always seem to work...

def butterfly(mstar=2.,Rin=0.1,Rout=200.,incl=45.,PA=0.,dist=140.,dv=0.1,npix=512,imres=0.01,offs=[0.,0.],beam=.5,nchans=15,chanstep=.1,voff=0.,outfile='mask.fits',freq0=230.538,imcen=[0.,0.]):
    ''' Calculate the location of emission associated with a Keplerian disk. Result is a fits image with 1's in the expected location of emission, and 0's everywhere else. Parameters allow for control of emission profile (mstar, Rin, Rout, dv, dist), its orientation (incl, PA, npix, imres, offs, beam) and the velocity sampling (nchans, chanstep, voff). [Edge of disk is cutoff if major axis is larger than imres*dist*npix]

    :param mstar (default=2):
    Stellar mass, in units of solar masses

    :param Rin (default=0.1):
    Inner radius of the disk in units of AU

    :param Rout (default=200.):
    Outer radius of the disk in units of AU

    :param incl (default=45.):
    Incliation (0=face-on, 90=edge-on)

    :param PA (default=0.):
    Position angle of the disk (measured east of north) in degrees

    :param dist (default=140.):
    Distance to the disk, in parsecs

    :param dv (default=0.1):
    Velocity broadening of emission. This is intrinsic to gas (ie turbulent or thermal broadening) rather than due to the finite channel width (which is handled by the chanstep keyword). Units of km/s.

    :param npix (default=512):
    Number of pixels across one dimension of the final image

    :param imres (default=0.01):
    Resolution of image in arcseconds

    :param offs (default=[0.,0.]):
    Positional offset from the center of the image, in units of arcseconds

    :peam beam (default=0.5):
    Beam FWHM in arcseconds. The image is convolved with this beam to approximate finite resolution of telescope. Assumes a circular beam. 

    :param nchans (default=15):
    Number of channels in the resulting spectrum

    :param chanstep (default=0.1):
    Width of a single channel (dv is the intrinsic broadening of the emission , e.g. turbulence or thermal broadening). Units of km/s

    :params voff (default=0.):
    Offset in velocity from the central channel (or if an even number of channels, offset from midpoint between central channels). NOT the systemic velocity of the star, but rather the difference between the systemic velocity and the velocity of the central channel. Units of km/s.

    :param outfile (default='mask.fits'):
    Name of the output fits file.

    :param freq0 (default=230.538):
    Rest frequency, in GHz. Needed to read the file into CASA or MIRIAD.

    :param imcen (default=[0.,0.]):
    RA and Dec, in decimal degrees, of the center of the image. If using the mask in CASA, this needs to be set to the phase center of the image.
    '''

    import numpy as np
    from astropy import constants as const
    from astropy.io import fits
    from scipy import ndimage
    from astropy.convolution import Gaussian2DKernel,convolve

    AU = const.au.cgs.value
    pc = const.pc.cgs.value
    Msun = const.M_sun.cgs.value
    G = const.G.cgs.value
    incl = np.radians(incl) #convert inclination from degrees to radians
    dist *= pc
    Rin *= AU
    Rout *= AU
    mstar *= Msun

    ## - Create image
    ra_image = np.linspace(-npix/2*imres,npix/2*imres,npix)
    dec_image = np.linspace(-npix/2*imres,npix/2*imres,npix)
    ra_image,dec_image = np.meshgrid(ra_image,dec_image)

    #define radius and phi for each pixel
    radius = np.sqrt((ra_image)**2+((dec_image)/np.cos(incl))**2)
    radius = np.radians(radius/3600.)*dist
    phi = np.arctan2((dec_image),(ra_image))
    
    
    vkep = np.sqrt(G*mstar/radius)/1e5
    vlos = vkep*np.cos(phi)*np.sin(incl)
    dv = dv*(radius/(100*AU))**(-.5)

    # Remove points outside of disk
    wremove = (radius<Rin) | (radius>Rout)
    vlos[wremove] = -100
    vmax = np.max(vlos[vlos>-90])
    vmin = np.min(vlos[vlos>-90])

    chanmin = (-nchans/2.+.5)*chanstep
    velo_steps = np.arange(nchans)*chanstep+chanmin+voff
    image = np.zeros((npix,npix,nchans))
        
    for i in range(nchans):
        w = (vlos+dv/2.>velo_steps[i]-chanstep/2.) & (vlos-dv/2.<(velo_steps[i]+chanstep/2.))
        image[:,:,i][w] = 1

    # Convolve with finite spatial resolution
        gauss = Gaussian2DKernel(beam/(imres*2*np.sqrt(2*np.log(2))))
    for i in range(nchans):
        image[:,:,i] = convolve(image[:,:,i],gauss,boundary=None)
    
    ## - Shift and rotate image to correct orientation
    image = ndimage.rotate(image,PA+180,reshape=False)
    pixshift = np.array([-1.,1.])*offs/(np.array([imres,imres]))
    image = ndimage.shift(image,(pixshift[0],pixshift[1],0),mode='nearest')
    image[image<.01]=0. #clean up artifacts from interpolation 
    image[image>.01]=1.

    ## - Make fits file
    hdr = fits.Header()
    hdr['SIMPLE'] = 'T'
    hdr['BITPIX'] = 32
    hdr['NAXIS'] = 3
    hdr['NAXIS1'] = npix
    hdr['NAXIS2'] = npix
    hdr['NAXIS3'] = nchans
    hdr['CDELT1'] = -1*imres/3600.
    hdr['CRPIX1'] = npix/2.+.5
    hdr['CRVAL1'] = imcen[0]
    hdr['CTYPE1'] = 'RA---SIN'
    hdr['CUNIT1'] = 'deg'
    hdr['CDELT2'] = imres/3600.
    hdr['CRPIX2'] = npix/2.+.5
    hdr['CRVAL2'] = imcen[1]
    hdr['CTYPE2'] = 'DEC--SIN'
    hdr['CUNIT2'] = 'deg'
    hdr['CTYPE3'] = 'VELO-LSR'
    hdr['CDELT3'] = chanstep*1e3
    hdr['CRPIX3'] = 1.
    hdr['CRVAL3'] = velo_steps[0]*1e3
    hdr['EPOCH'] = 2000.
    hdr['RESTFRQ'] = freq0
    hdu = fits.PrimaryHDU(image.T,hdr)
    hdu.writeto(outfile,clobber=True,output_verify='fix')
    


def contmask(Rin=0.1,Rout=200.,incl=45,PA=0.,dist=140.,npix=512,imres=0.01,offs=[0.,0.],beam=.5,outfile='mask.fits',freq0=230.538,imcen=[0.,0.]):
    '''Similar to butterfly, but for continuum instead of line images (ie only one spectral dimension) [Edge of disk is cutoff if major axis is larger than imres*dist*npix]

    :param Rin (default=0.1):
    Inner radius of the disk in units of AU

    :param Rout (default=200.):
    Outer radius of the disk in units of AU

    :param incl (default=45.):
    Incliation (0=face-on, 90=edge-on)

    :param PA (default=0.):
    Position angle of the disk (measured east of north) in degrees

    :param dist (default=140.):
    Distance to the disk, in parsecs

    :param npix (default=512):
    Number of pixels across one dimension of the final image

    :param imres (default=0.01):
    Resolution of image in arcseconds

    :param offs (default=[0.,0.]):
    Positional offset from the center of the image, in units of arcseconds

    :peam beam (default=0.5):
    Beam FWHM in arcseconds. The image is convolved with this beam to approximate finite resolution of telescope. Assumes a circular beam. 

    :param outfile (default='mask.fits'):
    Name of the output fits file.

    :param freq0 (default=230.538):
    Rest frequency, in GHz. Needed to read the file into CASA or MIRIAD.

    :param imcen (default=[0.,0.]):
    RA and Dec, in decimal degrees, of the center of the image. If using the mask in CASA, this needs to be set to the phase center of the image.

'''

    import numpy as np
    from astropy import constants as const
    from astropy.io import fits
    from scipy import ndimage
    from astropy.convolution import Gaussian2DKernel,convolve

    AU = const.au.cgs.value
    pc = const.pc.cgs.value
    incl = np.radians(incl) #convert inclination from degrees to radians
    dist *= pc
    Rin *= AU
    Rout *= AU

    ## - Create image
    ra_image = np.linspace(-npix/2*imres,npix/2*imres,npix)
    dec_image = np.linspace(-npix/2*imres,npix/2*imres,npix)
    ra_image,dec_image = np.meshgrid(ra_image,dec_image)
    
    # define radius and phi for each pixel
    radius = np.sqrt((ra_image)**2+((dec_image)/np.cos(incl))**2)
    radius = np.radians(radius/3600.)*dist
    phi = np.arctan2(dec_image,ra_image)



    image = np.zeros((npix,npix))
    w = (radius>Rin) & (radius<Rout)
    image[w] = 1

    # Convolve with finite spatial resolution
    gauss = Gaussian2DKernel(beam/(imres*2*np.sqrt(2*np.log(2))))
    image = convolve(image,gauss,boundary=None)

    # Shift and rotate image to correct orientation
    image = ndimage.rotate(image,PA+180,reshape=False)
    pixshift = np.array([-1,1])*offs/(np.array([imres,imres]))
    image = ndimage.shift(image,(pixshift[0],pixshift[1]),mode='nearest')
    image[image<0.01] = 0 #clean up artifacts from interpolation
    image[image>0.01] = 1.

    ## - Make fits file
    hdr = fits.Header()
    hdr['SIMPLE'] = 'T'
    hdr['BITPIX'] = 32
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = npix
    hdr['NAXIS2'] = npix
    hdr['CDELT1'] = -1*imres/3600.
    hdr['CRPIX1'] = npix/2.+.5
    hdr['CRVAL1'] = imcen[0]
    hdr['CTYPE1'] = 'RA---SIN'
    hdr['CUNIT1'] = 'deg'
    hdr['CDELT2'] = imres/3600.
    hdr['CRPIX2'] = npix/2.+.5
    hdr['CRVAL2'] = imcen[1]
    hdr['CTYPE2'] = 'DEC--SIN'
    hdr['CUNIT2'] = 'deg'
    hdr['EPOCH'] = 2000.
    hdr['RESTFRQ'] = freq0
    hdu = fits.PrimaryHDU(image.T,hdr)
    hdu.writeto(outfile,clobber=True,output_verify='fix')
    

