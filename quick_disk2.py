#Try to improve on quick_disk
#Use a series of gaussian rings. Each ring has an r0, I0, z, vphi, vr, vz, dz
#Start by fitting to the image plane. Eventually want to fit to the visibility domain.

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
#from galario import double as gdouble
from astropy.constants import c
#from vis_sample import vis_sample

def quick_disk2(pixel_size = 0.05, npixels = 512, nchans = 15, chanwidth = 0.3, beamsize=0.5):
    #file = '/Users/kevinflahertyastro/sample_data/dmtaush_co21sb.cm.fits'
    file = '/Users/kevinflahertyastro/sample_data/HD163296.CO32.regridded.cen15.cm.fits'
    hdr = fits.getheader(file)
    #nchans = hdr['naxis4']
    #chanwidth = hdr['cdelt4']/hdr['crval4']*c.cgs.value/1e5
    npixels = hdr['naxis1']
    pixel_size = np.abs(hdr['cdelt1'])*3600
    nchans = hdr['naxis3']
    chanwidth = np.abs(hdr['cdelt3'])/1e3
    beamsize = 3600*(hdr['bmaj']+hdr['bmin'])/2

    r0 = np.arange(0.,npixels/2*pixel_size,5*beamsize)
    sigma = beamsize/4.
    nrings = len(r0)
    ring_intensity = np.ones(nrings)
    zdisk = np.zeros(nrings)#-2
    inc = np.zeros(nrings)+np.radians(45)
    PA = np.zeros(nrings)+40

    vphi = np.ones(nrings)
    vr = np.zeros(nrings)
    vz = np.zeros(nrings)
    dv = np.zeros(nrings)+.5

    velocities = -nchans/2*chanwidth+np.arange(nchans)*chanwidth

    rings = {'r0':r0,'ring_intensity':ring_intensity,'zdisk':zdisk,'inc':inc,'PA':PA,'sigma':sigma,'nrings':nrings}
    velos = {'vphi':vphi,'vr':vr,'vz':vz,'dv':dv}
    image_props = {'pixel_size':pixel_size,'npixels':npixels,'velocities':velocities,'beamsize':beamsize,'nchans':nchans}

    #image = make_model(rings,velos,image_props)

    params = np.concatenate((ring_intensity,zdisk,inc,PA,vphi,vr,vz,dv))

    print(image_log_like(params,file,r0,sigma,nrings,image_props))

#Need a function that compare the model image and the data and returns a log-likelihood
#Need a function that runs some sort of MCMC model with the inputs

def image_log_like(params,data_filename,r0,sigma,nrings,image_props):
    '''Calculate the log-likelihood between and model image and the cleaned image. Not as good as fitting the visibilities, but still might be useful.
    params is the (long) list of parameters.
    '''
    ring_intensity = np.array(params[:nrings])
    zdisk = np.array(params[nrings:2*nrings])
    inc = np.array(params[2*nrings:3*nrings])
    PA = np.array(params[3*nrings:4*nrings])

    vphi = np.array(params[4*nrings:5*nrings])
    vr = np.array(params[5*nrings:6*nrings])
    vz = np.array(params[6*nrings:7*nrings])
    dv = np.array(params[7*nrings:])

    rings = {'r0':r0,'ring_intensity':ring_intensity,'zdisk':zdisk,'inc':inc,'PA':PA,'sigma':sigma,'nrings':nrings}
    velos = {'vphi':vphi,'vr':vr,'vz':vz,'dv':dv}

    image = make_model(rings,velos,image_props).T

    data = fits.open(data_filename)
    data_image = data[0].data.squeeze()
    noise = calc_noise(data_image)

    #The model images look empty when plotting HD 163296, but not for DM Tau...
    #That is weird because nothing changes in the model images between those two...
    plt.subplot(2,2,1)
    plt.contourf(image[3,:,:],100,cmap=plt.cm.afmhot)
    plt.colorbar()
    plt.gca().set_aspect(1)
    plt.subplot(2,2,2)
    plt.contourf(data_image[3,:,:],100,cmap=plt.cm.afmhot)
    plt.colorbar()
    plt.gca().set_aspect(1)
    plt.subplot(2,2,3)
    plt.contourf(image[10,:,:],100,cmap=plt.cm.afmhot)
    plt.colorbar()
    plt.gca().set_aspect(1)
    plt.subplot(2,2,4)
    plt.contourf(data_image[10,:,:],100,cmap=plt.cm.afmhot)
    plt.colorbar()
    plt.gca().set_aspect(1)

    #need to convolve model by the beam before calculating the chi-squared
    chi = (data_image-image)**2./noise**2.

    return -0.5*chi.sum()


def log_like2(params,data_filename):
    '''Calculate the log-likelihood between a model and the data. This uses vis_sample.
    params is the (long) list of parameters.
    '''
    ### I installed vis_sample via conda, but it doesn't recognize that the package exists...
    # Create the model image
    ring_intensity = np.array(params[:nrings])
    zdisk = np.array(params[nrings:2*nrings])
    inc = np.array(params[2*nrings:3*nrings])
    PA = np.array(params[3*nrings:4*nrings])

    vphi = np.array(params[4*nrings:5*nrings])
    vr = np.array(params[5*nrings:6*nrings])
    vz = np.array(params[6*nrings:7*nrings])
    dv = np.array(params[7*nrings:])

    rings = {'r0':r0,'ring_intensity':ring_intensity,'zdisk':zdisk,'inc':inc,'PA':PA,'sigma':sigma,'nrings':nrings}
    velos = {'vphi':vphi,'vr':vr,'vz':vz,'dv':dv}

    image = make_model(rings,velos,image_props)
    hdu = fits.PrimaryHDU(image)
    hdu.writeo('model.fits',overwrite=True,output_verify='fix')

    interp_vis = vis_sample(imagefile='model.fits',uvfile=data_filename)



def log_like(params,data_filename,r0,sigma,nrings,image_props):
    '''Calculate the log-likelihood between a model and data.
    params is the (long) list of parameters
    '''
    gdouble.threads(1)

    # - Read in object visibilities
    obj = fits.open(data_filename)
    freq0 = obj[0].header['crval4']
    u_obj,v_obj = (obj[0].data['UU']*freq0).astype(np.float64),(obj[0].data['VV']*freq0).astype(np.float64)
    vis_obj = (obj[0].data['data']).squeeze()
    if obj[0].header['telescop'] == 'ALMA':
        if obj[0].header['naxis3'] == 2:
            real_obj = (vis_obj[:,:,0,0]+vis_obj[:,:,1,0])/2.
            imag_obj = (vis_obj[:,:,0,1]+vis_obj[:,:,1,1])/2.
            weight_real = vis_obj[:,:,0,2]
            weight_imag = vis_obj[:,:,1,2]
        else:
            real_obj = vis_obj[::2,:,0]
            imag_obj = vis_obj[::2,:,1]
    obj.close()

    #Generate model visibilities
        #parse params list into the appropriate variables
        #need to know the number of rings to properly parse the params list...
    ring_intensity = np.array(params[:nrings])
    zdisk = np.array(params[nrings:2*nrings])
    inc = np.array(params[2*nrings:3*nrings])
    PA = np.array(params[3*nrings:4*nrings])

    vphi = np.array(params[4*nrings:5*nrings])
    vr = np.array(params[5*nrings:6*nrings])
    vz = np.array(params[6*nrings:7*nrings])
    dv = np.array(params[7*nrings:])

    rings = {'r0':r0,'ring_intensity':ring_intensity,'zdisk':zdisk,'inc':inc,'PA':PA,'sigma':sigma,'nrings':nrings}
    velos = {'vphi':vphi,'vr':vr,'vz':vz,'dv':dv}

    image = make_model(rings,velos,image_props)
    print(image.shape)
    #read/create the model image
    #Also need the number of pixels (nxy) and the size, in radians, of a pixel (dxy)
    real_model = np.zeros(real_obj.shape)
    imag_model = np.zeros(imag_obj.shape)
    dxy = np.radians(3600*image_props['pixel_size'])
    #image,u_obj,v_obj = np.require([image,u_obj,v_obj],requirements='C')
    for i in range(real_obj.shape[1]):
        vis = gdouble.sampleImage(np.flipud(image[:,:,i]).byteswap().newbyteorder(),dxy,u_obj,v_obj)
        #This breaks here...
        #I don't know why this works in single_model, but not here...
        #vis = gdouble.sampleImage(np.ascontiguousarray(image[:,:,i]),np.radians(3600*image_props['pixel_size']),u_obj,v_obj)
        real_model[:,i] = vis.real
        imag_model[:,i] = vis.imag

    weight_real[real_obj==0] = 0.
    weight_imag[imag_obj==0] = 0.

    #Include a prior on any of the variables? At the very least need to keep them physically realistic (e.g. r>0, intensity>0)

    chi = ((real_model-real_obj)**2.*weight_real)/sum() + ((imag_model-imag_obj)**2.*weight_imag).sum()
    return -0.5*chi


def make_model(rings,velos,image_props):
    '''Given a set of inputs, this code makes the model image

    pixel_size: arc-seconds
    chanwitdh: km/s
    beamsize: arc-seconds'''

    x = (np.arange(image_props['npixels'])-image_props['npixels']/2)*image_props['pixel_size']
    y = (np.arange(image_props['npixels'])-image_props['npixels']/2)*image_props['pixel_size']
    xm, ym = np.meshgrid(x,y)

    #Set up Gaussian rings
    #r0 = np.arange(0.,npixels/2*pixel_size,5*beamsize)
    #sigma = beamsize/4.
    #zdisk = np.zeros(len(r0))#-2
    #inc = np.zeros(len(r0))+np.radians(50)
    #ring_intensity = np.ones(len(r0))
    #vphi = np.ones(len(r0))
    #vr = np.zeros(len(r0))
    #z = np.zeros(len(r0))
    #v = np.zeros(len(r0))+.5
    #PA = np.zeros(len(r0))+90
    image = np.zeros((image_props['npixels'],image_props['npixels'],rings['nrings']))
    vlos = np.zeros((image_props['npixels'],image_props['npixels'],rings['nrings']))
    phi_disk = np.zeros((image_props['npixels'],image_props['npixels'],rings['nrings']))

    for i in range(rings['nrings']):
    #Rosenfeld et al. 2013, Teague et al. 2018 (although eq A4 is incorrect)
    #xdisk, ydisk = xm, ym/np.cos(inc[i])
        xdisk = xm*np.cos(np.radians(rings['PA'][i]-90))+ym*np.sin(np.radians(rings['PA'][i]-90))
    #ydisk = (-xm*np.sin(np.radians(PA[i]-90))+ym*np.cos(np.radians(PA[i]-90)))/np.cos(inc[i])
    #if inc[i]>0:
        ydisk = (-xm*np.sin(np.radians(rings['PA'][i]-90))+ym*np.cos(np.radians(rings['PA'][i]-90))+rings['zdisk'][i]*np.tan(rings['inc'][i]))/np.cos(rings['inc'][i])
    #else:
#        ydisk = (-xm*np.sin(np.radians(PA[i]-90))+ym*np.cos(np.radians(PA[i]-90)))/np.cos(inc[i])
        rdisk = np.sqrt(xdisk**2.+ydisk**2.)
        image[:,:,i] = rings['ring_intensity'][i]*np.exp(-(rdisk-rings['r0'][i])**2./(2*rings['sigma']**2.))
        phi_disk[:,:,i] = np.arctan2(ydisk,xdisk)


    #plt.contour(xm,ym,image,100,cmap=plt.cm.Greys)
    #plt.xlim(-npixels*pixel_size/2,npixels*pixel_size/2)
    #plt.ylim(-npixels*pixel_size/2,npixels*pixel_size/2)
    #plt.gca().set_aspect(1)

    #Teague et al. 2019
        vlos[:,:,i] = velos['vphi'][i]*np.sin(rings['inc'][i])*np.cos(phi_disk[:,:,i]) + velos['vr'][i]*np.sin(rings['inc'][i])*np.sin(phi_disk[:,:,i]) + velos['vz'][i]*np.cos(rings['inc'][i])
    #vlos = vr[i]*np.sin(inc[i])*np.sin(phi_disk)
    #vlos = vz[i]*np.cos(inc[i])
    #plt.contour(xm,ym,vlos,50,cmap=plt.cm.rainbow)
    #plt.colorbar()




    #Above gives the velocity and intensity at each position from each of the rings
    #For a given channel, look for the velocities from the different rings that fall within that channel. Then add the flux from the positions in all of those rings.

    #chanmin = -nchans/2*chanwidth
    final_image = np.zeros((image_props['npixels'],image_props['npixels'],image_props['nchans']))

    for ichan in range(image_props['nchans']):
        vchan = image_props['velocities'][ichan]


        for i in range(rings['nrings']):
            #w = (vlos[:,:,i]>(vchan-(dv[i]+chanwidth)/2)) & (vlos[:,:,i]<(vchan+(dv[i]+chanwidth)/2))
            #this is how disk_model handles things
            #It might work if dv is seen very genericaly. It could be (non-)thermal broadening or the channel width.
            final_image[:,:,ichan] += image[:,:,i]*np.exp(-((vchan-vlos[:,:,i])**2./velos['dv'][i])**2.)
        #plt.contour(xm,ym,image[:,:,i],100,cmap=plt.cm.Greys)
        #plt.contour(xm,ym,final_image[:,:,ichan],100,cmap=plt.cm.afmhot)
        #plt.gca().set_aspect(1)

    return final_image

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
            low = npix//2-3*imx//2
        if npix/2+3*imx/2>npix:
            high = -1
        else:
            high = npix//2+3*imx//2
        for i in range(nfreq):
            noise1[i] = np.std(image[i,low:npix//2-imx//2,low:npix//2-imx//2])
            noise2[i] = np.std(image[i,low:npix//2-imx//2,npix//2+imx//2:high])
            noise3[i] = np.std(image[i,npix//2+imx//2:high,low:npix//2-imx//2])
            noise4[i] = np.std(image[i,npix//2+imx//2:high,npix//2+imx//2:high])
            noise5[i] = np.std(image[i,low:high,low:high])
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

        noise1 = np.std(image[low:npix//2-imx//2,low:npix//2-imx//2])
        noise2 = np.std(image[low:npix//2-imx//2,npix/2+imx//2:high])
        noise3 = np.std(image[npix//2+imx//2:high,low:npix//2-imx//2])
        noise4 = np.std(image[npix//2+imx//2:high,npix//2+imx//2:high])
        noise5 = np.std(image[low:high,low:high])
        #print(noise1,noise2,noise3,noise4,noise5)
        return np.mean([noise1,noise2,noise3,noise4])
