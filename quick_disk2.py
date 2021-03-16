#Try to improve on quick_disk
#Use a series of gaussian rings. Each ring has an r0, I0, z, vphi, vr, vz, dz
#Start by fitting to the image plane. Eventually want to fit to the visibility domain.

import numpy as np
import matplotlib.pyplot as plt

def quick_disk2(pixel_size = 0.05, npixels = 512, nchans = 15, chanwidth = 0.3, beamsize=0.5):
    r0 = np.arange(0.,npixels/2*pixel_size,5*beamsize)
    sigma = beamsize/4.
    nrings = len(r0)
    ring_intensity = np.ones(nrings)
    zdisk = np.zeros(nrings)#-2
    inc = np.zeros(nrings)+np.radians(50)
    PA = np.zeros(nrings)+90

    vphi = np.ones(nrings)
    vr = np.zeros(nrings)
    vz = np.zeros(nrings)
    dv = np.zeros(nrings)+.5

    velocities = -nchans/2*chanwidth+np.arange(nchans)*chanwidth

    rings = {'r0':r0,'ring_intensity':ring_intensity,'zdisk':zdisk,'inc':inc,'PA':PA,'sigma':sigma,'nrings':nrings}
    velos = {'vphi':vphi,'vr':vr,'vz':vz,'dv':dv}
    image_props = {'pixel_size':pixel_size,'npixels':npixels,'velocities':velocities,'beamsize':beamsize,'nchans':nchans}

    image = make_model(rings,velos,image_props)


#Need a function that compare the model image and the data and returns a log-likelihood
#Need a function that runs some sort of MCMC model with the inputs

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
        plt.contour(xm,ym,final_image[:,:,ichan],100,cmap=plt.cm.afmhot)
        plt.gca().set_aspect(1)

    return final_image
