#Try to improve on quick_disk
#Use a series of gaussian rings. Each ring has an r0, I0, z, vphi, vr, vz, dz
#Start by fitting to the image plane. Eventually want to fit to the visibility domain.

def quick_disk2(pixel_size = 0.05, npixels = 512, nchans = 15, chanwidth = 0.3, beamsize=0.5):
    '''pixel_size: arc-seconds
    chanwitdh: km/s
    beamsize: arc-seconds'''

    x = (np.arange(npixels)-npixels/2)*pixel_size
    y = (np.arange(npixels)-npixels/2)*pixel_size
    xm, ym = np.meshgrid(x,y)

    #Set up Gaussian rings
    r0 = np.linspace(0,npixels*pixel_size,beamsize/2.)
    sigma = beamsize/4.

    
