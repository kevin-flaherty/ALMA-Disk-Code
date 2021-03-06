def var_vis(file,uvwidth=30,collapse=False,realimag=False):
    ''' Calculate the weight based on the variance in a visibility map at each u,v point and each channel. The codes estimate the variance among the 50 closest uv-points in a limited range in uv-space. 

    :param file:
    Name of the visibility fits file for which the weights will be calculated. The file is assumed to contain a spectral line (ie. it contains a spectral dimension). Use var_vis_cont if you have continuum data that has been averaged along the spectral dimension.

    :param uvwidth:
    Distance, in klam, over which to look for the closest 50 visibility points. If 50 visibility points are not found within a distance of uvwidth, then a weight of 0 is recorded. Increasing uvwidth ensures that enough visibility points are found, although at the expense of longer computational time. A note will be printed every time the code cannot find enough nearby points, and if this happens often enough then uvwidth should be increased, and the code re-run. This is especially important since points at large uv-distances are most susceptable to a lack of nearby points for the dispersion calculation, and choosing too small a value for uvwidth could introduce an artifical bias against these points by applying a weight of zero to valid data. 

    :param collapse:
    Calculates the average across the spectral windows, rather than calculating an average in each spectral window separatly. This is necessary if you are using line-free channels to calculate the dispersion.

    :param realimag:
    Estimates one weight for both the real and imaginary part, based exclusively on the real part of the visibilities. Otherwise a separate weight is calculated for the real and imaginary part of the visibilities. 
''' 
    from astropy.io import fits
    import numpy as np

    im = fits.open(file)
    u,v = im[0].data['UU'],im[0].data['VV']
    freq0 = im[0].header['crval4']
    klam = freq0/1e3
    # for now assume it is ALMA data
    vis = (im[0].data['data']).squeeze()
    if vis.shape[2] == 2:
        real = (vis[:,:,0,0]+vis[:,:,1,0])/2.
        imag = (vis[:,:,0,1]+vis[:,:,1,1])/2.
    else:
        real = vis[:,:,0]
        imag = vis[:,:,1]
    im.close()

    nuv = u.size
    nfreq = (real.shape)[1]
    uv = np.sqrt(u**2+v**2)
    nclose = 40 #number of nearby visibility points to use when measuring the dispersion
    #uvwidth = 30 #area around a particular uv point to consider when searching for the nearest nclose neighbors (smaller numbers help make the good run faster, but could result in many points for which the weight cannot be calculated and is left at 0)
    max_dist = np.zeros(nuv)
    
    import time
    start=time.time()
    if collapse:
        if realimag:
            weight = np.zeros(nuv)
        else:
            weight = np.zeros((nuv,2))
    else:
        if realimag:
            weight = np.zeros((nuv,nfreq))
        else:
            weight = np.zeros((nuv,nfreq,2))
    nclose_arr = np.zeros(len(u))
    for iuv in range(nuv):
        w = (np.abs(u-u[iuv])*klam < uvwidth) & (np.abs(v-v[iuv])*klam < uvwidth)
        s = np.argsort(np.sqrt((v[w]-v[iuv])**2+(u[w]-u[iuv])**2))
        wf = (real[w,0][s] !=0)
        nclose_arr[iuv] = wf.sum()
        if wf.sum()>nclose:
            if collapse:
                if realimag:
                    weight[iuv] = 1/np.std(real[w,:][s][wf][:nclose])**2 #the :nclose strides over baselines (the first dimension of real) instead of over frequency (which is what I want...)
                else:
                    weight[iuv,0] = 1/np.std(real[w,:][s][wf][:nclose])**2
                    weight[iuv,1] = 1/np.std(imag[w,:][s][wf][:nclose])**2
            else:
                if realimag:
                    for ifreq in range(nfreq):
                        weight[iuv,ifreq]=1/np.std(real[w,ifreq][s][wf][:nclose])**2
                else:
                    for ifreq in range(nfreq):
                        weight[iuv,ifreq,0]=1/np.std(real[w,ifreq][s][wf][:nclose])**2
                        weight[iuv,ifreq,1]=1/np.std(imag[w,ifreq][s][wf][:nclose])**2
        else:
            #print iuv,wf.sum(),np.sqrt(u[iuv]**2+v[iuv]**2)*klam
            print 'Not enough vis points near uv={:0.2f} klam. Only found {:0.0f} nearby points when {:0.0f} are needed'.format(np.sqrt(u[iuv]**2+v[iuv]**2)*klam,wf.sum(),nclose+1)

    print 'Elapsed time (hrs): ',(time.time()-start)/3600.


    return weight

#hdu=fits.PrimaryHDU(weight)
#hdu.writeto('mydata_weights.fits')


def var_vis_cont(file,uvwidth=30,collapse=False,realimag=False):
    ''' Calculate the weight based on the variance in a visibility map at each u,v point and each channel. The codes estimate the variance among the 50 closest uv-points in a limited range in uv-space. 

    :param file:
    Name of the visibility fits file for which the weights will be calculated. The file is assumed to not have a spectral dimension (ie. it contains averaged continuum data). Use var_vis if you have line data.

    :param uvwidth:
    Distance, in klam, over which to look for the closest 50 visibility points. If 50 visibility points are not found within a distance of uvwidth, then a weight of 0 is recorded. Increasing uvwidth ensures that enough visibility points are found, although at the expense of longer computational time. A note will be printed every time the code cannot find enough nearby points, and if this happens often enough then uvwidth should be increased, and the code re-run. This is especially important since points at large uv-distances are most susceptable to a lack of nearby points for the dispersion calculation, and choosing too small a value for uvwidth could introduce an artifical bias against these points by applying a weight of zero to valid data. 

    :param realimag:
    Estimates one weight for both the real and imaginary part, based exclusively on the real part of the visibilities. Otherwise a separate weight is calculated for the real and imaginary part of the visibilities.

''' 
    from astropy.io import fits
    import numpy as np

    im = fits.open(file)
    u,v = im[0].data['UU'],im[0].data['VV']
    freq0 = im[0].header['crval4']
    klam = freq0/1e3
    # for now assume it is ALMA data
    vis = (im[0].data['data']).squeeze()
    if vis.shape[1] == 2:
        real = (vis[:,0,0]+vis[:,1,0])/2.
        imag = (vis[:,0,1]+vis[:,1,1])/2.
    else:
        real = vis[:,0]
        imag = vis[:,1]
    im.close()

    nuv = u.size
    uv = np.sqrt(u**2+v**2)
    nclose = 50 #number of nearby visibility points to use when measuring the dispersion
    #uvwidth = 30 #area around a particular uv point to consider when searching for the nearest nclose neighbors (smaller numbers help make the good run faster, but could result in many points for which the weight cannot be calculated and is left at 0)
    max_dist = np.zeros(nuv)
    
    import time
    start=time.time()
    if realimag:
        weight = np.zeros(nuv)
    else:
        weight = np.zeros((nuv,2))
    nclose_arr = np.zeros(len(u))
    for iuv in range(nuv):
        w = (np.abs(u-u[iuv])*klam < uvwidth) & (np.abs(v-v[iuv])*klam < uvwidth)
        s = np.argsort(np.sqrt((v[w]-v[iuv])**2+(u[w]-u[iuv])**2))
        wf = (real[w][s] !=0)
        nclose_arr[iuv] = wf.sum()
        if wf.sum()>nclose:
            if realimag:
                weight[iuv] = 1/np.std(real[w][s][wf][:nclose])**2 
            else:
                weight[iuv,0] = 1/np.std(real[w][s][wf][:nclose])**2 
                weight[iuv,1] = 1/np.std(imag[w][s][wf][:nclose])**2 
        else:
            print 'Not enough vis points at uv={:0.2f}klam. Only found {:0.0f} nearby points when {:0.0f} are needed'.format(np.sqrt(u[iuv]**2+v[iuv]**2)*klam,wf.sum(),nclose+1)
    print 'Elapsed time (hrs): ',(time.time()-start)/3600.

    return weight

#To save the variable weight as a fits file, use the following commands:
#hdu=fits.PrimaryHDU(weight)
#hdu.writeto('mydata_weights.fits')


def add_weight_to_vis(file,weight,binned=False):
    ''' Take an array containing weights generated by var_vis or var_vis_cont and add it to a visibility fits file in place of the previous weights.

    :param file:
    Name of the visibility fits file.

    :param weight:
    Array containing the weights.

    :param binned:
    Set to True if the frequency axis has been binned down so that there is only one channel (e.g. binned continuum data).

'''

    from astropy.io import fits
    import numpy as np

    vis = fits.open(file)
    
    if weight.shape[-1]==2:
        if binned:
            vis[0].data['data'][:,0,0,0,0,0,2] = weight[:,0]
            vis[0].data['data'][:,0,0,0,0,1,2] = weight[:,1]
        else:
            #vis[0].data['data'][:,0,0,0,:,0,2] = weight[:,0,np.newaxis]*np.ones(vis[0].data['data'].shape[4])
            #vis[0].data['data'][:,0,0,0,:,1,2] = weight[:,1,np.newaxis]*np.ones(vis[0].data['data'].shape[4])
            vis[0].data['data'][:,0,0,0,:,0,2] = weight[:,:,0]
            vis[0].data['data'][:,0,0,0,:,1,2] = weight[:,:,1]
    else:
        if binned:
            vis[0].data['data'][:,0,0,0,0,2] = weight
            vis[0].data['data'][:,0,0,0,1,2] = weight
        else:
            vis[0].data['data'][:,0,0,0,:,0,2] = weight[:,np.newaxis]*np.ones(vis[0].data['data'].shape[4])
            vis[0].data['data'][:,0,0,0,:,1,2] = weight[:,np.newaxis]*np.ones(vis[0].data['data'].shape[4])


    vis.writeto(file,clobber=True)
