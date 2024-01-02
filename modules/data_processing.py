import numpy as np
from pathlib import Path
import time
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.optimize import curve_fit
from scipy.ndimage import convolve1d
from scipy.stats import norm

totlim = 30
hot_pixels = np.array([51060,50791,57285,42572,52023,38582,45206,24953,52148,58034,58748,14758])
timeRecalibrationFactor = 2e-06

def smoothing(hist, gauss_std_ns = 4, bin_width_ns = 1.5625):
    '''
    Apply Gaussian smoothing to a histogram.

    This function convolves a histogram with a Gaussian filter to smooth the
    distribution. The Gaussian filter is created based on the specified standard
    deviation and bin width, and it's normalized so that its sum equals 1.

    Parameters
    ----------
    - hist (array_like): The input histogram to be smoothed. It should be a 3-dimensional array.
    - gauss_std_ns (float, optional): The standard deviation of the Gaussian filter in nanoseconds. 
                                      Default is 6 ns.
    - bin_width_ns (float, optional): The width of each bin in the histogram in nanoseconds. 
                                      Default is 1.5625 ns.

    Returns
    -------
    - array_like: The smoothed histogram, which has the same shape as the input histogram.
    
    The function uses the `convolve1d` method from `scipy.ndimage.filters` for convolution,
    with the 'nearest' mode to handle boundaries.
    '''
    gauss_filter = norm.pdf(np.arange(-10,10)*bin_width_ns, scale = gauss_std_ns)
    gauss_filter *= 1/gauss_filter.sum()
    return convolve1d(hist,weights=gauss_filter, mode = 'reflect')

def radius(x,y,center):
    '''
    Calculates the distance from a given center on the Chip.

    Parameters
    ----------
    - x (array_like): Array of the x coordinates
    - y (array_like): Array of the y coordinates
    - center (array_like): x- and y- coordinate of origin

    Returns
    -------
    - array_like: distances from the center
    '''
    return np.sqrt((x-center[0])**2 + (y-center[1])**2)

def plot_rad_dist(df, center, bins = np.arange(0,121,1), label = None, color = None):
    '''
    Creates a radial plot

    Parameters
    ----------
    - x (array_like): Array of the x coordinates
    - y (array_like): Array of the y coordinates
    - center (array_like): x- and y- coordinate of origin

    Returns
    -------
    - array_like: histogram of radial distribution
    - array_like: edges for the histogram
    '''

    bins2 = bins
    vals, edges =  np.histogram(radius((df/256).astype(int),df%256,center),#most_frequent_point(class_arr)),
            bins = bins2)
    return vals, edges

def Rep_Func_default(class_arr):
    '''
    Alex part
    '''

    info = []

    columns = ['x','y','toa','tot','label']
    df = pd.DataFrame(class_arr,columns = columns)
    suitable_events = df[df['label']!=-1]
    rougue_events = df[df['label']==-1]

    cond = (suitable_events.groupby('label').sum()['tot'] > totlim)

    sex = suitable_events.groupby('label').min()[cond]['x']
    sey = suitable_events.groupby('label').min()[cond]['y']
    setoa = suitable_events.groupby('label').min()[cond]['toa']
    setot = suitable_events.groupby('label').sum()[cond]['tot']

    se = pd.concat([sex,sey,setoa,setot],axis = 1)
    re = (rougue_events[rougue_events['tot'] > totlim]).drop('label',axis =1)
    rep_arr = pd.concat([se,re],axis = 0)
    return rep_arr, info

def clustering(data, 
            label_offset = 0, 
            data_handler = None,
            eps= 2, 
            split = 1,
            make_rep = True, 
            Rep_Func = Rep_Func_default, 
            timeRecalibrationFactor = 1 / 300000, 
            normalize_t = 1 , 
            e0_toa = 0):
    '''
    TODO Alex part
    '''

    min_samples = 2
    n_jobs = 1 # n_jobs determines cpu-cure utilization (-1 uses all cores)
    
    start = time.time()
    # Cluster
    chunks = np.array_split(data, split,axis = 1)
    l_off = label_offset
    class_arr = np.zeros(5)
    for i,chunk in enumerate(chunks):
        
        # Sort by ToA, then normalize ToAs
        index = chunk[1,:].argsort()
        matrixID = chunk[0,:][index]
        timesoA = (chunk[1,:][index]*normalize_t - e0_toa)
        timesoT = chunk[2,:][index]
        
        # prepare individual coordinate arrays
        # tCoordinates normlized for clustering, for all other purposes use timesoA
        xCoordinates = (matrixID / 256).astype(int)
        yCoordinates = (matrixID % 256).astype(int)
        tCoordinates = timesoA * timeRecalibrationFactor 
        events = np.array(list(zip(xCoordinates,yCoordinates,tCoordinates)))

        # Apply Clustering
        # n_jobs determines cpu-cure utilization (-1 uses all cores)
        labels = DBSCAN(eps=eps,min_samples=min_samples,n_jobs=n_jobs).fit_predict(events)
        labels += l_off + 1
        labels[labels == l_off] = -1
        l_off = labels.max()    

        # create list of events with their classification (full data, but classified into clusters)
        classifiedEvents = np.array(list(zip(xCoordinates,yCoordinates,timesoA,timesoT,labels)))
        if data_handler:
            data_handler.save_class_data(matrixID,timesoA,timesoT,labels)

        # GENERATE BASIC DATA ABOUT LABELS
        class_arr = np.vstack([class_arr, classifiedEvents])
    class_arr = class_arr[1:,:]
    end = time.time()
    # print(end - start)

    if make_rep:
        rep_arr, info = Rep_Func(class_arr)
        if data_handler:
            data_handler.save_rep_data(np.array(rep_arr['x']*256 + rep_arr['y']),np.array(rep_arr['toa']),np.array(rep_arr['tot']))
    return class_arr, rep_arr

def data_processing(file_path, offsets):
    '''
    Function is used to parallelise data processing by loading the files from a measurement.
    It can be adjusted to perform different evaluations and return the results for single data packages.

    Parameters
    ----------
    - file_path (string): Path to the processing folder
    - offsets (stacked array_like): includes all relevant offsets from photon offset to bounds of histograms

    Returns
    -------
    - Results from performed measurements.
    '''

    # Load data package
    try:
        data = np.load(file_path)
    except: return

    # Path to the repository. Necessary cause its the location of the pixel delay calibration.
    rep_path = "C:/Users/admin/Desktop/Coding/QuOEPP/"

    # Extract necessary parameters
    photon_offset = offsets[0]
    lower = offsets[1]
    upper = offsets[2]
    bg_lower = offsets[3]
    bg_upper = offsets[4]

    delta = offsets[5]
    binning = offsets[6]
    offset = offsets[7]

    correction = offsets[8]
    #sub_correction = offsets[9]
    sub_correction = False

    bin= int(delta*2/binning)
    bins = np.linspace(-delta, delta, bin+1)
    hist = np.zeros(bin)

    try:
        arr1 = data['photons1']
        if arr1.shape[0] > 0:
            arr1=arr1-photon_offset
            arr1*=10
            #print(arr1)
    except Exception as e:
         print("No photon data in package")
         return

    arr2 = data['photons2']
    if arr2.shape[0]>0:
        arr2=arr2-photon_offset
        arr2*=10
        #print(arr2)
    arr = np.concatenate((arr1, arr2))
    photons = np.sort(arr[arr>0])

    try:
        electrons_toa = data['clustered_toa']
        electrons_tot = data['clustered_tot']
        electrons_xy = data['clustered_xy']

        # Delete hot pixels
        id = np.isin(electrons_xy,hot_pixels)
        electrons_toa = np.delete(electrons_toa,id)
        electrons_tot = np.delete(electrons_tot,id)
        electrons_xy = np.delete(electrons_xy,id)

        print("Clustered data present")
        print('Number of clustered electron events: {}'.format(electrons_toa.shape[0]))
        print('Number of photon events: {}'.format(photons.shape[0]))
    
    except Exception as e:
        print("Clustered data is generated")

        electrons_sort = np.argsort(data['toas'])

        electrons_toa = data['toas'][electrons_sort]*15625
        electrons_tot = data['tots'][electrons_sort]*25
        electrons_xy = data['addresses'][electrons_sort]

        # Delete hot pixels
        id = np.isin(electrons_xy,hot_pixels)
        electrons_toa = np.delete(electrons_toa,id)
        electrons_tot = np.delete(electrons_tot,id)
        electrons_xy = np.delete(electrons_xy,id)

        print('Number of electron events: {}'.format(electrons_toa.shape[0]))
        print('Number of photon events: {}'.format(photons.shape[0]))

        data_cluster =np.vstack([electrons_xy, electrons_toa, electrons_tot])

        class_arr, rep_arr = clustering(data_cluster,
                                        #label_offset = l_off,
                                        split = 1,
                                        eps = 3,
                                        timeRecalibrationFactor = timeRecalibrationFactor,
                                        normalize_t = 1)
        
        electrons_toa = np.array(rep_arr['toa'])
        electrons_tot = np.array(rep_arr['tot'])
        electrons_xy = np.array(rep_arr['x'])*256 + np.array(rep_arr['y'])

        electrons_sort=np.argsort(electrons_toa)

        electrons_toa = electrons_toa[electrons_sort]
        electrons_tot = electrons_tot[electrons_sort]
        electrons_xy = electrons_xy[electrons_sort]
    
        np.savez(file_path,addresses=data['addresses'], toas=data['toas'], tots=data['tots'], photons1=data['photons1'], photons2=data['photons2'], trigger=data['trigger'],clustered_toa=electrons_toa,clustered_tot=electrons_tot,clustered_xy=electrons_xy)

        print('Number of clustered electron events: {}'.format(electrons_toa.shape[0]))


        ############################################################

    # Pixel delay Calibration
    if correction == True:
        pixel_offsets = np.load(rep_path+"correction_map.npz")['correction_map'].ravel()
        pixel_delay = pixel_offsets[electrons_xy.astype(int)]
        electrons_toa += pixel_delay

    # Histogram
    for ph_set in np.array_split(photons, 1000)[:]:
        if ph_set.shape[0] != 0:
            es = electrons_toa[(electrons_toa>ph_set.min()-delta) & (electrons_toa<ph_set.max()+delta)] - offset
            diff_arr = es[:,np.newaxis]-ph_set
            diff_arr = diff_arr.ravel()
            hist_data = diff_arr[np.abs(diff_arr) < delta]
            _, edge = np.histogram(hist_data, bins=bins)
            hist += _

    if sub_correction == True:
        # New Method of improving fwhm (experimental)
        edge_off = 30*int(1e4)
        max = edge[hist.argmax()]

        hist_dens = np.zeros(bin)

        for ph_set in np.array_split(photons, 1000)[:]:
            if ph_set.shape[0] != 0:
                es = electrons_toa[(electrons_toa>ph_set.min()-edge_off) & (electrons_toa<ph_set.max()+edge_off)] - offset
                diff_arr = es[:,np.newaxis]-ph_set
                diff_arr = diff_arr.ravel()
                hist_data = diff_arr[np.abs(diff_arr) < edge_off]
                _, edge = np.histogram(hist_data, bins=bins)
                hist_dens += _

        def gauss_function(x, a, x0, sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
        
        bin_heights = hist_dens/np.sum(hist_dens)
        bin_centers = edge[:-1] + np.diff(edge) / 2

        # Gauß-Kurve an die Daten anpassen
        max0 = np.max(bin_heights)
        mean0 = 0
        sigma0 = edge_off/2
        try:
            popt, pcov = curve_fit(gauss_function, bin_centers, bin_heights, p0=[max0, mean0, sigma0])
            mean, sigma = popt[1], popt[2]
            if np.abs(mean) < edge_off: 
                offset += mean
                hist = np.zeros(bin)
                for ph_set in np.array_split(photons, 1000)[:]:
                    if ph_set.shape[0] != 0:
                        es = electrons_toa[(electrons_toa>ph_set.min()-delta) & (electrons_toa<ph_set.max()+delta)] - offset
                        diff_arr = es[:,np.newaxis]-ph_set
                        diff_arr = diff_arr.ravel()
                        hist_data = diff_arr[np.abs(diff_arr) < delta]
                        _, edge = np.histogram(hist_data, bins=bins)
                        hist += _
        except: pass


    # Coincidences
    i,j = 0,0
    coins = np.empty((0,))

    while (i<electrons_toa.shape[0]-10) and (j<photons.shape[0]-10):
        diff = electrons_toa[i] - photons[j] - offset                                                                                              
        if diff > lower and diff < upper:
            coins = np.append(coins,i)
            i += 1
            j += 1
        elif diff >= upper:
            j += 1
        else:
            i += 1

    print('Found {} coincidences at {} ns offset'.format(coins.shape[0],offset/1e4))

    # Background removal
    i,j = 0,0
    noise = np.empty((0,))
    while (i<electrons_toa.shape[0]-10) and (j<photons.shape[0]-10):
        diff = electrons_toa[i] - photons[j] - offset
        if diff > bg_lower and diff < bg_upper:
            noise = np.append(noise,i)
            i += 1
            j += 1
        elif diff >= bg_upper:
            j += 1
        else:
            i += 1

    print('{} background coincidences found'.format(noise.shape[0]))

    # Electrons per photon
    i,j = 0,0
    herald_count = np.empty((0,))

    while (i<electrons_toa.shape[0]-10) and (j<photons.shape[0]-10):
        diff = electrons_toa[i] - photons[j] - offset                                                                                              
        if diff > lower and diff < upper:
            herald_count = np.append(herald_count,j)
            i += 1
        elif diff >= upper:
            j += 1
        else:
            i += 1

    herald = np.bincount(herald_count.astype(int), minlength=photons.shape[0])

    # Correlated Data
    try:
        # Circular Profile
        center_id = np.bincount(electrons_xy.astype(int)).argmax()
        center = ((center_id/256).astype(int),(center_id%256).astype(int))
        circular_raw, raw_edge = plot_rad_dist(electrons_xy,center)
        
        circular_coin, coin_edge = plot_rad_dist(electrons_xy[coins.astype(int)],center)
        circular_noise, bg_edge = plot_rad_dist(electrons_xy[noise.astype(int)],center)

        circular_noise = circular_coin - circular_noise

        # Plots
        # Raw Data
        data1 = electrons_xy
        pic1 = np.array(data1)
        pic1[-1] = 256*256-1
        pic1 = pic1.astype(np.uint())
        bc1 = np.bincount(pic1).reshape(-1,256)

        # Coincidence Data
        data2 = data1[coins.astype(int)]
        pic2 = np.array(data2)
        pic2[-1] = 256*256-1
        pic2 = pic2.astype(np.uint())
        bc2 = np.bincount(pic2).reshape(-1,256)

        # Background substracted
        data3 = data1[noise.astype(int)]
        pic3 = np.array(data3)
        pic3[-1] = 256*256-1
        pic3 = pic3.astype(np.uint())
        bc3 = np.bincount(pic3).reshape(-1,256)
        bc3 = bc2 - bc3

        return hist, edge, electrons_toa.shape[0], photons.shape[0],herald, bc1, bc2, bc3,coins.shape[0], electrons_toa[coins.astype(int)],circular_raw,raw_edge,circular_coin,coin_edge,circular_noise,bg_edge,center_id
    except Exception as e:
        print(f'Error while returning values in {Path(file_path).name}: {str(e)}')
        return hist, edge, electrons_toa.shape[0], photons.shape[0], herald
        return 
    

def calibration_processing(file_path, offsets):
    '''
    Function is used to process calibration data and return desired calibrations.

    Parameters
    ----------
    - file_path (string): Path to the processing folder
    - offsets (stacked array_like): includes all relevant offsets from photon offset to bounds of histograms

    Returns
    -------
    - Results from performed calibration.
    '''

    # Load data package
    try:
        data = np.load(file_path)
    except: return

    # Path to the repository. Necessary cause its the location of the pixel delay calibration.
    rep_path = "C:/Users/admin/Desktop/Coding/QuOEPP/"

    # Extract necessary parameters
    photon_offset = offsets[0]
    lower = offsets[1]
    upper = offsets[2]
    bg_lower = offsets[3]
    bg_upper = offsets[4]

    delta = offsets[5]
    binning = offsets[6]
    offset = offsets[7]

    correction = offsets[8]

    edge_off = 50*int(1e4)

    bin= int(delta*2/binning)
    bins = np.linspace(-delta, delta, bin+1)
    hist = np.zeros(bin)
    hist_dens = np.zeros(bin)

    try:
        arr1 = data['photons1']
        if arr1.shape[0] > 0:
            arr1=arr1-photon_offset
            arr1*=10
            #print(arr1)
    except Exception as e:
         print("No photon data in package")
         return

    arr2 = data['photons2']
    if arr2.shape[0]>0:
        arr2=arr2-photon_offset
        arr2*=10
        #print(arr2)
    arr = np.concatenate((arr1, arr2))
    photons = np.sort(arr[arr>0])

    try:
        electrons_toa = data['clustered_toa']
        electrons_tot = data['clustered_tot']
        electrons_xy = data['clustered_xy']

        # Delete hot pixels
        id = np.isin(electrons_xy,hot_pixels)
        electrons_toa = np.delete(electrons_toa,id)
        electrons_tot = np.delete(electrons_tot,id)
        electrons_xy = np.delete(electrons_xy,id)

        print("Clustered data present")
        print('Number of clustered electron events: {}'.format(electrons_toa.shape[0]))
        print('Number of photon events: {}'.format(photons.shape[0]))
    
    except Exception as e:
        print("Clustered data is generated")

        electrons_sort = np.argsort(data['toas'])

        electrons_toa = data['toas'][electrons_sort]*15625
        electrons_tot = data['tots'][electrons_sort]*25
        electrons_xy = data['addresses'][electrons_sort]

        # Delete hot pixels
        id = np.isin(electrons_xy,hot_pixels)
        electrons_toa = np.delete(electrons_toa,id)
        electrons_tot = np.delete(electrons_tot,id)
        electrons_xy = np.delete(electrons_xy,id)

        print('Number of electron events: {}'.format(electrons_toa.shape[0]))
        print('Number of photon events: {}'.format(photons.shape[0]))

        data_cluster =np.vstack([electrons_xy, electrons_toa, electrons_tot])

        class_arr, rep_arr = clustering(data_cluster,
                                        #label_offset = l_off,
                                        split = 1,
                                        eps = 3,
                                        timeRecalibrationFactor = timeRecalibrationFactor,
                                        normalize_t = 1)
        
        electrons_toa = np.array(rep_arr['toa'])
        electrons_tot = np.array(rep_arr['tot'])
        electrons_xy = np.array(rep_arr['x'])*256 + np.array(rep_arr['y'])

        electrons_sort=np.argsort(electrons_toa)

        electrons_toa = electrons_toa[electrons_sort]
        electrons_tot = electrons_tot[electrons_sort]
        electrons_xy = electrons_xy[electrons_sort]
    
        np.savez(file_path,addresses=data['addresses'], toas=data['toas'], tots=data['tots'], photons1=data['photons1'], photons2=data['photons2'], trigger=data['trigger'],clustered_toa=electrons_toa,clustered_tot=electrons_tot,clustered_xy=electrons_xy)

        print('Number of clustered electron events: {}'.format(electrons_toa.shape[0]))


        ############################################################

    # Histogram
    for ph_set in np.array_split(photons, 1000)[:]:
        if ph_set.shape[0] != 0:
            es = electrons_toa[(electrons_toa>ph_set.min()-delta) & (electrons_toa<ph_set.max()+delta)] - offset
            diff_arr = es[:,np.newaxis]-ph_set
            diff_arr = diff_arr.ravel()
            hist_data = diff_arr[np.abs(diff_arr) < delta]
            _, edge = np.histogram(hist_data, bins=bins)
            hist += _


    # Coincidences
    i,j = 0,0
    coins = np.empty((0,))
    tdiff = np.empty((0,))

    while (i<electrons_toa.shape[0]-10) and (j<photons.shape[0]-10):
        diff = electrons_toa[i] - photons[j] - offset                                                                                              
        if diff > lower and diff < upper:
            coins = np.append(coins,i)
            tdiff = np.append(tdiff,diff)
            i += 1
            j += 1
        elif diff >= upper:
            j += 1
        else:
            i += 1

    xy = electrons_xy[coins.astype(int)]

    return xy.astype(int), tdiff.astype(int), hist