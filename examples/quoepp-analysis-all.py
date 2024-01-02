import os
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from tkinter import filedialog
from pathlib import Path
import tkinter
from matplotlib_scalebar.scalebar import ScaleBar
from sklearn.cluster import DBSCAN
import time
import pandas as pd

root = tkinter.Tk()
root.withdraw() # use to hide tkinter window

hot_pixels = np.array([(114,160),(116,199),(103,198),(76,166),(55,203),(121,97),(249,56),(150,176),(197,223),(166,57),(124,229),(87,178),(89,170),(122,103),(165,155),(165,112),(114,116),(165,115)])


# Path to Folder and Name of Folder
num_cores = 16
binning = 20
binning *= int(1e4)
delta = 1000
delta *= int(1e4)
offset = -60
offset *= int(1e4)
radi = 50
camera_length = 60
scale = 55/5.5/camera_length
linthresh_raw = 1e-5
vmax_raw = 1e-4
linthresh_coin = 5*1e-8
vmax_coin = 1e-6
downscaling=10
scaling = 1e-4
cluster = True
cluster_eps = 3 
timeRecalibrationFactor = 2e-06
norm='symlog'

def data_processing(file_path, photon_offset):
    data = np.load(file_path)

    bin=int(delta*2/binning)
    bins = np.linspace(-delta, delta, bin+1)
    hist = np.zeros(bin)

    electrons_sort = np.argsort(data['toas'])

    electrons_toa = data['toas'][electrons_sort]*15625
    electrons_tot = data['tots'][electrons_sort]*25
    electrons_xy = data['addresses'][electrons_sort]

    arr1 = data['photons1']
    if arr1.shape[0] > 0:
        arr1-=photon_offset
        arr1*=10
        #print(arr1)

    arr2 = data['photons2']
    if arr2.shape[0]>0:
        arr2-=photon_offset
        arr2*=10
        #print(arr2)
    arr = np.concatenate((arr1, arr2))
    photons = np.sort(arr[arr>0])

    print(Path(file_path).name)

    print('Number of electron events: {}'.format(electrons_toa.shape[0]))
    print('Number of photon events: {}'.format(photons.shape[0]))

    if cluster == True:
        data =np.vstack([electrons_xy, electrons_toa, electrons_tot])

        def Rep_Func_default(class_arr):
            totlim = 2.5
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
            '''data.shape = ()  
            e0_toa ... estimate of trigger time in  0.1 ps Units
            normalize_t ... rescales the time data
            timeRecalibrationFactor = 1 / 300000 #* (10**6 / 640) # recalibration of time for dimension equality (for fitting)
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

        class_arr, rep_arr = clustering(data,
                                        #label_offset = l_off,
                                        split = 1,
                                        eps = 3,
                                        timeRecalibrationFactor = timeRecalibrationFactor,
                                        normalize_t = 1)
        #l_off = int(class_arr[:,4])
        electrons_toa = np.array(rep_arr['toa'])
        electrons_tot = np.array(rep_arr['tot'])
        electrons_xy = np.array(rep_arr['x'])*256 + np.array(rep_arr['y'])

        electrons_sort=np.argsort(electrons_toa)

        electrons_toa = electrons_toa[electrons_sort]
        electrons_tot = electrons_tot[electrons_sort]
        electrons_xy = electrons_xy[electrons_sort]

        print('Number of clustered electron events: {}'.format(electrons_toa.shape[0]))

    for ph_set in np.array_split(photons, 1000)[:]:
        if ph_set.shape[0] != 0:
            es = electrons_toa[(electrons_toa>ph_set.min()-delta) & (electrons_toa<ph_set.max()+delta)] - offset
            diff_arr = es[:,np.newaxis]-ph_set
            diff_arr = diff_arr.ravel()
            hist_data = diff_arr[np.abs(diff_arr)<delta]
            _, edge = np.histogram(hist_data, bins=bins)
            hist += _


    i,j = 0,0
    coins = np.empty((0,)) 
    while (i<electrons_toa.shape[0]-10) and (j<photons.shape[0]-10):
        diff = electrons_toa[i] - offset - photons[j]
        if np.abs(diff) < delta:
            coins = np.append(coins,i)
            i += 1
            j += 1
        elif diff >= 0:
            j += 1
        else:
            i += 1

    print('Found {} coincidences at {} ns offset'.format(coins.shape[0],offset/1e4))

    # Raw Data
    data1 = electrons_xy
    pic1 = np.array(data1)
    pic1[-1] = 256*256-1
    pic1 = pic1.astype(np.uint())
    bc1 = np.bincount(pic1).reshape(-1,256)
    bc1[hot_pixels[:,1],hot_pixels[:,0]] = 0

    # Correlated Data
    try:
        data2 = data1[coins.astype(int)]
        pic2 = np.array(data2)
        pic2[-1] = 256*256-1
        pic2 = pic2.astype(np.uint())
        bc2 = np.bincount(pic2).reshape(-1,256)
        bc2[hot_pixels[:,1],hot_pixels[:,0]] = 0
        return hist, edge, bc1, bc2, electrons_toa.shape[0], photons.shape[0],coins.shape[0], electrons_toa[coins.astype(int)]
    except Exception as e:
        print(f'Error while processing {Path(file_path).name}: {str(e)}')
        return

def main(url):
    process_folders = [os.path.join(url, f) for f in os.listdir(url) if os.path.isdir(os.path.join(url, f))]

    folder_path = []

    for x in process_folders:
        folder_path.append([os.path.join(x, f) for f in os.listdir(x) if os.path.isdir(os.path.join(x, f))])
    folder_path=[num for elem in folder_path for num in elem]
    print(folder_path)

    for i in folder_path:
        print(i)
        process_path = i
        name = Path(process_path).name
        path = os.path.join(process_path, name)
        data_path = os.path.join(process_path, 'data')

        # Define the specific data format you want to search for (e.g., ".txt" or ".csv")
        target_format = ".npz"  # Change this to your desired format

        # Initialize an empty list to store the matching file paths
        process_list = []

        # Walk through the directory tree and find files with the specified format
        for root, _, files in os.walk(process_path):
            for filename in files:
                if filename.endswith(target_format):
                    process_list.append(os.path.join(root, filename))

        folder_list = [os.path.join(process_path, f) for f in os.listdir(process_path) if os.path.isdir(os.path.join(process_path, f))]
        folder_number = [f for f in os.listdir(process_path) if os.path.isdir(os.path.join(process_path, f))]

        photon_offsets = np.zeros((len(folder_number),))

        for folder in folder_list:
            package = os.path.join(folder, 'data')
            package_number = int(Path(folder).name)-1
            data_list = sorted([os.path.join(package, f) for f in os.listdir(package) if f.endswith('.npz')], key=os.path.getmtime)

            for data in data_list:
                package = np.load(data, mmap_mode='r')
                photon_offsets[package_number] = 0
                try:
                    if photon_offsets[package_number] == 0:
                        photon_offsets[package_number] = package['trigger'][0]
                        print(r'Photon offset corrected: {} ns in package {}'.format(photon_offsets[package_number]/1e4,package_number+1))
                        break
                except Exception as e:
                    print(f'Error while processing {data}: {str(e)}')

        pool = mp.Pool(num_cores)
        results = pool.starmap(data_processing, [(data, int(photon_offsets[int(Path(data).parent.parent.name)-1])) for data in process_list])
        pool.close()
        pool.join()

        hist = []
        bc1 = []
        bc2 = []
        raw_electrons = []
        raw_photons = []
        coincidences = []
        hist_toa = []

        for i in results:
            try:
                hist.append(i[0])
                bc1.append(i[2])
                bc2.append(i[3])
                raw_electrons.append(i[4])
                raw_photons.append(i[5])
                coincidences.append(i[6])
                hist_toa.append(i[7])
            except Exception as e:
                print(f'Error while getting results: {str(e)}')
                pass
        
        hist_sum = np.sum(hist, axis=0)
        sum_electrons = np.sum(raw_electrons)
        sum_photons = np.sum(raw_photons)
        sum_coincidences = np.sum(coincidences)
        sum_rawplot = np.sum(np.array(bc1)/sum_electrons, axis=0)
        sum_coinplot = np.sum(np.array(bc2)/sum_electrons, axis=0)
        hist_toa=np.concatenate(hist_toa)
        ################################################

        edge = results[0][1]

        center = np.unravel_index(np.argmax(sum_rawplot), sum_rawplot.shape)
        # center = (128,128)

        # Checks data type. jpg and png have shape (y,x,misc)
        y, x = np.indices((sum_rawplot.shape))

        # Creates an array of distances from the centre to the set radius of the profile. 
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

        # Convert array data to integer.
        r = r.astype(int)

        # Count bins from 0 to r in data. ravel() returns single entries of the whole array.
        tbin = np.bincount(r.ravel(), sum_rawplot.ravel())
        # Bincounting for normalizing the profile.
        nr = np.bincount(r.ravel())

        # Normalize radial profile and return.
        circularprofile_raw = tbin / nr

        # Count bins from 0 to r in data. ravel() returns single entries of the whole array.
        tbin = np.bincount(r.ravel(), sum_coinplot.ravel())
        # Bincounting for normalizing the profile.
        nr = np.bincount(r.ravel())

        # Normalize radial profile and return.
        circularprofile_coin = tbin / nr

        # Averaging of the plot.
        avg_y=[]
        for k in range(len(circularprofile_coin)-4+1):
            avg_y.append(np.mean(circularprofile_coin[k:k+4]))

        circularprofile_coin = avg_y[0:radi]

        # Averaging of the plot.
        avg_y=[]
        for k in range(len(circularprofile_coin)-4+1):
            avg_y.append(np.mean(circularprofile_raw[k:k+4]))

        circularprofile_raw = avg_y[0:radi]

        xscale = np.array(range(0,radi,10))

        # FWHM
        # Find the bin centers
        bin_centers = 0.5 * (edge[1:] + edge[:-1])

        # Find the half-maximum value
        half_max = np.max(hist_sum) / 2

        # Find the indices of the bins where the histogram values are closest to the half-maximum value
        left_idx = np.argmin(np.abs(hist_sum[:len(hist_sum)//2] - half_max))
        right_idx = np.argmin(np.abs(hist_sum[len(hist_sum)//2:] - half_max)) + len(hist_sum)//2

        # Calculate the FWHM
        fwhm = edge[right_idx] - edge[left_idx]

        #####################################################

        scalebar1 = ScaleBar(scale,units="um",dimension='si-length',length_fraction=0.25,location="lower left",box_alpha=0.0, scale_formatter=lambda value, unit: f"{value} {unit[:-1]}rad",color='#FFFFFF')
        scalebar2 = ScaleBar(scale,units="um",dimension='si-length',length_fraction=0.25,location="lower left",box_alpha=0.0, scale_formatter=lambda value, unit: f"{value} {unit[:-1]}rad",color='#FFFFFF')

        # Plot the histogram
        fig, ax = plt.subplots(2,3,figsize=(18,12))

        symnorm=SymLogNorm(linthresh=np.min(sum_rawplot[sum_rawplot>0]), linscale=scaling,vmax=np.max(sum_rawplot))

        fig.suptitle(name)

        ax[0][1].set_title('Raw data')
        im = ax[0][1].imshow(sum_rawplot,cmap='hot',norm=symnorm,origin='lower')
        fig.colorbar(im)
        ax[0][1].add_artist(scalebar1)

        ax[0][2].set_title(r'Raw Profile at ({},{})'.format(center[0],center[1]))
        ax[0][2].grid(True)
        ax[0][2].plot(circularprofile_raw)
        ax[0][2].set_yscale('log')
        ax[0][2].set_xticks(np.round(xscale, 3),np.round(xscale*scale, 3))
        ax[0][2].set_xlabel("Radius [urad]")
        ax[0][2].set_ylabel("Counts")

        ax[0][0].set_title('binning: {} ns, delta = {} ns, FWHM = {} ns'.format(binning/1e4, delta/1e4, fwhm/1e5))
        ax[0][0].grid(True)
        ax[0][0].plot(edge[10:-1]/1e7, hist_sum[10:]/np.mean(hist_sum), drawstyle= 'steps-pre')
        ax[0][0].set_xlabel(r'$\Delta t \;/ \mathrm{\mu s}$')
        ax[0][0].set_ylabel('Counts/bin')

        ax[1][1].set_title('{} electrons \n {} photons \n {} coincidences'.format(sum_electrons, sum_photons, sum_coincidences))
        im = ax[1][1].imshow(sum_coinplot,cmap='hot',norm='linear',vmax=np.max(sum_coinplot)/3,origin='lower')
        fig.colorbar(im)
        ax[1][1].add_artist(scalebar2)

        ax[1][2].set_title(r'Coincidence Profile at ({},{})'.format(center[0],center[1]))
        ax[1][2].grid(True)
        ax[1][2].plot(circularprofile_coin)
        ax[1][2].set_yscale('log')
        ax[1][2].set_xticks(np.round(xscale, 3),np.round(xscale*scale, 3))
        ax[1][2].set_xlabel("Radius [urad]")
        ax[1][2].set_ylabel("Counts")

        ax[1][0].set_title('ToA over Measurement time')
        ax[1][0].hist(hist_toa/1e13, bins=100)
        ax[1][0].set_xlabel(r'Measurement Time [s]')
        ax[1][0].set_ylabel('Counts/bin')

        plt.savefig(process_path+'\\'+name+'_range {} ns_binning {} ns_offset {} ns.png'.format(delta/1e4, binning/1e4, offset/1e4))

if __name__ == '__main__':
    url = str(filedialog.askdirectory())
    main(url)
