from sklearn.preprocessing import StandardScaler
# for now we'll use DBSCAN, but this still needs to be evaluated
from sklearn.cluster import DBSCAN
from npy_append_array import NpyAppendArray
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import njit
from matplotlib.colors import SymLogNorm, LogNorm
import os
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import LinearSegmentedColormap

class coin():

    def __init__(self,path,name):
        self.path = path
        self.name = name

    def clustering(self,eps,min_samples,dataPercentage):
        path = self.path
        url = Path(path)
        # Load and sort all electron files in url
        e_files = list(url.glob('**/electrons/*.npy'))
        dummy = [str(f).split('\\')[-1].lower() for f in e_files]   # Convert to lower case
        dummy = [int(f.split('.')[0]) for f in dummy]   # Convert to lower case
        dummy = np.argsort(np.array(dummy))
        e_files = [e_files [i]for i in dummy]

        # Get lowest value for offset correction
        e0_toa =  np.load(str(e_files[0]))[1,:].min()*15625 # 0.1 ps Units!!!!!

        # Setup files for large data array
        file_xy = url / 'out_xy.npy'
        file_toa  = url / 'out_toa.npy'


        npaa_xy = NpyAppendArray(file_xy,delete_if_exists=True)
        npaa_toa = NpyAppendArray(file_toa,delete_if_exists=True)

        # Convert files: load -> sort by toa -> append  
        for file in e_files:
            arr_e = np.load(str(file))[:2,:]
            index = arr_e[1,:].argsort()
            npaa_xy.append(arr_e[0,:][index])
            npaa_toa.append((arr_e[1,:][index]*15625-e0_toa))

        npaa_xy.close()
        npaa_toa.close()

        # Load data
        data_xy=np.load(file_xy,mmap_mode="r")
        data_toa=np.load(file_toa,mmap_mode="r")

        inputData = np.array([data_xy,data_toa])

        print("Data to Cluster: {}".format(inputData.shape[1]))

        # prepare individual coordinate arrays
        endPoint = int(inputData[0].size * dataPercentage) # determine array limit based on data percentage

        xCoordinates = (inputData[0,:endPoint] % 256).astype(int)
        yCoordinates = (inputData[0,:endPoint] / 256).astype(int)
        tCoordinates = inputData[1,:endPoint]

        print("Reduced data: {}".format(tCoordinates.shape[0]))

        # create list of events 
        events = np.array(list(zip(xCoordinates,yCoordinates,tCoordinates)))

        # SCALE DATA
        # this should actually not really be necessary (will have to try without)
        scaler = StandardScaler()
        normalizedData = scaler.fit_transform(events)


        # Apply Clustering
        #model = DBSCAN(eps=0.002,min_samples=3,n_jobs=-1).fit(normalizedData)
        # labels = model.fit_predict(normalizedData)
        # n_jobs determines cpu-cure utulization (-1 uses all cores)
        labels = DBSCAN(eps=eps,min_samples=min_samples,n_jobs=-1).fit_predict(normalizedData)

        # create list of events with their classification
        #classifiedEvents = np.array(list(zip(xCoordinates,yCoordinates,tCoordinates,labels)))

        rogueEvents = events[labels == -1]
        electronEvents = events[labels != -1]
        np.save(url / 'out_clustered.npy',electronEvents)
        # xArea = (100,160)
        # yArea = (40,100)

        xArea = (0,255)
        yArea = (0,255)

        fig, ax = plt.subplots(1,2,figsize=(12,6))
        ax[0].set(title="Rogue Events",xlim=xArea,ylim=yArea)
        ax[0].scatter(rogueEvents[:,0],rogueEvents[:,1],s=0.5)

        ax[1].set(title="Electron (clustered) Events",xlim=xArea,ylim=yArea)
        ax[1].scatter(electronEvents[:,0],electronEvents[:,1],s=0.5)
        plt.show()
        
    def data_merge(self):

        ##############################################################################################

        data_path=self.path / 'data'
        url = Path(self.path)

        data_list = sorted(data_path.iterdir(), key=os.path.getctime)

        ##############################################################################################

        # Setup files for large data array
        file_xy = url / 'out_xy.npy'
        file_toa  = url / 'out_toa.npy'
        file_ph = url / 'out_ph.npy'

        npaa_xy = NpyAppendArray(file_xy,delete_if_exists=True)
        npaa_toa = NpyAppendArray(file_toa,delete_if_exists=True)
        npaa_ph = NpyAppendArray(file_ph,delete_if_exists=True)

        photon_offset = 0

        # Search for photon offset 
        for file in data_list:
            try:
                if (photon_offset == 0):
                    photon_offset = np.load(str(file))['trigger'][0]
                    print("Photon offset corrected: {}".format(photon_offset))
                    break
            except: pass

        # Convert files: load -> sort by toa -> append

        for file in data_list:
            arr_toa = np.load(str(file))['toas']
            arr_xy = np.load(str(file))['addresses']
            index = arr_toa[:].argsort()
            npaa_xy.append(arr_xy[:][index])
            npaa_toa.append((arr_toa[:][index]*15625))

            arr1 = np.load(str(file))['photons1']
            if arr1.shape[0] > 0:
                arr1-=photon_offset
                arr1*=10

            arr2 = np.load(str(file))['photons2']
            if arr2.shape[0]>0:
                arr2-=photon_offset
                arr2*=10
            arr = np.concatenate((arr1, arr2))
            arr = arr[arr>0]
            npaa_ph.append(np.sort(arr))

        npaa_xy.close()
        npaa_toa.close()
        npaa_ph.close()

    def histogram(self,parameters,offset,delta,binning,split,save=False,cluster=False,show=False):
        path = self.path
        url = Path(path)
        name = self.name
        delta*=int(1e4)
        # offset*=int(1e4)
        self.delta = delta
        self.offset = offset

        # Setup files for large data array
        file_xy = url / 'out_xy.npy'
        file_toa  = url / 'out_toa.npy'
        file_ph = url / 'out_ph.npy'

        if (cluster == True): data_toa=np.load(url / 'out_clustered.npy')[:,2]
        else: data_toa = np.load(file_toa, mmap_mode="r")

        data_ph = np.load(file_ph)

        ##############################################################################################

        # Processing parameters
        seconds = time.time() # for timing 
        #delta = 5000000 # histogram range in units of 0.1. ps  
        #split = 1000 # in how many chunks to split the data, affects performance and memory consumption!
        #offset = 0 #1300000 offset added to electron time 
        #binning = 1000 # resolution of the histogram = delta/binning

        #Setup for histogram
        bins = np.linspace(-delta, delta, binning+1)
        hist = np.zeros(binning)
        collected = 0

        # Correlate
        #@jit(nopython=True)
        @njit
        def correlating(data_toa, data_ph, hist, delta_t, collected, split, offset, bins):
            for ph_set in np.array_split(data_ph,split)[:]:
                if ph_set.shape[0] != 0:
                    es = data_toa[(data_toa>ph_set.min()-delta_t) & (data_toa<ph_set.max()+delta_t)]
                    arr_e = es - offset
                    arr_ph = ph_set
                    
                    diff_arr = (arr_e.reshape(-1,1) - arr_ph.reshape(-1,1).transpose()).ravel()
                    hist_data = diff_arr[(np.abs(diff_arr) < delta_t)]

                    collected += hist_data.size
                    _, edge = np.histogram(hist_data, bins = bins) 
                    hist += _
            return hist, edge, collected

        hist, edge, collected = correlating(data_toa, data_ph[1:-2],hist, delta, collected, split, offset, bins)
        dt = time.time()-seconds

        # Find the bin centers
        bin_centers = 0.5 * (edge[1:] + edge[:-1])

        # Find the half-maximum value
        half_max = np.max(hist) / 2

        # Find the indices of the bins where the histogram values are closest to the half-maximum value
        left_idx = np.argmin(np.abs(hist[:len(hist)//2] - half_max))
        right_idx = np.argmin(np.abs(hist[len(hist)//2:] - half_max)) + len(hist)//2

        # Calculate the FWHM
        fwhm = edge[right_idx] - edge[left_idx]

        print('delta = {} ns'.format(delta/1e4))
        print('split = {}'.format(split))
        print('binning = {} ns'.format(2*delta/binning/1e4))
        print('offset = {} ns'.format(offset/1e4))
        print('Collected {} data points in {:.2f} seconds = {:.2f} points per second'.format(collected,dt,collected/dt))

        # Plot histogram
        plt.style.use('seaborn-whitegrid')
        cycler = plt.style.library['fivethirtyeight']['axes.prop_cycle']

        plt.rc('axes',  prop_cycle=cycler)
        plt.rc('lines', linewidth=2)
        plt.rc('image', cmap = 'Greys')
          
        plt.figure()
        plt.title('{}'.format(name)+'\nbinning = {} ns, FWHM = {} ns'.format(2*delta/binning/1e4,fwhm/1e5))
        plt.plot(edge[:-1]/1e7, hist, drawstyle= 'steps-pre')
        plt.xlabel(r'$\Delta t \;/ \mathrm{\mu s}$')
        file_name=name+"_"+str(delta/1e4)+"_coinc.png"
        if (save == True) : plt.savefig(path / file_name)
        #if (save == True and cluster == True) : plt.savefig("..\\results\\"+name+"_"+str(delta/1e4)+"_coinc_clustered.png")

        if show == True: plt.show()
        else: plt.close()
        np.save(url / 'out_hist.npy',hist)
        np.save(url / 'out_edges.npy',edge)
        return bins[hist.argmax()], hist, edge, fwhm

    def coincidences(self,offset,delta,cluster=False):
        path = self.path
        url = Path(path)
        delta*=int(1e4)
        offset*=int(1e4)
        self.delta = delta
        self.offset = offset

        # Load data 
        file_xy = url / 'out_xy.npy'
        file_toa  = url / 'out_toa.npy'
        file_ph = url / 'out_ph.npy'

        if (cluster == True): 
            data_toa=np.load(url / 'out_clustered.npy')[:,2]
            print("Clustered")
        else: 
            data_toa = np.load(file_toa, mmap_mode="r")
            print("Not clustered")

        photons = np.load(file_ph) 

        print('Number of electron events: {}'.format(data_toa.size))
        print('Number of photon events: {}'.format(photons.size))
        
        # Coincidence counting
        @njit
        def coincidence_counting(data_toa,photons,offset,delta):
            i,j = 0,0
            coins = np.empty((0,)) 
            while (i<data_toa.size-10) and (j<photons.size-10):
                diff = data_toa[i] - offset - photons[j]

                if np.abs(diff) < delta:
                    coins = np.append(coins,i)
                    i += 1
                    j += 1
                elif diff >= 0:
                    j += 1
                else:
                    i += 1
            return coins

        coin = coincidence_counting(data_toa,photons,offset,delta)
        print('Found {} coincidences at {} ns offset'.format(len(coin),offset/1e8))
        np.save(url / 'out_coin.npy',coin.astype(int))

        # Load xy data
        file_xy = url / 'out_xy.npy'
        data_xy = np.load(file_xy, mmap_mode="r")

        # Raw Data
        data1 = data_xy
        pic1 = np.array(data1)
        pic1[-1] = 256*256-1
        pic1 = pic1.astype(np.uint())
        bc1 = np.bincount(pic1).reshape(-1,256)/pic1.size

        # Correlated Data
        data2 = data_xy[coin.astype(int)]
        pic2 = np.array(data2)
        pic2[-1] = 256*256-1
        pic2 = pic2.astype(np.uint())
        bc2 = np.bincount(pic2).reshape(-1,256)/pic1.size


        return coin.astype(int), bc1, bc2

    def plot(self, delta,offset,thresh,map,scale,save=False,cluster=False,show=False,figsize=(3,3),xlim=(0,255),ylim=(0,255)):
        path = self.path
        url = Path(path) 
        name = self.name
        scaling = 1e-4

        file_xy = url / 'out_xy.npy'
        file_coin = url / 'out_coin.npy'
        data_xy = np.load(file_xy, mmap_mode="r")
        coincidences = np.load(file_coin)

        if cluster == True: data_clust = np.load(url / 'out_clustered.npy')

        # fig, ax = plt.subplot_mosaic([['1', '2', '3'],['4', '4', '4']],
        #                       constrained_layout=True,figsize=figsize)
        
        fig, ax = plt.subplot_mosaic([['1', '2']],
                              constrained_layout=True,figsize=figsize)
        
        fig.suptitle(self.name)
        plt.grid(False)
        scale = 0.1615
        scalebar1 = ScaleBar(scale,units="um",dimension='si-length',length_fraction=0.25,location="lower left",box_alpha=0.0, scale_formatter=lambda value, unit: f"{value} {unit[:-1]}rad",color='#FFFFFF')
        scalebar2 = ScaleBar(scale,units="um",dimension='si-length',length_fraction=0.25,location="lower left",box_alpha=0.0, scale_formatter=lambda value, unit: f"{value} {unit[:-1]}rad",color='#FFFFFF')
        scalebar3 = ScaleBar(scale,units="um",dimension='si-length',length_fraction=0.25,location="lower left",box_alpha=0.0, scale_formatter=lambda value, unit: f"{value} {unit[:-1]}rad",color='#FFFFFF')
        hot_pixels = np.array([(116,199),(103,198),(76,166),(55,203),(121,97)])

        # Raw Data
        data1 = data_xy
        pic1 = np.array(data1)
        pic1[-1] = 256*256-1
        pic1 = pic1.astype(np.uint())
        bc1 = np.bincount(pic1).reshape(-1,256)/pic1.size
        bc1[hot_pixels[:,1],hot_pixels[:,0]]=0

        lognorm1 = LogNorm(vmin=thresh, vmax=1)

        ax['1'].set(title="Raw Data: {} electrons".format(data1.shape[0]),xlim=xlim,ylim=ylim)
        ax['1'].grid(False)
        img0 = ax['1'].imshow(bc1,cmap=map, norm=lognorm1)
        fig.colorbar(img0,ax=ax['1'])
        ax['1'].add_artist(scalebar1)

        # Correlated Data
        data2 = data_xy[coincidences]
        pic2 = np.array(data2)
        pic2[-1] = 256*256-1
        pic2 = pic2.astype(np.uint())
        bc2 = np.bincount(pic2).reshape(-1,256)/pic1.size
        bc2[hot_pixels[:,1],hot_pixels[:,0]]=0
        #bc2 = np.bincount(pic2).reshape(-1,256)
        #img = ax.imshow(bc1 -bc2, cmap = 'RdBu', vmin = -0.0025,vmax = 0.0025 )

        lognorm2 = LogNorm(vmin=thresh, vmax=np.max(bc2))

        ax['2'].set(title="{} Coincidences".format(len(coincidences)),xlim=xlim,ylim=ylim)
        ax['2'].grid(False)
        img1 = ax['2'].imshow(bc2,cmap=map)
        fig.colorbar(img1,ax=ax['2'])
        ax['2'].add_artist(scalebar2)

        bc2 = np.bincount(pic2).reshape(-1,256)/pic1.size

        # # Clustered Data
        # data3 = data_clust[self.coincidences(offset,delta,cluster=True)]
        # bc3 = np.histogram2d(data3[:,1],data3[:,0],range=[[0,256],[0,256]],bins=256)[0]/data3.shape[0]

        # lognorm3 = LogNorm(vmin=thresh, vmax=np.max(bc3))

        # ax['3'].set(title='Clustered data',xlim=xlim,ylim=ylim)
        # img3 = ax['3'].imshow(bc3,cmap=map)
        # fig.colorbar(img3,ax=ax['3'])

        file_name=name+"_"+str(delta)+"_donut.png"
        if (save == True) : plt.savefig(path / file_name)

        np.save(url / 'out_rawcount.npy',bc1)
        np.save(url / 'out_coincount.npy',bc2)

        if show == True: plt.show()
        else: plt.close()

        return bc1, bc2, data1.shape[0], len(coincidences)

    def circular_profile_raw(self, bc1, bc2, radi, parameters, save=False):
            path = self.path
            url = Path(path) 
            name = self.name
            scaling = 1e-4
            save_path = parameters['path']

            file_xy = url / 'out_xy.npy'
            file_coin = url / 'out_coin.npy'

            #print(np.unravel_index(np.argmin(bc2-bc1), bc2.shape))
            center = np.unravel_index(np.argmax(bc2), bc2.shape)

            # Checks data type. jpg and png have shape (y,x,misc)
            y, x = np.indices((bc1.shape))

            # Creates an array of distances from the centre to the set radius of the profile. 
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

            # Convert array data to integer.
            r = r.astype(int)

            # Count bins from 0 to r in data. ravel() returns single entries of the whole array.
            tbin = np.bincount(r.ravel(), bc1.ravel())
            # Bincounting for normalizing the profile.
            nr = np.bincount(r.ravel())

            # Normalize radial profile and return.
            circularprofile = tbin / nr

            # Customizing the plot.

            #plt.plot(circularprofile[:radi], label=name)

            # Averaging of the plot.
            avg_y=[]
            for k in range(len(circularprofile)-4+1):
                avg_y.append(np.mean(circularprofile[k:k+4]))
            plt.plot(avg_y[0:radi],label=name+'_raw')
            circularprofile = avg_y[0:radi]

            xscale = np.array(range(0,radi,10))
            #scale = parameters['pixel_scale']
            scale = 0.1615
            plt.xticks(np.round(xscale, 3),np.round(xscale*scale, 3))
            plt.yscale("log")
            plt.xlabel("Radius [urad]")
            plt.ylabel("Counts")
            plt.legend()
            if save is True: plt.savefig(save_path+name+'_rawradial.png')
            plt.close('all')
            return circularprofile, xscale

    def circular_profile_coin(self, bc1, bc2, radi, parameters, save=False):
        path = self.path
        url = Path(path) 
        name = self.name
        scaling = 1e-4
        save_path=parameters['path']

        file_xy = url / 'out_xy.npy'
        file_coin = url / 'out_coin.npy'

        #print(np.unravel_index(np.argmin(bc2-bc1), bc2.shape))
        center = np.unravel_index(np.argmax(bc2), bc2.shape)

        # Checks data type. jpg and png have shape (y,x,misc)
        y, x = np.indices((bc2.shape))

        # Creates an array of distances from the centre to the set radius of the profile. 
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

        # Convert array data to integer.
        r = r.astype(int)

        # Count bins from 0 to r in data. ravel() returns single entries of the whole array.
        tbin = np.bincount(r.ravel(), (bc2).ravel())
        # Bincounting for normalizing the profile.
        nr = np.bincount(r.ravel())

        # Normalize radial profile and return.
        circularprofile = tbin / nr

        # Customizing the plot.

        #plt.plot(circularprofile[:radi], label=name)

        # Averaging of the plot.
        avg_y=[]
        for k in range(len(circularprofile)-4+1):
            avg_y.append(np.mean(circularprofile[k:k+4]))
        plt.plot(avg_y[0:radi],label=name+'_coin')

        circularprofile = avg_y[0:radi]

        xscale = np.array(range(0,radi,10))
        #scale = parameters['pixel_scale']
        scale = 0.1615
        plt.xticks(np.round(xscale, 3),np.round(xscale*scale, 3))
        plt.yscale("log")
        plt.xlabel("Radius [urad]")
        plt.ylabel("Counts")
        plt.legend()
        if save is True: plt.savefig(save_path+name+'_coinradial.png')
        plt.close('all')
        return circularprofile, xscale

    def circular_profile_diff(self, bc1, bc2, radi, parameters, save=False):
        path = self.path
        url = Path(path) 
        name = self.name
        scaling = 1e-4
        save_path=parameters['path']

        file_xy = url / 'out_xy.npy'
        file_coin = url / 'out_coin.npy'

        #print(np.unravel_index(np.argmin(bc2-bc1), bc2.shape))
        center = np.unravel_index(np.argmax(bc2), bc2.shape)

        # Checks data type. jpg and png have shape (y,x,misc)
        y, x = np.indices((bc2.shape))

        # Creates an array of distances from the centre to the set radius of the profile. 
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

        # Convert array data to integer.
        r = r.astype(int)

        # Count bins from 0 to r in data. ravel() returns single entries of the whole array.
        tbin = np.bincount(r.ravel(), (bc2-bc1).ravel())
        # Bincounting for normalizing the profile.
        nr = np.bincount(r.ravel())

        # Normalize radial profile and return.
        circularprofile = tbin / nr

        # Customizing the plot.

        #plt.plot(circularprofile[:radi], label=name)

        # Averaging of the plot.
        avg_y=[]
        for k in range(len(circularprofile)-4+1):
            avg_y.append(np.mean(circularprofile[k:k+4]))
        plt.plot(avg_y[0:radi],label=name+'_coin')

        circularprofile = avg_y[0:radi]

        xscale = np.array(range(0,radi,10))
        scale = parameters['pixel_scale']
        plt.xticks(np.round(xscale, 3),np.round(xscale*scale, 3))
        plt.yscale("log")
        plt.xlabel("Radius [urad]")
        plt.ylabel("Counts")
        plt.legend()
        if save is True: plt.savefig(save_path+name+'_coinradial.png')
        plt.close('all')
        return circularprofile, xscale

    def result_hist(path, parameters, name, hist, edge, delta, save = False, show = True, cluster = False):

        url = Path(path)

        # Find the bin centers
        bin_centers = 0.5 * (edge[1:] + edge[:-1])

        # Find the half-maximum value
        half_max = np.max(hist) / 2

        # Find the indices of the bins where the histogram values are closest to the half-maximum value
        left_idx = np.argmin(np.abs(hist[:len(hist)//2] - half_max))
        right_idx = np.argmin(np.abs(hist[len(hist)//2:] - half_max)) + len(hist)//2

        # Calculate the FWHM
        fwhm = edge[right_idx] - edge[left_idx]

        # Plot histogram
        plt.style.use('seaborn-whitegrid')
        cycler = plt.style.library['fivethirtyeight']['axes.prop_cycle']

        plt.rc('axes',  prop_cycle=cycler)
        plt.rc('lines', linewidth=2)
        plt.rc('image', cmap = 'Greys')
          
        plt.figure()
        plt.title('{}'.format(name)+' \n acquisition: {}s, range = {} ns'.format(parameters['tp3_exposure_time']*parameters['corr_sum'],delta/1e4)+', binning = {} ns, FWHM = {} ns'.format(2*delta/parameters['corr_binning']/1e4,fwhm/1e5))
        plt.plot(edge[:-1]/1e7, hist, drawstyle= 'steps-pre')
        plt.xlabel(r'$\Delta t \;/ \mathrm{\mu s}$')

        file_name=name+"_"+str(delta/1e4)+"_coin.png"
        if (save == True) : plt.savefig(url / file_name)

        if (save == True and cluster == True) : plt.savefig("..\\results\\"+name+"_"+str(delta/1e4)+"_coinc_clustered.png")

        if show == True: plt.show()
        else: plt.close()
        np.save(url / 'out_hist_{}.npy'.format(delta),hist)

    def result_coin(save_path, parameters, name, bc1, bc2, ecount, coin, delta, save = False, show = True, figsize=(12,3), thresh = 0.0001, scaling=1e-4, xlim=(0,255),ylim=(0,255)):
        
        url = Path(save_path)

        fig, ax = plt.subplot_mosaic([['1', '2', '3']],
                              constrained_layout=True,figsize=figsize)
        
        fig.suptitle(name)
        plt.grid(False)

        map='hot'
        

        # Raw Data

        #lognorm1 = LogNorm(vmin=thresh, vmax=100)
        lognorm1 = SymLogNorm(linthresh=thresh, linscale=scaling, vmin=np.min(bc1-bc2), vmax=1)

        ax['1'].set(title="Raw Data: {} electrons".format(ecount),xlim=xlim,ylim=ylim)
        img0 = ax['1'].imshow(bc1,cmap=map, norm=lognorm1)
        fig.colorbar(img0,ax=ax['1'])

        # Correlated Datati 2023a
        lognorm2 = LogNorm(vmin=thresh, vmax=np.max(bc2))

        ax['2'].set(title="{} Coincidences".format(coin),xlim=xlim,ylim=ylim)
        img1 = ax['2'].imshow(bc2,cmap=map)
        fig.colorbar(img1,ax=ax['2'])

        # # Clustered Data
        # data3 = data_clust[self.coincidences(offset,delta,cluster=True)]
        # bc3 = np.histogram2d(data3[:,1],data3[:,0],range=[[0,256],[0,256]],bins=256)[0]/data3.shape[0]

        # lognorm3 = LogNorm(vmin=thresh, vmax=np.max(bc3))

        # ax['3'].set(title='Clustered data',xlim=xlim,ylim=ylim)
        # img3 = ax['3'].imshow(bc3,cmap=map)
        # fig.colorbar(img3,ax=ax['3'])

        # Difference
        symlognorm = SymLogNorm(linthresh=thresh, linscale=scaling, vmin=np.min(bc1-bc2), vmax=np.max(bc1-bc2))

        lognorm3 = LogNorm(vmin=thresh, vmax=np.max(np.abs(bc2-bc1)))
        ax['3'].set(title=r'$\tau = {}$ ns'.format(delta),xlim=xlim,ylim=ylim)
        img2 = ax['3'].imshow(bc2-bc1, cmap = map, vmax=0.002, vmin=-0.002)
        fig.colorbar(img2,ax=ax['3'])

        #display_diff(data_xy, data_xy[coins], axes[2])
        scalebar = ScaleBar(scale,units="um",dimension='si-length',length_fraction=0.25,location="lower left",box_alpha=0.0, scale_formatter=lambda value, unit: f"{value} {unit[:-1]}rad")
        #ax[1].add_artist(scalebar)

        file_name=name+"_"+str(delta/1e4)+"_donut.png"
        if (save == True) : plt.savefig(url / file_name)

        if show == True: plt.show()
        else: plt.close()

    def result_hist(path, parameters, name, hist, edge, delta, save = False, show = True, cluster = False):

        url = Path(path)

        # Find the bin centers
        bin_centers = 0.5 * (edge[1:] + edge[:-1])

        # Find the half-maximum value
        half_max = np.max(hist) / 2

        # Find the indices of the bins where the histogram values are closest to the half-maximum value
        left_idx = np.argmin(np.abs(hist[:len(hist)//2] - half_max))
        right_idx = np.argmin(np.abs(hist[len(hist)//2:] - half_max)) + len(hist)//2

        # Calculate the FWHM
        fwhm = edge[right_idx] - edge[left_idx]

        # Plot histogram
        plt.style.use('seaborn-whitegrid')
        cycler = plt.style.library['fivethirtyeight']['axes.prop_cycle']

        plt.rc('axes',  prop_cycle=cycler)
        plt.rc('lines', linewidth=2)
        plt.rc('image', cmap = 'Greys')
          
        plt.figure()
        plt.title('{}'.format(name)+' \n acquisition: {}s, range = {} ns'.format(parameters['tp3_exposure_time']*parameters['corr_sum'],delta/1e4)+', binning = {} ns, FWHM = {} ns'.format(2*delta/parameters['corr_binning']/1e4,fwhm/1e5))
        plt.plot(edge[:-1]/1e7, hist, drawstyle= 'steps-pre')
        plt.xlabel(r'$\Delta t \;/ \mathrm{\mu s}$')

        file_name=name+"_"+str(delta/1e4)+"_coin.png"
        if (save == True) : plt.savefig(url / file_name)

        if (save == True and cluster == True) : plt.savefig("..\\results\\"+name+"_"+str(delta/1e4)+"_coinc_clustered.png")

        if show == True: plt.show()
        else: plt.close()
        np.save(url / 'out_hist_{}.npy'.format(delta),hist)

    def result_coin(save_path, parameters, name, bc1, bc2, ecount, coin, delta, save = False, show = True, figsize=(12,3), thresh = 0.0001, scaling=1e-4, xlim=(0,255),ylim=(0,255)):
        
        url = Path(save_path)

        fig, ax = plt.subplot_mosaic([['1', '2', '3']],
                              constrained_layout=True,figsize=figsize)
        
        fig.suptitle(name)
        plt.grid(False)

        map='hot'
        

        # Raw Data

        #lognorm1 = LogNorm(vmin=thresh, vmax=100)
        lognorm1 = SymLogNorm(linthresh=thresh, linscale=scaling, vmin=np.min(bc1-bc2), vmax=1)

        ax['1'].set(title="Raw Data: {} electrons".format(ecount),xlim=xlim,ylim=ylim)
        img0 = ax['1'].imshow(bc1,cmap=map, norm=lognorm1)
        fig.colorbar(img0,ax=ax['1'])

        # Correlated Datati 2023a
        lognorm2 = LogNorm(vmin=thresh, vmax=np.max(bc2))

        ax['2'].set(title="{} Coincidences".format(coin),xlim=xlim,ylim=ylim)
        img1 = ax['2'].imshow(bc2,cmap=map)
        fig.colorbar(img1,ax=ax['2'])

        # # Clustered Data
        # data3 = data_clust[self.coincidences(offset,delta,cluster=True)]
        # bc3 = np.histogram2d(data3[:,1],data3[:,0],range=[[0,256],[0,256]],bins=256)[0]/data3.shape[0]

        # lognorm3 = LogNorm(vmin=thresh, vmax=np.max(bc3))

        # ax['3'].set(title='Clustered data',xlim=xlim,ylim=ylim)
        # img3 = ax['3'].imshow(bc3,cmap=map)
        # fig.colorbar(img3,ax=ax['3'])

        # Difference
        symlognorm = SymLogNorm(linthresh=thresh, linscale=scaling, vmin=np.min(bc1-bc2), vmax=np.max(bc1-bc2))

        lognorm3 = LogNorm(vmin=thresh, vmax=np.max(np.abs(bc2-bc1)))
        ax['3'].set(title=r'$\tau = {}$ ns'.format(delta),xlim=xlim,ylim=ylim)
        img2 = ax['3'].imshow(bc2-bc1, cmap = map, vmax=0.002, vmin=-0.002)
        fig.colorbar(img2,ax=ax['3'])

        #display_diff(data_xy, data_xy[coins], axes[2])
        scalebar = ScaleBar(scale,units="um",dimension='si-length',length_fraction=0.25,location="lower left",box_alpha=0.0, scale_formatter=lambda value, unit: f"{value} {unit[:-1]}rad")
        #ax[1].add_artist(scalebar)

        file_name=name+"_"+str(delta/1e4)+"_donut.png"
        if (save == True) : plt.savefig(url / file_name)

        if show == True: plt.show()
        else: plt.close()


class coin_backup():

    def __init__(self,path,name):
        self.path = path
        self.name = name

    def clustering(self,eps,min_samples,dataPercentage):
        path = self.path
        url = Path(path)
        # Load and sort all electron files in url
        e_files = list(url.glob('**/electrons/*.npy'))
        dummy = [str(f).split('\\')[-1].lower() for f in e_files]   # Convert to lower case
        dummy = [int(f.split('.')[0]) for f in dummy]   # Convert to lower case
        dummy = np.argsort(np.array(dummy))
        e_files = [e_files [i]for i in dummy]

        # Get lowest value for offset correction
        e0_toa =  np.load(str(e_files[0]))[1,:].min()*15625 # 0.1 ps Units!!!!!

        # Setup files for large data array
        file_xy = url / 'out_xy.npy'
        file_toa  = url / 'out_toa.npy'


        npaa_xy = NpyAppendArray(file_xy,delete_if_exists=True)
        npaa_toa = NpyAppendArray(file_toa,delete_if_exists=True)

        # Convert files: load -> sort by toa -> append  
        for file in e_files:
            arr_e = np.load(str(file))[:2,:]
            index = arr_e[1,:].argsort()
            npaa_xy.append(arr_e[0,:][index])
            npaa_toa.append((arr_e[1,:][index]*15625-e0_toa))

        npaa_xy.close()
        npaa_toa.close()

        # Load data
        data_xy=np.load(file_xy,mmap_mode="r")
        data_toa=np.load(file_toa,mmap_mode="r")

        inputData = np.array([data_xy,data_toa])

        print("Data to Cluster: {}".format(inputData.shape[1]))

        # prepare individual coordinate arrays
        endPoint = int(inputData[0].size * dataPercentage) # determine array limit based on data percentage

        xCoordinates = (inputData[0,:endPoint] % 256).astype(int)
        yCoordinates = (inputData[0,:endPoint] / 256).astype(int)
        tCoordinates = inputData[1,:endPoint]

        print("Reduced data: {}".format(tCoordinates.shape[0]))

        # create list of events 
        events = np.array(list(zip(xCoordinates,yCoordinates,tCoordinates)))

        # SCALE DATA
        # this should actually not really be necessary (will have to try without)
        scaler = StandardScaler()
        normalizedData = scaler.fit_transform(events)


        # Apply Clustering
        #model = DBSCAN(eps=0.002,min_samples=3,n_jobs=-1).fit(normalizedData)
        # labels = model.fit_predict(normalizedData)
        # n_jobs determines cpu-cure utulization (-1 uses all cores)
        labels = DBSCAN(eps=eps,min_samples=min_samples,n_jobs=-1).fit_predict(normalizedData)

        # create list of events with their classification
        #classifiedEvents = np.array(list(zip(xCoordinates,yCoordinates,tCoordinates,labels)))

        rogueEvents = events[labels == -1]
        electronEvents = events[labels != -1]
        np.save(url / 'out_clustered.npy',electronEvents)
        # xArea = (100,160)
        # yArea = (40,100)

        xArea = (0,255)
        yArea = (0,255)

        fig, ax = plt.subplots(1,2,figsize=(12,6))
        ax[0].set(title="Rogue Events",xlim=xArea,ylim=yArea)
        ax[0].scatter(rogueEvents[:,0],rogueEvents[:,1],s=0.5)

        ax[1].set(title="Electron (clustered) Events",xlim=xArea,ylim=yArea)
        ax[1].scatter(electronEvents[:,0],electronEvents[:,1],s=0.5)
        plt.show()
        

    def histogram(self,offset,delta,binning,split,save=False,cluster=False, show=False):
        path = self.path
        url = Path(path)
        name = self.name
        delta*=int(1e4)
        offset*=int(1e4)
        self.delta = delta
        self.offset = offset

        ##############################################################################################

        # data_list = list(url.glob('*.npz'))
        # dummy = [str(f).split('\\')[-1].lower() for f in data]   # Convert to lower case
        # dummy = [int(f.split('-')[0]) for f in dummy]   # Convert to lower case
        # dummy = np.argsort(np.array(dummy))
        # data_list = [data[i] for i in dummy]

        data_path=path / 'data'

        data_list = sorted(data_path.iterdir(), key=os.path.getctime)

        ##############################################################################################

        # Setup files for large data array
        file_xy = url / 'out_xy.npy'
        file_toa  = url / 'out_toa.npy'
        file_ph = url / 'out_ph.npy'

        npaa_xy = NpyAppendArray(file_xy,delete_if_exists=True)
        npaa_toa = NpyAppendArray(file_toa,delete_if_exists=True)
        npaa_ph = NpyAppendArray(file_ph,delete_if_exists=True)

        photon_offset = 0

        # Convert files: load -> sort by toa -> append  
        for file in data_list:
            arr_toa = np.load(str(file))['toas']
            arr_xy = np.load(str(file))['addresses']
            index = arr_toa[:].argsort()
            npaa_xy.append(arr_xy[:][index])
            npaa_toa.append((arr_toa[:][index]*15625))

            if (photon_offset == 0):
                photon_offset = np.load(str(file))['trigger'][0]
                print("Photon offset corrected: {}".format(photon_offset))

            arr1 = np.load(str(file))['photons1']
            if arr1.shape[0] > 0:
                arr1-=photon_offset
                arr1*=10

            arr2 = np.load(str(file))['photons2']
            if arr2.shape[0]>0:
                arr2-=photon_offset
                arr2*=10
            arr = np.concatenate((arr1, arr2))
            arr = arr[arr>0]
            npaa_ph.append(np.sort(arr))

        npaa_xy.close()
        npaa_toa.close()
        npaa_ph.close()

        if (cluster == True): data_toa=np.load(url / 'out_clustered.npy')[:,2]
        else: data_toa = np.load(file_toa, mmap_mode="r")

        data_ph = np.load(file_ph)

        ##############################################################################################

        # Processing parameters
        seconds = time.time() # for timing 
        #delta = 5000000 # histogram range in units of 0.1. ps  
        #split = 1000 # in how many chunks to split the data, affects performance and memory consumption!
        #offset = 0 #1300000 offset added to electron time 
        #binning = 1000 # resolution of the histogram = delta/binning

        #Setup for histogram
        bins = np.linspace(-delta, delta, binning+1)
        hist = np.zeros(binning)
        collected = 0

        # Correlate
        #@jit(nopython=True)
        @njit
        def correlating(data_toa, data_ph, hist, delta_t, collected, split, offset, bins):
            for ph_set in np.array_split(data_ph,split)[:]:

                if ph_set.shape[0] != 0:
                    es = data_toa[(data_toa>ph_set.min()-delta_t) & (data_toa<ph_set.max()+delta_t)]
                    arr_e = es - offset
                    arr_ph = ph_set
                    
                    diff_arr = (arr_e.reshape(-1,1) - arr_ph.reshape(-1,1).transpose()).ravel()
                    hist_data = diff_arr[(np.abs(diff_arr) < delta_t)]

                    collected += hist_data.size
                    _, edge = np.histogram(hist_data, bins = bins) 
                    hist += _
            return hist, edge, collected

        hist, edge, collected = correlating(data_toa, data_ph,hist, delta, collected, split, offset, bins)
        dt = time.time()-seconds

        print('delta = {} ns'.format(delta/1e4))
        print('split = {}'.format(split))
        print('binning = {} ns'.format(2*delta/binning/1e4))
        print('offset = {} ns'.format(offset/1e4))
        print('Collected {} data points in {:.2f} seconds = {:.2f} points per second'.format(collected,dt,collected/dt))

        # Plot histogram
        plt.style.use('seaborn-whitegrid')
        cycler = plt.style.library['fivethirtyeight']['axes.prop_cycle']

        plt.rc('axes',  prop_cycle=cycler)
        plt.rc('lines', linewidth=2)
        plt.rc('image', cmap = 'Greys')
          
        plt.figure()
        plt.title('{}'.format(name)+' \n range = {} ns'.format(delta/1e4)+', binning = {} ns, offset = {} ns, max coinc = {} ns'.format(2*delta/binning/1e4,offset/1e4,bins[hist.argmax()]/1e4))
        plt.plot(edge[:-1]/1e7, hist, drawstyle= 'steps-pre')
        plt.xlabel(r'$\Delta t \;/ \mathrm{\mu s}$')
        if (save == True) : plt.savefig(path+name+"_"+str(delta/1e4)+"_coinc.png")
        if (save == True and cluster == True) : plt.savefig("..\\results\\"+name+"_"+str(delta/1e4)+"_coinc_clustered.png")

        if show == True: plt.show()
        else: plt.close()
        return bins[hist.argmax()], hist


    def coincidences(self,offset,delta,cluster=False):
        path = self.path
        url = Path(path)
        delta*=int(1e4)
        offset*=int(1e4)
        self.delta = delta
        self.offset = offset

        # Load photon files 
        ph_file1 = list(url.glob('**/photons/*.1.npy'))
        ph_file2 = list(url.glob('**/photons/*.2.npy'))

        # Load data 
        file_xy = url / 'out_xy.npy'
        file_toa  = url / 'out_toa.npy'

        if (cluster == True): 
            data_toa=np.load(url / 'out_clustered.npy')[:,2]
            print("Clustered")
        else: 
            data_toa = np.load(file_toa, mmap_mode="r")
            print("Not clustered")

        data_ph_1 = np.load(str(ph_file1[0]))*10 # Photon time stamps are converted to units of 0.1 ps
        data_ph_1 = data_ph_1[data_ph_1>0] 
        data_ph_2 = np.load(str(ph_file2[0]))*10 # Photon time stamps are converted to units of 0.1 ps
        data_ph_2 = data_ph_2[data_ph_2>0]
        photons = np.concatenate((data_ph_1,data_ph_2))
        photons = np.sort(photons) 

        print('Number of electron events: {}'.format(data_toa.size))
        print('Number of photon events: {}'.format(photons.size))
        
        # Coincidence counting
        @njit
        def coincidence_counting(data_toa,photons,offset,delta):
            i,j = 0,0
            coins = np.empty((0,)) 
            while (i<data_toa.size-10) and (j<photons.size-10):
                diff = data_toa[i] - offset - photons[j]

                if np.abs(diff) < delta:
                    coins = np.append(coins,i)
                    i += 1
                    #j += 1
                elif diff >= 0:
                    j += 1
                else:
                    i += 1
            return coins

        coin = coincidence_counting(data_toa,photons,offset,delta)
        print('Found {} coincidences at {} offset \n'.format(len(coin),offset))

        return coin.astype(int)


    def plot(self,thresh,map,scale,delta,offset,save=False,cluster=False,figsize=(3,3),xlim=(0,255),ylim=(0,255)):
        path = self.path
        url = Path(path) 
        name = self.name
        scaling = 1e-4

        file_xy = url / 'out_xy.npy'
        data_xy = np.load(file_xy, mmap_mode="r")
        data_clust = np.load(url / 'out_clustered.npy')

        fig, ax = plt.subplot_mosaic([['1', '2', '3'],['4', '4', '4']],
                              constrained_layout=True,figsize=figsize)

        # Raw Data
        data1 = data_xy
        pic1 = np.array(data1)
        pic1[-1] = 256*256-1
        pic1 = pic1.astype(np.uint())
        bc1 = np.bincount(pic1).reshape(-1,256)/pic1.size

        lognorm1 = LogNorm(vmin=thresh, vmax=np.max(bc1))

        ax['1'].set(title="Raw Data",xlim=xlim,ylim=ylim)
        img0 = ax['1'].imshow(bc1,cmap=map, norm=lognorm1)
        fig.colorbar(img0,ax=ax['1'])

        # Correlated Data
        data2 = data_xy[self.coincidences(offset,delta)]
        pic2 = np.array(data2)
        pic2[-1] = 256*256-1
        pic2 = pic2.astype(np.uint())
        bc2 = np.bincount(pic2).reshape(-1,256)/pic2.size
        #bc2 = np.bincount(pic2).reshape(-1,256)
        #img = ax.imshow(bc1 -bc2, cmap = 'RdBu', vmin = -0.0025,vmax = 0.0025 )

        lognorm2 = LogNorm(vmin=thresh, vmax=np.max(bc2))

        ax['2'].set(title="Correlated Data",xlim=xlim,ylim=ylim)
        img1 = ax['2'].imshow(bc2,cmap=map)
        fig.colorbar(img1,ax=ax['2'])

        # Clustered Data
        data3 = data_clust[self.coincidences(offset,delta,cluster=True)]
        bc3 = np.histogram2d(data3[:,1],data3[:,0],range=[[0,256],[0,256]],bins=256)[0]/data3.shape[0]

        lognorm3 = LogNorm(vmin=thresh, vmax=np.max(bc3))

        ax['3'].set(title='Clustered data',xlim=xlim,ylim=ylim)
        img3 = ax['3'].imshow(bc3,cmap=map)
        fig.colorbar(img3,ax=ax['3'])

        # Difference
        symlognorm = SymLogNorm(linthresh=thresh, linscale=scaling, vmin=np.min(bc1-bc2), vmax=np.max(bc1-bc2))

        if cluster == True:
            ax['4'].set(title=r'Clustered: $\tau = {}$ ns; offset = {} ns'.format(delta,offset),xlim=xlim,ylim=ylim)
            img2 = ax['4'].imshow(bc1-bc3, cmap = map, norm = symlognorm)

        else: 
            ax['4'].set(title=r'$\tau = {}$ ns; offset = {} ns'.format(delta,offset),xlim=xlim,ylim=ylim)
            img2 = ax['4'].imshow(bc1-bc2, cmap = map, norm = symlognorm)
        fig.colorbar(img2,ax=ax['4'])

        #display_diff(data_xy, data_xy[coins], axes[2])
        #scalebar = ScaleBar(scale,units="um",dimension='si-length',length_fraction=0.25,location="lower left",box_alpha=0.0, scale_formatter=lambda value, unit: f"{value} {unit[:-1]}rad")
        #ax[1].add_artist(scalebar)
        
        if (save == True) : plt.savefig(path+name+"_"+str(delta)+"_donut.png")

        plt.show()