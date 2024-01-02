from sklearn.cluster import DBSCAN
import numpy as np 
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def timeRecalibrationFactor(delta_t_ns):
    delta_t_100fs = delta_t_ns*10000
    return  1 / delta_t_100fs # delta_t is rescaled to 1

def unwrap_matrixID(matrixID):
    xCoordinates = (matrixID / 256).astype(int)
    yCoordinates = (matrixID % 256).astype(int)
    return xCoordinates,yCoordinates

def wrap_matrixID(x_arr,y_arr):
    matrixID = (x_arr*256 + y_arr).astype(int)
    return matrixID

def Rep_Func_default(class_arr):
    '''
    Input: 
        ndarray with shape (5,n), Row represent x,y,toa,tot,labels
    Output:
        ndarray with shape (4,n), Row represent x,y,toa,tot
        info ... not implemented 
    '''
    totlim = 30
    info = []

    columns = ['x','y','toa','tot','label']
    df = pd.DataFrame(class_arr.transpose(),columns = columns)
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
    
    df.sort_values(by=['toa'])
    rep_arr = np.array(rep_arr).transpose()
    return rep_arr, info

def clustering(data, 
               label_offset = 0, 
               eps= 2, 
               make_rep = True, 
               Rep_Func = Rep_Func_default, 
               timeRecalibrationFactor = 1 / 500000, #50 ns = 1 cluster time bin 
               normalize_t = 1 , 
               e0_toa = 0, 
               n_jobs = 3, # n_jobs determines cpu-cure utilization (-1 uses all cores)
               min_samples = 2):
    '''
    Compute the histogram of a dataset.

    Parameters
    ----------
    data : ndarray
        Input data. data.shape = (3,n): matrixID, toa in 0.1 ps units, tot in 25 ns units.
    e0_toa : int or float, optional
        trigger time to be subtracted in  0.1 ps units, default is 0
    normalize_t : int or float, optional
        Rescales the time data. This is needed if toa needs converting to units of 0.1 ps. Leave out otherwise, default = 1. 
    timeRecalibrationFactor : int or float, optional
        used to determine the time scale of one cluster, default is 1 / 500000 which corresponds to 1 delta_t = 50 ns

    Returns
    -------
    clustered_data : ndarray
        clustered_data.shape = (5,n): x,y, toa in 0.1 ps units, tot in 25 ns units, label.
    rep_arr : ndarray
        rep_arr.shape = (4,n): x,y, toa in 0.1 ps units, tot in 25 ns units.
    label_offset : int
        labels.max(), the last used label has to be passed on to do global labelling  
    '''
    # Sort by ToA, then normalize ToAs
    index = data[1,:].argsort()
    matrixID = data[0,:][index]
    timesoA = (data[1,:][index]*normalize_t - e0_toa)
    timesoT = data[2,:][index]
    
    # prepare individual coordinate arrays
    # tCoordinates normlized for clustering, for all other purposes use timesoA
    xCoordinates,yCoordinates = unwrap_matrixID(matrixID) 
    tCoordinates = timesoA * timeRecalibrationFactor 
    events = np.array(list(zip(xCoordinates,yCoordinates,tCoordinates)))

    # Apply Clustering
    # n_jobs determines cpu-cure utilization (-1 uses all cores)
    labels = DBSCAN(eps=eps,min_samples=min_samples,n_jobs=n_jobs).fit_predict(events)
    labels += label_offset + 1
    labels[labels == label_offset] = -1
    label_offset = labels.max()    

    # create list of events with their classification (full data, but classified into clusters)
    class_arr = np.array(list(zip(xCoordinates,yCoordinates,timesoA,timesoT,labels))).transpose()
    if make_rep:
        rep_arr, info = Rep_Func(class_arr)
    return class_arr, rep_arr, label_offset

def mask(data):
    return data, np.zeros(data.shape[1])

def MaskingAndClustering(cluster_parameters, e_files,global_labeling=False , load_func = None, save_func = None, n_jobs = 3):
    '''
    RunClustering applies the clustering algorithm  to a list of files
    It handles loading and saving the data 
    '''
    label_offset = 0
    for e_file in tqdm(e_files): 
        if load_func is None:
            with np.load(e_file) as e_file_npz:
                data = np.vstack([e_file_npz[j] for j in ['addresses', 'toas', 'tots']])
        else:
            data = load_func(e_file)
        
        data, masked = mask(data)

        class_arr, rep_arr, label_offset =  clustering(data,
                                                        label_offset = label_offset,
                                                        #Rep_Func = Rep_Func_test1,
                                                        eps = cluster_parameters['eps'], 
                                                        timeRecalibrationFactor = cluster_parameters['timeRecalibration'],
                                                        n_jobs = n_jobs,
                                                        normalize_t = 15625)
        if not global_labeling:
            label_offset = 0
            
        labels = class_arr[-1,:]
        
        if save_func is None:
            with np.load(e_file) as e_file_npz:
                np.savez(e_file,
                        addresses=e_file_npz['addresses'], 
                        toas=e_file_npz['toas'], 
                        tots=e_file_npz['tots'], 
                        labels =labels, 
                        photons1 = e_file_npz['photons1'], 
                        photons2=e_file_npz['photons2'], 
                        trigger=e_file_npz['trigger'],
                        clustered_xy = wrap_matrixID(rep_arr[0,:],rep_arr[1,:]),
                        clustered_toa =rep_arr[2],
                        clustered_tot=rep_arr[3])
        else:
            data = save_func(e_file, data, class_arr, rep_arr, cluster_parameters)
    return 


