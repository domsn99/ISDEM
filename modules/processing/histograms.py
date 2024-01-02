import numpy as np

# Correlate
#@jit(nopython=True)
#@njit


def correlating(data_toa, photons, hist_range = 200, offset = 0, edges = None, split = 1000 ):
    '''
    Input:
        data_toa ... ndarray of electron toas in 0.1 ps units 
        photons ... ndarray of photon detection times in 0.1 ps units 
        range ... range of the histogram in units of ns, if 100 is given, the histogram will cover approx. (-100,100) 
        edges ... optional, default is 15625 ps bins centered around offset up to the range.   
        offset ... optional, center of the histogram, default is 0  

    Output:
        histogram
        edges  
    '''
    if edges is None:
        num_bins = 2*np.floor((hist_range*10000 - 15625/2)/15625) + 1
        edges = np.arange(num_bins+1)*15625 - num_bins*15625/2
    
    hist = np.zeros(edges.shape[0]-1)
    for ph_set in np.array_split(photons,split)[:]:
        if ph_set.shape[0] != 0:
            arr_e = data_toa[((data_toa + offset) > (ph_set.min()-hist_range*10000)) & ((data_toa + offset) < (ph_set.max()+hist_range*10000))]
            arr_e += offset
            arr_ph = ph_set
            
            diff_arr = (arr_e.reshape(-1,1) - arr_ph.reshape(-1,1).transpose()).ravel()        
            _, edge = np.histogram(diff_arr, bins = edges) 
            hist += _    
    return hist, edges

def correlating_3d(x,y,data_toa, photons, hist_range = 200, offset = 0, edges = None, split = 1000 ):
    '''
    Input:
        data_toa ... ndarray of electron toas in 0.1 ps units 
        photons ... ndarray of photon detection times in 0.1 ps units 
        range ... range of the histogram in units of ns, if 100 is given, the histogram will cover approx. (-100,100) 
        edges ... optional, default is 15625 ps bins centered around offset up to the range.   
        offset ... optional, center of the histogram, default is 0  

    Output:
        histogram
        edges  
    '''
    if edges is None:
        num_bins = 2*np.floor((hist_range*10000 - 15625/2)/15625) + 1
        edges = np.arange(num_bins+1)*15625 - num_bins*15625/2
        x_edges, y_edges = np.arange(0,257)-0.5, np.arange(0,257)-0.5
    
    hist = np.zeros((x_edges.shape[0]-1,y_edges.shape[0]-1,edges.shape[0]-1))
    for ph in photons[(photons > np.abs(edges[0]) + np.abs(offset)) & (photons < photons[-1]-np.abs(edges[0]) - np.abs(offset))]:
        cond = ((data_toa + offset) > (ph-hist_range*10000)) & ((data_toa + offset) < (ph+hist_range*10000))
        arr_e = data_toa[cond]
        arr_x = x[cond]
        arr_y = y[cond]
        
        arr_e += offset
        
        diff_arr = np.vstack([arr_x, arr_y, (arr_e - ph)]).transpose()        
        if diff_arr.shape[0]!=0:
            _, edge = np.histogramdd(diff_arr, bins = [x_edges, y_edges, edges]) 
            hist += _    
    return hist, edges

def correlating_2d_matrixID(matrixID,data_toa, photons, hist_range = 200, offset = 0, edges = None, split = 1000 ):
    '''
    Input:
        data_toa ... ndarray of electron toas in 0.1 ps units 
        photons ... ndarray of photon detection times in 0.1 ps units 
        range ... range of the histogram in units of ns, if 100 is given, the histogram will cover approx. (-100,100) 
        edges ... optional, default is 15625 ps bins centered around offset up to the range.   
        offset ... optional, center of the histogram, default is 0  

    Output:
        histogram
        edges  
    '''
    if edges is None:
        num_bins = 2*np.floor((hist_range*10000 - 15625/2)/15625) + 1
        edges = np.arange(num_bins+1)*15625 - num_bins*15625/2
        matrixID_edges= np.arange(0,256*256+1)-0.5
    
    hist = np.zeros((matrixID_edges.shape[0]-1,edges.shape[0]-1))
    for ph in photons[(photons > np.abs(edges[0]) + np.abs(offset)) & (photons < photons[-1]-np.abs(edges[0]) - np.abs(offset))]:
        cond = ((data_toa + offset) > (ph-hist_range*10000)) & ((data_toa + offset) < (ph+hist_range*10000))
        arr_e = data_toa[cond]
        arr_matrixID = matrixID[cond]

        arr_e += offset
        
        diff_arr = np.vstack([arr_matrixID, (arr_e - ph)]).transpose()        
        if diff_arr.shape[0]!=0:
            _, edge = np.histogramdd(diff_arr, bins = [matrixID_edges, edges]) 
            hist += _    
    return hist, edges

def rebin_histogram_subset(hist, edges, new_edges):
    """
    Rebin a histogram given by `hist` and `edges` to a new histogram with `new_edges`.
    This function assumes `new_edges` is a subset of `edges`.
    
    Parameters:
    - hist: array-like, the counts in each bin of the histogram.
    - edges: array-like, the edges of the bins in the original histogram.
    - new_edges: array-like, the edges of the bins in the rebinned histogram.
    
    Returns:
    - new_values: array, the counts in each bin of the rebinned histogram.
    """
    # Create an array to hold the new values
    new_values = []
    
    # Create a set for faster membership testing with new_edges
    new_edges_set = set(new_edges)
    
    # Initialize sum for the current new bin
    current_bin_sum = 0
    
    # Iterate over the original edges
    for i in range(len(edges) - 1):
        # Add the value to the current bin sum
        current_bin_sum += hist[i]
        
        # If the next edge is in the new_edges, it is the end of the current new bin
        if edges[i + 1] in new_edges_set:
            # Append the current sum to the new values
            new_values.append(current_bin_sum)
            # Reset the sum for the next new bin
            current_bin_sum = 0
    
    return np.array(new_values)