import numpy as np
from modules.processing.histograms import correlating, correlating_3d
from modules.processing.coincidence_counting import coincidence_counting
from modules.processing.clustering import unwrap_matrixID
from tqdm import tqdm

def get_trigger(submeasurement_files):
    trigger_arr = np.hstack([np.load(file)['trigger'] for file in submeasurement_files])
    return trigger_arr.min(), trigger_arr

def load_data(e_file, trigger):
    '''
    Load Data from a .npz file
    '''
    with np.load(e_file) as e_file_npz:  
        x,y = unwrap_matrixID(e_file_npz['clustered_xy'])
        toas = e_file_npz['clustered_toa']
        photons1 = e_file_npz['photons1']
        if photons1.shape[0] > 0:
            photons1 -= trigger
            photons1 *= 10
        return x,y, toas, photons1
    
def load_data_corrected(e_file, trigger, correction_map):
    '''
    Load Data from a .npz file
    '''
    with np.load(e_file) as e_file_npz:  
        x,y = unwrap_matrixID(e_file_npz['clustered_xy'])
        corrections = correction_map[5,y]
        toas = e_file_npz['clustered_toa'] + corrections
        photons1 = e_file_npz['photons1']
        if photons1.shape[0] > 0:
            photons1 -= trigger
            photons1 *= 10
        return x,y, toas, photons1
    
def load_raw_data(e_file, trigger):
    '''
    Load Data from a .npz file
    '''
    with np.load(e_file) as e_file_npz:  
        x,y = unwrap_matrixID(e_file_npz['addresses'])
        toas = e_file_npz['toas']*15625 
        photons1 = e_file_npz['photons1']
        if photons1.shape[0] > 0:
            photons1 -= trigger
            photons1 *= 10
        return x,y, toas, photons1

def load_raw_data_corrected(e_file, trigger, correction_map):
    '''
    Load Data from a .npz file
    '''
    with np.load(e_file) as e_file_npz:  
        x,y = unwrap_matrixID(e_file_npz['addresses'])
        corrections = correction_map[5,y]
        toas = e_file_npz['toas']*15625  + corrections
        photons1 = e_file_npz['photons1']
        if photons1.shape[0] > 0:
            photons1 -= trigger
            photons1 *= 10
        return x,y, toas, photons1

def compute_histogram(submeasurement_files, submeasurement_path = None, load_data = load_data): 
    trigger,_ = get_trigger(submeasurement_files)
    hist, edges = correlating(np.array([0]), np.array([0]))

    for e_file in submeasurement_files:
        x, y, toas, photons1 = load_data(e_file, trigger)
        _, edges = correlating(toas, photons1)
        hist += _ 
    if submeasurement_path:
        np.savez(submeasurement_path/'histograms.npz', hist = hist, edges = edges)
    return hist, edges 

def compute_histogram3d(submeasurement_files, submeasurement_path = None, load_data = load_data): 
    trigger,_ = get_trigger(submeasurement_files)
    hist, edges = correlating_3d(np.array([0]),np.array([0]),np.array([0]), np.array([0]))

    for e_file in tqdm(submeasurement_files):
        x, y, toas, photons1 = load_data(e_file, trigger)
        _, edges = correlating_3d(x,y,toas, photons1)
        hist += _ 
    if submeasurement_path:
        np.savez(submeasurement_path/'histogram3d.npz', hist = hist, edges = edges)
    return hist, edges 


def compute_coincidence(submeasurement_files, offset, tau, submeasurement_path = None, load_data = load_data): 
    trigger, _ = get_trigger(submeasurement_files)
    x_coins, y_coins, toa_coins = [],[],[]

    for e_file in submeasurement_files:
        x, y, toas, photons1 = load_data(e_file, trigger)
        coins = coincidence_counting(toas,
                                    photons1,
                                    offset,
                                    tau)
        x_coins += [x[coins]]
        y_coins += [y[coins]]
        toa_coins += [toas[coins]]

    x_coins = np.concatenate(x_coins)
    y_coins = np.concatenate(y_coins)
    toa_coins = np.concatenate(toa_coins)

    coin_file_name = 'Tau{}ns_Off{}ns'.format(tau/1e4,offset/1e4)
    if submeasurement_path:
        np.savez(submeasurement_path/('coins'+ coin_file_name +'.npz'),x = x_coins, y = y_coins, toa_coins= toa_coins)
    return x_coins, y_coins, toa_coins

def compute_hist_raw(submeasurement_files, submeasurement_path = None, load_data = load_data): 
    trigger, _ = get_trigger(submeasurement_files)
    hist_raw = []

    for e_file in submeasurement_files:
        x, y, _, _ = load_data(e_file, trigger)
        dhist_raw, xedges,yedges = np.histogram2d(x,y, bins = (np.arange(0,257)-0.5,(np.arange(0,257)-0.5)))
        hist_raw += [dhist_raw]

    hist_raw = np.dstack(hist_raw)
    if submeasurement_path:
        np.savez(submeasurement_path/('hist_raw.npz'), hist_raw = hist_raw)
    return hist_raw, xedges,yedges

def compute_all(submeasurement_files, submeasurement_path, offset, tau, load_data = load_data):
    trigger, _ = get_trigger(submeasurement_files)

    hist, edges = correlating(np.array([0]), np.array([0]))
    x_coins, y_coins, toa_coins = [],[],[]
    hist_raw = []

    for e_file in submeasurement_files:
        x, y, toas, photons1 = load_data(e_file, trigger)
        
        _, edges = correlating(toas, photons1)
        hist += _
        
        coins = coincidence_counting(toas,
                                    photons1,
                                    offset,
                                    tau)
        x_coins += [x[coins]]
        y_coins += [y[coins]]
        toa_coins += [toas[coins]]

        dhist_raw, xedges,yedges = np.histogram2d(x,y, bins = (np.arange(0,257)-0.5,(np.arange(0,257)-0.5)))
        hist_raw += [dhist_raw]


    x_coins = np.concatenate(x_coins)
    y_coins = np.concatenate(y_coins)
    toa_coins = np.concatenate(toa_coins)
    
    hist_raw = np.dstack(hist_raw)

    coin_file_name = 'Tau{}ns_Off{}ns'.format(tau/1e4,offset/1e4)
    np.savez(submeasurement_path/('coins'+ coin_file_name +'.npz'),x = x_coins, y = y_coins, toa_coins= toa_coins)
    np.savez(submeasurement_path/('hist_raw.npz'), hist_raw = hist_raw)
    np.savez(submeasurement_path/'histograms.npz', hist = hist, edges = edges)
    return