import numpy as np 

def load_func_corrected(e_file, correction_map):
    with np.load(e_file) as e_file_npz:
        x, y = unwrap_matrixID(e_file_npz['addresses'])
        matrixID = e_file_npz['addresses']
        toa = e_file_npz['toas'] + correction_map[x,y]/15625
        tot = e_file_npz['tots']
    return np.vstack([matrixID,toa,tot])