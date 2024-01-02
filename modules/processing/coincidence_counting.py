import numpy as np

# Coincidence counting
#@njit
def coincidence_counting(data_toa,photons,offset,delta):
    i,j = 0,0
    coins = np.empty((0,),dtype = int) 
    while (i<data_toa.size-10) and (j<photons.size-10):
        diff = data_toa[i] - (photons[j] + offset)
        if np.abs(diff) < delta:
            coins = np.append(coins,i)
            i += 1
            #j += 1
        elif diff >= 0:
            j += 1
        else:
            i += 1
    return coins

def coincidence_counting_nearest(data_toa,photons,offset,delta):
    i,j = 0,0
    coins = np.empty((0,)) 
    while (i<data_toa.size-10) and (j<photons.size-10):
        diff = (data_toa[i]+ offset) - photons[j]
        if np.abs(diff) < delta:
            coins = np.append(coins,i)
            i += 1
            #j += 1
        elif diff >= 0:
            j += 1
        else:
            i += 1
    return coins, diffs