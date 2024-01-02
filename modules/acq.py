import numpy as np
import os
import shutil
import time
import multiprocessing as mp
from pathlib import Path

def createFolder(sum, path):
    for i in range(sum):
        folder_path = path + '\\' + str(i+1) + '\\data\\'
        try:
            shutil.rmtree(folder_path)
        except OSError as e:
            print(f"Error: {folder_path} : {e.strerror}")
        os.makedirs(folder_path)

def electron_acquisition(path, name, parameters, electrons, callback=None):
    folder_path = path / "data"
    
    if callback is None:
        def dataFunc(addresses, toas, tots, metadata):
            # Save data in file
            efilePrinter(addresses, toas, tots, metadata, path=folder_path)
        callback = dataFunc

    electrons.device.trigger.setShutterDuration(parameters['tp3_exposure_time'])
    electrons.device.chip.acq.setDataCallback(electrons.acq_ch, callback)

    electrons.device.chip.acq.begin(electrons.acq_ch)
    electrons.device.trigger.trigger()
    electrons.device.chip.acq.end(electrons.acq_ch)
    return

def efilePrinter(addresses, toas, tots, metadata, path = '..\\default\\data\\'):
    '''
    - This is a callback function for the TP3 acquisition
    - It only outputs electron data, no photons 
    - it saves the data as is in units of fast clockcycles 
    '''
    toas-=metadata['shutterStart']
    print("Saving: "+str(toas[0]*1.5625/10**9)+"-"+str(toas[-1]*1.5625/10**9))
    file_name = str(toas[0]*1.5625/10**9)+"-"+str(toas[-1]*1.5625/10**9)+".npz"
    url = path / file_name
    np.savez(url,addresses=addresses, toas=toas, tots=tots)

def correlation_acquisition(path, name, parameters, electrons, photons, callback=None):
    folder_path = path / "data"

    if callback is None:
        def dataFunc(addresses, toas, tots, metadata):
            # Save data in file
            corrfilePrinter(addresses, toas, tots, metadata, photons.stream.getData(), parameters, path=folder_path)
        callback = dataFunc

    electrons.device.trigger.setShutterDuration(parameters['tp3_exposure_time'])
    electrons.device.chip.acq.setDataCallback(electrons.acq_ch, callback)
    
    electrons.device.chip.acq.begin(electrons.acq_ch)
    time.sleep(0.1)
    electrons.device.trigger.trigger()
    electrons.device.chip.acq.end(electrons.acq_ch)
    return

def corrfilePrinter(addresses, toas, tots, metadata, photons, parameters, path = '..\\default\\data\\'):
    '''
    - This is a callback function for the TP3 acquisition module (<tp3_object>.device.chip.acq)
    - It outputs electron data as is in units of fast clockcycles 
    - It outputs photon data as is in units of ps  
    '''
    toas-=metadata['shutterStart']
    package = int(Path(path).parent.name)-1
    acq_time = package*int(parameters['tp3_exposure_time'])
    print("Saving: "+str(acq_time+toas[0]*1.5625/10**9)+"-"+str(acq_time+toas[-1]*1.5625/10**9))
    file_name = str(acq_time+toas[0]*1.5625/10**9)+"-"+str(acq_time+toas[-1]*1.5625/10**9)+".npz"
    url = path / file_name
    channels = photons.getChannels()
    timetags = photons.getTimestamps()

    trigger = timetags[channels == parameters['tt_trigger_channel']]

    data_ph1 = timetags[channels == parameters['SPDM1_channel']]
    data_ph2 = timetags[channels == parameters['SPDM2_channel']]
    np.savez(url,addresses=addresses, toas=toas, tots=tots, photons1=data_ph1, photons2=data_ph2, trigger=trigger)
    