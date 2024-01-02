import TimeTagger as TT
import shutil
import numpy as np
import os

def setup(parameters):
    tagger = TT.createTimeTagger()
    tagger.setTriggerLevel(parameters['tt_trigger_channel'],
                        parameters['tt_trigger_lvl'])
    tagger.setTriggerLevel(parameters['tt_clock_channel'],
                        parameters['tt_clock_lvl'])
    tagger.setTriggerLevel(parameters['SPDM1_channel'],
                        parameters['SPDM_lvl'])
    tagger.setTriggerLevel(parameters['SPDM2_channel'],
                        parameters['SPDM_lvl'])
    #tagger.setEventDivider(parameters['tt_clock_channel'],16)
    tagger.setSoftwareClock(parameters['tt_clock_channel'],
                            parameters['tp3_clockout_frequency'])

    print('Software Clock input channel = {}'.format(tagger.getSoftwareClockState().input_channel))
    print('Software Clock enabled = {}'.format(tagger.getSoftwareClockState().enabled))
    print('Software Clock locked  = {}'.format(tagger.getSoftwareClockState().enabled))
    return tagger

class tag:
    def __init__(self,tagger,parameters):
        self.parameters = parameters
        self.tagger = tagger
        self.trigger = 0
        return

    def measure(self,event_buffer_size = 100000000):
        self.stream = TT.TimeTagStream(tagger = self.tagger,
                                                n_max_events = event_buffer_size,
                                                channels = [self.parameters[key] for key in ['tt_trigger_channel',
                                                                                       'SPDM1_channel',
                                                                                       'SPDM2_channel']])
        return
    
    def get_trigger(self):
        return self.trigger
    
    def getData(self):
        return self.stream.getData().getTimestamps()
    
    def getData2(self):
        data = self.stream.getData()
        channels = data.getChannels()
        timetags = data.getTimestamps()

        if self.trigger is None:
            self.trigger = timetags[channels == self.parameters['tt_trigger_channel']]

        tt_SPDM1 = timetags[channels ==  self.parameters['SPDM1_channel']]
        tt_SPDM2 = timetags[channels == self.parameters['SPDM2_channel']]
        return tt_SPDM1, tt_SPDM2, self.trigger
    
    def get_Data(self):
        self.data = self.stream.getData()
        self.stream.stop()
        self.stream.clear()
        #print('Stream is running = {}'.format(self.stream.isRunning()))
        print('Number of Time Tag = {}'.format(self.data.getTimestamps().shape))
        self.timetags1, self.timetags2, self.trigger_time = self.evaluate(self.data, self.parameters)
        #self.timetags1, self.timetags2 = normalize(self.data, self.parameters)
        return self.timetags1, self.timetags2, self.trigger_time
    
    def dumpData(self):
        self.data = self.stream.getData()
        print('Number of Time Tag = {}'.format(self.data.getTimestamps().shape))
        timetags1, timetags2, trigger_time = self.evaluate(self.data, self.parameters)
        return 

    def normalize(data, parameters):
        trigger_time = data.getTimestamps()[data.getChannels()== parameters['tt_trigger_channel']]
        if trigger_time.size == 1:
            timetags1 = data.getTimestamps()[data.getChannels()== parameters['SPDM1_channel']] - trigger_time
            timetags2 = data.getTimestamps()[data.getChannels()== parameters['SPDM2_channel']] - trigger_time
            return timetags1,timetags2
        elif trigger_time.size == 0:
            raise ValueError('Error: No Trigger signals detected')
        else:
            raise ValueError('Error: Multipe Trigger signals detected {}'.format(trigger_time.size))

    def evaluate(self, data, parameters):
        event_buffer_size = 100000000
        if data.size == event_buffer_size:
            print('TimeTagStream buffer is filled completely. Events arriving after the buffer has been filled have been discarded. Please increase the buffer size not to miss any events.')
        print('Collected time stamps: {}'.format((data.getEventTypes() == 0).sum()))
        print('---------------------------------------')
        print('Event types:')
        print('0 TimeTag | 1 Error | 2 OverflowBegin | 3 OverflowEnd | 4 MissedEvents')
        print(np.bincount(data.getEventTypes()))
        print('Missed events: {}'.format(data.getMissedEvents()))

        channels = data.getChannels()
        timetags = data.getTimestamps()

        tt_trigger = timetags[channels == parameters['tt_trigger_channel']]
        tt_SPDM1 = timetags[channels ==  parameters['SPDM1_channel']]
        tt_SPDM2 = timetags[channels == parameters['SPDM2_channel']]
        print('Trigger time stamps: {}'.format(tt_trigger))
        if tt_trigger.size == 1:
            tt_SPDM1 -= tt_trigger[0]
            tt_SPDM2 -= tt_trigger[0]
            print('Calculating reduced time tags')
        elif tt_trigger.size == 0:
            raise ValueError('Error: No Trigger signals detected')
        else:
            raise ValueError('Error: Multipe Trigger signals detected {}'.format(tt_trigger.size))
        return tt_SPDM1,tt_SPDM2, tt_trigger

    def photon_filePrinter(tags_SPDM1, tags_SPDM2,path):
            folder_path = path + "\\data\\photons"
            try:
                url = folder_path + '\\' + str(tags_SPDM1[0])+"."+str(tags_SPDM1[-1])
                np.save(url+'.1.npy',tags_SPDM1)
                np.save(url+'.2.npy',tags_SPDM2)
                print("Photon data saved")
            except: print("Couldn't save photon data")
            return
    
    def prep_photon_folder(self,path):
        self.path = path
        folder_path = self.path + "\\data\\photons"
        try:
            shutil.rmtree(folder_path)
        except OSError as e:
            print(f"Error: {folder_path} : {e.strerror}")
        os.makedirs(folder_path)
        return

    def save_data(self,path = '..'):
        self.prep_photon_folder(path)
        tag.photon_filePrinter(self.timetags1, self.timetags2, path)
        return

    def acquire():
        print("To Do")
        return
    
    def __del__(self):
        TT.freeTimeTagger(self.tagger)
        return
    
    def shutdown(self):
        TT.freeTimeTagger(self.tagger)
        return