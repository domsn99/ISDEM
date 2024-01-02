from modules.acq import *
import advs
import time
import shutil
import os

logSeverityDict = {
        0: 'EMERG',
        1: 'ALERT',
        2: 'CRIT',
        3: 'ERR',
        4: 'WARNING',
        5: 'NOTICE',
        6: 'INFO',
        7: 'DEBUG',
        }


def logFunction(message, severity):
    if (severity < 0):  # Turn off debug messages
        severityString = logSeverityDict[severity]
        print(severityString + ': ' + message)

# def filePrinter(addresses, toas, tots,path = '..'):
#     url = path +"\\data\\electrons\\"+str(int(toas[0]*1562.5))+"."+str(int(toas[np.size(toas)-1]*1562.5))+".npy"
#     np.save(url,[addresses, toas, tots])


class tp3:

    def __init__(self, parameters):
        print(logFunction)
        print(advs.getRuntimeVersion())
        print(advs.getCompileTimeVersion())

        self.device = advs.Device()
        self.device.setLogFunction(logFunction)
        #self.device.open('usb:sn:9F12C985')
        self.device.open('usb:ix:0')

        boardCount = self.device.board.getCount()
        for i in range(boardCount):
            boardId = self.device.board.getId(i)
            if not self.device.board.isPowered(boardId):
                self.device.board.powerOn(boardId)

        self.device.chip.refresh()
        # self.device.config.load('0')

        self.device.trigger.selectMode('software-start-shutter-duration')
        self.device.chip.selectMode('0:tpx3-dd-toa-tot-vec')
        self.device.chip.setThreshold('0', parameters['tp3_threshold_voltage'])

        ## External clock/trigger settings
        self.device.extsync.selectOption('clock-output:yes')
        self.device.extsync.selectOption('trigger-out:level')
        self.device.extsync.selectOption('input-polarity:active-high')
        self.device.extsync.selectOption('output-polarity:active-low')

        self.acq_ch = self.device.chip.acq.getChannelId(0)

        self.sleep = parameters['tp3_sleep_time']
        return 

    def shutdown(self):
        boardCount = self.device.board.getCount()
        for i in reversed(range(boardCount)):
            boardId = self.device.board.getId(i)
            self.device.board.powerOff(boardId)
        self.device.close()
        return 
    
    def acquire(self, exposure_time = 1,callback = None, path = '..\\',name='default'):
        folder_path = path + name + "\\data\\electrons"
        try:
            shutil.rmtree(folder_path)
        except OSError as e:
            print(f"Error: {folder_path} : {e.strerror}")
        os.makedirs(folder_path)
        
        # if callback is None:
        #     def dataFunc(addresses, toas, tots):
        #         # Save adresses, toas and tots in file
        #         filePrinter(addresses, toas, tots, path = path)
        #     callback = dataFunc

        if callback is None:
            def dataFunc(acq_data, metadata):
                # Save adresses, toas and tots in file
                filePrinter(acq_data, metadata, path=path)
            callback = dataFunc

        self.device.trigger.setShutterDuration(exposure_time)
        self.device.chip.acq.setDataCallback(self.acq_ch, callback)
        self.device.chip.acq.begin(self.acq_ch)
        time.sleep(self.sleep)
        self.device.trigger.trigger()
        self.device.chip.acq.end(self.acq_ch)
        return

    def __del__(self):
        self.shutdown()
        return