import advs
import numpy as np


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


def main():
    device = advs.Device()

    # Custom logging function
    def logFunction(message, severity):
        if (severity < 7):  # Turn off DEBUG messages
            severityString = logSeverityDict[severity]
            print(severityString + ': ' + message)

    device.setLogFunction(logFunction)

    # Open device specified by a locator.
    device.open('usb:ix:0')

    boardCount = device.board.getCount()
    for i in range(boardCount):
        boardId = device.board.getId(i)
        if not device.board.isPowered(boardId):
            device.board.powerOn(boardId)

    device.chip.refresh()

    # device.config.load('0')

    # device.bias.setBias('0', 50)
    device.chip.setThreshold('0', 1)
    device.chip.selectMode('0:tpx3-dd-toa-tot')
    device.trigger.selectMode('software-start-shutter-duration')
    device.trigger.setShutterDuration(1)

    # Custom data function. Gets called on data from each block.
    # acq_data - numpy array of 128bit data packets (which contain address, toa, tot).
    # metadata - contains acquisition metadata including shutter start (with with 1.5625ns resolution)
    def dataFunc(acq_data, metadata):
        # Print address, toa (in seconds), tot of first hit in the block
        print("shutter start: " + str(metadata['shutterStart']*1.5625/10**9))
        print(acq_data[0]['address'], acq_data[0]
              ['toa']*1.5625/10**9, acq_data[0]['tot'])

    acq_ch = device.chip.acq.getChannelId(0)

    device.chip.acq.setDataCallback(acq_ch, dataFunc)

    device.chip.acq.begin(acq_ch)

    device.trigger.trigger()

    device.chip.acq.end(acq_ch)

    for i in reversed(range(boardCount)):
        boardId = device.board.getId(i)
        device.board.powerOff(boardId)
    device.close()


if __name__ == '__main__':
    main()
