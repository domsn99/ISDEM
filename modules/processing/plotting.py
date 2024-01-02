from matplotlib import pyplot as plt
import numpy as np 


# Plot bidirectional histogram 
def plot_histogram(hist, edges):
    fig = plt.figure()
    plt.title('delta = {} ns'.format(edges.min()/1e4)+'\nbin width = {} ns\nmax coinc @ {} ns'.format((edges[1]-edges[0])/1e4,edges[hist.argmax()]/1e4))
    plt.plot(edges[:-1]/1e4, hist, drawstyle= 'steps-pre')
    plt.xlabel(r'$\Delta t \;/ \mathrm{\mu s}$')
    plt.xlim(edges.min()/1e4,edges.max()/1e4)
    return fig

def plot_hist_image(hist, fig, ax, scaling = 1,norm = None, label = None ):
    ax.imshow(hist, 
              extent = [0,255*scaling,0,255*scaling], 
              norm = norm,
              label = label)
    ax.grid(False)
    ax.set_xlim(scaling*(np.array([60,198])-0.5))
    ax.set_ylim(scaling*(np.array([60,198])-0.5))
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x / μrad')
    ax.set_ylabel('y / μrad')
    return 

def plot_image(x,y , fig, ax, scaling,norm = None, label = None ):
    ax.hist2d(scaling*x,
            scaling*y,
            bins = (scaling*(np.arange(0,257)-0.5),scaling*(np.arange(0,257)-0.5)),
            norm =norm,
            label = label)
    ax.set_xlim(scaling*(np.array([60,198])-0.5))
    ax.set_ylim(scaling*(np.array([60,198])-0.5))
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x / μrad')
    ax.set_ylabel('y / μrad')
    return 

def radius(x,y,center):
     return np.sqrt((x-center[0])**2+(y-center[1])**2)

def rad_hist_from_image(hist,center,edges):
      rad_hist = np.zeros(edges.shape[0]-1)
      for x in range(256):
        for y in range(256):
             if hist[x,y] != 0:
               rad_hist += np.histogram(radius(x,y,center)*np.ones(hist[x,y]), bins = edges)[0]
      return rad_hist, edges

def rad_hist_from_events(x,y, center, edges):
     data = radius(x,y,center)
     return np.histogram(data, bins = edges)
        

def plot_rad_hist(rad_hist,edges , fig, ax,  center, scaling, bins = np.arange(2,121,2), label = None, color = None):
        ax.fill_between(edges[:-1]*scaling,
                        rad_hist,
                        alpha = 0.5,
                        step = 'post',
                        #drawstyle = 'blah',
                        color = color,
                        label = label)

        ax.set(xlabel = 'r /μrad',
                ylabel = 'Counts',
                xlim = (0,edges[-1]*scaling))
        return fig, ax

hot_pixels = [(7,103),(178,226),(182,150),(114,160),(116,199),(103,198),(76,166),(55,203),(121,97),(249,56),(150,176),(197,223),(166,57),(124,229),(87,178),(89,170),(122,103),(165,155),(165,112),(114,116),(165,115)]

def mask_from_image(hist,hot_pixels = hot_pixels):
    '''
    hist
    '''
    hot_pixel_mask = np.full(hist.shape, False)
    for y,x in hot_pixels:
        hot_pixel_mask[x,y] = True
    return hot_pixel_mask

def mask_from_xylist(x,y, hot_pixels = hot_pixels):
    hot_pix_arr = np.array(hot_pixels)
    return np.bitwise_not(np.isin(x_coin, hot_pix_arr[:,1]) & np.isin(y_coin, hot_pix_arr[:,0])) 
