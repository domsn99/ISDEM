from scipy.ndimage import convolve1d
from scipy.stats import norm 

def smoothing(hist, gauss_std_ns = 6, bin_width_ns = 1.5625):
    '''
    Apply Gaussian smoothing to a histogram.

    This function convolves a histogram with a Gaussian filter to smooth the
    distribution. The Gaussian filter is created based on the specified standard
    deviation and bin width, and it's normalized so that its sum equals 1.

    Parameters
    ----------
    - hist (array_like): The input histogram to be smoothed. It should be a 3-dimensional array.
    - gauss_std_ns (float, optional): The standard deviation of the Gaussian filter in nanoseconds. 
                                      Default is 6 ns.
    - bin_width_ns (float, optional): The width of each bin in the histogram in nanoseconds. 
                                      Default is 1.5625 ns.

    Returns
    -------
    - array_like: The smoothed histogram, which has the same shape as the input histogram.
    
    The function uses the `convolve1d` method from `scipy.ndimage.filters` for convolution,
    with the 'nearest' mode to handle boundaries.
    '''
    gauss_filter = norm.pdf(np.arange(-10,10)*bin_width_ns, scale = gauss_std_ns)
    gauss_filter *= 1/gauss_filter.sum()
    return convolve1d(hist,weights=gauss_filter, mode = 'reflect')

