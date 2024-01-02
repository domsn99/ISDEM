import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
import time

from modules.processing.save_and_load import * 
from modules.processing.clustering import * 
from modules.processing.histograms import * 
from modules.processing.plotting import * 
from modules.processing.coincidence_counting import *
from modules.processing.submeasurement_processing import *

def docstring_example():
    '''
    Compute the histogram of a dataset.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars or str, optional
        description

    Returns
    -------
    hist : array
        description
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.


    See Also
    --------
    histogramdd, bincount, searchsorted, digitize, histogram_bin_edges

    Notes
    -----

    Examples
    --------
    '''
    return
