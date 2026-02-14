# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 11:18:17 2024

@author: Hiroto
"""

import numpy as np

def get_phase(xin, N, arg):
    """
    Calculate the phase of xin with respect to the period N, 
    ranging from -N/2 to N/2.

    Parameters:
    xin : array_like
        Input can be any arbitrary number, vector, or matrix.
    N : float
        The period (non-zero real number).
    arg : str
        Determines the output ('int' or 'one').

    Returns:
    a : ndarray
        Phase of xin based on the period N.
    """

    x = np.ravel(xin)
    b = np.column_stack((x/N - np.floor(x/N), x/N - np.ceil(x/N)))

    ind = np.argsort(np.abs(b), axis=1)
    out = b[np.arange(b.shape[0]), ind[:, 0]].reshape(xin.shape)

    if arg == 'int':
        a = N * out
    elif arg == 'one':
        a = out
    else:
        raise ValueError("arg must be 'int' or 'one'")
    
    return a
