# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 11:53:40 2024

@author: Hiroto
"""

import numpy as np

def get_cat_error(predall1, classall1, predall2, classall2, group_bound):
    """
    Calculate classification error metrics based on predictions and class labels.
    
    Parameters:
    predall1 : array_like
        Predictions from the first set.
    classall1 : array_like
        True classes for the first set.
    predall2 : array_like
        Predictions from the second set.
    classall2 : array_like
        True classes for the second set.
    
    Returns:
    aaa : ndarray
        Normalized counts of misclassifications and correct classifications in different categories.
    """
    
    M2PPCmiscat = np.sum(((predall1 > group_bound) & (classall1 <= group_bound) | (predall1 <= group_bound) & (classall1 > group_bound)) &
                         ((predall2 > group_bound) & (classall2 <= group_bound) | (predall2 <= group_bound) & (classall2 > group_bound)))

    M2miscatPPCnomis = np.sum(((predall1 > group_bound) & (classall1 <= group_bound) | (predall1 <= group_bound) & (classall1 > group_bound)) &
                              ((predall2 <= group_bound) & (classall2 <= group_bound) | (predall2 > group_bound) & (classall2 > group_bound)))

    M2nomisPPCmiscat = np.sum(((predall1 <= group_bound) & (classall1 <= group_bound) | (predall1 > group_bound) & (classall1 > group_bound)) &
                              ((predall2 <= group_bound) & (classall2 > group_bound) | (predall2 > group_bound) & (classall2 <= group_bound)))

    M2nomisPPCnomis = np.sum(((predall1 <= group_bound) & (classall1 <= group_bound) | (predall1 > group_bound) & (classall1 > group_bound)) &
                             ((predall2 <= group_bound) & (classall2 <= group_bound) | (predall2 > group_bound) & (classall2 > group_bound)))

    total = M2PPCmiscat + M2miscatPPCnomis + M2nomisPPCmiscat + M2nomisPPCnomis

    # Return the normalized counts
    aaa = np.array([M2PPCmiscat, M2miscatPPCnomis, M2nomisPPCmiscat, M2nomisPPCnomis]) / total

    return aaa



from get_phase import get_phase

def get_temp_error(predall1, classall1, predall2, classall2, Groupnum18,corrrange):
    """
    Calculate the temporal error rate based on the differences between predictions and classes.
    
    Parameters:
    predall1 : array_like
        Predictions for the first set.
    classall1 : array_like
        True classes for the first set.
    predall2 : array_like
        Predictions for the second set.
    classall2 : array_like
        True classes for the second set.
    Groupnum18 : int
        Period or group number used in the get_phase function.
    
    Returns:
    error_rate : ndarray
        The error rate for each group (1 to 18).
    """


    # Calculate phase differences and errors
    preddiff = np.round(get_phase(predall1 - predall2, Groupnum18, 'int'))
    prederror1 = np.round(get_phase(predall1 - classall1, Groupnum18, 'int'))
    prederror2 = np.round(get_phase(predall2 - classall2, Groupnum18, 'int'))

    error = np.zeros(Groupnum18)
    offset_ind=np.ceil(Groupnum18/2)-1
    for i in range(Groupnum18):
        error[i] = np.sum((preddiff == i - offset_ind) & (np.abs(prederror1) >= corrrange) & (np.abs(prederror2) >= corrrange))

    sumall = np.sum(error)
    error_rate = error / sumall if sumall != 0 else error

    return error_rate
