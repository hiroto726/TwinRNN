import numpy as np

def confscore(confmatrix, Range):
    """
    This function returns the rate of correct prediction of a confusion matrix.
    The confusion matrix can be 3D, with each page being a confusion matrix.
    
    Parameters:
    confmatrix (numpy.ndarray): 3D confusion matrix.
    Range (int): Scalar value representing the range.
    
    Returns:
    numpy.ndarray: The score of the confusion matrix with size of (np.shape(confmatrix)[2])
    """
    if confmatrix.ndim==2:
        confmatrix=confmatrix[:,:,np.newaxis]
    # Normalize the confusion matrix
    confmatrix2 = confmatrix / np.sum(confmatrix, axis=(0, 1))
    confmatrix2=np.transpose(confmatrix2,axes=(2,0,1))
    
    # Create the cormat matrix
    cormat = np.zeros((confmatrix.shape[0], confmatrix.shape[1]))
    for ii in np.arange(1, Range * 2 + 2):
        cormat += np.roll(np.eye(confmatrix2.shape[-1]), ii - 1 - Range, axis=0)
    
    # Calculate the score
    Score = cormat[np.newaxis,:, :] * confmatrix2
    Score = np.squeeze(np.sum(Score, axis=(1, 2)))
    return Score


def confscore_axis0(confmatrix, Range):
    """
    Compute correct‐prediction rates when each confmatrix[i,:,:] is one confusion matrix.
    
    Parameters:
    -----------
    confmatrix : np.ndarray
        Either shape (nClasses, nClasses) for a single matrix, or
        shape (nMats, nClasses, nClasses) when you have multiple.
    Range : int
        How far off–diagonals you still count as “correct” (in each direction).
        
    Returns:
    --------
    score : np.ndarray
        1D array of length nMats giving the normalized “correct‐prediction” rate
        within +/- Range of the diagonal.
    """
    # ensure shape (nMats, C, C)
    if confmatrix.ndim == 2:
        confmatrix = confmatrix[np.newaxis, :, :]
    elif confmatrix.ndim != 3:
        raise ValueError("confmatrix must be 2D or 3D")
    
    nMats, C, C2 = confmatrix.shape
    if C != C2:
        raise ValueError("confmatrix must be square along last two dims")
    
    # normalize each matrix so sum over each is 1
    sums = confmatrix.sum(axis=(1, 2), keepdims=True)
    conf_norm = confmatrix / sums  # shape (nMats, C, C)
    
    # build the “correct‐region” mask matrix of shape (C, C)
    #  where entries within +/-Range of the diagonal are 1, else 0
    mask = np.zeros((C, C), dtype=float)
    for shift in range(-Range, Range+1):
        mask += np.eye(C, k=shift)
    
    # broadcast multiply and sum
    #  → shape (nMats,)
    score = (conf_norm * mask[np.newaxis, :, :]).sum(axis=(1, 2))
    return score


def confscore_last2(confmatrix, Range):
    """
    Compute “correct‐prediction” rates treating the last two axes of `confmatrix`
    as individual confusion matrices.

    Parameters
    ----------
    confmatrix : array-like, shape (..., C, C)
        A single confusion matrix of shape (C, C), or a stack of them of shape
        (..., C, C).  The final two dimensions must be square.
    Range : int
        How many off‐diagonals on each side to count as “correct”.
        Range=0 uses only the main diagonal; Range=1 includes the 1st off‐diagonals, etc.

    Returns
    -------
    score : numpy.ndarray, shape (...)
        The “correct‐prediction” rate for each matrix, with the same leading shape
        as `confmatrix.shape[:-2]`.
    """
    arr = np.asarray(confmatrix, dtype=float)
    if arr.ndim < 2:
        raise ValueError("confmatrix must have at least two dimensions")
    C1, C2 = arr.shape[-2], arr.shape[-1]
    if C1 != C2:
        raise ValueError("Last two dimensions of confmatrix must be equal (square matrices)")

    # Flatten all leading dims into one batch dimension
    lead_shape = arr.shape[:-2]
    mats = arr.reshape(-1, C1, C1)  # shape = (N, C, C)
    N = mats.shape[0]

    # Normalize each matrix so its entries sum to 1
    sums = mats.sum(axis=(1,2), keepdims=True)
    mats_norm = mats / sums

    # Build mask of size (C, C) that is 1 on main diagonal ± Range
    mask = np.zeros((C1, C1), dtype=float)
    for shift in range(-Range, Range+1):
        mask += np.eye(C1, k=shift)

    # Compute score for each matrix
    #   multiply elementwise and sum over the C×C plane
    scores = (mats_norm * mask[np.newaxis, :, :]).sum(axis=(1,2))

    # Reshape back to the original leading shape
    return scores.reshape(lead_shape)



def confscore_last2_nan(confmatrix, Range):
    """
    Compute “correct‐prediction” rates treating the last two axes of `confmatrix`
    as individual confusion matrices, ignoring any NaNs.

    Parameters
    ----------
    confmatrix : array-like, shape (..., C, C)
        One or more confusion matrices.
    Range : int
        How many off‐diagonals on each side to count as “correct”.

    Returns
    -------
    score : numpy.ndarray, shape (...,)
        The “correct‐prediction” rate for each matrix.
    """
    arr = np.asarray(confmatrix, dtype=float)
    if arr.ndim < 2:
        raise ValueError("confmatrix must have at least two dimensions")
    C1, C2 = arr.shape[-2], arr.shape[-1]
    if C1 != C2:
        raise ValueError("Last two dimensions of confmatrix must be square")

    # Flatten leading dims
    lead_shape = arr.shape[:-2]
    mats = arr.reshape(-1, C1, C1)  # (N, C, C)

    # Normalize each matrix so its (non-NaN) entries sum to 1
    sums = np.nansum(mats, axis=(1,2), keepdims=True)              # shape (N,1,1)
    mats_norm = mats / sums                                       # NaNs propagate if all-NaN

    # Build mask of 1s on diagonal ± Range
    mask = sum(np.eye(C1, k=shift) for shift in range(-Range, Range+1))

    # Compute nan‐aware sum over correct entries
    scores = np.nansum(mats_norm * mask[np.newaxis, :, :], axis=(1,2))

    # Restore leading shape
    return scores.reshape(lead_shape)


def ensure_column_matrix(arr):
    if arr.ndim == 1:
        return np.atleast_2d(arr).T  # Convert 1D array to (n, 1)
    return arr  # Leave multi-dimensional arrays unchanged

def confmat(Class,Pred):
    Class=ensure_column_matrix(Class)
    Pred=ensure_column_matrix(Pred)
    class_num=int(np.max(Class))+1
    Correct=np.zeros((np.shape(Class)[0],class_num))
    Predict=np.zeros((np.shape(Class)[0],class_num))
    for i in np.arange(class_num):
        Correct[:,i]=np.squeeze(1*(Class==i))
        Predict[:,i]=np.squeeze(1*(Pred==i))
    
    confmatrix=np.matmul(np.transpose(Predict),Correct)
    return confmatrix

def confmat_num(Class,Pred, class_num):
    Class=ensure_column_matrix(Class)
    Pred=ensure_column_matrix(Pred)
    Correct=np.zeros((np.shape(Class)[0],class_num))
    Predict=np.zeros((np.shape(Class)[0],class_num))
    for i in np.arange(class_num):
        Correct[:,i]=np.squeeze(1*(Class==i))
        Predict[:,i]=np.squeeze(1*(Pred==i))
    
    confmatrix=np.matmul(np.transpose(Predict),Correct)
    return confmatrix

