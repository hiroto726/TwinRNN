import numpy as np
def pairwise_corr(A, B):
    """
    Computes the column-wise Pearson correlation coefficient matrix between two matrices A and B.
    
    Parameters:
    A : ndarray of shape (n, m)
        A matrix where each column represents a variable, and each row is an observation.
    B : ndarray of shape (n, p)
        Another matrix with the same number of rows (observations) but possibly different columns.
    
    Returns:
    corr : ndarray of shape (m, p)
        A matrix where element (i, j) is the Pearson correlation coefficient between A[:, i] and B[:, j].
    """
    # Number of rows (observations)
    n = A.shape[0]

    # Center the data by subtracting the column means
    A_centered = A -np.mean(A, axis=0)
    B_centered = B -np.mean(B, axis=0)
    

    # Compute covariance between columns:
    # This gives an m x p matrix where element (i,j) is the covariance between A[:,i] and B[:,j]
    cov = np.dot(A_centered.T, B_centered) / (n - 1)

    # Compute the standard deviations (using ddof=1 for sample standard deviation)
    std_A = np.std(A_centered, axis=0, ddof=1)
    std_B = np.std(B_centered, axis=0, ddof=1)

    # Use broadcasting to divide each covariance by the product of corresponding std devs
    corr = cov / np.outer(std_A, std_B)
    
    return corr