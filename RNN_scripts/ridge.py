# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 20:40:59 2025

@author: RHIRAsimulation
"""

import numpy as np
from scipy import linalg

def ridge_fast(yraw, Xraw, k):
    """
    Fast ridge regression for scalar or vector k.
    
    Parameters
    ----------
    yraw : array_like, shape (n, s)
        Response matrix (n samples, s responses).
    Xraw : array_like, shape (n, p)
        Predictor matrix.
    k : float or array_like, shape (L,)
        Ridge penalty(s).
    
    Returns
    -------
    B : ndarray, shape (p+1, s, L)
        Coefficients. B[0,:,i] is intercept for k[i], B[1:, :, i] are slopes.
    """
    # --- Center & scale ---
    Xraw = np.asarray(Xraw)
    yraw = np.asarray(yraw)
    n, p = Xraw.shape
    _, s = yraw.shape
    k_arr = np.atleast_1d(k).astype(float)
    L = k_arr.size

    Xmean = Xraw.mean(axis=0)
    Xstd  = Xraw.std(axis=0, ddof=0)
    ymean = yraw.mean(axis=0)

    X = (Xraw - Xmean) / Xstd
    Y = yraw - ymean

    # Precompute
    XtX = X.T @ X        # (p, p)
    Xty = X.T @ Y        # (p, s)

    # Prepare output
    B = np.empty((p+1, s, L), dtype=X.dtype)

    if L == 1:
        # --- scalar k: Cholesky solve ---
        k0 = k_arr[0]
        A = XtX + k0 * np.eye(p)
        # numpy.linalg.cholesky returns lower-triangular L such that A = L @ L.T
        Lmat = np.linalg.cholesky(A)
        # Solve Lmat @ Z = Xty  ->  Z = solve(Lmat, Xty)
        Z = linalg.solve_triangular(Lmat, Xty, lower=True)
        # Solve Lmat.T @ B0 = Z -> B0 = solve(Lmat.T, Z)
        B0 = linalg.solve_triangular(Lmat.T, Z, lower=False)

        # rescale slopes & compute intercept
        B1 = B0 / Xstd[:, None]
        intercept = ymean - (Xmean @ B1)

        B[0, :, 0]   = intercept
        B[1:, :, 0]  = B1

    else:
        # multiple k: choose eigen or SVD path
        if n < p:
            # SVD path: X = U S Vt
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            # U: (n,p), S: (p,), Vt: (p,p)
            # compute U.T @ Y once
            Uty = U.T @ Y    # (p,s)

            # for each k, B0 = V @ diag(S/(S^2 + k)) @ U.T @ Y
            # vectorize over k:
            S2 = S**2        # (p,)
            inv = (S / (S2[:,None] + k_arr[None,:]))  # (p, L)
            # middle: (p, s, L)
            middle = Uty[:,:,None] * inv[:,None,:]
            B0_all = (Vt.T[:,:,None] @ middle)  # (p, s, L)
        else:
            # eigen-decomp of XtX: XtX = Q Lambda Q.T
            lamb, Q = np.linalg.eigh(XtX)
            # (p,), (p,p)
            QXty = Q.T @ Xty                # (p, s)
            # build inv diag: 1/(lamb + k)
            inv = 1.0 / (lamb[:,None] + k_arr[None,:])  # (p, L)
            # (p, s, L)
            middle = QXty[:,:,None] * inv[:,None,:]
            B0_all = (Q[:,:,None] @ middle)   # (p, s, L)

        # rescale slopes
        B1_all = B0_all / Xstd[:,None,None]  # (p, s, L)
        # intercepts: ymean (1,s) minus Xmean(1,p) @ B1_all (p,s,L)
        intercepts = (ymean[None,:,None] - (Xmean[None,:,None] @ B1_all)).reshape(1,s,L)

        B[0:1,:,:]   = intercepts
        B[1:,:,:]    = B1_all

    return B


import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

def ridge_path_eval(X, Y, n_alphas=21, random_state=0):
    """
    Splits X,Y into train/validation halves, fits Ridge for alphas
    logarithmically spaced from 1e-10 to 1e+10, and returns:
      R   : p1×n_alphas matrix of Pearson r’s
      EV  : p1×n_alphas matrix of explained variances
      alphas: array of length n_alphas
    """
    # 1) define penalty grid
    alphas = np.logspace(-10, 10, n_alphas)
    
    # 2) split rows half/half
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.5, random_state=random_state, shuffle=True
    )
    
    p1 = Y.shape[1]
    R  = np.zeros((p1, n_alphas))
    EV = np.zeros((p1, n_alphas))
    
    # 3) loop over alphas
    for i, α in enumerate(alphas):
        model = Ridge(alpha=α, fit_intercept=True)
        model.fit(X_train, Y_train)           # multi‐target fit
        Y_pred = model.predict(X_val)         # shape (n/2 × p1)
        
        # 4) for each target dimension j compute:
        #    - Pearson r
        #    - explained variance
        for j in range(p1):
            # pearsonr returns (r, pval)
            r_val, _ = pearsonr(Y_val[:, j], Y_pred[:, j])
            R[j, i]  = r_val
            
            # explained variance: 1 − Var[y−ŷ]/Var[y]
            EV[j, i] = explained_variance_score(Y_val[:, j], Y_pred[:, j])
    
    return R, EV, alphas