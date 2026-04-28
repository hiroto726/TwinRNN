# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 00:44:43 2025

@author: Hiroto
"""

import numpy as np
from sklearn.cross_decomposition import CCA

class CCA_SVD:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        # Store means for centering later.
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = Y.mean(axis=0)
        Xc = X - self.x_mean_
        Yc = Y - self.y_mean_

        # Compute covariance matrices.
        Cxx = Xc.T @ Xc
        Cyy = Yc.T @ Yc
        Cxy = Xc.T @ Yc

        # Compute the inverse square root of Cxx using SVD.
        Ux, sx, _ = np.linalg.svd(Cxx)
        A = Ux @ np.diag(1.0 / np.sqrt(sx)) @ Ux.T

        # Compute the inverse square root of Cyy using SVD.
        Uy, sy, _ = np.linalg.svd(Cyy)
        B = Uy @ np.diag(1.0 / np.sqrt(sy)) @ Uy.T

        # Form the matrix for SVD.
        M = A @ Cxy @ B

        # SVD on the whitened cross-covariance.
        U, s, Vt = np.linalg.svd(M)
        # Keep only the desired number of components.
        U = U[:, :self.n_components]
        Vt = Vt[:self.n_components, :]
        s = s[:self.n_components]
        self.canonical_correlations_ = s

        # Canonical weights in the original space.
        self.x_weights_ = A @ U         # shape: (n_features_x, n_components)
        self.y_weights_ = B @ Vt.T        # shape: (n_features_y, n_components)

        # For prediction, compute loadings that map from canonical scores to Y.
        Y_scores = Yc @ self.y_weights_
        # Solve the least-squares problem: Y_scores * y_loadings = Yc.
        #self.y_loadings_, _, _, _ = np.linalg.lstsq(Y_scores, Yc, rcond=None)

        return self

    def transform(self, X, Y=None):
        X = np.asarray(X)
        Xc = X - self.x_mean_
        X_scores = Xc @ self.x_weights_
        if Y is not None:
            Y = np.asarray(Y)
            Yc = Y - self.y_mean_
            Y_scores = Yc @ self.y_weights_
            return X_scores, Y_scores
        else:
            return X_scores

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(X, Y)

    def predict(self, X):
        X = np.asarray(X)
        Xc = X - self.x_mean_
        # Compute canonical scores for new X.
        U_new = Xc @ self.x_weights_
        # Predict Y by mapping the canonical scores back to the original Y space.
        Y_pred = U_new @ self.y_loadings_ + self.y_mean_
        return Y_pred