# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 22:41:44 2025

@author: RHIRAsimulation
"""
import numpy as np
class PLS_SVD:
    def __init__(self, n_components=2, scale=False):
        """
        Initializes the PLS model using the SVD method.

        Parameters:
        -----------
        n_components : int
            Number of latent components to extract.
        scale : bool
            Whether to scale (standardize) the data.
        """
        self.n_components = n_components
        self.scale = scale

    def fit(self, X, Y):
        """
        Fits the PLS model using the SVD of the cross-covariance matrix.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Predictor matrix.
        Y : array-like, shape (n_samples, n_targets)
            Response matrix.

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        # Convert to float arrays.
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        # Center the data.
        self.x_mean = np.mean(X, axis=0)
        self.y_mean = np.mean(Y, axis=0)
        Xc = X - self.x_mean
        Yc = Y - self.y_mean

        # Optionally scale the data.
        if self.scale:
            self.x_std = np.std(Xc, axis=0, ddof=1)
            self.y_std = np.std(Yc, axis=0, ddof=1)
            # Avoid division by zero.
            self.x_std[self.x_std == 0] = 1
            self.y_std[self.y_std == 0] = 1
            Xc = Xc / self.x_std
            Yc = Yc / self.y_std

        # Save the centered (and scaled) matrices for later use.
        self.Xc = Xc
        self.Yc = Yc

        # Compute the cross-covariance matrix.
        C = Xc.T @ Yc

        # Perform SVD on the cross-covariance matrix.
        # C = U * S * V^T
        U, s, Vt = np.linalg.svd(C, full_matrices=False)

        # Extract the first n_components singular vectors.
        self.weights_x = U[:, :self.n_components]     # weights for X
        self.weights_y = Vt.T[:, :self.n_components]    # weights for Y
        self.singular_values = s[:self.n_components]    # corresponding singular values

        # Compute latent scores for X and Y.
        self.x_scores = Xc @ self.weights_x
        self.y_scores = Yc @ self.weights_y

        # Optionally, compute loadings.
        eps = 1e-12  # Small constant to prevent division by zero
        
        # X loadings: projecting Xc onto the latent scores
        x_scores_norm_sq = np.sum(self.x_scores**2, axis=0)
        x_scores_norm_sq_safe = np.where(x_scores_norm_sq == 0, eps, x_scores_norm_sq)
        self.x_loadings = (Xc.T @ self.x_scores) / x_scores_norm_sq_safe
        
        # Y loadings: projection of Yc on the latent scores
        y_scores_norm_sq = np.sum(self.y_scores**2, axis=0)
        y_scores_norm_sq_safe = np.where(y_scores_norm_sq == 0, eps, y_scores_norm_sq)
        self.y_loadings = (Yc.T @ self.y_scores) / y_scores_norm_sq_safe
        return self

    def transform(self, X, Y):
        """
        Projects new data onto the latent space for both X and Y.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New predictor data.
        Y : array-like, shape (n_samples, n_targets)
            New response data.

        Returns:
        --------
        T_x : array, shape (n_samples, n_components)
            The latent scores for X.
        T_y : array, shape (n_samples, n_components)
            The latent scores for Y.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        Xc = X - self.x_mean
        Yc = Y - self.y_mean
        if self.scale:
            Xc = Xc / self.x_std
            Yc = Yc / self.y_std

        T_x = Xc @ self.weights_x
        T_y = Yc @ self.weights_y
        return T_x, T_y

    def fit_transform(self, X, Y):
        """
        Fits the model and transforms X and Y into their latent spaces.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Predictor matrix.
        Y : array-like, shape (n_samples, n_targets)
            Response matrix.

        Returns:
        --------
        T_x : array, shape (n_samples, n_components)
            The latent scores for X.
        T_y : array, shape (n_samples, n_components)
            The latent scores for Y.
        """
        self.fit(X, Y)
        return self.transform(X, Y)
