import numpy as np

class cPCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, w):
        X = np.asarray(X)
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError("X must be 3D: (trials, time, cells)")
        if not (0 <= w <= 1):
            raise ValueError("w must be in [0,1]")
        
        # means
        mu_trial = X.mean(axis=1, keepdims=True)  # (trials,1,cells)
        mu_time  = X.mean(axis=0, keepdims=True)  # (1,time,cells)

        # residuals
        R_within = X - mu_trial                         # (tr, t, c)
        R_between= np.transpose(X - mu_time, (1,0,2))   # (t, tr, c)

        # covariances
        C_within = np.matmul(R_within.transpose(0,2,1), R_within)    # (tr, c, c)
        C_between= np.matmul(R_between.transpose(0,2,1), R_between)# (t, c, c)

        # sum across trials/time
        Cw = C_within.sum(axis=0)
        Cb = C_between.sum(axis=0)

        # normalize
        Cw = Cw / np.trace(Cw)
        Cb = Cb / np.trace(Cb)

        # composite matrix
        M = (1-w)*Cw - w*Cb

        # eigendecomposition
        vals, vecs = np.linalg.eigh(M)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]

        self.explained_variance_ = vals
        self.components_ = vecs     # shape (cells, cells)
        return self

    def transform(self, X):
        X = np.asarray(X)
        V = self.components_
        if self.n_components:
            V = V[:, :self.n_components]
        # project
        return X@V

    def fit_transform(self, X, w):
        self.fit(X, w)
        return self.transform(X)
