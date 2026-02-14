import numpy as np

class CCA_ortho:
    """
    Orthogonal Canonical Correlation Analysis (rc-OCCA) with optional variance normalization.
    
    Parameters
    ----------
    n_components : int
        Number of canonical components to extract.
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.P_a = None        # Orthonormal weight vectors (unit norm)
        self.P_b = None
        self.W_a = None        # Scaled weight vectors (unit variance)
        self.W_b = None
        self.mean_a = None
        self.mean_b = None


    def get_1_cca_old(self, Xc, Yc):
        # assume Xc and Yc to be centered
        # get first sets of components with highest correlation

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
        U = U[:, 0]
        Vt = Vt[0, :]
        s = s[0]

        # Canonical weights in the original space.
        x_weights = A @ U         # shape: (n_features_x, n_components)
        y_weights = B @ Vt.T        # shape: (n_features_y, n_components)
        return x_weights, y_weights
    
    def get_1_cca(self, Xc, Yc):
        """
        Compute one orthogonalized CCA direction on centered data Xc, Yc.
        Returns (p_a, p_b) of unit Euclidean norm.
        """
        # ——————————————————————————————
        # 1) Economy SVD on the data matrices
        # Xc: (n_samples, n_features_x)
        Ux, sx, VxT = np.linalg.svd(Xc, full_matrices=False)
        Uy, sy, VyT = np.linalg.svd(Yc, full_matrices=False)
        tol_x = np.max(sx) * max(Xc.shape) * np.finfo(float).eps
        inv_sx = np.where(sx > tol_x, 1.0 / sx, 0.0)
        tol_y = np.max(sy) * max(Yc.shape) * np.finfo(float).eps
        inv_sy = np.where(sy > tol_y, 1.0 / sy, 0.0)
        
        U, s, Vt = np.linalg.svd(Ux.T @ Uy, full_matrices=False)
        
        # 4) take only the leading singular vectors
        u1 = U[:, 0]       # shape (n_feats_x,)
        v1 = Vt.T[:, 0]    # shape (n_feats_y,)
    
        # 5) back-project into original space
        #    (equivalent to Vx @ diag(1/sx) @ u1 )
        p_a = VxT.T @ (inv_sx * u1)
        p_b = VyT.T @ (inv_sy * v1)
        return p_a, p_b

    def fit(self, X_a, X_b):
        """
        Fit the orthogonal CCA model and compute both unit-norm and unit-variance coefficients.
        
        Parameters
        ----------
        X_a : array-like, shape (n_samples, n_features_a)
        X_b : array-like, shape (n_samples, n_features_b)
        """
        X_a = np.asarray(X_a, dtype=float)
        X_b = np.asarray(X_b, dtype=float)
        if X_a.shape[0] != X_b.shape[0]:
            raise ValueError("X_a and X_b must have the same number of samples")
        
        # Center data
        self.mean_a = X_a.mean(axis=0)
        self.mean_b = X_b.mean(axis=0)
        A = X_a - self.mean_a
        B = X_b - self.mean_b
        
        n_samples, fa = A.shape
        _, fb = B.shape
        
        P_a = np.zeros((fa, self.n_components))
        P_b = np.zeros((fb, self.n_components))
        
        A_def, B_def = A.copy(), B.copy()
        
        # Iteratively extract components by deflation
        for i in range(self.n_components):
            p_a, p_b=self.get_1_cca(A_def, B_def)
            
            a_norm=np.linalg.norm(p_a)
            b_norm=np.linalg.norm(p_b)
            
            # get components orthogonal to all of the previous components
            #if i > 0:
            if i>0:
                prev_a = P_a[:, :i]
                prev_b = P_b[:, :i]
                p_a -= prev_a @ (prev_a.T @ p_a)
                p_b -= prev_b @ (prev_b.T @ p_b) 
            
            #print(f"2norm ratio A:{np.linalg.norm(p_a)/a_norm}, norm ratio B:{np.linalg.norm(p_b)/b_norm}")
            # Normalize to unit Euclidean norm
            p_a /= np.linalg.norm(p_a)
            p_b /= np.linalg.norm(p_b)
            
            P_a[:, i] = p_a
            P_b[:, i] = p_b
            
            # Deflation
            u_score = A_def @ p_a
            v_score = B_def @ p_b
            A_def -= np.outer(u_score, p_a)
            B_def -= np.outer(v_score, p_b)
            #A_def = A - A @ P_a @ P_a.T
            #B_def = B -B @ P_b @ P_b.T

        
        self.P_a = P_a
        self.P_b = P_b
        
        # Compute variance of each component on the centered data
        U = A @ P_a
        V = B @ P_b
        std_u = U.std(axis=0, ddof=0)
        std_v = V.std(axis=0, ddof=0)
        std_u[std_u == 0] = 1.0
        std_v[std_v == 0] = 1.0
        
        # Scale weights to achieve unit variance in projected scores
        self.W_a = P_a / std_u
        self.W_b = P_b / std_v
        
        return self

    def transform(self, X_a, X_b, scale=False):
        """
        Apply the learned CCA transformation to new data.
        
        Parameters
        ----------
        X_a : array-like, shape (n_samples, n_features_a)
        X_b : array-like, shape (n_samples, n_features_b)
        scale : bool, default=False
            If True, uses the unit-variance coefficients (W_a, W_b);
            otherwise uses unit-norm coefficients (P_a, P_b).
        
        Returns
        -------
        U : ndarray, shape (n_samples, n_components)
            Canonical variates for X_a.
        V : ndarray, shape (n_samples, n_components)
            Canonical variates for X_b.
        """
        if self.P_a is None or self.P_b is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        A = np.asarray(X_a, dtype=float) - self.mean_a
        B = np.asarray(X_b, dtype=float) - self.mean_b
        
        if scale:
            U = A @ self.W_a
            V = B @ self.W_b
        else:
            U = A @ self.P_a
            V = B @ self.P_b
        
        return U, V

    def fit_transform(self, X_a, X_b, scale=False):
        """
        Fit the model and return the canonical variates.
        
        Parameters
        ----------
        X_a, X_b : array-like
        scale : bool, default=False
            If True, returns unit-variance variates; else unit-norm.
        """
        self.fit(X_a, X_b)
        return self.transform(X_a, X_b, scale=scale)

"""
# test with random data
from CCA_SVD import CCA_SVD

from OrthoCCA import CCA_ortho
# --- Generate synthetic test data ---
rng = np.random.RandomState(0)
n_samples = 1000
latent_dim = 100
features_a = 200
features_b = 300

Z = rng.randn(n_samples, latent_dim)
A_true = rng.randn(latent_dim, features_a)
B_true = rng.randn(latent_dim, features_b)
X_a =  rng.randn(n_samples, features_a)
X_b =  rng.randn(n_samples, features_b)

# --- Fit and compare ---
cca_svd = CCA_SVD(latent_dim)
cca_ortho = CCA_ortho(latent_dim)

U_svd, V_svd = cca_svd.fit_transform(X_a, X_b)
U_ortho, V_ortho = cca_ortho.fit_transform(X_a, X_b, scale=False)

corrs_svd = [np.corrcoef(U_svd[:, i], V_svd[:, i])[0, 1] for i in range(latent_dim)]
corrs_ortho = [np.corrcoef(U_ortho[:, i], V_ortho[:, i])[0, 1] for i in range(latent_dim)]



# corrs_svd and corrs_ortho should be sequences of length n_components
components = range(1, len(corrs_svd) + 1)

plt.figure()
plt.plot(components, corrs_svd, marker='o', label='CCA_SVD')
plt.plot(components, corrs_ortho, marker='o', label='CCA_ortho')

plt.xlabel('Component')
plt.ylabel('Canonical Correlation')
plt.title('Canonical Correlation by Component')
plt.grid(True)
plt.legend()
plt.show()


plt.figure()
plt.imshow(cca_ortho.P_a.T@cca_ortho.P_a)
plt.colorbar()


plt.figure()
cca_svd.x_weights_/=np.linalg.norm(cca_svd.x_weights_ , axis=0, keepdims=True)
plt.imshow(np.abs(cca_svd.x_weights_.T@cca_ortho.P_a))
plt.colorbar()
"""