import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, RNN
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform
from sklearn.cross_decomposition import CCA, PLSSVD
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from RNNcustom_2_fix2 import RNNCustom2Fix2
from RNNcustom_2_fix_2_full_brown import RNNCustom2Fix2full_brown_2
from CustomConstraintWithMax import IEWeightandLim, IEWeightOut
from WInitial_3 import OrthoCustom3
from RNNcustom_2_perturb_noise_prob import RNNCustom2FixPerturb_noise_prob,RNNCustom2FixPerturb_noise_prob_brown
from CCA_SVD import CCA_SVD
from column_corr import pairwise_corr

class Test:
    def __init__(self, min_dur, max_dur, dt, dim_method, Dim=100, lin_method="act_avg"):
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.dt = dt
        self.dim_method = dim_method
        self.fit_method = lin_method
        self.Dim = Dim

        
    def xlogx(self, p):
        p = np.asarray(p)  # Convert p to a numpy array (works for scalars as well)
        return np.where(p > 0, p * np.log(p), 0.0)

    
    def Seq_ind(self,act_avg_A,mean_range):
        """
        act_avg_A: (time, ncells)
        """
        data6 = act_avg_A[:self.min_dur, :]
        data12 =act_avg_A[self.min_dur:, :]
        ncell = data6.shape[1]

        # Reduce bins
        #mean_range = 50
        data6_2 = np.zeros((data6.shape[0] // mean_range, data6.shape[1]))
        data12_2 = np.zeros((data12.shape[0] // mean_range, data12.shape[1]))

        for k in range(data6_2.shape[0]):
            data6_2[k, :] = np.mean(data6[mean_range * k:mean_range * (k + 1), :], axis=0)
        for k in range(data12_2.shape[0]):
            data12_2[k, :] = np.mean(data12[mean_range * k:mean_range * (k + 1), :], axis=0)

        data6 = data6_2
        data12 = data12_2

        maxind6 = np.argmax(data6, axis=0)
        maxind12 = np.argmax(data12, axis=0)
        onehot6 = np.zeros_like(data6)
        onehot12 = np.zeros_like(data12)

        for k in range(data6.shape[1]):
            onehot6[maxind6[k], k] = 1
        for k in range(data12.shape[1]):
            onehot12[maxind12[k], k] = 1

        p6 = np.sum(onehot6, axis=1) / ncell
        p12 = np.sum(onehot12, axis=1) / ncell

        peak_ent=np.zeros(2)
        temp_spar=np.zeros(2)
        Sql=np.zeros(2)
        peak_ent[0] = np.sum(-self.xlogx(p6)) / np.log(data6.shape[0])
        peak_ent[1]= np.sum(-self.xlogx(p12)) / np.log(data12.shape[0])

        data6_norm = data6 / np.sum(data6, axis=1, keepdims=True)
        data12_norm = data12 / np.sum(data12, axis=1, keepdims=True)
        temp_spar[0] = 1 - np.mean(np.sum(-self.xlogx(data6_norm), axis=1)) / np.log(ncell)
        temp_spar[1] = 1 - np.mean(np.sum(-self.xlogx(data12_norm), axis=1)) / np.log(ncell)

        Sql[0] = np.sqrt(peak_ent[0] * temp_spar[0])
        Sql[1] = np.sqrt(peak_ent[1] * temp_spar[1])

        return peak_ent, temp_spar, Sql