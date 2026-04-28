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
from RNNcustom_2_perturb_noise_prob import (
    RNNCustom2FixPerturb_noise_prob,
    RNNCustom2FixPerturb_noise_prob_brown,
    RNNCustom2FixPerturb_noise_dir_prob_brown,
    RNNCustom2FixPerturb_noise_prob_brown_lyapunov,
    RNNCustom2FixPerturb_noise_prob_brown_noise_jacobian,
    RNNCustom2FixPerturb_noise_prob_synch_jacobian,
    RNNCustom2FixPerturb_noise_dir_prob_brown_return_noise,
    RNNCustom2FixPerturb_noise_prob_brown_noise_jacobian_2,
    RNNCustom2FixPerturb_noise_dir_prob_brown_ratio,
    RNNCustom2FixPerturb_noise_prob_brown_noise_jacobian_output
)
from CCA_SVD import CCA_SVD
from PLS_SVD import PLS_SVD
from OrthoCCA import CCA_ortho
from column_corr import pairwise_corr
from scipy.linalg import null_space
from QRDCell2 import (QRDcell2_flat_2,
                      Get_norm_ratio,
                      CumulativeJacobian, 
                      EigOrSingJacobian,
                      Get_lyap_dir,
                      Noise_jacob,
                      QRDcell2_flat_half,
                      Running_correlation,
                      DiagonalLayer,
                      LE_axis_diff,
                      Jxy_val)
import warnings

class PerturbDecodeAnalyze:
    def __init__(self, min_dur, max_dur, dt, dim_method, Dim=100, lin_method="act_avg",nohigh=True):
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.dt = dt
        self.dim_method = dim_method
        self.fit_method = lin_method
        self.Dim = Dim
        self.nohigh=nohigh
        
    def build_masks(self, nUnit, nInh, con_prob, seed):
        random_matrix = tf.random.uniform([nUnit - nInh, nUnit], minval=0, maxval=1, seed=seed)
        # Apply threshold to generate binary values
        mask_A_1 = tf.cast(tf.random.uniform([nUnit - nInh, nUnit], minval=0, maxval=1) < con_prob, dtype=tf.int32)
        mask_A = tf.concat([mask_A_1, tf.zeros([nInh, nUnit], dtype=tf.int32)], 0)
        return mask_A
    
    def column_wise_brown(self, length,rank,exp,freq_scale=1.0):
        # shape: (length, rank)
        white_noise = np.random.normal(size=(length, rank))
        fft_noise = np.fft.fft(white_noise, axis=0)
        freqs = np.fft.fftfreq(length) * freq_scale
        freqs[0] = freqs[1]  # avoid 0
        scale_factor = 1.0 / (np.abs(freqs) ** exp)   # shape (length,)
        # Apply scaling to each column
        scaled_fft_noise = fft_noise * scale_factor[:, None]
        brown_1d = np.fft.ifft(scaled_fft_noise, axis=0).real    
        return brown_1d

    def column_wise_brown_nohigh(self, length, rank, exp, freq_scale=100.0):
        # Generate white noise of shape (length, rank)
        white_noise = np.random.normal(size=(length, rank))
        # Compute FFT along the time dimension (axis=0)
        fft_noise = np.fft.fft(white_noise, axis=0)
        # Create frequency bins in Hz: d=1/sampling_rate ensures the correct scaling
        freqs = np.fft.fftfreq(length, d=1/freq_scale)
        # Avoid division by zero at f=0; assign it a value from the next frequency bin
        freqs[0] = freqs[1]
        # Compute the scaling factor: 1/|f|^exp for each frequency bin
        scale_factor = 1.0 / (np.abs(freqs) ** exp)
        # Eliminate all frequency components with period < 1 second (i.e., f > 1 Hz)
        scale_factor[np.abs(freqs) > 1] = 0
        # Apply the scaling factor column-wise to the FFT noise components
        scaled_fft_noise = fft_noise * scale_factor[:, None]
        # Transform the scaled data back to the time domain and take the real part
        brown_1d = np.fft.ifft(scaled_fft_noise, axis=0).real    
        return brown_1d

    def column_wise_noise(self,length, rank, exp, freq_scale=100.0):
        if self.nohigh is True:
            noise_weights=self.column_wise_brown_nohigh(length,rank,exp, freq_scale=freq_scale)
        else:
            noise_weights=self.column_wise_brown(length,rank,exp, freq_scale=freq_scale)
        return noise_weights
    
    #noise_weights=tf.convert_to_tensor(column_wise_brown(nUnit,rank,exp=2).T) #(rank, nUnit)
    def create_noise_weights(self, length,rank,exp,scale):
        noise_weights=self.column_wise_noise(length,rank,exp)
        noise_weights=scale * (noise_weights - np.min(noise_weights,axis=0,keepdims=True))/ (np.max(noise_weights,axis=0, keepdims=True) - np.min(noise_weights,axis=0, keepdims=True))-0.5*scale
        return noise_weights.T    # rank, time

    def create_brown_noise_rank(self, time,rank,sample_size,exp,scale):
        noise_inputs=self.column_wise_noise(time,rank*sample_size,exp=exp)
        noise_inputs = scale * (noise_inputs - np.min(noise_inputs,axis=0,keepdims=True)) / (np.max(noise_inputs,axis=0, keepdims=True) - np.min(noise_inputs,axis=0, keepdims=True))-0.5*scale
        noise_inputs=np.array(np.split(noise_inputs, indices_or_sections=sample_size, axis=1))#(samplesize,time,rank)
        return noise_inputs #(samplesize,time,rank)
    
    def build_model(self, nUnit,nInh,nInput,con_prob,maxval,ReLUalpha,tau,seed1):
        A_mask=self.build_masks(nUnit,nInh, con_prob,seed=seed1)
        B_mask=self.build_masks(nUnit,nInh, con_prob,seed=seed1)
        visible = Input(shape=(None,nInput)) 
        #vis_noise=GaussianNoiseAdd(stddev=0.01, seed=seed1)(visible)# used to be 0.01*np.sqrt(tau*2)
        #hidden1 = SimpleRNN(nUnit,activation='tanh', use_bias=False, batch_size=batch_sz, stateful=False, input_shape=(None, 1), return_sequences=True)(vis_noise)
    
        
        # the code below incorporated options to train input kernel within RNN layer
        hidden1=RNN(RNNCustom2Fix2(nUnit, 
                              output_activation=tf.keras.layers.ReLU(max_value=1000),
                              input_activation=tf.keras.layers.LeakyReLU(alpha=ReLUalpha),
                              use_bias=False,
                              kernel_initializer=initializers.RandomNormal(mean=0., stddev=1/np.sqrt(nInput), seed=seed1), # kernel initializer should be random normal
                              recurrent_initializer=OrthoCustom3(gain=0.5, seed=seed1 , nUnit=nUnit, nInh=nInh, conProb=con_prob),
                              recurrent_constraint=IEWeightandLim(nInh=nInh,A_mask=A_mask,B_mask=B_mask,maxval=maxval),
                              kernel_trainable=True,
                              seed=seed1,
                              tau=tau, 
                              noisesd=0.08), # used to be 0.05*np.sqrt(tau*2)
                    stateful=False, 
                    input_shape=(None, nInput), 
                    return_sequences=True,
                    activity_regularizer=l2(0.01)# used to be 0.0001, 0.000001
                    #recurrent_regularizer=l2(0.000001)
                    )(visible)
        #  hidden2 = Dense(10, activation='relu')(hidden1)
        output_A = Dense(2, activation='tanh',kernel_initializer=GlorotUniform(seed=seed1), kernel_constraint=IEWeightOut(nInh=nInh))(hidden1[0])
        output_B = Dense(2, activation='tanh',kernel_initializer=GlorotUniform(seed=seed1), kernel_constraint=IEWeightOut(nInh=nInh))(hidden1[1])
        output=Concatenate(axis=2)([output_A,output_B])
        model = Model(inputs=visible, outputs=output)
        return model    
    
    def build_model_perturb_noise_prob(self, nUnit, nInh, nInput, con_prob, maxval, ReLUalpha,
                                       pert_ind, pert_which, seed1, pert_noisesd, tau):
        A_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        B_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        visible = Input(shape=(None, nInput))
        # Optionally add noise: vis_noise = GaussianNoiseAdd(stddev=0.01, seed=seed1)(visible)
        hidden1 = RNN(
            RNNCustom2FixPerturb_noise_prob(
                nUnit,
                output_activation=tf.keras.layers.ReLU(max_value=1000),
                input_activation=tf.keras.layers.LeakyReLU(alpha=ReLUalpha),
                use_bias=False,
                kernel_initializer=initializers.RandomNormal(mean=0., stddev=1/np.sqrt(nInput), seed=seed1),
                recurrent_initializer=OrthoCustom3(gain=0.5, seed=seed1, nUnit=nUnit, nInh=nInh, conProb=con_prob),
                recurrent_constraint=IEWeightandLim(nInh=nInh, A_mask=A_mask, B_mask=B_mask, maxval=maxval),
                kernel_trainable=True,
                seed=seed1,
                tau=tau,
                noisesd=0.08,
                perturb_ind=pert_ind,
                pert_which=pert_which,
                pert_noisesd=pert_noisesd
            ),
            stateful=False,
            input_shape=(None, nInput),
            return_sequences=True,
            activity_regularizer=l2(0.01)
        )(visible)
        output_A = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(hidden1[0])
        output_B = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(hidden1[1])
        output = Concatenate(axis=2)([output_A, output_B])
        model = Model(inputs=visible, outputs=output)
        return model

    def build_model_brown(self,nUnit,nInh,nInput,con_prob,maxval,ReLUalpha,seed1,tau,rank,exp=1,noise_weights=None):
        A_mask=self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        B_mask=self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        visible = Input(shape=(None,nInput+rank)) 
        #vis_noise=GaussianNoiseAdd(stddev=0.01, seed=seed1)(visible)# used to be 0.01*np.sqrt(tau*2)
        #hidden1 = SimpleRNN(nUnit,activation='tanh', use_bias=False, batch_size=batch_sz, stateful=False, input_shape=(None, 1), return_sequences=True)(vis_noise)
        if noise_weights is None:
            self.noise_weights=self.create_noise_weights(nUnit,rank,exp,2)
            print('Random noise weight created')
        else:
            self.noise_weights=noise_weights.copy()
        
        # Create your custom RNN cell
        rnn_cell = RNNCustom2Fix2full_brown_2(
            nUnit,
            output_activation=tf.keras.layers.ReLU(max_value=1000),
            input_activation=tf.keras.layers.LeakyReLU(alpha=ReLUalpha),
            use_bias=False,
            kernel_initializer=initializers.RandomNormal(mean=0., stddev=1/np.sqrt(nInput), seed=seed1),
            recurrent_initializer=OrthoCustom3(gain=0.5, seed=seed1, nUnit=nUnit, nInh=nInh, conProb=con_prob),
            recurrent_constraint=IEWeightandLim(nInh=nInh, A_mask=A_mask, B_mask=B_mask, maxval=maxval),
            kernel_trainable=True,
            seed=seed1,
            tau=tau,
            noisesd=0.08,
            noise_weights=tf.convert_to_tensor(self.noise_weights),
        )
        
        # Create the RNN layer with your custom cell and set it to stateless
        rnn_layer = RNN(
            rnn_cell,
            stateful=False,  # now stateless
            return_sequences=True,
            activity_regularizer=l2(0.1)
        )
        
        # Call the RNN layer; initial states will be automatically set to zeros
        hidden_outputs = rnn_layer(visible)
        
        # Since the custom cell returns concatenated outputs, split them back into po_A and po_B sequences
        po_A_sequence, po_B_sequence = tf.split(hidden_outputs, num_or_size_splits=2, axis=-1)
        
        # Define the output layers
        output_A = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(po_A_sequence)
        output_B = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(po_B_sequence)
        
        # Concatenate the outputs
        output = Concatenate(axis=-1)([output_A, output_B])
        
        # Define the model with only the visible input
        model = Model(inputs=visible, outputs=output)
        return model



    def build_model_perturb_noise_prob_brown(self, nUnit, nInh, nInput, con_prob, maxval, ReLUalpha,
                                       pert_ind, pert_which, seed1, pert_noisesd, tau, rank,exp=1,noise_weights=None):
        A_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        B_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        visible = Input(shape=(None,nInput+rank)) 
        
        if noise_weights is None:
            self.noise_weights=self.create_noise_weights(nUnit,rank,exp,2)
            print('Random noise weight created')
        else:
            self.noise_weights=noise_weights.copy()
        
        
        rnn_cell = RNNCustom2FixPerturb_noise_prob_brown(
            nUnit,
            output_activation=tf.keras.layers.ReLU(max_value=1000),
            input_activation=tf.keras.layers.LeakyReLU(alpha=ReLUalpha),
            use_bias=False,
            kernel_initializer=initializers.RandomNormal(mean=0., stddev=1/np.sqrt(nInput), seed=seed1),
            recurrent_initializer=OrthoCustom3(gain=0.5, seed=seed1, nUnit=nUnit, nInh=nInh, conProb=con_prob),
            recurrent_constraint=IEWeightandLim(nInh=nInh, A_mask=A_mask, B_mask=B_mask, maxval=maxval),
            kernel_trainable=True,
            seed=seed1,
            tau=tau,
            noisesd=0.08,
            perturb_ind=pert_ind,
            pert_which=pert_which,
            pert_noisesd=pert_noisesd,
            noise_weights=tf.convert_to_tensor(self.noise_weights),
        )
        
        # Create the RNN layer with your custom cell and set it to stateless
        rnn_layer = RNN(
            rnn_cell,
            stateful=False,  # now stateless
            return_sequences=True,
            activity_regularizer=l2(0.1)
        )
        
        # Call the RNN layer; initial states will be automatically set to zeros
        hidden_outputs = rnn_layer(visible)
        
        # Since the custom cell returns concatenated outputs, split them back into po_A and po_B sequences
        po_A_sequence, po_B_sequence = tf.split(hidden_outputs, num_or_size_splits=2, axis=-1)
        
        # Define the output layers
        output_A = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(po_A_sequence)
        output_B = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(po_B_sequence)
        
        # Concatenate the outputs
        output = Concatenate(axis=-1)([output_A, output_B])
        
        # Define the model with only the visible input
        model = Model(inputs=visible, outputs=output)
        return model

    def build_model_perturb_noise_prob_brown_dir(self, nUnit, nInh, nInput, con_prob, maxval, ReLUalpha,
                                       pert_ind, pert_which, seed1, pert_noisesd, tau, rank,exp=1,noise_weights=None,
                                       noise_vec=None, sync_noise=True):
        A_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        B_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        visible = Input(shape=(None,nInput+rank)) 
        
        if noise_weights is None:
            self.noise_weights=self.create_noise_weights(nUnit,rank,exp,2)
            print('Random noise weight created')
        else:
            self.noise_weights=noise_weights.copy()
        
        
        rnn_cell = RNNCustom2FixPerturb_noise_dir_prob_brown(
            nUnit,
            output_activation=tf.keras.layers.ReLU(max_value=1000),
            input_activation=tf.keras.layers.LeakyReLU(alpha=ReLUalpha),
            use_bias=False,
            kernel_initializer=initializers.RandomNormal(mean=0., stddev=1/np.sqrt(nInput), seed=seed1),
            recurrent_initializer=OrthoCustom3(gain=0.5, seed=seed1, nUnit=nUnit, nInh=nInh, conProb=con_prob),
            recurrent_constraint=IEWeightandLim(nInh=nInh, A_mask=A_mask, B_mask=B_mask, maxval=maxval),
            kernel_trainable=True,
            seed=seed1,
            tau=tau,
            noisesd=0.08,
            perturb_ind=pert_ind,
            pert_which=pert_which,
            pert_noisesd=pert_noisesd,
            noise_weights=tf.convert_to_tensor(self.noise_weights),
            noise_vec=noise_vec,
            sync_noise=sync_noise,
        )
        
        # Create the RNN layer with your custom cell and set it to stateless
        rnn_layer = RNN(
            rnn_cell,
            stateful=False,  # now stateless
            return_sequences=True,
            activity_regularizer=l2(0.1)
        )
        
        # Call the RNN layer; initial states will be automatically set to zeros
        hidden_outputs = rnn_layer(visible)
        
        # Since the custom cell returns concatenated outputs, split them back into po_A and po_B sequences
        po_A_sequence, po_B_sequence = tf.split(hidden_outputs, num_or_size_splits=2, axis=-1)
        
        # Define the output layers
        output_A = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(po_A_sequence)
        output_B = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(po_B_sequence)
        
        # Concatenate the outputs
        output = Concatenate(axis=-1)([output_A, output_B])
        
        # Define the model with only the visible input
        model = Model(inputs=visible, outputs=output)
        return model

    def build_model_perturb_noise_prob_brown_dir_ratio(self, nUnit, nInh, nInput, con_prob, maxval, ReLUalpha,
                                       pert_ind, pert_which, seed1, pert_noisesd, tau, rank,exp=0.5,noise_weights=None,
                                       pert_noise_mat=None, sync_noise=True):
        A_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        B_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        visible = Input(shape=(None,nInput+rank+pert_noise_mat[0].shape[1])) 
        
        if noise_weights is None:
            self.noise_weights=self.create_noise_weights(nUnit,rank,exp,2)
            print('Random noise weight created')
        else:
            self.noise_weights=noise_weights.copy()
        
        
        rnn_cell = RNNCustom2FixPerturb_noise_dir_prob_brown_ratio(
            nUnit,
            output_activation=tf.keras.layers.ReLU(max_value=1000),
            input_activation=tf.keras.layers.LeakyReLU(alpha=ReLUalpha),
            use_bias=False,
            kernel_initializer=initializers.RandomNormal(mean=0., stddev=1/np.sqrt(nInput), seed=seed1),
            recurrent_initializer=OrthoCustom3(gain=0.5, seed=seed1, nUnit=nUnit, nInh=nInh, conProb=con_prob),
            recurrent_constraint=IEWeightandLim(nInh=nInh, A_mask=A_mask, B_mask=B_mask, maxval=maxval),
            kernel_trainable=True,
            seed=seed1,
            tau=tau,
            noisesd=0.08,
            perturb_ind=pert_ind,
            pert_which=pert_which,
            pert_noisesd=pert_noisesd,
            noise_weights=tf.convert_to_tensor(self.noise_weights),
            pert_noise_mat=pert_noise_mat,
            sync_noise=sync_noise,
        )
        
        # Create the RNN layer with your custom cell and set it to stateless
        rnn_layer = RNN(
            rnn_cell,
            stateful=False,  # now stateless
            return_sequences=True,
            activity_regularizer=l2(0.1)
        )
        
        # Call the RNN layer; initial states will be automatically set to zeros
        hidden_outputs = rnn_layer(visible)
        
        # Since the custom cell returns concatenated outputs, split them back into po_A and po_B sequences
        po_A_sequence, po_B_sequence = tf.split(hidden_outputs, num_or_size_splits=2, axis=-1)
        
        # Define the output layers
        output_A = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(po_A_sequence)
        output_B = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(po_B_sequence)
        
        # Concatenate the outputs
        output = Concatenate(axis=-1)([output_A, output_B])
        
        # Define the model with only the visible input
        model = Model(inputs=visible, outputs=output)
        return model

    def makeInput(self, x, In_ons, pert_ind):
        In_ons2 = []
        for i in range(np.shape(In_ons)[0]):
            # Find the first index where the condition is met
            index = np.argmax(In_ons[i, :] >= pert_ind[i, 0]) if np.any(In_ons[i, :] >= pert_ind[i, 0]) else None
            if index is not None:
                x[i, In_ons[i, index] - 1:, :] = np.random.normal(
                    loc=0.0, scale=0.01, size=x[i, In_ons[i, index] - 1:, :].shape)
                In_ons2.append(In_ons[i, :index - 1])
            else:
                In_ons2.append(In_ons[i, :])
        return x, In_ons2

    def avgAct(self, activities, In_ons):
        dura = [self.min_dur, self.max_dur]
        dur0 = In_ons[0, 1] - In_ons[0, 0]
        ind1 = np.argmin(np.abs(dur0 - np.array(dura)))
        act_avg = np.zeros((self.min_dur + self.max_dur, activities.shape[2]))
        kk = 0
        for i in range(activities.shape[0]):
            In_time = In_ons[i, 1 + ((ind1 + i + 1) % 2) : -2 : 2]
            for j in In_time:
                act_avg += np.squeeze(activities[i, j : j + self.min_dur + self.max_dur, :])
                kk += 1
        act_avg /= kk
        return act_avg

    def avgAct2(self, activities, In_ons):
        avg_dur = (self.min_dur + self.max_dur) / 2
        dur0 = In_ons[0, 1] - In_ons[0, 0]
        ind1 = np.argmin(np.abs(dur0 - np.array([self.min_dur, self.max_dur])))
        act_avg = np.zeros((self.min_dur + self.max_dur, activities.shape[2]))
        kk = 0
        for i in range(activities.shape[0]):
            for j in range(1,In_ons.shape[1] - 1):
                if (In_ons[i, j + 1] - In_ons[i, j] <= avg_dur and
                    In_ons[i, j] + self.min_dur + self.max_dur <= activities.shape[1]):
                    act_avg += np.squeeze(activities[i, In_ons[i, j] : In_ons[i, j] + self.min_dur + self.max_dur, :])
                    kk += 1
        act_avg /= kk
        return act_avg

    def avgAct2_stack(self, activities, In_ons):
        """
        Extracts and stacks all valid activity segments of length (min_dur+max_dur)
        for each trial & onset in `In_ons`.  Returns an array of shape
          (n_segments, min_dur+max_dur, activities.shape[2])
        """
        combined = self.min_dur + self.max_dur
        segments = []
    
        n_trials, n_onsets = In_ons.shape
        for i in range(n_trials):
            for j in range(1, n_onsets - 1):
                start = In_ons[i, j]
                end   = start + combined
                # check that this segment fits within the time-axis
                if (In_ons[i, j + 1] - start <= (combined / 2)  and  end <= activities.shape[1]):
                    seg = activities[i, start:end, :]   # shape = (combined, n_channels)
                    segments.append(seg)
    
        if len(segments) == 0:
            # no valid segments found → return empty array
            return np.zeros((0, combined, activities.shape[2]))
    
        return np.stack(segments, axis=0)


    def concatAct(self, activities, In_ons):
        segments = []
        for i in range(activities.shape[0]):
            for j in range(1, In_ons.shape[1] - 1):
                if (In_ons[i, j + 1] - In_ons[i, j] <= (self.min_dur + self.max_dur) / 2 and
                    In_ons[i, j] + self.min_dur + self.max_dur <= activities.shape[1]):
                    segment = np.squeeze(activities[i, In_ons[i, j] : In_ons[i, j] + self.min_dur + self.max_dur, :])
                    segments.append(segment)
        
        if segments:
            act_concat = np.concatenate(segments, axis=0)  # Concatenate along the time axis
        else:
            act_concat = np.zeros((0, activities.shape[2]))  # Return an empty array if no segments
        
        return act_concat

    def concatAct_sliced(self, activities, In_ons):
        # batch, time, units
        final_act_concat = []  # List to store act_concat for each slice
        
        for i in range(activities.shape[0]):  # Iterate over the first dimension
            segments = []
            
            for j in range(1, In_ons.shape[1] - 1):
                if (In_ons[i, j + 1] - In_ons[i, j] <= (self.min_dur + self.max_dur) / 2 and
                    In_ons[i, j] + self.min_dur + self.max_dur <= activities.shape[1]):
                    segment = np.squeeze(activities[i, In_ons[i, j] : In_ons[i, j] + self.min_dur + self.max_dur, :])
                    segments.append(segment)
            
            if segments:
                act_concat = np.concatenate(segments, axis=0)  # Concatenate segments within this slice
            else:
                act_concat = np.zeros((0, activities.shape[2]))  # Empty array if no valid segments
            
            final_act_concat.append(act_concat)  # Store each act_concat in the list
        
        return np.array(final_act_concat)  # Returns a list where each element corresponds to a slice

    def concatAct_with_class(self, activities, In_ons, class_per_trial):
        segments = []
        class_vec=[]
        avg_dur=0.5*(self.min_dur+self.max_dur)
        t_per_class=int((self.min_dur+self.max_dur)/class_per_trial)
        min_dur_class=int(class_per_trial*(self.min_dur)/(self.min_dur+self.max_dur))
        for i in range(activities.shape[0]):
            for j in range(1, In_ons.shape[1] - 1):
                if In_ons[i, j] + self.max_dur <= activities.shape[1]:
                    if In_ons[i, j + 1] - In_ons[i, j] <= avg_dur:
                        segment = np.squeeze(activities[i, In_ons[i, j] : In_ons[i, j] + self.min_dur, :])
                        class_sub=np.repeat(np.arange(min_dur_class),t_per_class,axis=0)
                    else:
                        segment = np.squeeze(activities[i, In_ons[i, j] : In_ons[i, j] + self.max_dur, :])
                        class_sub=np.repeat(np.arange(min_dur_class,class_per_trial),t_per_class,axis=0)
                    
                    segments.append(segment)
                    class_vec.append(class_sub)
        
        if segments:
            act_concat = np.concatenate(segments, axis=0)  # Concatenate along the time axis
            class_vec = np.concatenate(class_vec, axis=0)
        else:
            act_concat = np.zeros((0, activities.shape[2]))  # Return an empty array if no segments
        
        return act_concat, class_vec

    def concatAct_with_class_min_max(self, activities, In_ons, class_per_trial):
        segments_min = []
        segments_max = []
        class_min=[]
        class_max=[]
        avg_dur=0.5*(self.min_dur+self.max_dur)
        t_per_class=int((self.min_dur+self.max_dur)/class_per_trial)
        min_dur_class=int(class_per_trial*(self.min_dur)/(self.min_dur+self.max_dur))
        for i in range(activities.shape[0]):
            for j in range(1, In_ons.shape[1] - 1):
                if In_ons[i, j] + self.max_dur <= activities.shape[1]:
                    if In_ons[i, j + 1] - In_ons[i, j] <= avg_dur:
                        segment = np.squeeze(activities[i, In_ons[i, j] : In_ons[i, j] + self.min_dur, :])
                        segments_min.append(segment)
                        class_sub=np.repeat(np.arange(min_dur_class),t_per_class,axis=0)
                        class_min.append(class_sub)
                    else:
                        segment = np.squeeze(activities[i, In_ons[i, j] : In_ons[i, j] + self.max_dur, :])
                        segments_max.append(segment)
                        class_sub=np.repeat(np.arange(min_dur_class,class_per_trial),t_per_class,axis=0)
                        class_max.append(class_sub)
        
        
        act_cat_min = np.stack(segments_min, axis=-1)  # Concatenate along the time axis
        act_cat_max = np.stack(segments_max, axis=-1)  # Concatenate along the time axis
        class_minvec = np.stack(class_min, axis=-1)
        class_maxvec = np.stack(class_max, axis=-1)
        return act_cat_min, act_cat_max, class_minvec, class_maxvec

    def concatAct_with_in_ons_new(self, activities, in_ons_new, class_per_trial):
        segments_min = []
        segments_max = []
        class_min=[]
        class_max=[]
        avg_dur=0.5*(self.min_dur+self.max_dur)
        t_per_class=int((self.min_dur+self.max_dur)/class_per_trial)
        min_dur_class=int(class_per_trial*(self.min_dur)/(self.min_dur+self.max_dur))
        
        
        
        for i in range(activities.shape[0]):
            for j in range(1, in_ons_new[0][i].shape[0] - 1):
                if in_ons_new[0][i][j] + self.min_dur<activities.shape[1]:
                    seg_min=np.squeeze(activities[i, in_ons_new[0][i][j] : in_ons_new[0][i][j] + self.min_dur, :])
                    segments_min.append(seg_min)
                    class_sub=np.repeat(np.arange(min_dur_class),t_per_class,axis=0)
                    class_min.append(class_sub)   
                
            for j in range(1, in_ons_new[1][i].shape[0] - 1):
                if in_ons_new[1][i][j] + self.max_dur< activities.shape[1]:
                    seg_max=np.squeeze(activities[i, in_ons_new[1][i][j] : in_ons_new[1][i][j] + self.max_dur, :])
                    segments_max.append(seg_max)
                    class_sub=np.repeat(np.arange(min_dur_class,class_per_trial),t_per_class,axis=0)
                    class_max.append(class_sub)                  
        
        
        act_cat_min = np.stack(segments_min, axis=-1)  # Concatenate along the time axis
        act_cat_max = np.stack(segments_max, axis=-1)  # Concatenate along the time axis
        class_minvec = np.stack(class_min, axis=-1)
        class_maxvec = np.stack(class_max, axis=-1)
        return act_cat_min, act_cat_max, class_minvec, class_maxvec    
    
    def concatAct_sliced(self, activities, In_ons):
        # batch, time, units
        final_act_concat = []  # List to store act_concat for each slice
        
        for i in range(activities.shape[0]):  # Iterate over the first dimension
            segments = []
            
            for j in range(1, In_ons.shape[1] - 1):
                if (In_ons[i, j + 1] - In_ons[i, j] <= (self.min_dur + self.max_dur) / 2 and
                    In_ons[i, j] + self.min_dur + self.max_dur <= activities.shape[1]):
                    segment = np.squeeze(activities[i, In_ons[i, j] : In_ons[i, j] + self.min_dur + self.max_dur, :])
                    segments.append(segment)
            
            if segments:
                act_concat = np.concatenate(segments, axis=0)  # Concatenate segments within this slice
            else:
                act_concat = np.zeros((0, activities.shape[2]))  # Empty array if no valid segments
            
            final_act_concat.append(act_concat)  # Store each act_concat in the list
        
        return np.array(final_act_concat)  # Returns a list where each element corresponds to a slice

    
    def concatAct_list(self, activities, In_ons):
        segments = []
        for i in range(activities.shape[0]):
            for j in range(1, In_ons.shape[1] - 1):
                if (In_ons[i, j + 1] - In_ons[i, j] <= (self.min_dur + self.max_dur) / 2 and
                    In_ons[i, j] + self.min_dur + self.max_dur <= activities.shape[1]):
                    segment = np.squeeze(activities[i, In_ons[i, j] : In_ons[i, j] + self.min_dur + self.max_dur, :])
                    segments.append(segment)
               
        return np.array(segments)
    
    
    def make_in_ons(self, x):
        In_ons=[]
        for k in range(x.shape[-1]):
            In_ons_1=[]
            for i in range(x.shape[0]):
                in_ons_sub=np.where((x[i][1:,k]>0.5) & (x[i][:-1,k] <0.5))[0] + 1
                In_ons_1.append(in_ons_sub)
            In_ons.append(In_ons_1)
        return In_ons
    
    def sort_and_scale2(self, act_avg_A,start,stop):
        # act_avg_A is matrix with size (time, nCells)
        # this function first sort cells accorcding to the time of maximum activity from start to stop
        # then this function scale the value so that the maximum is 1.
        act_avg_A=np.array(act_avg_A)
        maxtime=np.argmax(act_avg_A[start:stop+1,:],0)
        maxtime_sort=np.argsort(maxtime)
        act_avg_A=act_avg_A[:,maxtime_sort]
        act_avg_new=act_avg_A/np.maximum(np.max(act_avg_A[start:stop+1,:],0),1e-10) # scale the matrix to set maximum to 1
        return act_avg_new, maxtime_sort    
    
    def makeInOut_sameint(self, sample_size, trial_num, inputdur, nInput, interval):
        noise_range = 0
        max_dur_max = np.ceil((1 + noise_range) * self.max_dur)
        min_dur_max = np.ceil((1 + noise_range) * self.min_dur)
        total_time_orig = int(min_dur_max * np.floor(trial_num / 2) +
                              max_dur_max * np.ceil(trial_num / 2) + 300 / self.dt)
        trial_num = int(2 * np.ceil(trial_num / 2))
        total_time = int(min_dur_max * np.floor(trial_num / 2) +
                         max_dur_max * np.ceil(trial_num / 2) + 300 / self.dt)
        x = np.zeros((sample_size, total_time, nInput))
        y = -0.5 * np.ones((sample_size, total_time, 2))
        In_ons = np.zeros((sample_size, trial_num), dtype=np.int64)
        for i in range(sample_size):
            vec = total_time
            for j in range(trial_num):
                vecbf = vec
                if j % 2 == interval:
                    vec -= self.min_dur + random.randint(-int(self.min_dur * noise_range),
                                                         int(self.min_dur * noise_range))
                else:
                    vec -= self.max_dur + random.randint(-int(self.max_dur * noise_range),
                                                         int(self.max_dur * noise_range))
                In_ons[i, -j - 1] = vec
                in_start = vec
                Dur = vecbf - in_start
                x[i, vec : vec + inputdur, :] = 1
                y[i, in_start : vecbf, 0] = np.power(np.linspace(0, 1, num=Dur), 4) - 0.5
                y[i, in_start : vecbf, 1] = np.arange(-0.5,-0.5+(Dur/self.max_dur)-1e-10,1/self.max_dur) # aboslute timing 1
        x += np.random.normal(loc=0.0, scale=0.01, size=x.shape)
        y = np.tile(y, (1, 1, 2))
        x = x[:, :total_time_orig, :]
        y = y[:, :total_time_orig, :]
        return x, y, In_ons
    
    def makeInOut_sameint_brown(self,sample_size,trial_num,inputdur,nInput,interval,brown_scale, rank, exp, noise_range=0):
        max_dur_max = np.ceil((1 + noise_range) * self.max_dur)
        min_dur_max = np.ceil((1 + noise_range) * self.min_dur)
        total_time_orig = int(min_dur_max * np.floor(trial_num / 2) +
                              max_dur_max * np.ceil(trial_num / 2) + 300 / self.dt)
        trial_num = int(2 * np.ceil(trial_num / 2))
        total_time = int(min_dur_max * np.floor(trial_num / 2) +
                         max_dur_max * np.ceil(trial_num / 2) + 300 / self.dt)
        x = np.zeros((sample_size, total_time, nInput))
        y = -0.5 * np.ones((sample_size, total_time, 2))
        noise_inputs=self.create_brown_noise_rank(total_time,rank,sample_size,exp,brown_scale) #(samplesize,time,rank)
        In_ons = np.zeros((sample_size, trial_num), dtype=np.int64)
        for i in range(sample_size):
            vec = total_time
            for j in range(trial_num):
                vecbf = vec
                if j % 2 == interval:
                    vec -= self.min_dur + random.randint(-int(self.min_dur * noise_range),
                                                         int(self.min_dur * noise_range))
                else:
                    vec -= self.max_dur + random.randint(-int(self.max_dur * noise_range),
                                                         int(self.max_dur * noise_range))
                In_ons[i, -j - 1] = vec
                in_start = vec
                Dur = vecbf - in_start
                x[i, vec : vec + inputdur, :] = 1
                y[i, in_start : vecbf, 0] = np.power(np.linspace(0, 1, num=Dur), 4) - 0.5
                y[i, in_start : vecbf, 1] = np.arange(-0.5,-0.5+(Dur/self.max_dur)-1e-10,1/self.max_dur) # aboslute timing 1
        x += np.random.normal(loc=0.0, scale=0.01, size=x.shape)
        x=np.concatenate((x,noise_inputs),axis=2)
        y = np.tile(y, (1, 1, 2))
        x = x[:, :total_time_orig, :]
        y = y[:, :total_time_orig, :]
        return x, y, In_ons

    def makeInOut_sameint_brown_stateful(self,sample_size,trial_num,inputdur,
                                         nInput,interval,brown_scale,
                                         rank, exp, 
                                         batch_num, noise_range=0):
        max_dur_max = np.ceil((1 + noise_range) * self.max_dur)
        min_dur_max = np.ceil((1 + noise_range) * self.min_dur)
        total_time_orig = int(min_dur_max * np.floor(trial_num / 2) +
                              max_dur_max * np.ceil(trial_num / 2) + 300 / self.dt)
        trial_num = int(2 * np.ceil(trial_num / 2))
        total_time = int(min_dur_max * np.floor(trial_num / 2) +
                         max_dur_max * np.ceil(trial_num / 2) + 300 / self.dt)
        total_time=int(np.ceil(total_time/batch_num)*batch_num)
        x = np.zeros((sample_size, total_time, nInput))
        Wbin=np.zeros_like(x)
        y = -0.5 * np.ones((sample_size, total_time, 2))
        noise_inputs=self.create_brown_noise_rank(total_time,rank,sample_size,exp,brown_scale) #(samplesize,time,rank)
        In_ons = np.zeros((sample_size, trial_num), dtype=np.int64)
        for i in range(sample_size):
            vec = total_time
            for j in range(trial_num):
                vecbf = vec
                if j % 2 == interval:
                    vec -= self.min_dur + random.randint(-int(self.min_dur * noise_range),
                                                         int(self.min_dur * noise_range))
                else:
                    vec -= self.max_dur + random.randint(-int(self.max_dur * noise_range),
                                                         int(self.max_dur * noise_range))
                In_ons[i, -j - 1] = vec
                in_start = vec
                Dur = vecbf - in_start
                x[i, vec : vec + inputdur, :] = 1
                y[i, in_start : vecbf, 0] = np.power(np.linspace(0, 1, num=Dur), 4) - 0.5
                y[i, in_start : vecbf, 1] = np.arange(-0.5,-0.5+(Dur/self.max_dur)-1e-10,1/self.max_dur) # aboslute timing 1
            Wbin[i,In_ons[i,1]:]=1
        
        x += np.random.normal(loc=0.0, scale=0.01, size=x.shape)
        x=np.concatenate((x,noise_inputs),axis=2)
        y = np.tile(y, (1, 1, 2))
        xsplit=np.split(x,batch_num,axis=1)
        ysplit=np.split(y,batch_num,axis=1)
        Wbinsplit=np.split(Wbin,batch_num,axis=1)
        # x_reconstructed = np.concatenate(xsplit, axis=1)  # undo the split
        return xsplit, ysplit, In_ons, Wbinsplit #(Batch_num, sample_size, time, noutput)


    def z_score_with_zero_handling(self, A, dim=0):
        mean_A = np.mean(A, axis=dim)
        std_A = np.std(A, axis=dim)
        std_A[std_A == 0] = 1
        return (A - mean_A) / std_A

    def make_pertind(self, In_ons, trial1, ind1):
        addvec = np.array([int(ind1 / self.dt)])
        pert_ind = In_ons[:, [trial1]] + addvec
        return pert_ind

    def makeit2d(self, actpart_A):
        a, b, c = np.shape(actpart_A)
        mat = np.zeros((a * c, b))
        for i in range(c):
            mat[a * i : a * (i + 1), :] = actpart_A[:, :, i]
        return mat

    def Act_2dsort(self, activities, In_ons):
        dura = [self.min_dur, self.max_dur]
        dur0 = In_ons[0, 1] - In_ons[0, 0]
        ind1 = np.argmin(np.abs(dur0 - np.array(dura)))
        act_avg = np.zeros((self.min_dur + self.max_dur, activities.shape[2]))
        for i in range(activities.shape[0]):
            In_time = In_ons[i, 1 + ((ind1 + i + 1) % 2) : -2 : 2]
            for j in In_time:
                addmat = np.squeeze(activities[i, j : j + self.min_dur + self.max_dur, :])
                act_avg = np.concatenate((act_avg, addmat), axis=0)
        act_avg = act_avg[self.min_dur + self.max_dur :, :]
        return act_avg

    def make_classying_classes(self, act_stack_A, act_stack_B, Class_per_sec):
        classleng = int(1000 / (self.dt * Class_per_sec))
        class_per_trial = int((self.min_dur + self.max_dur) / classleng)
        class_A = np.arange(0, class_per_trial)
        class_A = np.repeat(class_A, classleng)
        trial_rep_A = int(act_stack_A.shape[0] / (self.min_dur + self.max_dur))
        class_A_train = np.tile(class_A, trial_rep_A)
        trial_rep_B = int(act_stack_B.shape[0] / (self.min_dur + self.max_dur))
        class_B_train = np.tile(class_A, trial_rep_B)
        return class_A_train, class_B_train

    def make_classying_classes_2(self, size_A, size_B, Class_per_sec):
        classleng = int(1000 / (self.dt * Class_per_sec))
        class_per_trial = int((self.min_dur + self.max_dur) / classleng)
        class_A = np.arange(0, class_per_trial)
        class_A = np.repeat(class_A, classleng)
        trial_rep_A = int(size_A / (self.min_dur + self.max_dur))
        class_A_train = np.tile(class_A, trial_rep_A)
        trial_rep_B = int(size_B / (self.min_dur + self.max_dur))
        class_B_train = np.tile(class_A, trial_rep_B)
        return class_A_train, class_B_train

    def remove_inactive(self,act_A,act_B):
        self.act_log_A=np.sum(np.power(act_A,2),0)>0
        self.act_log_B=np.sum(np.power(act_B,2),0)>0
        return act_A[:,self.act_log_A], act_B[:,self.act_log_B]
    
    def remove_inactive_transform(self, act_A, act_B, dim=1):
        # Convert boolean masks to indices
        indices_A = np.where(self.act_log_A)[0]
        indices_B = np.where(self.act_log_B)[0]
        return np.take(act_A, indices_A, axis=dim), np.take(act_B,indices_B,axis=dim)

    def remove_inactive_1(self,act_A):
        self.act_log_C=np.sum(np.power(act_A,2),0)>0
        return act_A[:,self.act_log_C]
    
    def remove_inactive_transform_1(self, act_A, dim=1):
        # Convert boolean masks to indices
        indices_C = np.where(self.act_log_C)[0]
        return np.take(act_A, indices_C, axis=dim)

    
    def trial_avg(self,stack_act):
        # calculates average activity in a trial for each column
        stack_reshape=stack_act.reshape(-1,(self.max_dur+self.min_dur),np.shape(stack_act)[1])
        stack_avg=np.mean(stack_reshape,axis=0)
        return stack_avg
    
    
    def intra_inter_var(self,stack_act):
        # calculates intra trial variance/inter trial variance
        stack_reshape=stack_act.reshape(-1,(self.max_dur+self.min_dur),np.shape(stack_act)[1])# trial, time, components
        inter_var=np.sum(np.std(stack_reshape,axis=0),0)#-> time, components-> components
        intra_var=np.sum(np.std(stack_reshape,axis=1),0)#-> trial, components-> components
        
        return intra_var/inter_var # (components,)
        
        
    def reduce_dimension_pre(self, act_avg_A, act_avg_B, act_stack_A, act_stack_B, methodname='pca', Dim=100):
        self.Dim_pre=Dim
        self.method_pre=methodname.lower()
        if methodname.lower() == "pca":
            # Create PCA objects
            self.method_A_pre = PCA()
            self.method_B_pre = PCA()
            if self.fit_method.lower() == "act_avg":
                # Fit on averaged data and transform both sets
                proj_A_train_avg = self.method_A_pre.fit_transform(act_avg_A)[:, :Dim]
                proj_B_train_avg = self.method_B_pre.fit_transform(act_avg_B)[:, :Dim]
                proj_A_train = self.method_A_pre.transform(act_stack_A)[:, :Dim]
                proj_B_train = self.method_B_pre.transform(act_stack_B)[:, :Dim]
            elif self.fit_method.lower() == "act_stack":
                # Fit on stacked data and transform both sets
                proj_A_train = self.method_A_pre.fit_transform(act_stack_A)[:, :Dim]
                proj_B_train = self.method_B_pre.fit_transform(act_stack_B)[:, :Dim]
                proj_A_train_avg = self.method_A_pre.transform(act_avg_A)[:, :Dim]
                proj_B_train_avg = self.method_B_pre.transform(act_avg_B)[:, :Dim]
            else:
                raise ValueError("Unknown fit_method. Choose 'act_avg' or 'act_stack'.")
            return (proj_A_train, proj_B_train, proj_A_train_avg, proj_B_train_avg)

        elif methodname.lower() == "cca":
            if self.fit_method.lower() == "act_avg":
                fit_A, fit_B = act_avg_A, act_avg_B
            elif self.fit_method.lower() == "act_stack":
                fit_A, fit_B = act_stack_A, act_stack_B
            else:
                raise ValueError("Unknown fit_method. Choose 'act_avg' or 'act_stack'.")
            n_comp = min(fit_A.shape[1], fit_B.shape[1])
            self.method_cca_pre = CCA_SVD(n_components=n_comp)
            if self.fit_method.lower() == "act_avg":
                proj_A_train_avg, proj_B_train_avg = self.method_cca_pre.fit_transform(act_avg_A, act_avg_B)
                proj_A_train, proj_B_train = self.method_cca_pre.transform(act_stack_A, act_stack_B)
            else:
                proj_A_train, proj_B_train = self.method_cca_pre.fit_transform(act_stack_A, act_stack_B)
                proj_A_train_avg, proj_B_train_avg = self.method_cca_pre.transform(act_avg_A, act_avg_B)
            return proj_A_train[:, :Dim], proj_B_train[:, :Dim], proj_A_train_avg[:, :Dim], proj_B_train_avg[:, :Dim]

        elif methodname.lower() == "cca_ortho":
            if self.fit_method.lower() == "act_avg":
                fit_A, fit_B = act_avg_A, act_avg_B
            elif self.fit_method.lower() == "act_stack":
                fit_A, fit_B = act_stack_A, act_stack_B
            else:
                raise ValueError("Unknown fit_method. Choose 'act_avg' or 'act_stack'.")
            n_comp = min(fit_A.shape[1], fit_B.shape[1])
            self.method_cca_pre = CCA_ortho(n_components=n_comp)
            if self.fit_method.lower() == "act_avg":
                proj_A_train_avg, proj_B_train_avg = self.method_cca_pre.fit_transform(act_avg_A, act_avg_B)
                proj_A_train, proj_B_train = self.method_cca_pre.transform(act_stack_A, act_stack_B)
            else:
                proj_A_train, proj_B_train = self.method_cca_pre.fit_transform(act_stack_A, act_stack_B)
                proj_A_train_avg, proj_B_train_avg = self.method_cca_pre.transform(act_avg_A, act_avg_B)
            return proj_A_train[:, :Dim], proj_B_train[:, :Dim], proj_A_train_avg[:, :Dim], proj_B_train_avg[:, :Dim]



        elif methodname.lower() == "pls":
            if self.fit_method.lower() == "act_avg":
                fit_A, fit_B = act_avg_A, act_avg_B
            elif self.fit_method.lower() == "act_stack":
                fit_A, fit_B = act_stack_A, act_stack_B
            else:
                raise ValueError("Unknown fit_method. Choose 'act_avg' or 'act_stack'.")
            n_comp = min(fit_A.shape[1], fit_B.shape[1])
            self.method_pls_pre = PLS_SVD(n_components=n_comp)
            if self.fit_method.lower() == "act_avg":
                proj_A_train_avg, proj_B_train_avg = self.method_pls_pre.fit_transform(act_avg_A, act_avg_B)
                proj_A_train, proj_B_train = self.method_pls_pre.transform(act_stack_A, act_stack_B)
            else:
                proj_A_train, proj_B_train = self.method_pls_pre.fit_transform(act_stack_A, act_stack_B)
                proj_A_train_avg, proj_B_train_avg = self.method_pls_pre.transform(act_avg_A, act_avg_B)
            return proj_A_train[:, :Dim], proj_B_train[:, :Dim], proj_A_train_avg[:, :Dim], proj_B_train_avg[:, :Dim]

        else:
            raise ValueError("Unknown methodname. Choose 'pca', 'cca', or 'pls'.")
    
    
    def transform_stack_pre(self, act_stack_A, act_stack_B):
        if act_stack_A.ndim == 2:
            act_stack_A = act_stack_A[..., np.newaxis]
        if act_stack_B.ndim == 2:
            act_stack_B = act_stack_B[..., np.newaxis]

        T, _, n_trials = act_stack_A.shape
        proj_A = np.zeros((T, self.Dim_pre, n_trials))
        proj_B = np.zeros((T, self.Dim_pre, n_trials))
        method = self.method_pre.lower()

        for i in range(n_trials):
            X_A = act_stack_A[:, :, i]
            X_B = act_stack_B[:, :, i]

            if method == "pca":
                A_t = self.method_A_pre.transform(X_A)[:, :self.Dim_pre]
                B_t = self.method_B_pre.transform(X_B)[:, :self.Dim_pre]

            elif method == "cca" or method == "cca_ortho":
                A_t, B_t = self.method_cca_pre.transform(X_A, X_B)
                A_t = A_t[:, :self.Dim_pre]
                B_t = B_t[:, :self.Dim_pre]

            elif method == "pls":
                A_t, B_t = self.method_pls_pre.transform(X_A, X_B)
                A_t = A_t[:, :self.Dim_pre]
                B_t = B_t[:, :self.Dim_pre]

            else:
                raise ValueError("Unknown dim_method. Choose 'pca', 'cca', or 'pls'.")

            proj_A[:, :, i] = A_t
            proj_B[:, :, i] = B_t

        return proj_A, proj_B    
        
        
    def reduce_dimension(self, act_avg_A, act_avg_B, act_stack_A, act_stack_B):
        if self.dim_method.lower() == "pca":
            act_avg_C = np.concatenate((act_avg_A, act_avg_B), axis=1)
            act_stack_C = np.concatenate((act_stack_A, act_stack_B), axis=1)
            self.method_A = PCA()
            self.method_B = PCA()
            self.method_C = PCA()
            if self.fit_method.lower() == "act_avg":
                self.method_A.fit(act_avg_A)
                self.method_B.fit(act_avg_B)
                self.method_C.fit(act_avg_C)
            elif self.fit_method.lower() == "act_stack":
                self.method_A.fit(act_stack_A)
                self.method_B.fit(act_stack_B)
                self.method_C.fit(act_stack_C)
            else:
                raise ValueError("Unknown fit_method. Choose 'act_avg' or 'act_stack'.")
            proj_A_train = self.method_A.transform(act_stack_A)
            proj_B_train = self.method_B.transform(act_stack_B)
            proj_C_train = self.method_C.transform(act_stack_C)
            return proj_A_train, proj_B_train, proj_C_train

        elif self.dim_method.lower() == "cca":
            if self.fit_method.lower() == "act_avg":
                fit_A, fit_B = act_avg_A, act_avg_B
            elif self.fit_method.lower() == "act_stack":
                fit_A, fit_B = act_stack_A, act_stack_B
            else:
                raise ValueError("Unknown fit_method. Choose 'act_avg' or 'act_stack'.")
            n_comp = min(fit_A.shape[1], fit_B.shape[1])
            self.method_cca = CCA_SVD(n_components=n_comp)
            self.method_cca.fit(fit_A, fit_B)
            proj_A_train, proj_B_train = self.method_cca.transform(act_stack_A, act_stack_B)
            return proj_A_train, proj_B_train


        elif self.dim_method.lower() == "cca_ortho":
            if self.fit_method.lower() == "act_avg":
                fit_A, fit_B = act_avg_A, act_avg_B
            elif self.fit_method.lower() == "act_stack":
                fit_A, fit_B = act_stack_A, act_stack_B
            else:
                raise ValueError("Unknown fit_method. Choose 'act_avg' or 'act_stack'.")
            n_comp = min(fit_A.shape[1], fit_B.shape[1])
            self.method_cca = CCA_ortho(n_components=n_comp)
            self.method_cca.fit(fit_A, fit_B)
            proj_A_train, proj_B_train = self.method_cca.transform(act_stack_A, act_stack_B)
            return proj_A_train, proj_B_train


        elif self.dim_method.lower() == "pls":
            if self.fit_method.lower() == "act_avg":
                fit_A, fit_B = act_avg_A, act_avg_B
            elif self.fit_method.lower() == "act_stack":
                fit_A, fit_B = act_stack_A, act_stack_B
            else:
                raise ValueError("Unknown fit_method. Choose 'act_avg' or 'act_stack'.")
            n_comp = min(fit_A.shape[1], fit_B.shape[1])  # Use fit_A and fit_B here
            self.method_pls = PLS_SVD(n_components=n_comp)
            self.method_pls.fit(fit_A, fit_B)
            proj_A_train, proj_B_train = self.method_pls.transform(act_stack_A, act_stack_B)
            return proj_A_train, proj_B_train
        else:
            raise ValueError("Unknown dim_method. Choose 'pca', 'cca', or 'pls'.")

    def reduce_dim_transform(self,act_A, act_B):
        if self.dim_method.lower() == "pca":
            act_A_trans=self.method_A.transform(act_A)
            act_B_trans=self.method_B.transform(act_B)
        elif self.dim_method.lower() == "cca" or self.dim_method.lower() == "cca_ortho":
            act_A_trans, act_B_trans=self.method_cca.transform(act_A, act_B)
        elif self.dim_method.lower() == "pls":
            act_A_trans, act_B_trans=self.method_pls.transform(act_A, act_B)
        return act_A_trans, act_B_trans
            
        

    def get_transfomation_matrix(self, rescale=False):
        # Determine pre-transformation matrices and means for both groups
        if self.method_pre.lower() == "pca":
            pre_trans_A = self.method_A_pre.components_.T
            pre_trans_B = self.method_B_pre.components_.T
            pre_mean_A = self.method_A_pre.mean_
            pre_mean_B = self.method_B_pre.mean_
        elif self.method_pre.lower() == "cca":
            pre_trans_A = self.method_cca_pre.x_weights_
            pre_trans_B = self.method_cca_pre.y_weights_
            pre_mean_A = self.method_cca_pre.x_mean_
            pre_mean_B = self.method_cca_pre.y_mean_
        elif self.method_pre.lower() == "cca_ortho":
            pre_mean_A = self.method_cca_pre.mean_a
            pre_mean_B = self.method_cca_pre.mean_b
            if rescale is True: 
                # the projected data has std=1
                pre_trans_A = self.method_cca_pre.W_a
                pre_trans_B = self.method_cca_pre.W_b
            else:
                # coefficient norm is 1 (default)
                pre_trans_A = self.method_cca_pre.P_a
                pre_trans_B = self.method_cca_pre.P_b

        elif self.method_pre.lower() == "pls":
            pre_trans_A = self.method_pls_pre.weights_x
            pre_trans_B = self.method_pls_pre.weights_y
            pre_mean_A = self.x_mean
            pre_mean_B = self.y_mean
        else:
            raise ValueError("Unknown method_pre. Choose 'pca', 'cca', or 'pls'.")
    
        # Determine post-transformation matrices and means for both groups
        if self.dim_method.lower() == "pca":
            post_trans_A = self.method_A.components_.T
            post_trans_B = self.method_B.components_.T
            # For PCA we assume no extra mean subtraction
            post_mean_A = self.method_A.mean_
            post_mean_B = self.method_B.mean_
        elif self.dim_method.lower() == "cca":
            post_trans_A = self.method_cca.x_weights_
            post_trans_B = self.method_cca.y_weights_
            post_mean_A = self.method_cca.x_mean_
            post_mean_B = self.method_cca.y_mean_
        elif self.dim_method.lower() == "cca_ortho":
            post_mean_A = self.method_cca.mean_a
            post_mean_B = self.method_cca.mean_b
            if rescale is True: 
                # the projected data has std=1
                post_trans_A = self.method_cca.W_a
                post_trans_B = self.method_cca.W_b
            else:
                # coefficient norm is 1 (default)
                post_trans_A = self.method_cca.P_a
                post_trans_B = self.method_cca.P_b
        elif self.dim_method.lower() == "pls":
            post_trans_A = self.method_pls.weights_x
            post_trans_B = self.method_pls.weights_y
            post_mean_A = self.method_pls.x_mean
            post_mean_B = self.method_pls.y_mean
        else:
            raise ValueError("Unknown dim_method. Choose 'pca', 'cca', 'cca_ortho', or 'pls'.")
    
        # Compose the overall transformation for group A and B
        linear_A = np.matmul(pre_trans_A[:,:np.shape(post_trans_A)[0]], post_trans_A)
        linear_B = np.matmul(pre_trans_B[:,:np.shape(post_trans_B)[0]], post_trans_B)
        bias_A = - (np.matmul(pre_mean_A, linear_A) + np.matmul(post_mean_A, post_trans_A))
        bias_B = - (np.matmul(pre_mean_B, linear_B) + np.matmul(post_mean_B, post_trans_B))
    
        # Return the overall transformation as lists for each group: [linear, bias]
        return [linear_A, bias_A.reshape(1, -1)], [linear_B, bias_B.reshape(1, -1)]

    def make_entire_trans_mat(self,nUnit, trans_A_sub, trans_B_sub, norm=False):
        trans_A=np.zeros((nUnit,np.shape(trans_A_sub[0])[1]))
        trans_B=np.zeros((nUnit,np.shape(trans_B_sub[0])[1]))
        trans_A[self.act_log_A,:]=trans_A_sub[0] # (nUnit, Dim)
        trans_B[self.act_log_B,:]=trans_B_sub[0]
        bias_A=trans_A_sub[1] # (1, Dim)
        bias_B=trans_B_sub[1]
        norms_A=np.linalg.norm(trans_A, axis=0, keepdims=True)
        norms_B=np.linalg.norm(trans_B, axis=0, keepdims=True)
        # normalize transformation matrix
        if norm is True:
            trans_A=np.divide(trans_A, norms_A, where=norms_A!=0)
            trans_B=np.divide(trans_B, norms_B, where=norms_B!=0)
            bias_A=np.divide(bias_A, norms_A, where=norms_A!=0)
            bias_B=np.divide(bias_B, norms_B, where=norms_B!=0)     
        
        return [trans_A,bias_A], [trans_B,bias_B]
    
    def get_ortho_vec(self,mat,ind):
        """
        Returns the projection matrix onto the subspace orthogonal to all columns of `mat`
        except for the column at index `ind`.
    
        Parameters:
            mat (np.ndarray): The input matrix of shape (n, m).
            ind (int): The index of the column to exclude from the orthogonal projection.
    
        Returns:
            np.ndarray: The projection matrix onto the orthogonal complement of the subspace 
                        spanned by all columns except the one at index `ind`.
        """
        cols = mat.shape[1]
        size0=mat.shape[0]
        new_order = [ind] + [j for j in range(cols) if j != ind]
        mat_reordered = mat[:, new_order]
        Q,R=np.linalg.qr(mat_reordered)
        Q_sub=Q[:,1:]
        ortho_mat=np.eye(size0)-Q_sub @ Q_sub.T
        return ortho_mat
        
    def get_ortho_subspace(self,mat,ind):
        """
        Returns the vector that is the projection of the column at index `ind` onto
        the subspace orthogonal to all the other columns of `mat`.

        Parameters:
            mat (np.ndarray): The input matrix of shape (n, m) with n > m.
            ind (int): The index of the column to exclude from the orthogonal projection.

        Returns:
            np.ndarray: A vector of shape (n,) that is the projection of mat[:, ind]
                        onto the null space of the other columns.
        """
        null=null_space(np.delete(mat.T,ind, axis=0))
        
        vec_ortho=null@(null.T@mat[:,ind])
        vec_ortho/=np.linalg.norm(vec_ortho,axis=0) 
        return vec_ortho
    
    
    def get_ortho_mat(self,mat):
        """
        Returns a matrix whose i-th column is the projection of mat[:, i]
        onto the subspace orthogonal to the other columns of mat.

        Parameters:
            mat (np.ndarray): The input matrix of shape (n, m) with n > m.

        Returns:
            np.ndarray: A matrix of shape (n, m), where the i-th column is the
                        orthogonal projection of mat[:, i] onto the null space
                        of the other columns.
        """
        mat_ortho=[]
        for i in range(np.shape(mat)[1]):
            vec_ortho=self.get_ortho_subspace(mat,i)
            mat_ortho.append(vec_ortho)
        return np.array(mat_ortho).T

    def remove_component(self, proj_train, remove_ind):
        if remove_ind==None:
            return proj_train
        else:
            return np.delete(proj_train, remove_ind, axis=1)

    def remove_component_stack(self, proj_list, remove_ind):
        if remove_ind==None:
            return proj_list
        else:
            return [np.delete(proj, remove_ind, axis=1) for proj in proj_list]

    def make_rand_weight(self,x,n,ind, w):
        """
        for each timepoint in x, make vector of size n.
        each vector is sampled randomly and first ind has weights w (0 to 1)
        """
        mat1=np.random.randn(x.shape[0], x.shape[1], ind)
        mat2=np.random.randn(x.shape[0], x.shape[1], n-ind)
        mat1/=np.linalg.norm(mat1,axis=2, keepdims=True)
        mat2/=np.linalg.norm(mat2,axis=2, keepdims=True)
        mat=np.concatenate((np.sqrt(w)*mat1, np.sqrt(1-w)*mat2), axis=2)
        mat/=np.linalg.norm(mat, axis=2, keepdims=True)
        return mat #(sample, time, n)
        

    def create_train_classifier(self, proj_train, class_train, n_estimators=100, bootstrap=True, n_jobs=-1):
        from sklearn.ensemble import RandomForestClassifier

        proj_train_A = proj_train[0]
        proj_train_B = proj_train[1]
        if self.dim_method.lower() == "pca":
            proj_train_C = proj_train[2]
            
        clf_A = RandomForestClassifier(n_estimators=n_estimators, bootstrap=bootstrap, n_jobs=n_jobs)
        clf_A.fit(proj_train_A[:, :self.Dim], class_train[0])
        clf_B = RandomForestClassifier(n_estimators=n_estimators, bootstrap=bootstrap, n_jobs=n_jobs)
        clf_B.fit(proj_train_B[:, :self.Dim], class_train[1])
        self.clf_A = clf_A
        self.clf_B = clf_B
        if self.dim_method.lower() == "pca":
            clf_C = RandomForestClassifier(n_estimators=n_estimators, bootstrap=bootstrap, n_jobs=n_jobs)
            clf_C.fit(proj_train_C[:, :self.Dim], class_train[0])
            self.clf_C = clf_C
            return clf_A, clf_B, clf_C
        else:
            return clf_A, clf_B

    def perturb_and_decode_noise_prob(self, trial1, ind1, pert_which, order, pert_noisesd, stop,
                                        sample_size, trial_num, inputdur, nInput,
                                        nUnit, nInh, con_prob, maxval, ReLUalpha, seed1, tau, model):
        x, y, In_ons = self.makeInOut_sameint(sample_size, trial_num, inputdur, nInput, order)
        pert_ind = In_ons[:, [trial1]] + ind1
        if stop:
            x, In_ons = self.makeInput(x, In_ons, pert_ind)
        model2 = self.build_model_perturb_noise_prob(nUnit=nUnit, nInh=nInh, nInput=nInput,
                                                     con_prob=con_prob, maxval=maxval, ReLUalpha=ReLUalpha,
                                                     pert_ind=pert_ind, pert_which=pert_which,
                                                     seed1=seed1, pert_noisesd=pert_noisesd, tau=tau)
        model2.set_weights(model.get_weights())
        predictions = model2.predict(x)
        outputs = [layer.output for layer in model2.layers[1:]]  # Exclude input layer
        activity_model2 = Model(inputs=model2.input, outputs=outputs)
        output_and_activities2 = activity_model2.predict(x)
        activities_A = output_and_activities2[0]
        activities_B = output_and_activities2[1]
        pert_ind_2 = np.zeros(pert_ind.shape)
        In_ons_2 = np.zeros(In_ons[:, trial1:].shape)
        trial2 = trial1
        int_diff = int(np.round((In_ons[0, trial2] - In_ons[0, trial1]) / self.min_dur) * self.min_dur)
        eightnum = int((trial_num - trial1) / 2)
        actpart_A = np.zeros((activities_A.shape[0], eightnum * (self.min_dur + self.max_dur), activities_A.shape[2]))
        actpart_B = np.zeros((activities_B.shape[0], eightnum * (self.min_dur + self.max_dur), activities_B.shape[2]))
        predictions2 = np.zeros((predictions.shape[0], eightnum * (self.min_dur + self.max_dur), predictions.shape[2]))
        for i in range(In_ons.shape[0]):
            actpart_A[i, :, :] = activities_A[i, In_ons[i, trial2] : In_ons[i, trial2] + eightnum * (self.min_dur + self.max_dur), :]
            actpart_B[i, :, :] = activities_B[i, In_ons[i, trial2] : In_ons[i, trial2] + eightnum * (self.min_dur + self.max_dur), :]
            predictions2[i, :, :] = predictions[i, In_ons[i, trial2] : In_ons[i, trial2] + eightnum * (self.min_dur + self.max_dur), :]
            pert_ind_2[i, :] = pert_ind[i, :] - (In_ons[i, trial2] - int_diff)
            In_ons_2[i, :] = In_ons[i, trial1:] - (In_ons[i, trial2] - int_diff)
        actpart_A = np.transpose(actpart_A, (1, 2, 0))
        actpart_B = np.transpose(actpart_B, (1, 2, 0))
        return actpart_A, actpart_B


    def perturb_and_decode_noise_prob2(self, trial1, ind1, pert_which, order, pert_noisesd, stop,
                                        sample_size, trial_num, inputdur, nInput,
                                        nUnit, nInh, con_prob, maxval, ReLUalpha, seed1, tau, model):
        x, y, In_ons = self.makeInOut_sameint(sample_size, trial_num, inputdur, nInput, order)
        pert_ind = In_ons[:, [trial1]] + ind1
        if stop:
            x, In_ons = self.makeInput(x, In_ons, pert_ind)
        model2 = self.build_model_perturb_noise_prob(nUnit=nUnit, nInh=nInh, nInput=nInput,
                                                     con_prob=con_prob, maxval=maxval, ReLUalpha=ReLUalpha,
                                                     pert_ind=pert_ind, pert_which=pert_which,
                                                     seed1=seed1, pert_noisesd=pert_noisesd, tau=tau)
        model2.set_weights(model.get_weights())
        predictions = model2.predict(x)
        outputs = [layer.output for layer in model2.layers[1:]]  # Exclude input layer
        activity_model2 = Model(inputs=model2.input, outputs=outputs)
        output_and_activities2 = activity_model2.predict(x)
        activities_A = output_and_activities2[0]
        activities_B = output_and_activities2[1]
        
        # reshape the activities
        actpart_A=self.concatAct_sliced(activities_A, In_ons)#batch, time, units
        actpart_B=self.concatAct_sliced(activities_B, In_ons)
        
        
        actpart_A = np.transpose(actpart_A, (1, 2, 0))# time, units, batch
        actpart_B = np.transpose(actpart_B, (1, 2, 0))
        return actpart_A, actpart_B


    def perturb_and_decode_noise_prob2_brown(self, trial1, ind1, pert_which, order, pert_noisesd, stop,
                                        sample_size, trial_num, inputdur, nInput,
                                        nUnit, nInh, con_prob, maxval, ReLUalpha, seed1, tau, model,
                                        brown_scale, rank,exp=1,noise_weights=None):
        x, y, In_ons = self.makeInOut_sameint_brown(sample_size, trial_num, inputdur, nInput, order,
                                                    brown_scale=brown_scale, rank=rank, exp=exp)
        self.x=x
        self.In_ons=In_ons
        pert_ind = In_ons[:, [trial1]] + ind1
        self.pert_ind=pert_ind
        if stop:
            x, In_ons = self.makeInput(x, In_ons, pert_ind)
        model2 = self.build_model_perturb_noise_prob_brown(nUnit=nUnit, nInh=nInh, nInput=nInput,
                                                     con_prob=con_prob, maxval=maxval, ReLUalpha=ReLUalpha,
                                                     pert_ind=pert_ind, pert_which=pert_which,
                                                     seed1=seed1, pert_noisesd=pert_noisesd, tau=tau,
                                                     rank=rank,exp=exp,noise_weights=noise_weights)
        
        model2.set_weights(model.get_weights())
        predictions = model2.predict(x)
        outputs = [layer.output for layer in model2.layers[1:]]  # Exclude input layer
        activity_model2 = Model(inputs=model2.input, outputs=outputs)
        output_and_activities2 = activity_model2.predict(x)
        activities_A = output_and_activities2[1]
        activities_B = output_and_activities2[2]
        self.activities_A=activities_A
        self.activities_B=activities_B
        
        # reshape the activities
        actpart_A=self.concatAct_sliced(activities_A, In_ons)#batch, time, units
        actpart_B=self.concatAct_sliced(activities_B, In_ons)
        self.perturbed_output=output_and_activities2[5]
        self.perturbed_output_y=y
        x_part=self.concatAct_sliced(x, In_ons)#batch, time, units
        self.x=x
        self.x_part=x_part
        
        #self.perturbed_output_all=output_and_activities2
        
        
        actpart_A = np.transpose(actpart_A, (1, 2, 0))# time, units, batch
        actpart_B = np.transpose(actpart_B, (1, 2, 0))
        return actpart_A, actpart_B

    def perturb_and_decode_noise_prob2_brown_dir(self, trial1, ind1, pert_which, order, pert_noisesd, stop,
                                        sample_size, trial_num, inputdur, nInput,
                                        nUnit, nInh, con_prob, maxval, ReLUalpha, seed1, tau, model,
                                        brown_scale, rank,exp=1,noise_weights=None, noise_vec=None,sync_noise=True):
        x, y, In_ons = self.makeInOut_sameint_brown(sample_size, trial_num, inputdur, nInput, order,
                                                    brown_scale=brown_scale, rank=rank, exp=exp)
        pert_ind = In_ons[:, [trial1]] + ind1
        if stop:
            x, In_ons = self.makeInput(x, In_ons, pert_ind)
        model2 = self.build_model_perturb_noise_prob_brown_dir(nUnit=nUnit, nInh=nInh, nInput=nInput,
                                                     con_prob=con_prob, maxval=maxval, ReLUalpha=ReLUalpha,
                                                     pert_ind=pert_ind, pert_which=pert_which,
                                                     seed1=seed1, pert_noisesd=pert_noisesd, tau=tau,
                                                     rank=rank,exp=exp,noise_weights=noise_weights,
                                                     noise_vec=noise_vec,
                                                     sync_noise=sync_noise)
        model2.set_weights(model.get_weights())
        #predictions = model2.predict(x)
        outputs = [layer.output for layer in model2.layers[1:]]  # Exclude input layer
        activity_model2 = Model(inputs=model2.input, outputs=outputs)
        output_and_activities2 = activity_model2.predict(x)
        activities_A = output_and_activities2[1]
        activities_B = output_and_activities2[2]
        
        # reshape the activities
        actpart_A=self.concatAct_sliced(activities_A, In_ons)#batch, time, units
        actpart_B=self.concatAct_sliced(activities_B, In_ons)
        self.perturbed_output=output_and_activities2[5]
        self.perturbed_output_y=y
        #self.perturbed_output_all=output_and_activities2
        
        
        actpart_A = np.transpose(actpart_A, (1, 2, 0))# time, units, batch
        actpart_B = np.transpose(actpart_B, (1, 2, 0))
        return actpart_A, actpart_B


    def perturb_and_decode_noise_prob2_brown_dir_ratio(self, trial1, ind1, pert_which, order, pert_noisesd, stop,
                                        sample_size, trial_num, inputdur, nInput,
                                        nUnit, nInh, con_prob, maxval, ReLUalpha, seed1, tau, model,
                                        brown_scale, rank,exp=1,
                                        noise_weights=None, pert_noise_mat=None,sync_noise=True,pert_w=0,bias_ind=2):
        x, y, In_ons = self.makeInOut_sameint_brown(sample_size, trial_num, inputdur, nInput, order,
                                                    brown_scale=brown_scale, rank=rank, exp=exp)
        pert_ind = In_ons[:, [trial1]] + ind1
        if stop:
            x, In_ons = self.makeInput(x, In_ons, pert_ind)
        
        # pert_noise_mat is a list of length 2 and each has shape (unit, 101)
        # create noise weights for weighting the noise component
        noise_w=self.make_rand_weight(x, pert_noise_mat[0].shape[1], bias_ind, pert_w) 
        x=np.concatenate((x,noise_w),axis=2) # sample, time, 3+100
        model2 = self.build_model_perturb_noise_prob_brown_dir_ratio(nUnit=nUnit, nInh=nInh, nInput=nInput,
                                                     con_prob=con_prob, maxval=maxval, ReLUalpha=ReLUalpha,
                                                     pert_ind=pert_ind, pert_which=pert_which,
                                                     seed1=seed1, pert_noisesd=pert_noisesd, tau=tau,
                                                     rank=rank,exp=exp,noise_weights=noise_weights,
                                                     pert_noise_mat=pert_noise_mat,
                                                     sync_noise=sync_noise)
        model2.set_weights(model.get_weights())
        #predictions = model2.predict(x)
        outputs = [layer.output for layer in model2.layers[1:]]  # Exclude input layer
        activity_model2 = Model(inputs=model2.input, outputs=outputs)
        output_and_activities2 = activity_model2.predict(x)
        activities_A = output_and_activities2[1]
        activities_B = output_and_activities2[2]
        
        # reshape the activities
        actpart_A=self.concatAct_sliced(activities_A, In_ons)#batch, time, units
        actpart_B=self.concatAct_sliced(activities_B, In_ons)
        self.perturbed_output=output_and_activities2[5]
        self.perturbed_output_y=y
        #self.perturbed_output_all=output_and_activities2
        
        
        actpart_A = np.transpose(actpart_A, (1, 2, 0))# time, units, batch
        actpart_B = np.transpose(actpart_B, (1, 2, 0))
        return actpart_A, actpart_B



    def activities_to_actpart(self,activities_A,In_ons):
        actpart_A=self.concatAct_qq(activities_A, In_ons)#batch, time, units
        actpart_A = np.transpose(actpart_A, (1, 2, 0))# time, units, batch
        return actpart_A
        


    def decode_time(self, actpart, clf_A=None, clf_B=None, clf_C=None, remove_ind=None):
        actpart_A, actpart_B = actpart
        # If actpart_A or actpart_B is 2D, expand dims to simulate shape[..., 1]
        if actpart_A.ndim == 2:
            actpart_A = np.expand_dims(actpart_A, axis=2)
        if actpart_B.ndim == 2:
            actpart_B = np.expand_dims(actpart_B, axis=2)
        
        actpart_C = np.concatenate((actpart_A, actpart_B), axis=1)

        pred_A = np.zeros((actpart_A.shape[0], actpart_A.shape[2]))
        pred_B = np.zeros((actpart_B.shape[0], actpart_B.shape[2]))
        if self.dim_method.lower() == "pca":
            pred_C = np.zeros((actpart_B.shape[0], actpart_B.shape[2]))
            for i in range(actpart_A.shape[2]):
                proj_A = self.method_A.transform(actpart_A[:, :, i])
                proj_B = self.method_B.transform(actpart_B[:, :, i])
                proj_C = self.method_C.transform(actpart_C[:, :, i])
                

                proj_A, proj_B, proj_C = self.remove_component_stack([proj_A, proj_B, proj_C],remove_ind)
                
                if clf_A==None:
                    pred_A[:, i] = self.clf_A.predict(proj_A[:, :self.Dim])
                else:
                    pred_A[:, i] = clf_A.predict(proj_A[:, :self.Dim])
                if clf_B==None:
                    pred_B[:, i] = self.clf_B.predict(proj_B[:, :self.Dim])
                else:
                    pred_B[:, i] = clf_B.predict(proj_B[:, :self.Dim])
                if clf_C==None:
                    pred_C[:, i] = self.clf_C.predict(proj_C[:, :self.Dim])
                else:
                    pred_C[:, i] =clf_C.predict(proj_C[:, :self.Dim])
            return pred_A, pred_B, pred_C
        
        elif self.dim_method.lower() == "cca":
            for i in range(actpart_A.shape[2]):
                proj_A, proj_B = self.method_cca.transform(actpart_A[:, :, i], actpart_B[:, :, i])
                

                proj_A, proj_B = self.remove_component_stack([proj_A, proj_B],remove_ind)
                
                if clf_A==None:
                    pred_A[:, i] = self.clf_A.predict(proj_A[:, :self.Dim])
                else:
                    pred_A[:, i] = clf_A.predict(proj_A[:, :self.Dim])
                if clf_B==None:
                    pred_B[:, i] = self.clf_B.predict(proj_B[:, :self.Dim])
                else:
                    pred_B[:, i] = clf_B.predict(proj_B[:, :self.Dim])
            return pred_A, pred_B
        
        elif self.dim_method.lower() == "pls":
            for i in range(actpart_A.shape[2]):
                proj_A, proj_B = self.method_pls.transform(actpart_A[:, :, i], actpart_B[:, :, i])
                

                proj_A, proj_B = self.remove_component_stack([proj_A, proj_B],remove_ind)
                
                if clf_A==None:
                    pred_A[:, i] = self.clf_A.predict(proj_A[:, :self.Dim])
                else:
                    pred_A[:, i] = clf_A.predict(proj_A[:, :self.Dim])
                if clf_B==None:
                    pred_B[:, i] = self.clf_B.predict(proj_B[:, :self.Dim])
                else:
                    pred_B[:, i] = clf_B.predict(proj_B[:, :self.Dim])
            return pred_A, pred_B
    
    def xlogx(self, p):
        """ Compute p*log(p) while handling p=0 cases properly. """
        a = np.zeros_like(p)
        valid = p > 0  # Only apply log where p > 0
        a[valid] = p[valid] * np.log(p[valid])
        return a

    
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
    
    
    # create models for calculating lyapunov exponents
    
    def build_model_perturb_noise_prob_brown_lyapunov(self,nUnit, nInh, nInput, sample_size, con_prob, maxval, ReLUalpha,
                                        pert_ind, pert_which, seed1, pert_noisesd, tau, rank,exp=1,noise_weights=None,
                                        stateful=False,start=0, option="lyapunov",test_mat=None, update_state=True):
        A_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        B_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        if stateful is False:
            visible = Input(shape=(None,nInput+rank)) 
        else:
            visible = Input(batch_shape=(sample_size, None, nInput+rank))
        
        if noise_weights is None:
            self.noise_weights=self.create_noise_weights(nUnit,rank,exp,2)
            print('Random noise weight created')
        else:
            self.noise_weights=noise_weights.copy()
        
        
        rnn_cell = RNNCustom2FixPerturb_noise_prob_brown_lyapunov(
            nUnit,
            output_activation=tf.keras.layers.ReLU(max_value=1000),
            input_activation=lambda x: tf.where(x > 0, x, ReLUalpha * x),
            use_bias=False,
            kernel_initializer=initializers.RandomNormal(mean=0., stddev=1/np.sqrt(nInput), seed=seed1),
            recurrent_initializer=OrthoCustom3(gain=0.5, seed=seed1, nUnit=nUnit, nInh=nInh, conProb=con_prob),
            recurrent_constraint=IEWeightandLim(nInh=nInh, A_mask=A_mask, B_mask=B_mask, maxval=maxval),
            kernel_trainable=True,
            seed=seed1,
            tau=tau,
            noisesd=0.08,
            perturb_ind=pert_ind,
            pert_which=pert_which,
            pert_noisesd=pert_noisesd,
            noise_weights=tf.convert_to_tensor(self.noise_weights),
        )
        
        # Create the RNN layer with your custom cell and set it to stateless
        rnn_layer = RNN(
            rnn_cell,
            stateful=stateful,  # now stateless
            return_sequences=True,
            activity_regularizer=l2(0.1)
        )
        
        # Call the RNN layer; initial states will be automatically set to zeros
        outputs = rnn_layer(visible)
        # Here, outputs is a list of two tensors.
        hidden_outputs = outputs[0]  # shape: (batch, timesteps, 2*nUnit)
        Jacobian = outputs[1]        # shape: (batch, timesteps, 4*nUnit^2)
        
        # Since the custom cell returns concatenated outputs, split them back into po_A and po_B sequences
        po_A_sequence, po_B_sequence = tf.split(hidden_outputs, num_or_size_splits=2, axis=-1)
        
        # Define the output layers
        output_A = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(po_A_sequence)
        output_B = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(po_B_sequence)
        
        # Concatenate the outputs
        output = Concatenate(axis=-1)([output_A, output_B])
        
        if option.lower()=="lyapunov":
            QRDCell=QRDcell2_flat_2(nUnit=nUnit,start=start)
            #hidden1[2] = tf.ensure_shape(hidden1[2], [batch_size, None, 4 * nUnit**2])
            Rlayer=RNN(QRDCell,stateful=stateful, return_sequences=True)# (batch_num,timestep,2*nUnits)
            Rmat=Rlayer(Jacobian) #size(batch, timesteps, 2*nUnit)
            model = Model(inputs=visible, outputs=[output,Rmat])
        elif option.lower()=="norm_scale":
            norm_cell=Get_norm_ratio(nUnit=nUnit, start=start, test_mat=test_mat, update_state=update_state)
            dummy_input = tf.zeros((sample_size, 1))  # Dummy input if needed (can also pass None in later versions)
            # Make sure to use the same dtype you expect (typically tf.float32)
            initial_state = norm_cell.get_initial_state(inputs=dummy_input, batch_size=sample_size, dtype=tf.float32)
            Rlayer=RNN(norm_cell,stateful=stateful, return_sequences=True)# (batch_num,timestep,2*nUnits)
            Rmat = Rlayer(Jacobian, initial_state=initial_state) #size(batch, timesteps, ncol)
            model = Model(inputs=visible, outputs=[output,Rmat])
        elif option.lower()=="lyap_dir" or option.lower()=="cca_diff_lyap":
            norm_cell=Get_lyap_dir(nUnit=nUnit, start=start, test_mat=test_mat, update_state=update_state)
            dummy_input = tf.zeros((sample_size, 1))  # Dummy input if needed (can also pass None in later versions)
            # Make sure to use the same dtype you expect (typically tf.float32)
            initial_state = norm_cell.get_initial_state(inputs=dummy_input, batch_size=sample_size, dtype=tf.float32)
            Rlayer=RNN(norm_cell,stateful=stateful, return_sequences=True)# (batch_num,timestep,2*nUnits)
            Rmat = Rlayer(Jacobian, initial_state=initial_state) #size(batch, timesteps, ncol)
            model = Model(inputs=visible, outputs=[output,Rmat]) 
        elif option.lower()=="cca_ax_diff":
            u,v=tf.split(test_mat, num_or_size_splits=2, axis=0)
            LE_out=LE_axis_diff(nUnit*2, u, v)(Jacobian)
            model = Model(inputs=visible, outputs=[output,LE_out]) 
        elif option.lower()=="jxy":
            u,v=tf.split(test_mat, num_or_size_splits=2, axis=0)
            LE_out=Jxy_val(nUnit*2, u, v)(Jacobian)
            model = Model(inputs=visible, outputs=[output,LE_out]) 
        elif option.lower()=="none":
            Rmat=None
            model = Model(inputs=visible, outputs=[output])
        elif option.lower()=="jacobian":
            model = Model(inputs=visible, outputs=[output, Jacobian])
        elif option.lower()=="jacobian_avg":
            norm_cell=CumulativeJacobian(nUnit=nUnit, start=start)
            dummy_input = tf.zeros((sample_size, 1))  # Dummy input if needed (can also pass None in later versions)
            initial_state = norm_cell.get_initial_state(inputs=dummy_input, batch_size=sample_size, dtype=tf.float32)
            Rlayer=RNN(norm_cell,stateful=stateful, return_sequences=False)# (batch_num,timestep,2*nUnits)
            Rmat = Rlayer(Jacobian, initial_state=initial_state) #size(batch, 1, (2*nUnit)^2)            
            model = Model(inputs=visible, outputs=[output, Rmat])
        elif option.lower()=="eigen":
            norm_cell=EigOrSingJacobian(nUnit=nUnit, start=start, calc_eig=True)
            dummy_input = tf.zeros((sample_size, 1))  # Dummy input if needed (can also pass None in later versions)
            initial_state = norm_cell.get_initial_state(inputs=dummy_input, batch_size=sample_size, dtype=tf.float32)
            Rlayer=RNN(norm_cell,stateful=stateful, return_sequences=True)# (batch_num,timestep,2*nUnits)
            Rmat = Rlayer(Jacobian, initial_state=initial_state) #size(batch, 1, (2*nUnit)^2)            
            model = Model(inputs=visible, outputs=[output, Rmat])   
        elif option.lower()=="singular":
            norm_cell=EigOrSingJacobian(nUnit=nUnit, start=start, calc_eig=False)
            dummy_input = tf.zeros((sample_size, 1))  # Dummy input if needed (can also pass None in later versions)
            initial_state = norm_cell.get_initial_state(inputs=dummy_input, batch_size=sample_size, dtype=tf.float32)
            Rlayer=RNN(norm_cell,stateful=stateful, return_sequences=True)# (batch_num,timestep,2*nUnits)
            Rmat = Rlayer(Jacobian, initial_state=initial_state) #size(batch, 1, (2*nUnit)^2)            
            model = Model(inputs=visible, outputs=[output, Rmat])                                   
        else:
            warnings.warn(f"Unknown option: {option}")
        return model



    def run_model_with_jacobian(self, trial1, ind1, pert_which, order, pert_noisesd, stop,
                                                   sample_size, trial_num, inputdur, nInput,
                                                   nUnit, nInh, con_prob, maxval, ReLUalpha, seed1, tau, model,
                                                   brown_scale, rank, exp=1, noise_weights=None,stateful=False, start=None,
                                                   option="lyapunov",test_mat=None, batch_size=10, update_state=True,
                                                   lr_AB=[None, None], lr_BA=[None,None],jacob_at_zero=False, project_mat=None):

        # Create input and output arrays using the helper function.
        if stateful is False:
            x, y, In_ons = self.makeInOut_sameint_brown(sample_size, trial_num, inputdur, nInput, order,
                                                          brown_scale=brown_scale, rank=rank, exp=exp)
        else:
            x, y, In_ons, Wbin= self.makeInOut_sameint_brown_stateful(sample_size, trial_num, 
                                                            inputdur, nInput, 
                                                            order,brown_scale=brown_scale, 
                                                            rank=rank, exp=exp,
                                                            batch_num=batch_size)
        
        if start is None:
            start=In_ons[0,2]
            self.start=start
        # Compute the perturbation indices.
        pert_ind = In_ons[:, [trial1]] + ind1
        
        # Optionally modify the input if stop is True.
        if stop:
            x, In_ons = self.makeInput(x, In_ons, pert_ind)
        
        # Build a new model that computes the lyapunov exponents with the provided parameters.
        if option.lower()=="noise_sing" or option.lower()=="noise_lyap_dir":
            model2 = self.build_model_perturb_noise_prob_brown_lyapunov_noise(
                nUnit, nInh, nInput, sample_size, con_prob, maxval, ReLUalpha,
                pert_ind, pert_which, seed1, pert_noisesd, tau, rank,exp=exp,noise_weights=noise_weights,
                stateful=stateful,start=start,option=option,
                test_mat=test_mat,update_state=update_state, project_mat=project_mat
            )    
        elif option.lower()=="synch_lyapunov" or option.lower()=="synch_norm_scale" or option.lower()=="synch_lyap_dir":
            model2 = self.build_model_synchronicity_jacobian(
                nUnit, nInh, nInput, sample_size, con_prob, maxval, ReLUalpha,
                pert_ind, pert_which, seed1, pert_noisesd, tau, rank,exp=exp,noise_weights=noise_weights,
                stateful=stateful,start=start,option=option,
                test_mat=test_mat,update_state=update_state,
                lr_AB=lr_AB, lr_BA=lr_BA,jacob_at_zero=jacob_at_zero
            )                
        else:
            model2 = self.build_model_perturb_noise_prob_brown_lyapunov(
                nUnit, nInh, nInput, sample_size, con_prob, maxval, ReLUalpha,
                pert_ind, pert_which, seed1, pert_noisesd, tau, rank,exp=exp,noise_weights=noise_weights,
                stateful=stateful,start=start,option=option,
                test_mat=test_mat,update_state=update_state
            )
            
        # Copy the weights from the provided model.
        self.model2=model2
        model2.set_weights(model.get_weights())
        model2.reset_states()
        
        # Run the prediction.
        if stateful is False:
            predictions = model2.predict(x) #(output, Rmat, A_activity, B_activity
        else:
            for i in range(len(x)):
                print(f"prediction: {i} out of {len(x)}")
                xin = tf.convert_to_tensor(x[i], dtype=tf.float32)
                xin=tf.stop_gradient(xin)
                pred = model2.predict_on_batch(xin)  # pred = (output, Rmat, A_activity, B_activity)
                if not isinstance(pred, (list, tuple)):
                    pred = [pred] #1, sample_size, time, output_dim
                if i==0:
                    outputs_split = [[] for _ in range(len(pred))]
                for j in range(len(pred)):
                    outputs_split[j].append(pred[j])
        
            # Concatenate each type of output across time (axis=1)
            predictions = [np.concatenate(output_list, axis=1) for output_list in outputs_split]
        
        self.In_ons_temp=In_ons
        return predictions #

    def get_lyapunov(self, Rmat, start):
        if start is None:
            start=self.start
        #Rmat: (batch, timesteps, 2*nUnit(or ncol))
        Rmat=Rmat[:,start+1:,:]#(batch, time, output_dim)
        Rmat=tf.math.abs(Rmat)
        Rmat=tf.math.log(Rmat)
        Rmat=tf.math.cumsum(Rmat,1)
        Rmat=tf.math.reduce_mean(Rmat,0)#(time, output__dim)
        
        timepoints=tf.shape(Rmat)[0]
        v=tf.reshape(tf.cast(tf.range(timepoints),dtype=tf.float32), (-1, 1))#(time, 1)
        v+=1
        # take the mean
        Rmat=Rmat/v
        
        return Rmat # (time, output_dim=2*nUnit)
    
    def get_lyapunov_along_axis(self,Rmat,start):
        if start is None:
            start=self.start
        #Rmat: (batch, timesteps, 2*nUnit(or ncol))
        Rmat=Rmat[:,start+1:,:]#(batch, time, ncol)        
        Rmat=tf.math.abs(Rmat)
        Rmat=tf.math.log(Rmat)   
        Rmat=tf.math.reduce_mean(Rmat,0)#(time, output__dim)
        
        return Rmat #(time, ncol) log ratio for each timestep



    def build_model_synchronicity_jacobian(self,nUnit, nInh, nInput, sample_size, con_prob, maxval, ReLUalpha,
                                        pert_ind, pert_which, seed1, pert_noisesd, tau, rank,exp=1,noise_weights=None,
                                        stateful=False,start=0, option="synch",test_mat=None, update_state=True,
                                        lr_AB=[None, None], lr_BA=[None,None],jacob_at_zero=False ):
        A_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        B_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        if stateful is False:
            visible = Input(shape=(None,nInput+rank)) 
        else:
            visible = Input(batch_shape=(sample_size, None, nInput+rank))
        
        if noise_weights is None:
            self.noise_weights=self.create_noise_weights(nUnit,rank,exp,2)
            print('Random noise weight created')
        else:
            self.noise_weights=noise_weights.copy()
        
        
        rnn_cell = RNNCustom2FixPerturb_noise_prob_synch_jacobian(
            nUnit,
            output_activation=tf.keras.layers.ReLU(max_value=1000),
            input_activation=lambda x: tf.where(x > 0, x, ReLUalpha * x),
            use_bias=False,
            kernel_initializer=initializers.RandomNormal(mean=0., stddev=1/np.sqrt(nInput), seed=seed1),
            recurrent_initializer=OrthoCustom3(gain=0.5, seed=seed1, nUnit=nUnit, nInh=nInh, conProb=con_prob),
            recurrent_constraint=IEWeightandLim(nInh=nInh, A_mask=A_mask, B_mask=B_mask, maxval=maxval),
            kernel_trainable=True,
            seed=seed1,
            tau=tau,
            noisesd=0.08,
            perturb_ind=pert_ind,
            pert_which=pert_which,
            pert_noisesd=pert_noisesd,
            noise_weights=tf.convert_to_tensor(self.noise_weights),
            lr_AB=lr_AB,
            lr_BA=lr_BA,
            jacob_at_zero=jacob_at_zero,
        )
        
        # Create the RNN layer with your custom cell and set it to stateless
        rnn_layer = RNN(
            rnn_cell,
            stateful=stateful,  # now stateless
            return_sequences=True,
            activity_regularizer=l2(0.1)
        )
        
        # Call the RNN layer; initial states will be automatically set to zeros
        outputs = rnn_layer(visible)
        # Here, outputs is a list of two tensors.
        hidden_outputs = outputs[0]  # shape: (batch, timesteps, 2*nUnit)
        J_delta_A = outputs[1]        # shape: (batch, timesteps, 4*nUnit^2)
        J_delta_B = outputs[2]        # shape: (batch, timesteps, 4*nUnit^2)
        
        # Since the custom cell returns concatenated outputs, split them back into po_A and po_B sequences
        po_A_sequence, po_B_sequence = tf.split(hidden_outputs, num_or_size_splits=2, axis=-1)
        
        # Define the output layers
        output_A = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(po_A_sequence)
        output_B = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(po_B_sequence)
        
        # Concatenate the outputs
        output = Concatenate(axis=-1)([output_A, output_B])
        nUnit_half=int(nUnit/2)
        if option.lower()=="synch_lyapunov":
            QRDCell_A=QRDcell2_flat_half(nUnit=nUnit,start=start)
            #hidden1[2] = tf.ensure_shape(hidden1[2], [batch_size, None, 4 * nUnit**2])
            Rlayer_A=RNN(QRDCell_A,stateful=stateful, return_sequences=True)# (batch_num,timestep,2*nUnits)
            Rmat_A=Rlayer_A(J_delta_A) #size(batch, timesteps, 2*nUnit)
            
            QRDCell_B=QRDcell2_flat_half(nUnit=nUnit,start=start)
            #hidden1[2] = tf.ensure_shape(hidden1[2], [batch_size, None, 4 * nUnit**2])
            Rlayer_B=RNN(QRDCell_B,stateful=stateful, return_sequences=True)# (batch_num,timestep,2*nUnits)
            Rmat_B=Rlayer_B(J_delta_B) #size(batch, timesteps, 2*nUnit)
            model = Model(inputs=visible, outputs=[output,Rmat_A, Rmat_B])
        
        elif option.lower()=="synch_norm_scale":
            
            test_mat_A, test_mat_B=np.split(test_mat,indices_or_sections=2, axis=0)
            norm_cell=Get_norm_ratio(nUnit=nUnit_half, start=start, test_mat=test_mat_A, update_state=update_state)
            dummy_input = tf.zeros((sample_size, 1))  # Dummy input if needed (can also pass None in later versions)
            # Make sure to use the same dtype you expect (typically tf.float32)
            initial_state = norm_cell.get_initial_state(inputs=dummy_input, batch_size=sample_size, dtype=tf.float32)
            
            
            norm_cell_A=Get_norm_ratio(nUnit=nUnit_half, start=start, test_mat=test_mat_A, update_state=update_state)
            Rlayer_A=RNN(norm_cell_A,stateful=stateful, return_sequences=True)# (batch_num,timestep,2*nUnits)
            Rmat_A = Rlayer_A(J_delta_A, initial_state=initial_state) #size(batch, timesteps, ncol)
            
            norm_cell_B=Get_norm_ratio(nUnit=nUnit_half, start=start, test_mat=test_mat_B, update_state=update_state)
            Rlayer_B=RNN(norm_cell_B,stateful=stateful, return_sequences=True)# (batch_num,timestep,2*nUnits)
            Rmat_B = Rlayer_B(J_delta_B, initial_state=initial_state) #size(batch, timesteps, ncol)    
              
            model = Model(inputs=visible, outputs=[output,Rmat_A, Rmat_B])
            
            
        elif option.lower()=="synch_lyap_dir":
            test_mat_A, test_mat_B=np.split(test_mat,indices_or_sections=2, axis=0)
            norm_cell=Get_lyap_dir(nUnit=nUnit_half, start=start, test_mat=test_mat_A, update_state=update_state)
            dummy_input = tf.zeros((sample_size, 1))  # Dummy input if needed (can also pass None in later versions)
            initial_state = norm_cell.get_initial_state(inputs=dummy_input, batch_size=sample_size, dtype=tf.float32)
            # Make sure to use the same dtype you expect (typically tf.float32)
            
            norm_cell_A=Get_lyap_dir(nUnit=nUnit_half, start=start, test_mat=test_mat_A, update_state=update_state)
            Rlayer_A=RNN(norm_cell_A,stateful=stateful, return_sequences=True)# (batch_num,timestep,2*nUnits)
            Rmat_A = Rlayer_A(J_delta_A, initial_state=initial_state) #size(batch, timesteps, ncol)
            
            norm_cell_B=Get_lyap_dir(nUnit=nUnit_half, start=start, test_mat=test_mat_B, update_state=update_state)
            Rlayer_B=RNN(norm_cell_B,stateful=stateful, return_sequences=True)# (batch_num,timestep,2*nUnits)
            Rmat_B = Rlayer_B(J_delta_B, initial_state=initial_state) #size(batch, timesteps, ncol)
                        
            model = Model(inputs=visible, outputs=[output,Rmat_A,Rmat_B])    
        return model
        
    def build_model_perturb_noise_prob_brown_lyapunov_noise(self,nUnit, nInh, nInput, sample_size, con_prob, maxval, ReLUalpha,
                                        pert_ind, pert_which, seed1, pert_noisesd, tau, rank,exp=1,noise_weights=None,
                                        stateful=False,start=0, option="lyapunov",test_mat=None, project_mat=None, update_state=True):
        A_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        B_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        if stateful is False:
            visible = Input(shape=(None,nInput+rank)) 
        else:
            visible = Input(batch_shape=(sample_size, None, nInput+rank))
        
        if noise_weights is None:
            self.noise_weights=self.create_noise_weights(nUnit,rank,exp,2)
            print('Random noise weight created')
        else:
            self.noise_weights=noise_weights.copy()
        
        if option.lower()=="noise_sing":
            rnn_cell = RNNCustom2FixPerturb_noise_prob_brown_noise_jacobian(
                nUnit,
                output_activation=tf.keras.layers.ReLU(max_value=1000),
                input_activation=lambda x: tf.where(x > 0, x, ReLUalpha * x),
                use_bias=False,
                kernel_initializer=initializers.RandomNormal(mean=0., stddev=1/np.sqrt(nInput), seed=seed1),
                recurrent_initializer=OrthoCustom3(gain=0.5, seed=seed1, nUnit=nUnit, nInh=nInh, conProb=con_prob),
                recurrent_constraint=IEWeightandLim(nInh=nInh, A_mask=A_mask, B_mask=B_mask, maxval=maxval),
                kernel_trainable=True,
                seed=seed1,
                tau=tau,
                noisesd=0.08,
                perturb_ind=pert_ind,
                pert_which=pert_which,
                pert_noisesd=pert_noisesd,
                noise_weights=tf.convert_to_tensor(self.noise_weights),
            )
        elif option.lower()=="noise_lyap_dir":
            rnn_cell = RNNCustom2FixPerturb_noise_prob_brown_noise_jacobian_2(
                nUnit,
                output_activation=tf.keras.layers.ReLU(max_value=1000),
                input_activation=lambda x: tf.where(x > 0, x, ReLUalpha * x),
                use_bias=False,
                kernel_initializer=initializers.RandomNormal(mean=0., stddev=1/np.sqrt(nInput), seed=seed1),
                recurrent_initializer=OrthoCustom3(gain=0.5, seed=seed1, nUnit=nUnit, nInh=nInh, conProb=con_prob),
                recurrent_constraint=IEWeightandLim(nInh=nInh, A_mask=A_mask, B_mask=B_mask, maxval=maxval),
                kernel_trainable=True,
                seed=seed1,
                tau=tau,
                noisesd=0.08,
                perturb_ind=pert_ind,
                pert_which=pert_which,
                pert_noisesd=pert_noisesd,
                noise_weights=tf.convert_to_tensor(self.noise_weights),
                noise_mat=test_mat,#[( m,512), (m,512)]
                project_mat=project_mat,
            )        
        # Create the RNN layer with your custom cell and set it to stateless
        rnn_layer = RNN(
            rnn_cell,
            stateful=stateful,  # now stateless
            return_sequences=True,
            activity_regularizer=l2(0.1)
        )
        
        # Call the RNN layer; initial states will be automatically set to zeros
        outputs = rnn_layer(visible)
        # Here, outputs is a list of two tensors.
        hidden_outputs = outputs[0]  # shape: (batch, timesteps, 2*nUnit)
        Jacobian = outputs[1]        # shape: (batch, timesteps, 2*nUnit*2)
        Jacobian_sub=outputs[2]
        
        # Since the custom cell returns concatenated outputs, split them back into po_A and po_B sequences
        po_A_sequence, po_B_sequence = tf.split(hidden_outputs, num_or_size_splits=2, axis=-1)
        
        # Define the output layers
        output_A = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(po_A_sequence)
        output_B = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(po_B_sequence)
        
        # Concatenate the outputs
        output = Concatenate(axis=-1)([output_A, output_B])   
        
        if option.lower()=="noise_sing":
            norm_cell=Noise_jacob(nUnit=nUnit, start=start, sing_or_eig=True)
            dummy_input = tf.zeros((sample_size, 1))  # Dummy input if needed (can also pass None in later versions)
            initial_state = norm_cell.get_initial_state(inputs=dummy_input, batch_size=sample_size, dtype=tf.float32)
            Rlayer=RNN(norm_cell,stateful=stateful, return_sequences=True)# (batch_num,timestep,2*nUnits)
            Rmat = Rlayer(Jacobian, initial_state=initial_state) #size(batch, 1, (2*nUnit)^2)            
            
            norm_cell_2=Noise_jacob(nUnit=nUnit, start=start, sing_or_eig=True)
            dummy_input_2 = tf.zeros((sample_size, 1))  # Dummy input if needed (can also pass None in later versions)
            initial_state_2 = norm_cell_2.get_initial_state(inputs=dummy_input_2, batch_size=sample_size, dtype=tf.float32)
            Rlayer_2=RNN(norm_cell_2,stateful=stateful, return_sequences=True)# (batch_num,timestep,2*nUnits)
            Rmat_2 = Rlayer_2(Jacobian_sub, initial_state=initial_state_2) #size(batch, 1, (2*nUnit)^2)               
            
            model = Model(inputs=visible, outputs=[output, Rmat, Rmat_2]) 
        elif option.lower()=="noise_lyap_dir":
            #outputs_state=outputs[0] 
            jacobian_A_flat=outputs[1]  #(batch, m,m)
            jacobian_B_flat=outputs[2]  #(batch, m,m)
            jacobian_A_flat_sub=outputs[3]   #(batch, m,m)
            jacobian_B_flat_sub=outputs[4] #(batch, m,m)
            
            m=test_mat[0].shape[0]
            diag_A=DiagonalLayer(m)(jacobian_A_flat)  # shape = (batch_size, n)
            diag_B=DiagonalLayer(m)(jacobian_B_flat)  # shape = (batch_size, n)
            diag_A_sub=DiagonalLayer(m)(jacobian_A_flat_sub)  # shape = (batch_size, n)
            diag_B_sub=DiagonalLayer(m)(jacobian_B_flat_sub)  # shape = (batch_size, n)
                   
            model = Model(inputs=visible, outputs=[output, diag_A, diag_B, diag_A_sub, diag_B_sub])  
        return model
    
    
    
    # perturb the model and record correlation of the activity with the perturbation noise
    def build_model_perturb_noise_prob_brown_corr(self, nUnit, nInh, nInput, con_prob, maxval, ReLUalpha,
                                       pert_ind, pert_which, seed1, pert_noisesd, tau, rank, exp, sample_size,
                                       stateful=True, noise_weights=None, noise_vec=None, sync_noise=True, 
                                       option="running_corr",mat_AB=None):
        A_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        B_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        if stateful is False:
            visible = Input(shape=(None,nInput+rank)) 
        else:
            visible = Input(batch_shape=(sample_size, None, nInput+rank))
        
        if noise_weights is None:
            self.noise_weights=self.create_noise_weights(nUnit,rank,exp,2)
            print('Random noise weight created')
        else:
            self.noise_weights=noise_weights.copy()
        
        rnn_cell = RNNCustom2FixPerturb_noise_dir_prob_brown_return_noise(
            nUnit,
            output_activation=tf.keras.layers.ReLU(max_value=1000),
            input_activation=tf.keras.layers.LeakyReLU(alpha=ReLUalpha),
            use_bias=False,
            kernel_initializer=initializers.RandomNormal(mean=0., stddev=1/np.sqrt(nInput), seed=seed1),
            recurrent_initializer=OrthoCustom3(gain=0.5, seed=seed1, nUnit=nUnit, nInh=nInh, conProb=con_prob),
            recurrent_constraint=IEWeightandLim(nInh=nInh, A_mask=A_mask, B_mask=B_mask, maxval=maxval),
            kernel_trainable=True,
            seed=seed1,
            tau=tau,
            noisesd=0.08,
            perturb_ind=pert_ind,
            pert_which=pert_which,
            pert_noisesd=pert_noisesd,
            noise_weights=tf.convert_to_tensor(self.noise_weights),
            noise_vec=noise_vec,
            sync_noise=sync_noise,
        )
        
        # Create the RNN layer with your custom cell and set it to stateless
        rnn_layer = RNN(
            rnn_cell,
            stateful=stateful,  # now stateless
            return_sequences=True,
            activity_regularizer=l2(0.1)
        )
        
        # Call the RNN layer; initial states will be automatically set to zeros
        rnn_out = rnn_layer(visible)
        po_A, po_B, noise_add_A, noise_add_B, bin_A, bin_B=rnn_out
                
        
        
        merged = Concatenate(axis=-1)(
            [po_A, po_B, noise_add_A, noise_add_B, bin_A, bin_B]
        )   # shape = (batch, time, 512+512+512+512+1+1)
        
        
        if option.lower()=="running_corr":
            corr_cell=Running_correlation(nUnit=nUnit)
            # 2) Build your stateful RNN exactly once (no initial_state kwarg)
            corr_rnn_layer = RNN(
                corr_cell,
                stateful=stateful,         # True
                return_sequences=True,
            )
            # 3) Call it on the merged tensor
            corr_RNN = corr_rnn_layer(merged)
        elif option.lower()=="cca_values":
            output_dim=tf.shape(mat_AB[0][0])[1]
            trans_A=mat_AB[0]
            trans_B=mat_AB[1]
            cca_A = Dense(
                units=output_dim,
                use_bias=True,  # or True if you also want a fixed bias
                kernel_initializer=initializers.Constant(trans_A[0]),
                bias_initializer=initializers.Constant(trans_A[1]),
                trainable=False
            )(po_A)       
            cca_B = Dense(
                units=output_dim,
                use_bias=True,  # or True if you also want a fixed bias
                kernel_initializer=initializers.Constant(trans_B[0]),
                bias_initializer=initializers.Constant(trans_B[1]),
                trainable=False
            )(po_B)        
            corr_RNN=Concatenate(axis=-1)([cca_A , cca_B])
            
        
        # Define the output layers
        output_A = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(po_A)
        output_B = Dense(2, activation='tanh',
                         kernel_initializer=GlorotUniform(seed=seed1),
                         kernel_constraint=IEWeightOut(nInh=nInh))(po_B)
        
        # Concatenate the outputs
        output = Concatenate(axis=-1)([output_A, output_B])
        bin_AB = Concatenate(axis=-1)([bin_A, bin_B])
        
        # out: (batch, time, -> )4, 2, 2
        model = Model(inputs=visible, outputs=[output, corr_RNN, bin_AB])
        return model
    
    def perturb_noise_corr(self, trial1, ind1, pert_which, order, pert_noisesd, stop,
                                        sample_size, trial_num, inputdur, nInput,
                                        nUnit, nInh, con_prob, maxval, ReLUalpha, seed1, tau, model,
                                        brown_scale, rank, exp, batch_size, stateful=True, 
                                        noise_weights=None, noise_vec=None,sync_noise=True,
                                        option="running_corr",mat_AB=None):
        # perturb state along noise_vec
        # if option is cca_values, then the state is projected to dimension specified by mat_AB and the difference is evaluated
        # Create input and output arrays using the helper function.
        if stateful is False:
            x, y, In_ons = self.makeInOut_sameint_brown(sample_size, trial_num, inputdur, nInput, order,
                                                          brown_scale=brown_scale, rank=rank, exp=exp)
        else:
            x, y, In_ons, Wbin= self.makeInOut_sameint_brown_stateful(sample_size, trial_num, 
                                                            inputdur, nInput, 
                                                            order,brown_scale=brown_scale, 
                                                            rank=rank, exp=exp,
                                                            batch_num=batch_size)

        start=In_ons[0,2]
        self.start=start
        # Compute the perturbation indices.
        pert_ind = In_ons[:, [trial1]] + ind1
        
        # Optionally modify the input if stop is True.
        if stop:
            x, In_ons = self.makeInput(x, In_ons, pert_ind)
            

        model2 = self.build_model_perturb_noise_prob_brown_corr(nUnit=nUnit, nInh=nInh, nInput=nInput,
                                                     con_prob=con_prob, maxval=maxval, ReLUalpha=ReLUalpha,
                                                     pert_ind=pert_ind, pert_which=pert_which,
                                                     seed1=seed1, pert_noisesd=pert_noisesd, tau=tau,
                                                     rank=rank,exp=exp,sample_size=sample_size,
                                                     stateful=stateful,noise_weights=noise_weights,
                                                     noise_vec=noise_vec,
                                                     sync_noise=sync_noise,
                                                     option=option,mat_AB=mat_AB)
            
        # Copy the weights from the provided model.
        self.model2=model2
        if option.lower()=="running_corr":
            model2.set_weights(model.get_weights())
        elif option.lower()=="cca_values":
            w_model  = model.get_weights()         # length == 6
            w_model2 = model2.get_weights()        # length == 8
            
            # 2) overwrite the corresponding entries
            for i in range(len(w_model)):
                w_model2[i] = w_model[i]
            
            # 3) push them back into model2
            model2.set_weights(w_model2)
        
        
        
        model2.reset_states()
        
        if not stateful:
            predictions = model2.predict(x)
        else:
            for i in range(len(x)):
                xin = tf.stop_gradient(tf.convert_to_tensor(x[i], tf.float32))
                pred = model2.predict_on_batch(xin)
                if not isinstance(pred, (list, tuple)):
                    pred = [pred]
                if i == 0:
                    outputs_split = [[] for _ in pred]
                for j in range(len(pred)):
                    outputs_split[j].append(pred[j])
    
            # stitch the time segments back together
            predictions = [
                np.concatenate(output_list, axis=1)
                for output_list in outputs_split
            ]
            
        self.In_ons_temp=In_ons
        return predictions #
    
    
    
    
    def RNNCustom2FixPerturb_noise_prob_brown_noise_jacobian_output(self,nUnit, nInh, nInput, sample_size, con_prob, maxval, ReLUalpha,
                                        pert_ind, pert_which, seed1, pert_noisesd, tau, rank,exp=0.5,noise_weights=None,
                                        stateful=False,start=0, option="lyapunov",test_mat=None, project_mat=None, update_state=True):
        A_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        B_mask = self.build_masks(nUnit, nInh, con_prob, seed=seed1)
        if stateful is False:
            visible = Input(shape=(None,nInput+rank)) 
        else:
            visible = Input(batch_shape=(sample_size, None, nInput+rank))
        
        if noise_weights is None:
            self.noise_weights=self.create_noise_weights(nUnit,rank,exp,2)
            print('Random noise weight created')
        else:
            self.noise_weights=noise_weights.copy()
        
        if option.lower()=="noise_sing":
            rnn_cell = RNNCustom2FixPerturb_noise_prob_brown_noise_jacobian_output(
                nUnit,
                output_activation=tf.keras.layers.ReLU(max_value=1000),
                input_activation=lambda x: tf.where(x > 0, x, ReLUalpha * x),
                use_bias=False,
                kernel_initializer=initializers.RandomNormal(mean=0., stddev=1/np.sqrt(nInput), seed=seed1),
                recurrent_initializer=OrthoCustom3(gain=0.5, seed=seed1, nUnit=nUnit, nInh=nInh, conProb=con_prob),
                recurrent_constraint=IEWeightandLim(nInh=nInh, A_mask=A_mask, B_mask=B_mask, maxval=maxval),
                kernel_trainable=True,
                seed=seed1,
                tau=tau,
                noisesd=0.08,
                perturb_ind=pert_ind,
                pert_which=pert_which,
                pert_noisesd=pert_noisesd,
                noise_weights=tf.convert_to_tensor(self.noise_weights),
            )
     
        # Create the RNN layer with your custom cell and set it to stateless
        rnn_layer = RNN(
            rnn_cell,
            stateful=stateful,  # now stateless
            return_sequences=True,
            activity_regularizer=l2(0.1)
        )
        
        # Call the RNN layer; initial states will be automatically set to zeros
        outputs = rnn_layer(visible)
        # Here, outputs is a list of two tensors.
        hidden_outputs = outputs[0]  # shape: (batch, timesteps, 4)
        Jacobian = outputs[1]        # shape: (batch, timesteps, 4*2)
        
        norm_cell=Noise_jacob(nUnit=2, start=start, sing_or_eig=True)
        dummy_input = tf.zeros((sample_size, 1))  # Dummy input if needed (can also pass None in later versions)
        initial_state = norm_cell.get_initial_state(inputs=dummy_input, batch_size=sample_size, dtype=tf.float32)
        Rlayer=RNN(norm_cell,stateful=stateful, return_sequences=True)# (batch_num,timestep,2*nUnits)
        Rmat = Rlayer(Jacobian, initial_state=initial_state) #size(batch, 1, (2*nUnit)^2)            
        
        model = Model(inputs=visible, outputs=[hidden_outputs, Rmat, Jacobian]) 
        return model
    
    def run_model_with_jacobian_output_noise(self, trial1, ind1, pert_which, order, pert_noisesd, stop,
                                                   sample_size, trial_num, inputdur, nInput,
                                                   nUnit, nInh, con_prob, maxval, ReLUalpha, seed1, tau, model,
                                                   brown_scale, rank, exp=1, noise_weights=None,stateful=False, start=None,
                                                   option="lyapunov",test_mat=None, batch_size=10, update_state=True,
                                                   lr_AB=[None, None], lr_BA=[None,None],jacob_at_zero=False, project_mat=None):

        # Create input and output arrays using the helper function.
        if stateful is False:
            x, y, In_ons = self.makeInOut_sameint_brown(sample_size, trial_num, inputdur, nInput, order,
                                                          brown_scale=brown_scale, rank=rank, exp=exp)
        else:
            x, y, In_ons, Wbin= self.makeInOut_sameint_brown_stateful(sample_size, trial_num, 
                                                            inputdur, nInput, 
                                                            order,brown_scale=brown_scale, 
                                                            rank=rank, exp=exp,
                                                            batch_num=batch_size)
        
        if start is None:
            start=In_ons[0,2]
            self.start=start
        # Compute the perturbation indices.
        pert_ind = In_ons[:, [trial1]] + ind1
        
        # Optionally modify the input if stop is True.
        if stop:
            x, In_ons = self.makeInput(x, In_ons, pert_ind)
        
        # Build a new model that computes the lyapunov exponents with the provided parameters.
        model2 = self.RNNCustom2FixPerturb_noise_prob_brown_noise_jacobian_output(
            nUnit, nInh, nInput, sample_size, con_prob, maxval, ReLUalpha,
            pert_ind, pert_which, seed1, pert_noisesd, tau, rank,exp=exp,noise_weights=noise_weights,
            stateful=stateful,start=start,option=option,
            test_mat=test_mat,update_state=update_state, project_mat=project_mat
        )    
        self.model2=model2
        
        # transfer the weight from model to model2
        rnn_core_w = model.layers[1].get_weights()   # [kernel, recurrent_kernel, bias]
        d2_w        = model.layers[3].get_weights()   # [dense2_kernel, dense2_bias]
        d3_w        = model.layers[4].get_weights()   # [dense3_kernel, dense3_bias]
        
        # 2) Concatenate into the order your custom cell expects:
        #    [ core_kernel, core_recurrent, core_bias,
        #      dense1_kernel, dense1_bias,
        #      dense2_kernel, dense2_bias ]
        new_weights = rnn_core_w + d2_w + d3_w
        
        # 3) Assign into your custom-cell RNN
        rnn_custom = model2.layers[1]    # absolute index of the RNN wrapper
        rnn_custom.set_weights(new_weights)
        
        #model2.summary()
        model2.reset_states()
        
        
        # 3) Compare them
        fun_w_all   = rnn_core_w + d2_w + d3_w
        
        # 2) Pull out the custom‐cell RNN weights
        rnn_custom_w = model2.layers[1].get_weights()
        
        # 3) Compare them
        all_same = True
        for idx, (wf, wc) in enumerate(zip(fun_w_all, rnn_custom_w)):
            same = np.allclose(wf, wc, atol=1e-6)
            print(f"Weight #{idx}: shape {wf.shape} — match? {same}")
            if not same:
                all_same = False
        
        print("\nAll weights identical:", all_same)
        
        
        # Run the prediction.
        if stateful is False:
            predictions = model2.predict(x) #(output, Rmat, A_activity, B_activity
        else:
            for i in range(len(x)):
                print(f"prediction: {i} out of {len(x)}")
                xin = tf.convert_to_tensor(x[i], dtype=tf.float32)
                xin=tf.stop_gradient(xin)
                pred = model2.predict_on_batch(xin)  # pred = (output, Rmat, A_activity, B_activity)
                if not isinstance(pred, (list, tuple)):
                    pred = [pred] #1, sample_size, time, output_dim
                if i==0:
                    outputs_split = [[] for _ in range(len(pred))]
                for j in range(len(pred)):
                    outputs_split[j].append(pred[j])
        
            # Concatenate each type of output across time (axis=1)
            predictions = [np.concatenate(output_list, axis=1) for output_list in outputs_split]
        
        self.In_ons_temp=In_ons
        return predictions #
        