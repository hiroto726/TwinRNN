# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 15:01:09 2024

@author: Hiroto
"""
import argparse
def main(args):
    # assess synchronicity of outputs after perturbation. ramp up starting from the latter half. different weights
    import os
    import shutil
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import random
    import scipy
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.decomposition import PCA
    
    # use stateful rnn
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, SimpleRNN, SimpleRNNCell, GaussianNoise, RNN, Concatenate
    from tensorflow.python.keras import backend, activations, constraints, initializers, regularizers, layers
    from tensorflow.python.keras.initializers import GlorotUniform
    import sys
    sys.path.append(r'C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_scripts\my2RNN_fix2')
    from RNNcustom_2_fix2 import RNNCustom2Fix2
    from CustomConstraintWithMax import IEWeightandLim, IEWeightOut
    from WInitial_3 import OrthoCustom3
    from get_folders import load_checkpoint_with_max_number
    #from GaussianNoiseCustom import GaussianNoiseAdd
    from tensorflow.keras.regularizers import l1, l2
    from parula_colormap import parula
    
    # set settings for saving svg images so that text are saved as texts
    # Apply your configurations 
    plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG 
    plt.rcParams['text.usetex'] = False
    
    figure_folders=r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\Analysis_folder\figures"
    analysis_folder=r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\Analysis_folder"
    
    savedirectory=[r'C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\t4stateful',
                   r'C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\t4stateful2',
                   r'C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\t4stateful3',
                   r'C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\t4stateful4',
                   r'C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\t4stateful5',
                   r'C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\t4stateful6']
    # os.chdir(savedirectory)
    
    # save options, if set to true, a new folder will be created and the weights and script will be saved in the folder
    max_iteration=50000
    
    
    #20240829
    weight_max=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    conProbability=[1.e-05, 3.e-05, 1.e-04, 3.e-04, 1.e-03, 3.e-03, 1.e-02, 3.e-02, 1.e-01,3.e-01,1.e+00]
    #model_index=[3173,3855,3660,2095,2909,1608,3160,2295,2641,3754,3754]
    
    
    
    def get_numbers_after_period(number):
      # Convert the number to a string
      number_str = f'{number:.10f}'.rstrip('0').rstrip('.')
      
      # Split the string at the period
      parts = number_str.split(".")
      
      # Check if there's a decimal part
      if len(parts) > 1:
          # Return the part after the period
          return parts[1]
      else:
          # If there's no period, return an empty string
          return "" 
      
        
    # each list contains max weight, directory, and model index
    foldername=[]
    best_models=[]
    for k in range(len(savedirectory)):
        #foldername_sub=[[fr"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\t4stateful\2RNNs_prob{get_numbers_after_period(conProbability[i])}_weightmax{get_numbers_after_period(weight_max[i])}_fix3"]for con, wei in zip(conProbability, weight_max)]
        foldername_sub=[os.path.join(savedirectory[k],fr"2RNNs_prob{get_numbers_after_period(con)}_weightmax{get_numbers_after_period(wei)}_fix3")for con, wei in zip(conProbability, weight_max)]
        foldername.append(foldername_sub)
        best_models.append([load_checkpoint_with_max_number(i)[0] for i in foldername_sub])
    
            
    
    
    seed=11
    seed1=int(seed)
    np.random.seed(seed1)
    tf.random.set_seed(seed1)
    random.seed(seed1)
    nUnit=512
    batch_sz=32
    min_dur=6000 # minimum duration
    max_dur=12000 # maximum duration
    # make input function
    inputdur=int(30) # duration of input in ms
    nInh=int(np.ceil(nUnit/5)) # number of inhibitory units (from dales law)
    nInput=1# number of input units
    sample_size=8
    tau=100
    dt=10 #discretization time stamp
    trial_num=3
    con_prob=0.005 # probability of connection between neurons
    maxval=0.003
    ReLUalpha=0.2
    
    # scale time parameters by dt
    min_dur=int(min_dur/dt) # minimum duration
    max_dur=int(max_dur/dt) # maximum duration
    # make input function
    inputdur=int(inputdur/dt) # duration of input in ms
    tau=tau/dt
    
    
    
    
    
    
    # make inputs and outputs
    def makeInOut(sample_size,trial_num,inputdur,nInput,min_dur,max_dur,dt):
        # make inputs and outputs
        # there are 2 kinds of input and for each input, output is a slowly increasing activity
        #total_time=100+2*inputdur+max_dur+100
        noise_range=1/20 #level of temporal noise with respect to duration
        max_dur_max=np.ceil((1+noise_range)*max_dur)
        min_dur_max=np.ceil((1+noise_range)*min_dur)
        
        total_time_orig=int(min_dur_max*np.floor(trial_num/2)+max_dur_max*np.ceil(trial_num/2)+300/dt)
        trial_num_orig=trial_num
        trial_num=2*np.ceil(trial_num/2)
        total_time=int(min_dur_max*np.floor(trial_num/2)+max_dur_max*np.ceil(trial_num/2)+300/dt)
        x=np.zeros((sample_size,total_time,nInput)) # x is the input
        y=-0.5*np.ones((sample_size,total_time,2)) # y is the ouput
        context_mag=0.2 # magnitude of contextual output
        
        minratio=min_dur/(max_dur+min_dur)
        
        # make random input for all samples
        In_ons=np.zeros((sample_size,int(trial_num)),dtype=np.int64)
        binvec=[0,1]
        for i in range(sample_size):
            vec=total_time
            for j in np.arange(trial_num):
                vecbf=vec
                if j % 2==binvec[i % 2]:
                    vec-=min_dur+random.randint(-int(min_dur*noise_range),int(min_dur*noise_range))
                    cont_output=context_mag
                else:
                    vec-=max_dur+random.randint(-int(max_dur*noise_range),int(max_dur*noise_range))
                    cont_output=-context_mag
                In_ons[i,int(-j-1)]=vec
                in_start=int(0.5*(vec+vecbf))
                Dur=vecbf-vec
                x[i,vec:vec+inputdur,:]=1
                #y[i,in_start:vecbf,0]=np.linspace(0,1,num=vecbf-in_start)-0.5  # relative timing 1
                y[i,vec:vecbf,0]=np.power(np.linspace(0,1,num=Dur),4)-0.5
                y[i,vec:vecbf,1]=np.arange(-0.5,-0.5+(Dur/max_dur)-1e-10,1/max_dur) # aboslute timing 1
    
                
        x+=np.random.normal(loc=0.0, scale=0.01, size=np.shape(x))
        y=np.tile(y,(1,1,2))    
        x=x[:,:total_time_orig,:]
        y=y[:,:total_time_orig,:]
        return x, y, In_ons
    
    def makeInOutphase(sample_size,trial_num,inputdur,nInput,min_dur,max_dur,dt):
        # there is no variability in durations!
        # make inputs and outputs
        # there are 2 kinds of input and for each input, output is a slowly increasing activity
        #total_time=100+2*inputdur+max_dur+100
        noise_range=0 #level of temporal noise with respect to duration
        max_dur_max=np.ceil((1+noise_range)*max_dur)
        min_dur_max=np.ceil((1+noise_range)*min_dur)
        
        total_time_orig=int(min_dur_max*np.floor(trial_num/2)+max_dur_max*np.ceil(trial_num/2)+300/dt)
        trial_num_orig=trial_num
        trial_num=2*np.ceil(trial_num/2)
        total_time=int(min_dur_max*np.floor(trial_num/2)+max_dur_max*np.ceil(trial_num/2)+300/dt)
        x=np.zeros((sample_size,total_time,nInput)) # x is the input
        y=-0.5*np.ones((sample_size,total_time,2)) # y is the ouput
        phase=np.zeros_like(x)
        context_mag=0.2 # magnitude of contextual output
        
        minratio=min_dur/(max_dur+min_dur)
        mid_dur=0.5*(min_dur+max_dur)
        
        # make random input for all samples
        In_ons=np.zeros((sample_size,int(trial_num)),dtype=np.int64)
        binvec=[0,1]
        for i in range(sample_size):
            vec=total_time
            for j in np.arange(trial_num):
                vecbf=vec
                if j % 2==binvec[i % 2]:
                    vec-=min_dur+random.randint(-int(min_dur*noise_range),int(min_dur*noise_range))
                    cont_output=context_mag
                else:
                    vec-=max_dur+random.randint(-int(max_dur*noise_range),int(max_dur*noise_range))
                    cont_output=-context_mag
                In_ons[i,int(-j-1)]=vec
                in_start=int(0.5*(vec+vecbf))
                Dur=vecbf-vec
                x[i,vec:vec+inputdur,:]=1
                #y[i,in_start:vecbf,0]=np.linspace(0,1,num=vecbf-in_start)-0.5  # relative timing 1
                y[i,vec:vecbf,0]=np.power(np.linspace(0,1,num=Dur),4)-0.5
                y[i,vec:vecbf,1]=np.arange(-0.5,-0.5+(Dur/max_dur)-1e-10,1/max_dur) # aboslute timing 1
                if Dur<mid_dur:
                    phase[i,vec:vecbf,0]=np.linspace(-np.pi,-np.pi+2*np.pi*minratio,num=Dur)
                else:
                    phase[i,vec:vecbf,0]=np.linspace(-np.pi+2*np.pi*minratio,np.pi,num=Dur)
                            
        x+=np.random.normal(loc=0.0, scale=0.01, size=np.shape(x))
        y=np.tile(y,(1,1,2))    
        x=x[:,:total_time_orig,:]
        y=y[:,:total_time_orig,:]
        return x, y, In_ons, phase
    
    
    
    
    def build_masks(nUnit,nInh, con_prob,seed):
        random_matrix = tf.random.uniform([nUnit-nInh,nUnit], minval=0, maxval=1,seed=seed)
        # Apply threshold to generate binary values
        mask_A_1 = tf.cast(tf.random.uniform([nUnit-nInh,nUnit], minval=0, maxval=1)< con_prob, dtype=tf.int32)
        mask_A=tf.concat([mask_A_1,tf.zeros([nInh,nUnit],dtype=tf.int32)],0)
        return mask_A
    
    
    def build_model(nUnit,nInh,nInput,con_prob,maxval,ReLUalpha,seed1):
        A_mask=build_masks(nUnit,nInh, con_prob,seed=seed1)
        B_mask=build_masks(nUnit,nInh, con_prob,seed=seed1)
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
    
    
    model=build_model(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha, seed1=seed1)
    print(model.summary())
    
    
    #% pertrubation with dedicated model
    from RNNcustom_2_perturb import RNNCustom2FixPerturb
    
    
    
    def build_model_perturb(nUnit,nInh,nInput,con_prob,maxval,ReLUalpha,pert_ind, pert_state,seed1):
        A_mask=build_masks(nUnit,nInh, con_prob,seed=seed1)
        B_mask=build_masks(nUnit,nInh, con_prob,seed=seed1)
        visible = Input(shape=(None,nInput)) 
        #vis_noise=GaussianNoiseAdd(stddev=0.01, seed=seed1)(visible)# used to be 0.01*np.sqrt(tau*2)
        #hidden1 = SimpleRNN(nUnit,activation='tanh', use_bias=False, batch_size=batch_sz, stateful=False, input_shape=(None, 1), return_sequences=True)(vis_noise)
    
        # the code below incorporated options to train input kernel within RNN layer
        hidden1=RNN(RNNCustom2FixPerturb(nUnit, 
                              output_activation=tf.keras.layers.ReLU(max_value=1000),
                              input_activation=tf.keras.layers.LeakyReLU(alpha=ReLUalpha),
                              use_bias=False,
                              kernel_initializer=initializers.RandomNormal(mean=0., stddev=1/np.sqrt(nInput), seed=seed1), # kernel initializer should be random normal
                              recurrent_initializer=OrthoCustom3(gain=0.5, seed=seed1 , nUnit=nUnit, nInh=nInh, conProb=con_prob),
                              recurrent_constraint=IEWeightandLim(nInh=nInh,A_mask=A_mask,B_mask=B_mask,maxval=maxval),
                              kernel_trainable=True,
                              seed=seed1,
                              tau=tau, 
                              noisesd=0.08,
                              perturb_ind=pert_ind,
                              pert_state=pert_state
                              ), # used to be 0.05*np.sqrt(tau*2)
                    stateful=False, 
                    input_shape=(None, nInput), 
                    return_sequences=True,
                    activity_regularizer=l2(0.01),# used to be 0.0001, 0.000001
                    #recurrent_regularizer=l2(0.000001)
                    )(visible)
        #  hidden2 = Dense(10, activation='relu')(hidden1)
        output_A = Dense(2, activation='tanh',kernel_initializer=GlorotUniform(seed=seed1), kernel_constraint=IEWeightOut(nInh=nInh))(hidden1[0])
        output_B = Dense(2, activation='tanh',kernel_initializer=GlorotUniform(seed=seed1), kernel_constraint=IEWeightOut(nInh=nInh))(hidden1[1])
        output=Concatenate(axis=2)([output_A,output_B])
        model = Model(inputs=visible, outputs=output)
        return model
    
    
    from RNNcustom_2_perturb_noise import RNNCustom2FixPerturb_noise
    def build_model_perturb_noise(nUnit,nInh,nInput,con_prob,maxval,ReLUalpha,pert_ind, pert_state,seed1, pert_noisesd):
        A_mask=build_masks(nUnit,nInh, con_prob,seed=seed1)
        B_mask=build_masks(nUnit,nInh, con_prob,seed=seed1)
        visible = Input(shape=(None,nInput)) 
        #vis_noise=GaussianNoiseAdd(stddev=0.01, seed=seed1)(visible)# used to be 0.01*np.sqrt(tau*2)
        #hidden1 = SimpleRNN(nUnit,activation='tanh', use_bias=False, batch_size=batch_sz, stateful=False, input_shape=(None, 1), return_sequences=True)(vis_noise)
    
        # the code below incorporated options to train input kernel within RNN layer
        hidden1=RNN(RNNCustom2FixPerturb_noise(nUnit, 
                              output_activation=tf.keras.layers.ReLU(max_value=1000),
                              input_activation=tf.keras.layers.LeakyReLU(alpha=ReLUalpha),
                              use_bias=False,
                              kernel_initializer=initializers.RandomNormal(mean=0., stddev=1/np.sqrt(nInput), seed=seed1), # kernel initializer should be random normal
                              recurrent_initializer=OrthoCustom3(gain=0.5, seed=seed1 , nUnit=nUnit, nInh=nInh, conProb=con_prob),
                              recurrent_constraint=IEWeightandLim(nInh=nInh,A_mask=A_mask,B_mask=B_mask,maxval=maxval),
                              kernel_trainable=True,
                              seed=seed1,
                              tau=tau, 
                              noisesd=0.08,
                              perturb_ind=pert_ind,
                              pert_state=pert_state,
                              pert_noisesd=pert_noisesd
                              ), # used to be 0.05*np.sqrt(tau*2)
                    stateful=False, 
                    input_shape=(None, nInput), 
                    return_sequences=True,
                    activity_regularizer=l2(0.01),# used to be 0.0001, 0.000001
                    #recurrent_regularizer=l2(0.000001)
                    )(visible)
        #  hidden2 = Dense(10, activation='relu')(hidden1)
        output_A = Dense(2, activation='tanh',kernel_initializer=GlorotUniform(seed=seed1), kernel_constraint=IEWeightOut(nInh=nInh))(hidden1[0])
        output_B = Dense(2, activation='tanh',kernel_initializer=GlorotUniform(seed=seed1), kernel_constraint=IEWeightOut(nInh=nInh))(hidden1[1])
        output=Concatenate(axis=2)([output_A,output_B])
        model = Model(inputs=visible, outputs=output)
        return model
    
    
    
    
    #% perturbation with many noise ->  then decode
    
    #build model
    from RNNcustom_2_perturb_noise_prob import RNNCustom2FixPerturb_noise_prob
    def build_model_perturb_noise_prob(nUnit,nInh,nInput,con_prob,maxval,ReLUalpha,pert_ind,pert_which, seed1, pert_noisesd):
        A_mask=build_masks(nUnit,nInh, con_prob,seed=seed1)
        B_mask=build_masks(nUnit,nInh, con_prob,seed=seed1)
        visible = Input(shape=(None,nInput)) 
        #vis_noise=GaussianNoiseAdd(stddev=0.01, seed=seed1)(visible)# used to be 0.01*np.sqrt(tau*2)
        #hidden1 = SimpleRNN(nUnit,activation='tanh', use_bias=False, batch_size=batch_sz, stateful=False, input_shape=(None, 1), return_sequences=True)(vis_noise)
    
        # the code below incorporated options to train input kernel within RNN layer
        hidden1=RNN(RNNCustom2FixPerturb_noise_prob(nUnit, 
                              output_activation=tf.keras.layers.ReLU(max_value=1000),
                              input_activation=tf.keras.layers.LeakyReLU(alpha=ReLUalpha),
                              use_bias=False,
                              kernel_initializer=initializers.RandomNormal(mean=0., stddev=1/np.sqrt(nInput), seed=seed1), # kernel initializer should be random normal
                              recurrent_initializer=OrthoCustom3(gain=0.5, seed=seed1 , nUnit=nUnit, nInh=nInh, conProb=con_prob),
                              recurrent_constraint=IEWeightandLim(nInh=nInh,A_mask=A_mask,B_mask=B_mask,maxval=maxval),
                              kernel_trainable=True,
                              seed=seed1,
                              tau=tau, 
                              noisesd=0.08,
                              perturb_ind=pert_ind,
                              pert_which=pert_which,
                              pert_noisesd=pert_noisesd
                              ), # used to be 0.05*np.sqrt(tau*2)
                    stateful=False, 
                    input_shape=(None, nInput), 
                    return_sequences=True,
                    activity_regularizer=l2(0.01),# used to be 0.0001, 0.000001
                    #recurrent_regularizer=l2(0.000001)
                    )(visible)
        #  hidden2 = Dense(10, activation='relu')(hidden1)
        output_A = Dense(2, activation='tanh',kernel_initializer=GlorotUniform(seed=seed1), kernel_constraint=IEWeightOut(nInh=nInh))(hidden1[0])
        output_B = Dense(2, activation='tanh',kernel_initializer=GlorotUniform(seed=seed1), kernel_constraint=IEWeightOut(nInh=nInh))(hidden1[1])
        output=Concatenate(axis=2)([output_A,output_B])
        model = Model(inputs=visible, outputs=output)
        return model
    
    
    # stop input after perturbation
    def makeInput(x,In_ons,pert_ind):
        In_ons2=[]
        for i in range(np.shape(In_ons)[0]):
            index = np.argmax(In_ons[i,:]>=pert_ind[i,0]) if np.any(In_ons[i,:]>=pert_ind[i,0]) else None
            x[i,In_ons[i,index]-1:,:]=np.random.normal(loc=0.0, scale=0.01, size=np.shape(x[i,In_ons[i,index]-1:,:]))
            In_ons2.append(In_ons[i,0:index-1])
        return x, In_ons2
    
    
    def avgAct(activities,In_ons,min_dur,max_dur):
        dura=[min_dur,max_dur]
        dur0=In_ons[0,1]-In_ons[0,0]
        ind1=np.argmin(dur0-dura)
        dur0_dur=dura[ind1]
        kk=0
        act_avg=np.zeros((min_dur+max_dur,np.shape(activities)[2]))
        for i in np.arange(np.shape(activities)[0]):
            In_time=In_ons[i,1+((ind1+i+1) % 2):-2:2]
            for j in In_time:
                act_avg+=np.squeeze(activities[i,j:j+min_dur+max_dur,:])
                kk=kk+1
        act_avg/=kk
        return act_avg
    
    def makeInOut_sameint(sample_size,trial_num,inputdur,nInput,min_dur,max_dur,dt,interval):
        # make intervals always start from min_dur or max_dur
        # make inputs and outputs
        # there are 2 kinds of input and for each input, output is a slowly increasing activity
        #total_time=100+2*inputdur+max_dur+100
        noise_range=0 #level of temporal noise with respect to duration
        max_dur_max=np.ceil((1+noise_range)*max_dur)
        min_dur_max=np.ceil((1+noise_range)*min_dur)
        total_time_orig=int(min_dur_max*np.floor(trial_num/2)+max_dur_max*np.ceil(trial_num/2)+300/dt)
        trial_num_orig=trial_num
        trial_num=2*np.ceil(trial_num/2)
        total_time=int(min_dur_max*np.floor(trial_num/2)+max_dur_max*np.ceil(trial_num/2)+300/dt)
        x=np.zeros((sample_size,total_time,nInput)) # x is the input
        y=-0.5*np.ones((sample_size,total_time,2)) # y is the ouput
        # make random input for all samples
        In_ons=np.zeros((sample_size,int(trial_num)),dtype=np.int64)
        for i in range(sample_size):
            vec=total_time
            for j in np.arange(trial_num):
                vecbf=vec
                if j % 2==interval:
                    vec-=min_dur+random.randint(-int(min_dur*noise_range),int(min_dur*noise_range))
                else:
                    vec-=max_dur+random.randint(-int(max_dur*noise_range),int(max_dur*noise_range))
                In_ons[i,int(-j-1)]=vec
                in_start=vec
                Dur=vecbf-in_start
                x[i,vec:vec+inputdur,:]=1
                y[i,in_start:vecbf,0]=np.power(np.linspace(0,1,num=Dur),4)-0.5        # relative timing 1
                #y[i,vec:vecbf,1]=np.arange(-0.5,-0.5+(Dur/max_dur)-1e-10,1/max_dur)
                y[i,in_start:vecbf,1]=np.power(np.linspace(0,1,num=Dur),4)-0.5 
        x+=np.random.normal(loc=0.0, scale=0.01, size=np.shape(x))
        y=np.tile(y,(1,1,2))
        x=x[:,:total_time_orig,:]
        y=y[:,:total_time_orig,:]
        return x, y, In_ons
    
    
    # perturb and decode
    def make_pertind(In_ons,trial1,ind1):
        addvec=np.array([int(ind1/dt)])
        pert_ind=In_ons[:,[trial1]]+addvec
        return pert_ind
    def makeit2d(actpart_A):
        [a,b,c]=np.shape(actpart_A)
        mat=np.zeros((a*c,b))
        for i in np.arange(c):
            mat[a*i:a*(i+1),:]=actpart_A[:,:,i]
        return mat
    
    def Act_2dsort(activities,In_ons,min_dur,max_dur):
        dura=[min_dur,max_dur]
        dur0=In_ons[0,1]-In_ons[0,0]
        ind1=np.argmin(dur0-dura)
        dur0_dur=dura[ind1]
        kk=0
        act_avg=np.zeros((min_dur+max_dur,np.shape(activities)[2]))
        for i in np.arange(np.shape(activities)[0]):
            In_time=In_ons[i,1+((ind1+i+1) % 2):-2:2]
            for j in In_time:
                addmat=np.squeeze(activities[i,j:j+min_dur+max_dur,:])
                act_avg=np.concatenate((act_avg,addmat),axis=0)
        act_avg=act_avg[min_dur+max_dur:,:]
        return act_avg
    
    def perturb_and_decode_noise_prob(trial1,ind1,pert_which,order,pca_A,pca_B,pca_C,clf_A,clf_B,clf_C,pert_noisesd,stop):
        x, y, In_ons=makeInOut_sameint(sample_size,trial_num,inputdur,nInput,min_dur,max_dur,dt,order)
            
        pert_ind=In_ons[:,[trial1]]+ind1
        if stop:
            x,In_ons2=makeInput(x,In_ons,pert_ind)
    
        #pert_ind=np.zeros((4,2))
        #model2=build_model_perturb(pert_ind, pert_state)
        model2=build_model_perturb_noise_prob(nUnit=nUnit, 
                                   nInh=nInh, 
                                   nInput=nInput, 
                                   con_prob=con_prob, 
                                   maxval=maxval ,
                                   ReLUalpha=ReLUalpha,
                                   pert_ind=pert_ind, 
                                   pert_which=pert_which,
                                   seed1=seed1,
                                   pert_noisesd=pert_noisesd)
        model2.set_weights(model.get_weights())
        predictions = model2.predict(x)
        
        #create model that output all intermediate layers
        outputs = [layer.output for layer in model2.layers[1:]]  # Exclude the input layer
        activity_model2 = Model(inputs=model2.input, outputs=outputs)
        output_and_activities2 = activity_model2.predict(x)
        activities_A = output_and_activities2[0]  # Activities of all intermediate layers
        activities_B=output_and_activities2[1]
        #predictions2=output_and_activities2[4]
        pert_ind_2=np.zeros((np.shape(pert_ind)))
        In_ons_2=np.zeros((np.shape(In_ons[:,trial1:])))
        
        trial2=trial1
        int_diff=int(np.round((In_ons[0,trial2]-In_ons[0,trial1])/min_dur)*min_dur)
        
        eightnum=int((trial_num-trial1)/2)
        actpart_A=np.zeros((np.shape(activities_A)[0],eightnum*(min_dur+max_dur),np.shape(activities_A)[2]))
        actpart_B=np.zeros((np.shape(activities_B)[0],eightnum*(min_dur+max_dur),np.shape(activities_B)[2]))
        predictions2=np.zeros((np.shape(predictions)[0],eightnum*(min_dur+max_dur),np.shape(predictions)[2]))
        # take predictions at the time of perturbation
        for i in np.arange(np.shape(In_ons)[0]):
            actpart_A[i,:,:]=activities_A[i,In_ons[i,trial2]:In_ons[i,trial2]+eightnum*(min_dur+max_dur),:]
            actpart_B[i,:,:]=activities_B[i,In_ons[i,trial2]:In_ons[i,trial2]+eightnum*(min_dur+max_dur),:]
            predictions2[i,:,:]=predictions[i,In_ons[i,trial2]:In_ons[i,trial2]+eightnum*(min_dur+max_dur),:]
            pert_ind_2[i,:]=pert_ind[i,:]-(In_ons[i,trial2]-int_diff)
            In_ons_2[i,:]=In_ons[i,trial1:]-(In_ons[i,trial2]-int_diff)
        
        actpart_A=np.transpose(actpart_A,(1,2,0))
        actpart_B=np.transpose(actpart_B,(1,2,0))
        #actpart_A= makeit2d(actpart_A)
        #actpart_B= makeit2d(actpart_B)
        
        
        pred_A=np.zeros((np.shape(actpart_A)[0],np.shape(actpart_A)[2]))
        pred_B=np.zeros((np.shape(actpart_B)[0],np.shape(actpart_B)[2]))
        pred_C=np.zeros((np.shape(actpart_B)[0],np.shape(actpart_B)[2]))
    
        
        actpart_C=np.concatenate((actpart_A,actpart_B),axis=1)
        for i in np.arange(np.shape(actpart_A)[2]):
            #dimension reduction
            proj_A=pca_A.transform(actpart_A[:,:,i])
            proj_B=pca_B.transform(actpart_B[:,:,i])
            proj_C=pca_C.transform(actpart_C[:,:,i])
            
            #decode    
            pred_A[:,i]=clf_A.predict(proj_A[:,:Dim])
            pred_B[:,i]=clf_B.predict(proj_B[:,:Dim])
            pred_C[:,i]=clf_C.predict(proj_C[:,:Dim])
        return pred_A, pred_B, pred_C, predictions2, pert_ind_2, In_ons_2
    
    
    
    #% remove a portion of inter rnn connections and study their synchronicity
    from Confmatrix import confscore, confmat
    def avgAct(activities,In_ons,min_dur,max_dur):
        dura=[min_dur,max_dur]
        dur0=In_ons[0,1]-In_ons[0,0]
        ind1=np.argmin(dur0-dura)
        dur0_dur=dura[ind1]
        kk=0
        act_avg=np.zeros((min_dur+max_dur,np.shape(activities)[2]))
        for i in np.arange(np.shape(activities)[0]):
            In_time=In_ons[i,1+((ind1+i+1) % 2):-2:2]
            for j in In_time:
                act_avg+=np.squeeze(activities[i,j:j+min_dur+max_dur,:])
                kk=kk+1
        act_avg/=kk
        return act_avg
    def Act_2dsort(activities,In_ons,min_dur,max_dur):
        dura=[min_dur,max_dur]
        dur0=In_ons[0,1]-In_ons[0,0]
        ind1=np.argmin(dur0-dura)
        dur0_dur=dura[ind1]
        kk=0
        act_avg=np.zeros((min_dur+max_dur,np.shape(activities)[2]))
        for i in np.arange(np.shape(activities)[0]):
            In_time=In_ons[i,1+((ind1+i+1) % 2):-2:2]
            for j in In_time:
                addmat=np.squeeze(activities[i,j:j+min_dur+max_dur,:])
                act_avg=np.concatenate((act_avg,addmat),axis=0)
        act_avg=act_avg[min_dur+max_dur:,:]
        return act_avg
    
    def reduce_nonzero_elements(Wr_A, m):
        n_nonzero = np.sum(Wr_A>0)
        k = n_nonzero - m
        if k <= 0:
            # No need to reduce; m is greater than or equal to the current number of non-zero elements
            return Wr_A.copy()
        
        # get index of nonzero elements
        nonzero_ind=np.argwhere(Wr_A>0)
        nonzero_linind=np.ravel_multi_index(nonzero_ind.T,np.shape(Wr_A)) # convert to linear index
        
        # choose random elements to set to 0
        indices_to_zero = np.random.choice(nonzero_linind, size=k, replace=False)
        ind2sub=np.unravel_index(indices_to_zero,np.shape(Wr_A))
        # set the appropriate index to 0
        Wr_A_new=Wr_A.copy()
        Wr_A_new[ind2sub]=0
        return Wr_A_new
    
    def set_nonzero_elements(Wr_A, m):
        # set number of 0 elements in Wr_A to m randomly
        n_nonzero = np.sum(Wr_A>0)
        k = n_nonzero - m
        if k <= 0:
            k=-k
            # get index of zero elements
            nonzero_ind=np.argwhere(Wr_A==0)
            nonzero_linind=np.ravel_multi_index(nonzero_ind.T,np.shape(Wr_A)) # convert to linear index
            
            # choose random elements to set to 0
            indices_to_zero = np.random.choice(nonzero_linind, size=k, replace=False)
            ind2sub=np.unravel_index(indices_to_zero,np.shape(Wr_A))
            # set the appropriate index to 0
            Wr_A_new=Wr_A.copy()
            Wr_A_new[ind2sub]=0
            
    
        else:
            # get index of nonzero elements
            nonzero_ind=np.argwhere(Wr_A>0)
            nonzero_linind=np.ravel_multi_index(nonzero_ind.T,np.shape(Wr_A)) # convert to linear index
            
            # choose random elements to set to 0
            indices_to_zero = np.random.choice(nonzero_linind, size=k, replace=False)
            ind2sub=np.unravel_index(indices_to_zero,np.shape(Wr_A))
            # set the appropriate index to 0
            Wr_A_new=Wr_A.copy()
            Wr_A_new[ind2sub]=0
        return Wr_A_new
    
    def sort_and_scale(act_avg_A,start,stop):
        # act_avg_A is matrix with size (time, nCells)
        # this function first sort cells accorcding to the time of maximum activity from start to stop
        # then this function scale the value so that the maximum is 1.
        act_avg_A=np.array(act_avg_A)
        maxtime=np.argmax(act_avg_A[start:stop+1,:],0)
        maxtime_sort=np.argsort(maxtime)
        act_avg_A=act_avg_A[:,maxtime_sort]
        act_avg_new=act_avg_A/np.maximum(np.max(act_avg_A[start:stop+1,:],0),1e-10) # scale the matrix to set maximum to 1
        return act_avg_new
    
    sample_size=12
    trial_num=8
    pert_state=0 # 0 to perturb RNN A and 1 to perturb RNN B
    pert_noisesd=0.8#1.5 original 0.8
    stop=False
    # take circular mean of the prediction
    option=0 #0 for circular, 1 for mean
    trial1=2
    
    
    pert_prob=1/100 # probability of perturbation original 1/200
    pert_A_prob=0.5 # probability of perturbing RNNA 
    
    order=1# order 0 starts all trials with max_dur, 1 starts with min_dur
    
    max_ind=int(np.floor((min_dur+max_dur)*(np.floor((trial_num-trial1)/2)*19/20)))
    pert_number=int(np.floor(max_ind*pert_prob))
    vectors=[]
    for i in range(sample_size):
        time0 = np.random.randint(0, max_ind, pert_number)
        time0.sort()
        time0=np.reshape(time0,(1,-1))
        vectors.append(time0)
    time_1 = np.concatenate(vectors, axis=0)
    
    pert_which=np.random.uniform(size=np.shape(time_1))
    pert_which=pert_which<pert_A_prob
    
    
    nonzero_num_ratio=np.array([1,0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0])
    repnum=3 # number of repetitions
    


    
    confavg_A_all=[]
    confavg_B_all=[]
    pred_diff_all2=[]
    k_values = args.k_values
    kloop=np.arange(k_values,k_values+1)
    #for k_ind in range(np.shape(best_models)[0]): 
    for k_ind in kloop:     
        pred_diff_cat=[]
        pred_diff_nan=np.empty((len(nonzero_num_ratio),len(conProbability)))
        pred_diff_nan[:]=np.nan
        pred_diff_ALL=[]
        confsc_A=[]
        confsc_B=[]
        pred_diff_all=np.zeros((len(nonzero_num_ratio),len(conProbability)))
        confavg_A=np.zeros((len(nonzero_num_ratio),len(conProbability)))
        confavg_B=np.zeros((len(nonzero_num_ratio),len(conProbability)))
    
                     
        for ss in range(len(conProbability)):
            modelind=ss
            conprobmodel=conProbability[modelind]
            
            pred_diff_ALL.append(np.zeros((len(nonzero_num_ratio),repnum)))
            confsc_A.append(np.zeros((len(nonzero_num_ratio),repnum)))
            confsc_B.append(np.zeros((len(nonzero_num_ratio),repnum)))
            
            nonzero_num=np.round(nUnit*(nUnit-nInh)*conProbability[ss]*nonzero_num_ratio).astype(np.int32)  
            
            for t in range(len(nonzero_num)):# the last index is for case when there is no change added to S_A and S_B
                #  analyze weight distribution
                
                max_range=[0,max_dur+min_dur] # time range to choose max firing time
                nExc=nUnit-nInh #number of excitory units
                
                for k in range(repnum): 
                    model=build_model(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha, seed1=seed1)
                    # load weights
                    checkpoint_filepath=best_models[k_ind][ss]
                    model.load_weights(checkpoint_filepath)
                    
                    
                    
                    #get average activity
                    outputs = [layer.output for layer in model.layers[1:]]  # Exclude the input layer
                    activity_model = Model(inputs=model.input, outputs=outputs)
                    trial_num=4
                    # Get the output and activities of all layers for the new input data
                    x, y, In_ons=makeInOut(sample_size,trial_num+4,inputdur,nInput,min_dur,max_dur,dt)
                    #xnew=sum(xnew,axis=2)
                    output_and_activities = activity_model.predict(x)
                    activities_A = output_and_activities[0]  # Activities of all intermediate layers
                    activities_B=output_and_activities[1]
                    act_avg_A=avgAct(activities_A,In_ons,min_dur,max_dur)
                    act_avg_B=avgAct(activities_B,In_ons,min_dur,max_dur)
                    
            
                    
                    
                    # the weights for this RNN
                    RNN_input_kernel=model.layers[1].get_weights()[0]
                    RNN_layer_Recurrent_kernel=model.layers[1].get_weights()[1]
                    dense_kernel_A=model.layers[2].get_weights()[0]
                    dense_bias_A=model.layers[2].get_weights()[1]
                    dense_kernel_B=model.layers[3].get_weights()[0]
                    dense_bias_B=model.layers[3].get_weights()[1]
                    
                    in_A,in_B=np.split(RNN_input_kernel,2, axis=1)
                    Wr_A, Wr_B, S_A, S_B=np.split(RNN_layer_Recurrent_kernel,4, axis=1)
                    if t!=0:
                        S_A_new=set_nonzero_elements(S_A,nonzero_num[t])
                        S_B_new=set_nonzero_elements(S_B,nonzero_num[t])
                    else:
                        S_A_new=S_A.copy()
                        S_B_new=S_B.copy()
                         
                    Recurrent_new=np.concatenate((Wr_A, Wr_B, S_A_new, S_B_new),axis=1)
                    all_weights=[]
                    all_weights.append( RNN_input_kernel)
                    all_weights.append(Recurrent_new)
                    
                    
                    # set the updated weights of the model
                    model.layers[1].set_weights(all_weights)
                    
                    
                    # perturb and decode
                    # create classifier for decoding
                    outputs = [layer.output for layer in model.layers[1:]]  # Exclude the input layer
                    activity_model = Model(inputs=model.input, outputs=outputs)
                    # Get the output and activities of all layers for the new input data
                    x, y, In_ons=makeInOut(2,8,inputdur,nInput,min_dur,max_dur,dt)
                    #xnew=sum(xnew,axis=2)
                    output_and_activities = activity_model.predict(x)
                    activities_A = output_and_activities[0]  # Activities of all intermediate layers
                    activities_B=output_and_activities[1]
                    act_avg_A=avgAct(activities_A,In_ons,min_dur,max_dur)
                    act_avg_B=avgAct(activities_B,In_ons,min_dur,max_dur)
                    act_stack_A=Act_2dsort(activities_A,In_ons,min_dur,max_dur)#(time,nUnit) time is multiple of mindur+maxdur
                    act_stack_B=Act_2dsort(activities_B,In_ons,min_dur,max_dur)
            
                    act_avg_A_sort=sort_and_scale(act_avg_A,min_dur,min_dur+max_dur)
                    act_avg_B_sort=sort_and_scale(act_avg_B,min_dur,min_dur+max_dur)

            
                    # make classifying classes
                    Class_per_sec=1
                    classleng=int(1000/(dt*Class_per_sec))   #amount of step equaling 1 class
                    class_per_trial=int((min_dur+max_dur)/classleng)
                    class_A=np.arange(0,class_per_trial)
                    class_A=np.repeat(class_A,classleng) #(time,nUnit)
                    trial_rep_A=int(np.shape(act_stack_A)[0]/(min_dur+max_dur))
                    class_A_train=np.tile(class_A,(trial_rep_A))
                    trial_rep_B=int(np.shape(act_stack_B)[0]/(min_dur+max_dur))
                    class_B_train=np.tile(class_A,(trial_rep_B))
                        
                    # reduce dimensions with pca
                    pca_A = PCA()
                    pca_A.fit(act_avg_A)
                    proj_A_train=pca_A.transform(act_stack_A)
                    pca_B = PCA()
                    pca_B.fit(act_avg_B)
                    proj_B_train=pca_B.transform(act_stack_B)
                    act_avg_C=np.concatenate((act_avg_A,act_avg_B),axis=1)
                    act_stack_C=np.concatenate((act_stack_A,act_stack_B),axis=1)
                    pca_C = PCA()
                    pca_C.fit(act_avg_C)
                    proj_C_train=pca_C.transform(act_stack_C)   
                    
                    # create classifier
                    Dim=100
                    #for RNN A
                    clf_A=RandomForestClassifier(n_estimators=100,bootstrap=True,n_jobs=-1)
                    clf_A.fit(proj_A_train[:,:Dim],class_A_train)
                    #for RNN B
                    clf_B=RandomForestClassifier(n_estimators=100,bootstrap=True,n_jobs=-1)
                    clf_B.fit(proj_B_train[:,:Dim],class_B_train)
                    #combine both RNN A and B
                    clf_C=RandomForestClassifier(n_estimators=100,bootstrap=True,n_jobs=-1)
                    clf_C.fit(proj_C_train[:,:Dim],class_A_train)    
                
                    # test accuracy of the model
                    x, y, In_ons=makeInOut(2,8,inputdur,nInput,min_dur,max_dur,dt)
                    #xnew=sum(xnew,axis=2)
                    output_and_activities = activity_model.predict(x)
                    activities_A = output_and_activities[0]  # Activities of all intermediate layers
                    activities_B=output_and_activities[1]
                    act_avg_A=avgAct(activities_A,In_ons,min_dur,max_dur)
                    act_avg_B=avgAct(activities_B,In_ons,min_dur,max_dur)
                    act_stack_A=Act_2dsort(activities_A,In_ons,min_dur,max_dur)#(time,nUnit) time is multiple of mindur+maxdur
                    act_stack_B=Act_2dsort(activities_B,In_ons,min_dur,max_dur)
                    
                    proj_A_test=pca_A.transform(act_stack_A)
                    proj_B_test=pca_B.transform(act_stack_B)
                    
                    #decode    
                    pred_A_test=clf_A.predict(proj_A_test[:,:Dim])
                    pred_B_test=clf_B.predict(proj_B_test[:,:Dim])
                    
                    confsc_A[ss][t,k]=confscore(confmat(class_A_train,pred_A_test),1)
                    confsc_B[ss][t,k]=confscore(confmat(class_B_train,pred_B_test),1)
        
                
                    predavg_A=[]
                    predavg_B=[]
                    predavg_C=[]
                    pred_Aall=[]
                    pred_Ball=[]
                    pred_Call=[]
                    predictions2=[]
                    pert_ind_2=[]
                    In_ons_2=[]
                    
                    #perturb A
                    pert_state=0
                    pred_A, pred_B, pred_C, A_predictions2, pert_ind_3, In_ons_a=perturb_and_decode_noise_prob(trial1,time_1,pert_which,order,pca_A,pca_B,pca_C,clf_A,clf_B,clf_C,pert_noisesd,stop)
                    predictions2.append(A_predictions2)
                    pert_ind_2.append(pert_ind_3)
                    In_ons_2.append(In_ons_a)
                    class_all=np.tile(class_A,int((trial_num-trial1)/2))
                
                    
                    if option==0:
                        predavg_A.append(scipy.stats.circmean(pred_A,high=class_per_trial,low=1,axis=1))
                        predavg_B.append(scipy.stats.circmean(pred_B,high=class_per_trial,low=1,axis=1))
                        predavg_C.append(scipy.stats.circmean(pred_C,high=class_per_trial,low=1,axis=1))
                    else:
                        predavg_A.append(np.mean(pred_A,axis=1))
                        predavg_B.append(np.mean(pred_B,axis=1))
                        predavg_C.append(np.mean(pred_C,axis=1))
                            
                    pred_Aall.append(pred_A)
                    pred_Ball.append(pred_B)
                    pred_Call.append(pred_C)
                    
                    # calculate prediction difference of decoded results 20241003
                    from get_phase import get_phase
                    pred_diff= np.round(get_phase(pred_A - pred_B, class_per_trial, 'int')) 
                    pred_diff_ALL[ss][t,k]=np.linalg.norm(pred_diff)        
                print(f'{k_ind}: conprob {ss+1} out of {len(conProbability)}, loop {t+1} out of {len(nonzero_num)}')                                                            
            
            pred_diff_ss=np.array(pred_diff_ALL[ss])
            pred_diff_all[:,ss]=np.mean(pred_diff_ss,axis=1)
            x=nonzero_num
            fig,axs=plt.subplots(1,1)
            axs.plot(x,pred_diff_all[:,ss])
            axs.set_xscale('log')
            axs.set_title('Decoded difference over connection probability')
            axs.vlines(conprobmodel*(nUnit-nInh)*nUnit,np.min(pred_diff_all[:,ss]),np.max(pred_diff_all[:,ss]),'r')
            
            confavg_A[:,ss]=np.mean(np.array(confsc_A[ss]),axis=1)
            confavg_B[:,ss]=np.mean(np.array(confsc_B[ss]),axis=1)
        
        # save the data
        file_name = f"deletepert_k_{k_values}.npz"
        file_path = os.path.join(analysis_folder, file_name)
        np.savez(file_path, confavg_A=confavg_A, confavg_B=confavg_B, pred_diff_all=pred_diff_all)
        print("Save complete!")

        confavg_A_all.append(confavg_A)
        confavg_B_all.append(confavg_B)
        pred_diff_all2.append(pred_diff_all)
    
    
    # take the mean
    pred_diff_all=np.mean(pred_diff_all2,axis=0)
    confavg_A=np.mean(confavg_A_all,axis=0)
    confavg_B=np.mean(confavg_B_all,axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="State perturbation")
    parser.add_argument('--k_values',type=int,required=True,help='A list of numbers (e.g., --numbers 2 3 4)')
    args = parser.parse_args()
    main(args)