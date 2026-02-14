# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 23:24:41 2024

@author: RHIRAsimulation
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:18:05 2024

@author: Hiroto
"""


# assess synchronicity of outputs after perturbation. ramp up starting from the latter half. different weights
import os
import argparse
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
import json

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


def main(args):
    
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
    
    
    #% analyze synchronicity of the output
    # perturb both A and B, and decode using either A or B activity
    
    # stop input after perturbation
    def makeInput(x,In_ons,pert_ind):
        In_ons2=[]
        for i in range(np.shape(In_ons)[0]):
            index = np.argmax(In_ons[i,:]>=pert_ind[i,1]) if np.any(In_ons[i,:]>=pert_ind[i,1]) else None
            x[i,In_ons[i,index]-1:,:]=np.random.normal(loc=0.0, scale=0.01, size=np.shape(x[i,In_ons[i,index]-1:,:]))
            In_ons2.append(In_ons[i,0:index-1])
        return x, In_ons2
    
    # function to calculate output differences(mean squared error)
    def pred_diff(predictions,pert_ind):
        pred1=np.zeros(np.shape(pert_ind)[0])
        pred2=np.zeros(np.shape(pert_ind)[0])
        for i in range(np.shape(pert_ind)[0]):
            pred1_diff=predictions[i,pert_ind[i,1]:pert_ind[i,1]+min_dur+max_dur,0]-predictions[i,pert_ind[i,1]:pert_ind[i,1]+min_dur+max_dur,2]
            pred2_diff=predictions[i,pert_ind[i,1]:pert_ind[i,1]+min_dur+max_dur,1]-predictions[i,pert_ind[i,1]:pert_ind[i,1]+min_dur+max_dur,3]
            pred1[i]=np.linalg.norm(pred1_diff)
            pred2[i]=np.linalg.norm(pred2_diff)
        return pred1, pred2
    
    
    #function to get activity after perturbation
    def get_act_interest(activities_A,activities_B,pert_ind):
        act_A_interest=np.zeros((np.shape(pert_ind)[0]*(min_dur+max_dur),np.shape(activities_A)[2]))
        act_B_interest=np.zeros((np.shape(pert_ind)[0]*(min_dur+max_dur),np.shape(activities_B)[2]))
        for i in range(np.shape(pert_ind)[0]):
            act_A_interest[i*(min_dur+max_dur):(i+1)*(min_dur+max_dur),:]=activities_A[i,pert_ind[i,1]:pert_ind[i,1]+min_dur+max_dur,:]
            act_B_interest[i*(min_dur+max_dur):(i+1)*(min_dur+max_dur),:]=activities_B[i,pert_ind[i,1]:pert_ind[i,1]+min_dur+max_dur,:]
        Num=np.shape(pert_ind)[0]
        return act_A_interest, act_B_interest, Num #act_A_interest=(batch*(min+max_dur),nUnit)
            
    
    
    # make classifier
    
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
    
    
    # Dimension for decoding
    Dim=100
    
    
    
    # perturb RNN
    maxval=0.008
    sample_size=2
    step=18 # number of bins over min_dur+max_dur duration
    
    
    bin_width=(min_dur+max_dur)//step
    
    errorMat1=np.zeros([step,step,len(weight_max),2])# 4 th dimension specify whether it is perturbed by A or B (0 and 1)
    errorMat2=np.zeros([step,step,len(weight_max),2])
    count_mat=np.ones([step,step,len(weight_max),1])
    
    AB_decode_diff=np.zeros([step,step,len(weight_max),2]) # difference between decoded results for A and B
    A_decode_diff=np.zeros([step,step,len(weight_max),2])# difference between decoded results for A and actual time
    B_decode_diff=np.zeros([step,step,len(weight_max),2])
    C_decode_diff=np.zeros([step,step,len(weight_max),2])
    
    
    
    
    
    
    
    errorMat1_mean0 = []
    errorMat2_mean0 = []
    AB_decode_diff_mean0 = []
    A_decode_diff_mean0 = []
    B_decode_diff_mean0 = []
    C_decode_diff_mean0 = []
    
    
    aa=np.arange(8,len(weight_max))
    from Confmatrix import confmat, confscore
    k_values = args.k_values if isinstance(args.k_values, list) else [args.k_values]
    t_values = args.t_values if isinstance(args.t_values, list) else [args.t_values]
    for k_ind in k_values:
        for t in t_values:
        #for t in aa:
            maxval=weight_max[t]
            con_prob=conProbability[t]
        
        
        
            #build models
            model=build_model(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha, seed1=seed1)
            # load weights
            checkpoint_filepath=best_models[k_ind][t]
            model.load_weights(checkpoint_filepath)
        
        
            outputs = [layer.output for layer in model.layers[1:]]  # Exclude the input layer
            activity_model = Model(inputs=model.input, outputs=outputs)
            trial_num=8
            # Get the output and activities of all layers for the new input data
            x, y, In_ons=x, y, In_ons=makeInOut(2,8,inputdur,nInput,min_dur,max_dur,dt)
            #xnew=sum(xnew,axis=2)
            output_and_activities = activity_model.predict(x)
            activities_A = output_and_activities[0]  # Activities of all intermediate layers
            activities_B=output_and_activities[1]
            
            #output_and_activities = activity_model(x,training=False)
    
    
            #activities_A = output_and_activities[0][0]  # Activities of all intermediate layers
            #activities_B=output_and_activities[0][1]        
            
            act_avg_A=avgAct(activities_A,In_ons,min_dur,max_dur)
            act_avg_B=avgAct(activities_B,In_ons,min_dur,max_dur)
            act_stack_A=Act_2dsort(activities_A,In_ons,min_dur,max_dur)#(time,nUnit) time is multiple of mindur+maxdur
            act_stack_B=Act_2dsort(activities_B,In_ons,min_dur,max_dur)
            
            
            
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
            #for RNN A
            clf_A=RandomForestClassifier(n_estimators=100,bootstrap=True,n_jobs=-1)
            clf_A.fit(proj_A_train[:,:Dim],class_A_train)
            #for RNN B
            clf_B=RandomForestClassifier(n_estimators=100,bootstrap=True,n_jobs=-1)
            clf_B.fit(proj_B_train[:,:Dim],class_B_train)
            #combine both RNN A and B
            clf_C=RandomForestClassifier(n_estimators=100,bootstrap=True,n_jobs=-1)
            clf_C.fit(proj_C_train[:,:Dim],class_A_train)
            
            
        
            for i in range(step):
                for k in range(step):
                    x, y, In_ons, phase=makeInOutphase(sample_size,6,inputdur,nInput,min_dur,max_dur,dt)
                    #convert phase information (-pi,pi) to (-18,18)
                    phase=class_per_trial*(phase+np.pi)/(2*np.pi)
        
                    minik=min([i,k])
                    maxik=max([i,k])
                    pert_ind_raw=np.array([bin_width*minik,bin_width*maxik])
                    ind11=2
                    ind12=2
                    ind21=2
                    ind22=2
                    dur11=pert_ind_raw[0]
                    dur12=pert_ind_raw[1]
                    dur21=pert_ind_raw[0]
                    dur22=pert_ind_raw[1]
                    step11=i+round(step*min_dur/(min_dur+max_dur))
                    step12=k+round(step*min_dur/(min_dur+max_dur))
                    step21=i
                    step22=k        
                    
                    if pert_ind_raw[0]>=max_dur:
                        ind11=3
                        dur11=pert_ind_raw[0]-max_dur
                        
                    if pert_ind_raw[0]>=min_dur:
                        ind21=3
                        dur21=pert_ind_raw[0]-min_dur
                        
                    if pert_ind_raw[1]>=max_dur:
                        ind12=3
                        dur12=pert_ind_raw[1]-max_dur
                        
                    if pert_ind_raw[1]>=min_dur:
                        ind22=3
                        dur22=pert_ind_raw[1]-min_dur
                    
                    if i>=round(step*max_dur/(min_dur+max_dur)):
                        step11=i-round(step*max_dur/(min_dur+max_dur))
                        
                    if k>=round(step*max_dur/(min_dur+max_dur)):
                        step12=k-round(step*max_dur/(min_dur+max_dur))
                    
                    
                    if sample_size==4:
                        pert_ind=np.array([[In_ons[0,ind11]+int(dur11),In_ons[0,ind12]+int(dur12)]
                                            ,[In_ons[1,ind21]+int(dur21),In_ons[1,ind22]+int(dur22)]
                                            ,[In_ons[2,ind11]+int(dur11),In_ons[2,ind12]+int(dur12)]
                                            ,[In_ons[3,ind21]+int(dur21),In_ons[3,ind22]+int(dur22)]])
                    elif sample_size==2:
                        pert_ind=np.array([[In_ons[0,ind11]+int(dur11),In_ons[0,ind12]+int(dur12)]
                                            ,[In_ons[1,ind21]+int(dur21),In_ons[1,ind22]+int(dur22)]])                
                    
                    
                    
                    #x,In_ons=makeInput(x,In_ons,pert_ind) # if you want to stop the input after perturbation, run this, but it will ruin decoding
                    
                    
                    for ps in range(2):
                        # perturb RNNA and see the synchronicity
                        pert_state=ps # 0 to perturb RNN A and 1 to perturb RNN B            
                        model2=build_model_perturb(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha,pert_ind=pert_ind, pert_state=pert_state,seed1=seed1)
                        model2.set_weights(model.get_weights())
                        outputs = [layer.output for layer in model2.layers[1:]]  # Exclude the input layer
                        activity_model2 = Model(inputs=model2.input, outputs=outputs)
                        #activity_model2 = Model(inputs=model2.input, outputs=outputs,training=False)
                        
                        #predict
                        #output_and_activities = activity_model2.predict(x)
                        output_and_activities = activity_model2.predict_on_batch(x)
                        #output_and_activities = activity_model2(x, training=False)
                        predictions=output_and_activities[-1]
                        pred1, pred2=pred_diff(predictions,pert_ind)
                        errorMat1[step11,step12,t,ps]+=np.mean(pred1[0::2])
                        errorMat2[step11,step12,t,ps]+=np.mean(pred2[0::2])
                        errorMat1[step21,step22,t,ps]+=np.mean(pred1[1::2])
                        errorMat2[step21,step22,t,ps]+=np.mean(pred2[1::2])
                        
                        
                        # decode
                        activities_A = output_and_activities[0][0]  # Activities of all intermediate layers
                        activities_B=output_and_activities[0][1]
                        act_A_interest, act_B_interest, Num=get_act_interest(activities_A,activities_B,pert_ind) # act_A_interest=(batch*(min+max_dur),nUnit)
                        phase_interest,_,_=get_act_interest(phase,phase,pert_ind) # get the corresponding phase
                        
                        proj_A_test=pca_A.transform(act_A_interest) #PCA on decoding data
                        proj_B_test=pca_B.transform(act_B_interest)
                        proj_C_test=pca_C.transform(np.concatenate((act_A_interest,act_B_interest),axis=1))
                        
                        pred_A=clf_A.predict(proj_A_test[:,:Dim])# actual decoding
                        pred_B=clf_B.predict(proj_B_test[:,:Dim])# actual decoding
                        pred_C=clf_C.predict(proj_C_test[:,:Dim])# actual decoding
                        
                        
                        
                        # get difference of RNN A and RNN B decoding results
                        ABdiff=pred_A-pred_B
                        AB_phasediff=np.minimum(np.minimum(np.absolute(ABdiff-class_per_trial),np.absolute(ABdiff)),np.absolute(ABdiff+class_per_trial))
                        AB_phasediff=np.reshape(AB_phasediff,(min_dur+max_dur,-1))
                        AB_decode_diff[step11,step12,t,ps]=np.linalg.norm(np.mean(AB_phasediff[:,0::2],axis=1))    
                        AB_decode_diff[step21,step22,t,ps]=np.linalg.norm(np.mean(AB_phasediff[:,1::2],axis=1))
                        
                    
                        # get decoding offset from the actual phase
                        # RNN A
                        Adiff=pred_A-phase_interest
                        A_phasediff=np.minimum(np.minimum(np.absolute(Adiff-class_per_trial),np.absolute(Adiff)),np.absolute(Adiff+class_per_trial))
                        A_phasediff=np.reshape(A_phasediff,(min_dur+max_dur,-1))
                        A_decode_diff[step11,step12,t,ps]=np.linalg.norm(np.mean(A_phasediff[:,0::2],axis=1))    
                        A_decode_diff[step21,step22,t,ps]=np.linalg.norm(np.mean(A_phasediff[:,1::2],axis=1))
                        
                        #RNN B
                        Bdiff=pred_B-phase_interest
                        B_phasediff=np.minimum(np.minimum(np.absolute(Bdiff-class_per_trial),np.absolute(Bdiff)),np.absolute(Bdiff+class_per_trial))
                        B_phasediff=np.reshape(B_phasediff,(min_dur+max_dur,-1))
                        B_decode_diff[step11,step12,t,ps]=np.linalg.norm(np.mean(B_phasediff[:,0::2],axis=1))   
                        B_decode_diff[step21,step22,t,ps]=np.linalg.norm(np.mean(B_phasediff[:,1::2],axis=1))    
            
                        #RNN C (A and B combined)
                        Cdiff=pred_C-phase_interest
                        C_phasediff=np.minimum(np.minimum(np.absolute(Cdiff-class_per_trial),np.absolute(Cdiff)),np.absolute(Cdiff+class_per_trial))
                        C_phasediff=np.reshape(C_phasediff,(min_dur+max_dur,-1))
                        C_decode_diff[step11,step12,t,ps]=np.linalg.norm(np.mean(C_phasediff[:,0::2],axis=1))    
                        C_decode_diff[step21,step22,t,ps]=np.linalg.norm(np.mean(C_phasediff[:,1::2],axis=1))
                    
                  
                    
                    count_mat[step11,step12,t,0]+=1
                    count_mat[step21,step22,t,0]+=1
                print(f"{k_ind}:RNN {t+1} out of {len(weight_max)}: {i+1} out of {step}")
            
            errorMat1[:,:,t,:]/=count_mat[:,:,t]
            errorMat2[:,:,t,:]/=count_mat[:,:,t]
            AB_decode_diff[:,:,t,:]/=count_mat[:,:,t]
            A_decode_diff[:,:,t,:]/=count_mat[:,:,t]
            B_decode_diff[:,:,t,:]/=count_mat[:,:,t]
            C_decode_diff[:,:,t,:]/=count_mat[:,:,t]
            
            print(f"t={t}, k={k_ind}")
            
            
    
        errorMat1_mean0.append(errorMat1)
        errorMat2_mean0.append(errorMat2)
        AB_decode_diff_mean0.append(AB_decode_diff)
        A_decode_diff_mean0.append(A_decode_diff)
        B_decode_diff_mean0.append(B_decode_diff)
        C_decode_diff_mean0.append(C_decode_diff)
    
    
    
    error_mean1=np.mean(errorMat1_mean0, axis=(0, 1,2)) # shape (len(weight_max),2)
    error_mean2=np.mean(errorMat2_mean0, axis=(0, 1,2))
    AB_decode_diff_mean=np.mean(AB_decode_diff_mean0, axis=(0, 1,2))
    A_decode_diff_mean=np.mean(A_decode_diff_mean0, axis=(0, 1,2))
    B_decode_diff_mean=np.mean(B_decode_diff_mean0, axis=(0, 1,2))
    C_decode_diff_mean=np.mean(C_decode_diff_mean0, axis=(0, 1,2))
    
    Allinfo = {
        "errorMat1_mean0": errorMat1_mean0,
        "errorMat2_mean0": errorMat2_mean0,
        "AB_decode_diff_mean0": AB_decode_diff_mean0,
        "A_decode_diff_mean0": A_decode_diff_mean0,
        "B_decode_diff_mean0": B_decode_diff_mean0,
        "C_decode_diff_mean0": C_decode_diff_mean0,
        "error_mean1": error_mean1,
        "error_mean2": error_mean2,
        "AB_decode_diff_mean": AB_decode_diff_mean,
        "A_decode_diff_mean": A_decode_diff_mean,
        "B_decode_diff_mean": B_decode_diff_mean,
        "C_decode_diff_mean": C_decode_diff_mean,
        "k_values":k_values,
        "t_values":t_values,
    }
    
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # Convert NumPy numbers to Python scalars
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]  # Handle lists recursively
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}  # Handle dictionaries recursively
        else:
            return obj  # Return other types as-is
    
    Allinfo2 = {key: convert_to_native(value) for key, value in Allinfo.items()}
    
    
    # save relevant data to a file
    json_file_path = os.path.join(analysis_folder, f"Allpert_k_{k_values}_t_{t_values}.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(Allinfo2, json_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="State perturbation")
    parser.add_argument('--t_values',type=int,nargs='+',required=True,help='A list of numbers (e.g., --numbers 2 3 4)')
    parser.add_argument('--k_values',type=int,nargs='+',required=True,help='A list of numbers (e.g., --numbers 2 3 4)')
    args = parser.parse_args()
    main(args)