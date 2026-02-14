# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:18:05 2024

@author: Hiroto
"""


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


#%% analyze synchronicity of the output

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
        

# perturb RNN
maxval=0.008
sample_size=4
step=18 # number of bins over min_dur+max_dur duration


bin_width=(min_dur+max_dur)//step
errorMat1=np.zeros([step,step,len(weight_max)]) # error of prediction 1
errorMat2=np.zeros([step,step,len(weight_max)])
count_mat=np.ones([step,step,len(weight_max)])

#aa=[5,6,7,8]
#loaded = np.load('arrays.npz')
#errorMat1=loaded['errorMat1']
#errorMat2=loaded['errorMat2']
for t in range(len(weight_max)):
#for t in aa:
    maxval=weight_max[t]
    con_prob=conProbability[t]
    for i in range(step):
        for k in range(step):
            x, y, In_ons=makeInOut(sample_size,6,inputdur,nInput,min_dur,max_dur,dt)
            pert_state=1 # 0 to perturb RNN A and 1 to perturb RNN B
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
            
            pert_ind=np.array([[In_ons[0,ind11]+int(dur11),In_ons[0,ind12]+int(dur12)]
                                ,[In_ons[1,ind21]+int(dur21),In_ons[1,ind22]+int(dur22)]
                                ,[In_ons[2,ind11]+int(dur11),In_ons[2,ind12]+int(dur12)]
                                ,[In_ons[3,ind21]+int(dur21),In_ons[3,ind22]+int(dur22)]])
            
            x,In_ons=makeInput(x,In_ons,pert_ind)
            
            #build models
            model=build_model(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha, seed1=seed1)
            # load weights
            checkpoint_filepath=os.path.join(foldername[t], f"epoch_{model_index[t]:05d}.ckpt")
            model.load_weights(checkpoint_filepath)
            
            model2=build_model_perturb(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha,pert_ind=pert_ind, pert_state=pert_state,seed1=seed1)
            model2.set_weights(model.get_weights())
            predictions = model2.predict(x)
            pred1, pred2=pred_diff(predictions,pert_ind)
            
            errorMat1[step11,step12,t]+=0.5*(pred1[0]+pred1[2])
            errorMat2[step11,step12,t]+=0.5*(pred2[0]+pred2[2])
            errorMat1[step21,step22,t]+=0.5*(pred1[1]+pred1[3])
            errorMat2[step21,step22,t]+=0.5*(pred2[1]+pred2[3])
            count_mat[step11,step12,t]+=1
            count_mat[step21,step22,t]+=1
        print(f"RNN{t+1} out of {len(weight_max)}, {i+1} out of {step}")
    
    errorMat1[:,:,t]/=count_mat[:,:,t]
    errorMat2[:,:,t]/=count_mat[:,:,t]


    # display results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the first matrix A
    im1 = axes[0].imshow(errorMat1[:,:,t], cmap='viridis')
    axes[0].set_title('Prediction 1')
    fig.colorbar(im1, ax=axes[0])
    
    # Plot the second matrix B
    im2 = axes[1].imshow(errorMat2[:,:,t], cmap='viridis')
    axes[1].set_title('Prediction 2')
    fig.colorbar(im2, ax=axes[1])
    fig.suptitle(f'MSE after perturbation {conProbability[t]}', fontsize=16)
    # Display the plots
    plt.tight_layout()
    plt.show()

error_mean1=np.mean(errorMat1, axis=(0, 1))
error_mean2=np.mean(errorMat2, axis=(0, 1))

plt.figure(figsize=(10, 6))

plt.plot(conProbability, error_mean1, marker='o', linestyle='-', label='Prediction 1')
plt.plot(conProbability, error_mean2, marker='o', linestyle='-', label='Prediction 2')
plt.xscale('log')
plt.title('L2 norm of prediction after perturbation')
plt.xlabel('max value')
plt.ylabel('squared difference')
plt.legend()

# Show the plot
plt.show()




#%% analyze synchronicity of the output
# perturb both A and B



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
        

# perturb RNN
maxval=0.008
sample_size=4
step=18 # number of bins over min_dur+max_dur duration


bin_width=(min_dur+max_dur)//step
errorMat1A=np.zeros([step,step,len(weight_max)]) # error of prediction 1
errorMat2A=np.zeros([step,step,len(weight_max)])
errorMat1B=np.zeros([step,step,len(weight_max)]) # error of prediction 1
errorMat2B=np.zeros([step,step,len(weight_max)])
count_mat=np.ones([step,step,len(weight_max)])

#aa=[4,5,6,7]
#loaded = np.load('arrays.npz')
#errorMat1=loaded['errorMat1']
#errorMat2=loaded['errorMat2']

#loaded.files
loaded = np.load(r"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4rampprobvarfix\errormat_outputdiff_1to6.npz")
errorMat1A=loaded['errorMat1A']
errorMat2A=loaded['errorMat2A']
errorMat1B=loaded['errorMat1B']
errorMat2B=loaded['errorMat2B']

aa=np.array([6,7,8,9])
#for t in range(len(weight_max)):
for t in aa:
    maxval=weight_max[t]
    con_prob=conProbability[t]
    for i in range(step):
        for k in range(step):
            x, y, In_ons=makeInOut(sample_size,6,inputdur,nInput,min_dur,max_dur,dt)

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
            
            pert_ind=np.array([[In_ons[0,ind11]+int(dur11),In_ons[0,ind12]+int(dur12)]
                                ,[In_ons[1,ind21]+int(dur21),In_ons[1,ind22]+int(dur22)]
                                ,[In_ons[2,ind11]+int(dur11),In_ons[2,ind12]+int(dur12)]
                                ,[In_ons[3,ind21]+int(dur21),In_ons[3,ind22]+int(dur22)]])
            
            x,In_ons=makeInput(x,In_ons,pert_ind)
            
            
            # perturb RNNA and see the synchronicity
            pert_state=0 # 0 to perturb RNN A and 1 to perturb RNN B
            #build models
            model=build_model(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha, seed1=seed1)
            # load weights
            checkpoint_filepath=os.path.join(foldername[t], f"epoch_{model_index[t]:05d}.ckpt")
            model.load_weights(checkpoint_filepath)
            
            model2=build_model_perturb(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha,pert_ind=pert_ind, pert_state=pert_state,seed1=seed1)
            model2.set_weights(model.get_weights())
            predictions = model2.predict(x)
            pred1, pred2=pred_diff(predictions,pert_ind)
            
            errorMat1A[step11,step12,t]+=0.5*(pred1[0]+pred1[2])
            errorMat2A[step11,step12,t]+=0.5*(pred2[0]+pred2[2])
            errorMat1A[step21,step22,t]+=0.5*(pred1[1]+pred1[3])
            errorMat2A[step21,step22,t]+=0.5*(pred2[1]+pred2[3])
            
            
            
            pert_state=1 # 0 to perturb RNN A and 1 to perturb RNN B
            #build models
            model=build_model(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha, seed1=seed1)
            # load weights
            checkpoint_filepath=os.path.join(foldername[t], f"epoch_{model_index[t]:05d}.ckpt")
            model.load_weights(checkpoint_filepath)
            
            model2=build_model_perturb(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha,pert_ind=pert_ind, pert_state=pert_state,seed1=seed1)
            model2.set_weights(model.get_weights())
            predictions = model2.predict(x)
            pred1, pred2=pred_diff(predictions,pert_ind)
            
            errorMat1B[step11,step12,t]+=0.5*(pred1[0]+pred1[2])
            errorMat2B[step11,step12,t]+=0.5*(pred2[0]+pred2[2])
            errorMat1B[step21,step22,t]+=0.5*(pred1[1]+pred1[3])
            errorMat2B[step21,step22,t]+=0.5*(pred2[1]+pred2[3])            
            
            
            
            count_mat[step11,step12,t]+=1
            count_mat[step21,step22,t]+=1
        print(f"RNN{t} out of {len(weight_max)}, {i} out of {step}")
    
    errorMat1A[:,:,t]/=count_mat[:,:,t]
    errorMat2A[:,:,t]/=count_mat[:,:,t]
    errorMat1B[:,:,t]/=count_mat[:,:,t]
    errorMat2B[:,:,t]/=count_mat[:,:,t]

    # display results
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    
    # Plot the first matrix A
    im1 = axes[0,0].imshow(errorMat1A[:,:,t], cmap='viridis')
    axes[0,0].set_title('Perturb A, Prediction 1')
    fig.colorbar(im1, ax=axes[0,0])
    
    # Plot the second matrix B
    im2 = axes[0,1].imshow(errorMat2A[:,:,t], cmap='viridis')
    axes[0,1].set_title('Perturb A, Prediction 2')
    fig.colorbar(im2, ax=axes[0,1])
    
    # Plot the first matrix A
    im1 = axes[1,0].imshow(errorMat1B[:,:,t], cmap='viridis')
    axes[1,0].set_title('Perturb B, Prediction 1')
    fig.colorbar(im1, ax=axes[1,0])
    
    # Plot the second matrix B
    im2 = axes[1,1].imshow(errorMat2B[:,:,t], cmap='viridis')
    axes[1,1].set_title('Perturb B, Prediction 2')
    fig.colorbar(im2, ax=axes[1,1])
    fig.suptitle(f'MSE after perturbation {weight_max[t]}', fontsize=16)
    
    # Display the plots
    plt.tight_layout()
    plt.show()

error_mean1A=np.mean(errorMat1A, axis=(0, 1))
error_mean2A=np.mean(errorMat2A, axis=(0, 1))
error_mean1B=np.mean(errorMat1B, axis=(0, 1))
error_mean2B=np.mean(errorMat2B, axis=(0, 1))

plt.figure(figsize=(10, 6))

plt.plot(conProbability, error_mean1A, marker='o', linestyle='-', label='Perturb A, Prediction 1')
plt.plot(conProbability, error_mean2A, marker='o', linestyle='-', label='Perturb A, Prediction 2')
plt.plot(conProbability, error_mean1B, marker='o', linestyle='-', label='Perturb B, Prediction 1')
plt.plot(conProbability, error_mean2B, marker='o', linestyle='-', label='Perturb B, Prediction 2')
plt.xscale('log')
plt.title('Squared differences of prediction after perturbation')
plt.xlabel('Connection probability')
plt.ylabel('squared difference')
plt.legend()

#if save:
#    plt.savefig(os.path.join(savepath,f"weight_distribution_each_other_{minInd}"),transparent=True,dpi=600)
# Show the plot
plt.show()



#%%
savepath=r"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4rampprobvarfix"
np.savez(os.path.join(savepath,'errormat_outputdiff_all.npz'),errorMat1A=errorMat1A, errorMat2A=errorMat2A, errorMat1B=errorMat1B, errorMat2B=errorMat2B, weight_max=weight_max, conProbability=conProbability,model_index=model_index)
# start from t=2

#%% show output for output differences
loaded = np.load(r"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4rampprobvarfix\errormat_outputdiff_all.npz")
errorMat1A=loaded['errorMat1A']
errorMat2A=loaded['errorMat2A']
errorMat1B=loaded['errorMat1B']
errorMat2B=loaded['errorMat2B']

plt.plot(np.delete(conProbability,2), np.delete(error_mean1A,2), marker='o', linestyle='-', label='Perturb A, Prediction 1')
plt.plot(np.delete(conProbability,2), np.delete(error_mean2A,2), marker='o', linestyle='-', label='Perturb A, Prediction 2')
plt.plot(np.delete(conProbability,2), np.delete(error_mean1B,2), marker='o', linestyle='-', label='Perturb B, Prediction 1')
plt.plot(np.delete(conProbability,2), np.delete(error_mean2B,2), marker='o', linestyle='-', label='Perturb B, Prediction 2')
plt.xscale('log')
plt.title('Squared differences of prediction after perturbation')
plt.xlabel('Connection probability')
plt.ylabel('squared difference')
plt.legend()

if save:
    #plt.savefig(os.path.join(savepath,f"Perturb_decode_{time_1}_{time_2}_{order}_{option}"),transparent=True,dpi=400)
    plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    
    # Your plotting code
    plt.savefig(f"output_diff_state_perturb_del.svg", format='svg')
    plt.savefig(f"output_diff_state_perturb_del.png", transparent=True,dpi=500)
plt.show()

#%% analyze synchronicity of the output
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
for k_ind in range(np.shape(best_models)[0]):
    for t in range(len(conProbability)):
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
        #output_and_activities = activity_model.predict(x)
        #activities_A = output_and_activities[0]  # Activities of all intermediate layers
        #activities_B=output_and_activities[1]
        
        output_and_activities = activity_model(x,training=False)


        activities_A = output_and_activities[0][0]  # Activities of all intermediate layers
        activities_B=output_and_activities[0][1]        
        
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
            print(f"RNN {t+1} out of {len(weight_max)}: {i+1} out of {step}")
        
        errorMat1[:,:,t,:]/=count_mat[:,:,t]
        errorMat2[:,:,t,:]/=count_mat[:,:,t]
        AB_decode_diff[:,:,t,:]/=count_mat[:,:,t]
        A_decode_diff[:,:,t,:]/=count_mat[:,:,t]
        B_decode_diff[:,:,t,:]/=count_mat[:,:,t]
        C_decode_diff[:,:,t,:]/=count_mat[:,:,t]
        
        print(f"t={t}, k={k}")
        
        
        # display results
        fig, axes = plt.subplots(2, 2, figsize=(10, 5))
        
        # Plot the first matrix A
        im1 = axes[0,0].imshow(errorMat1[:,:,t,0], cmap='viridis')
        axes[0,0].set_title('Perturb A, Prediction 1')
        fig.colorbar(im1, ax=axes[0,0])
        
        # Plot the second matrix B
        im2 = axes[0,1].imshow(errorMat2[:,:,t,0], cmap='viridis')
        axes[0,1].set_title('Perturb A, Prediction 2')
        fig.colorbar(im2, ax=axes[0,1])
        
        # Plot the first matrix A
        im1 = axes[1,0].imshow(errorMat1[:,:,t,1], cmap='viridis')
        axes[1,0].set_title('Perturb B, Prediction 1')
        fig.colorbar(im1, ax=axes[1,0])
        
        # Plot the second matrix B
        im2 = axes[1,1].imshow(errorMat2[:,:,t,1], cmap='viridis')
        axes[1,1].set_title('Perturb B, Prediction 2')
        fig.colorbar(im2, ax=axes[1,1])
        fig.suptitle(f'{k}: MSE after perturbation {conProbability[t]}', fontsize=16)
        
        # Display the plots
        plt.tight_layout()
        plt.show()

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



# plot output difference
plt.figure(figsize=(10, 6))
plt.plot(conProbability, error_mean1[:,0], marker='o', linestyle='-', label='Perturb A, Prediction 1')
plt.plot(conProbability, error_mean2[:,0], marker='o', linestyle='-', label='Perturb A, Prediction 2')
plt.plot(conProbability, error_mean1[:,1], marker='o', linestyle='-', label='Perturb B, Prediction 1')
plt.plot(conProbability, error_mean2[:,1], marker='o', linestyle='-', label='Perturb B, Prediction 2')
plt.xscale('log')
plt.title('Squared differences of prediction after perturbation')
plt.xlabel('Connection probability')
plt.ylabel('Squared difference')
plt.legend()
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
if save:
    plt.savefig(os.path.join(savepath,f"MSE after perturbation"),transparent=True,dpi=600)
    plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    
    # Your plotting code
    plt.savefig(f"MSE after perturbation.svg", format='svg')
# Show the plot
plt.show()


# plot decoding difference between A and B after perturbation 
plt.figure(figsize=(10, 6))
plt.plot(conProbability, AB_decode_diff_mean[:,0], marker='o', linestyle='-', label='Perturb A')
plt.plot(conProbability, AB_decode_diff_mean[:,1], marker='o', linestyle='-', label='Perturb B')

plt.xscale('log')
plt.title('Decoding offset between RNN A and B')
plt.xlabel('Connection probability')
plt.ylabel('Difference')
plt.legend()
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
if save:
    plt.savefig(os.path.join(savepath,f"Deocding difference"),transparent=True,dpi=600)
    plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    
    # Your plotting code
    plt.savefig(f"Deocding difference.svg", format='svg')
# Show the plot
plt.show()


# plot decoding difference from actual time 
plt.figure(figsize=(10, 6))
plt.plot(conProbability, A_decode_diff_mean[:,0], marker='o', linestyle='-', label='Decode A, Perturb A')
plt.plot(conProbability, A_decode_diff_mean[:,1], marker='o', linestyle='-', label='Decode A, Perturb B')
plt.plot(conProbability, B_decode_diff_mean[:,0], marker='o', linestyle='-', label='Decode B, Perturb A')
plt.plot(conProbability, B_decode_diff_mean[:,1], marker='o', linestyle='-', label='Decode B, Perturb B')
plt.plot(conProbability, C_decode_diff_mean[:,0], marker='o', linestyle='-', label='Decode C, Perturb A')
plt.plot(conProbability, C_decode_diff_mean[:,1], marker='o', linestyle='-', label='Decode C, Perturb B')

plt.xscale('log')
plt.title('Decoding offset from actual time')
plt.xlabel('Connection probability')
plt.ylabel('Difference')
plt.legend()
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
if save:
    plt.savefig(os.path.join(savepath,f"Decoding offset from actual time"),transparent=True,dpi=600)
    plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    
    # Your plotting code
    plt.savefig(f"Decoding offset from actual time.svg", format='svg')
# Show the plot
plt.show()

#%% load files and analyze
def getrelevantind(data,t_all,t):
    errorMat1_mean0_sub=np.squeeze(data) #(18,18,conprob,2)
    errorMat1_mean0_sub=errorMat1_mean0_sub[:,:,t_all[t][0]:t_all[t][-1]+1,:]#(18,18,3,2)    
    return np.array(errorMat1_mean0_sub)
    
    
import json
t_all=[[0,1,2],[3,4,5],[6,7,8],[9,10]]
k_all=[0,1,2,3,4,5]
data_all=[]
errorMat1_mean0=[]
errorMat2_mean0=[]
AB_decode_diff_mean0=[]
A_decode_diff_mean0=[]
B_decode_diff_mean0=[]
C_decode_diff_mean0=[]

for k in k_all:
    subdata=[]
    errorMat1_mean0_sub=[]
    errorMat2_mean0_sub=[]
    AB_decode_diff_mean0_sub=[]
    A_decode_diff_mean0_sub=[]
    B_decode_diff_mean0_sub=[]
    C_decode_diff_mean0_sub=[]


    for t in range(len(t_all)):
        json_file_path = os.path.join(analysis_folder, f"Allpert_k_[{k}]_t_{t_all[t]}.json")
        # Load the JSON file
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
        subdata.append(data)
        
        #errorMat1_mean0_sub=np.squeeze(data["errorMat1_mean0"]) #(18,18,conprob,2)
        #errorMat1_mean0_sub=errorMat1_mean0_sub[:,:,t_all[t][0]:t_all[t][-1]+1,:]#(18,18,3,2)
        errorMat1_mean0_sub.append(getrelevantind(data["errorMat1_mean0"],t_all,t))#(18,18,3,2) 
        errorMat2_mean0_sub.append(getrelevantind(data["errorMat2_mean0"],t_all,t))
        AB_decode_diff_mean0_sub.append(getrelevantind(data["AB_decode_diff_mean0"],t_all,t))
        A_decode_diff_mean0_sub.append(getrelevantind(data["A_decode_diff_mean0"],t_all,t))
        B_decode_diff_mean0_sub.append(getrelevantind(data["B_decode_diff_mean0"],t_all,t))
        C_decode_diff_mean0_sub.append(getrelevantind(data["C_decode_diff_mean0"],t_all,t))
        
    data_all.append(subdata)
    errorMat1_mean0.append(np.concatenate(errorMat1_mean0_sub, axis=2))#(6,18,18,conprob,2)
    errorMat2_mean0.append(np.concatenate(errorMat2_mean0_sub, axis=2))
    AB_decode_diff_mean0.append(np.concatenate(AB_decode_diff_mean0_sub, axis=2))
    A_decode_diff_mean0.append(np.concatenate(A_decode_diff_mean0_sub, axis=2))
    B_decode_diff_mean0.append(np.concatenate(B_decode_diff_mean0_sub, axis=2))
    C_decode_diff_mean0.append(np.concatenate(C_decode_diff_mean0_sub, axis=2))
    

# `data` now contains the Python object stored in the JSON file


#%% plot the loaded data
error_mean1=np.mean(errorMat1_mean0, axis=(0, 1,2)) # shape (len(weight_max),2)
error_mean2=np.mean(errorMat2_mean0, axis=(0, 1,2))
AB_decode_diff_mean=np.mean(AB_decode_diff_mean0, axis=(0, 1,2))
A_decode_diff_mean=np.mean(A_decode_diff_mean0, axis=(0, 1,2))
B_decode_diff_mean=np.mean(B_decode_diff_mean0, axis=(0, 1,2))
C_decode_diff_mean=np.mean(C_decode_diff_mean0, axis=(0, 1,2))



# plot output difference
plt.figure(figsize=(10, 6))
plt.plot(conProbability, error_mean1[:,0], marker='o', linestyle='-', label='Perturb A, Prediction 1')
plt.plot(conProbability, error_mean2[:,0], marker='o', linestyle='-', label='Perturb A, Prediction 2')
plt.plot(conProbability, error_mean1[:,1], marker='o', linestyle='-', label='Perturb B, Prediction 1')
plt.plot(conProbability, error_mean2[:,1], marker='o', linestyle='-', label='Perturb B, Prediction 2')
plt.xscale('log')
plt.title('Squared differences of prediction after perturbation')
plt.xlabel('Connection probability')
plt.ylabel('Squared difference')
plt.legend()
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
if save:
    plt.savefig(os.path.join(figure_folders,f"MSE after perturbation"),transparent=True,dpi=600)
    plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    
    # Your plotting code
    plt.savefig(os.path.join(figure_folders,f"MSE after perturbation.svg"), format='svg')
# Show the plot
plt.show()


# plot decoding difference between A and B after perturbation 
plt.figure(figsize=(10, 6))
plt.plot(conProbability, AB_decode_diff_mean[:,0], marker='o', linestyle='-', label='Perturb A')
plt.plot(conProbability, AB_decode_diff_mean[:,1], marker='o', linestyle='-', label='Perturb B')

plt.xscale('log')
plt.title('Decoding offset between RNN A and B')
plt.xlabel('Connection probability')
plt.ylabel('Difference')
plt.legend()
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
if save:
    plt.savefig(os.path.join(figure_folders,f"Deocding difference"),transparent=True,dpi=600)
    plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    
    # Your plotting code
    plt.savefig(os.path.join(figure_folders,f"Deocding difference.svg"), format='svg')
# Show the plot
plt.show()


# plot decoding difference from actual time 
plt.figure(figsize=(10, 6))
# take mean
concatenated = np.concatenate([A_decode_diff_mean, B_decode_diff_mean, C_decode_diff_mean], axis=1)
# Take mean along axis=1
mean_result = np.mean(concatenated, axis=1)
plt.plot(conProbability, mean_result, marker='o', linestyle='-',color='red', linewidth=4, label='mean')
plt.plot(conProbability, A_decode_diff_mean[:,0], marker=None, linestyle='-', label='Decode A, Perturb A')
plt.plot(conProbability, A_decode_diff_mean[:,1], marker=None, linestyle='-', label='Decode A, Perturb B')
plt.plot(conProbability, B_decode_diff_mean[:,0], marker=None, linestyle='-', label='Decode B, Perturb A')
plt.plot(conProbability, B_decode_diff_mean[:,1], marker=None, linestyle='-', label='Decode B, Perturb B')
plt.plot(conProbability, C_decode_diff_mean[:,0], marker=None, linestyle='-', label='Decode A+B, Perturb A')
plt.plot(conProbability, C_decode_diff_mean[:,1], marker=None, linestyle='-', label='Decode A+B, Perturb B')





plt.xscale('log')
plt.title('Decoding offset from actual time')
plt.xlabel('Connection probability')
plt.ylabel('Difference')
plt.legend()
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
if save:
    plt.savefig(os.path.join(figure_folders,f"Decoding offset from actual time"),transparent=True,dpi=600)
    plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    
    # Your plotting code
    plt.savefig(os.path.join(figure_folders,f"Decoding offset from actual time.svg"), format='svg')
# Show the plot
plt.show()

#%%
savepath=r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\t4rampprobvarfix"
np.savez(os.path.join(savepath,'perturb_decode_1to4.npz'),errorMat1=errorMat1,errorMat2=errorMat2,
         AB_decode_diff=AB_decode_diff, A_decode_diff=A_decode_diff,B_decode_diff=B_decode_diff,C_decode_diff=C_decode_diff,
         weight_max=weight_max, conProbability=conProbability,model_index=model_index)




#%% load errormat and make a graph
loaded = np.load(r"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4ramp_probvar\errormat_all7.npz")
errorMat1A1=loaded['errorMat1A']
errorMat2A1=loaded['errorMat2A']
errorMat1B1=loaded['errorMat1B']
errorMat2B1=loaded['errorMat2B']
conProbability1=loaded['conProbability']
model_index1=loaded['model_index']

loaded = np.load(r"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4ramp_probvar\errormat_all_second.npz")
errorMat1A2=loaded['errorMat1A']
errorMat2A2=loaded['errorMat2A']
errorMat1B2=loaded['errorMat1B']
errorMat2B2=loaded['errorMat2B']
conProbability2=loaded['conProbability']
model_index2=loaded['model_index']


errorMat1A=np.array(np.concatenate((errorMat1A1,errorMat1A2),axis=2))
errorMat2A=np.array(np.concatenate((errorMat2A1,errorMat2A2),axis=2))
errorMat1B=np.array(np.concatenate((errorMat1B1,errorMat1B2),axis=2))
errorMat2B=np.array(np.concatenate((errorMat2B1,errorMat2B2),axis=2))
conProbability=np.array(np.concatenate((conProbability1,conProbability2),axis=0))

sortind=np.argsort(conProbability,axis=0)
conProbability=np.sort(conProbability,axis=0)


errorMat1A=errorMat1A[:,:,sortind]
errorMat2A=errorMat2A[:,:,sortind]
errorMat1B=errorMat1B[:,:,sortind]
errorMat2B=errorMat2B[:,:,sortind]




error_mean1A=np.mean(errorMat1A, axis=(0, 1))
error_mean2A=np.mean(errorMat2A, axis=(0, 1))
error_mean1B=np.mean(errorMat1B, axis=(0, 1))
error_mean2B=np.mean(errorMat2B, axis=(0, 1))

plt.figure(figsize=(10, 6))

plt.plot(conProbability, error_mean1A, marker='o', linestyle='-', label='Perturb A, Output 1')
plt.plot(conProbability, error_mean2A, marker='o', linestyle='-', label='Perturb A, Output 2')
plt.plot(conProbability, error_mean1B, marker='o', linestyle='-', label='Perturb B, Output 1')
plt.plot(conProbability, error_mean2B, marker='o', linestyle='-', label='Perturb B, Output 2')
plt.xscale('log')
plt.title('Squared differences of prediction after perturbation')
plt.xlabel('Connection probability')
plt.ylabel('squared difference')
plt.legend()

#if save:
#    plt.savefig(os.path.join(savepath,f"weight_distribution_each_other_{minInd}"),transparent=True,dpi=600)
# Show the plot
plt.show()


#%% plot all
loaded = np.load(r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\t4ramp_probvar\errormat_all_add001.npz")
errorMat1A=loaded['errorMat1A']
errorMat2A=loaded['errorMat2A']
errorMat1B=loaded['errorMat1B']
errorMat2B=loaded['errorMat2B']
conProbability=loaded['conProbability']

error_mean1A=np.mean(errorMat1A, axis=(0, 1))
error_mean2A=np.mean(errorMat2A, axis=(0, 1))
error_mean1B=np.mean(errorMat1B, axis=(0, 1))
error_mean2B=np.mean(errorMat2B, axis=(0, 1))

error_mean1A = error_mean1A.reshape(-1, 1)
error_mean2A = error_mean2A.reshape(-1, 1)
error_mean1B = error_mean1B.reshape(-1, 1)
error_mean2B = error_mean2B.reshape(-1, 1)

# Concatenate along the second dimension (axis=1)
concatenated_errors = np.concatenate((error_mean1A, error_mean2A, error_mean1B, error_mean2B), axis=1)

# Calculate the mean along the second dimension (axis=1)
error_mean_all = np.mean(concatenated_errors, axis=1)

plt.figure()
alpha=0.7
plt.plot(conProbability, error_mean1A, marker=None, linestyle='-',alpha=alpha, label='Perturb A, Output 1')
plt.plot(conProbability, error_mean2A, marker=None, linestyle='-',alpha=alpha, label='Perturb A, Output 2')
plt.plot(conProbability, error_mean1B, marker=None, linestyle='-',alpha=alpha, label='Perturb B, Output 1')
plt.plot(conProbability, error_mean2B, marker=None, linestyle='-',alpha=alpha, label='Perturb B, Output 2')
plt.plot(conProbability, error_mean_all, marker=None, linestyle='-', linewidth=5, color='red', label='Mean')

plt.xscale('log')
plt.title('Squared differences of prediction after perturbation')
plt.xlabel('Connection probability')
plt.ylabel('squared difference')
plt.legend()
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#if save:
#    plt.savefig(os.path.join(savepath,f"weight_distribution_each_other_{minInd}"),transparent=True,dpi=600)
# Show the plot
plt.show()

#%% plot perturbtion and decord
loaded = np.load(r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\t4ramp_probvar\perturb_decode_all.npz")
errorMat1=loaded['errorMat1']
errorMat2=loaded['errorMat2']
conProbability=loaded['conProbability']
AB_decode_diff=loaded['AB_decode_diff']
A_decode_diff=loaded['A_decode_diff']
B_decode_diff=loaded['B_decode_diff']
C_decode_diff=loaded['C_decode_diff']

error_mean1=np.mean(errorMat1, axis=(0, 1)) # shape (len(weight_max),2)
error_mean2=np.mean(errorMat2, axis=(0, 1))
AB_decode_diff_mean=np.mean(AB_decode_diff, axis=(0, 1))
A_decode_diff_mean=np.mean(A_decode_diff, axis=(0, 1))
B_decode_diff_mean=np.mean(B_decode_diff, axis=(0, 1))
C_decode_diff_mean=np.mean(C_decode_diff, axis=(0, 1))

error_mean_all=np.mean(np.concatenate((error_mean1,error_mean2),axis=1),axis=1)


# plot output difference
plt.figure()
plt.plot(conProbability, error_mean1[:,0], marker=None, linestyle='-',alpha=0.5, label='Perturb A, Prediction 1')
plt.plot(conProbability, error_mean2[:,0], marker=None, linestyle='-', alpha=0.5, label='Perturb A, Prediction 2')
plt.plot(conProbability, error_mean1[:,1], marker=None, linestyle='-', alpha=0.5, label='Perturb B, Prediction 1')
plt.plot(conProbability, error_mean2[:,1], marker=None, linestyle='-', alpha=0.5, label='Perturb B, Prediction 2')
plt.plot(conProbability, error_mean_all, marker=None, linestyle='-',linewidth=5, color='red', label='Mean')
plt.xscale('log')
plt.title('Squared differences of prediction after perturbation')
plt.xlabel('Connection probability')
plt.ylabel('Squared difference')
plt.legend()
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#if save:
#    plt.savefig(os.path.join(savepath,f"weight_distribution_each_other_{minInd}"),transparent=True,dpi=600)
# Show the plot
plt.show()


# plot decoding difference between A and B after perturbation 
plt.figure()
plt.plot(conProbability, AB_decode_diff_mean[:,0], marker=None, linestyle='-', alpha=0.5,label='Perturb A')
plt.plot(conProbability, AB_decode_diff_mean[:,1], marker=None, linestyle='-', alpha=0.5,label='Perturb B')
plt.plot(conProbability, np.mean(AB_decode_diff_mean,axis=1), marker=None, linestyle='-',linewidth=5, color='red', label='Mean')

plt.xscale('log')
plt.title('Decoding offset between RNN A and B')
plt.xlabel('Connection probability')
plt.ylabel('Difference')
plt.legend()
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#if save:
#    plt.savefig(os.path.join(savepath,f"weight_distribution_each_other_{minInd}"),transparent=True,dpi=600)
# Show the plot
plt.show()



perA_mean=np.mean(np.concatenate((A_decode_diff_mean[:,[0]],B_decode_diff_mean[:,[0]],C_decode_diff_mean[:,[0]]),axis=1),axis=1)
perB_mean=np.mean(np.concatenate((A_decode_diff_mean[:,[1]],B_decode_diff_mean[:,[1]],C_decode_diff_mean[:,[1]]),axis=1),axis=1)
# plot decoding difference from actual time 
plt.figure()
plt.plot(conProbability, A_decode_diff_mean[:,0], marker=None, linestyle='-',alpha=0.5, label='Decode A, Perturb A')
plt.plot(conProbability, A_decode_diff_mean[:,1], marker=None, linestyle='-', alpha=0.5,label='Decode A, Perturb B')
plt.plot(conProbability, B_decode_diff_mean[:,0], marker=None, linestyle='-', alpha=0.5,label='Decode B, Perturb A')
plt.plot(conProbability, B_decode_diff_mean[:,1], marker=None, linestyle='-', alpha=0.5,label='Decode B, Perturb B')
plt.plot(conProbability, C_decode_diff_mean[:,0], marker=None, linestyle='-', alpha=0.5, label='Decode C, Perturb A')
plt.plot(conProbability, C_decode_diff_mean[:,1], marker=None, linestyle='-', alpha=0.5,label='Decode C, Perturb B')
plt.plot(conProbability, perA_mean, marker=None, linestyle='-',linewidth=5, color='red', label='Perturb A, mean')
plt.plot(conProbability, perB_mean, marker=None, linestyle='-', linewidth=5, color='blue', label='Perturb B, mean')

plt.xscale('log')
plt.title('Decoding offset from actual time')
plt.xlabel('Connection probability')
plt.ylabel('Difference')
plt.legend()
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#if save:
#    plt.savefig(os.path.join(savepath,f"weight_distribution_each_other_{minInd}"),transparent=True,dpi=600)
# Show the plot
plt.show()
#%% show weight distribution


save=False
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

Bin=100

exexavg_A=np.zeros((2*Bin+1,len(weight_max)))
inexavg_A=np.zeros((2*Bin+1,len(weight_max)))
ininavg_A=np.zeros((2*Bin+1,len(weight_max)))
exinavg_A=np.zeros((2*Bin+1,len(weight_max)))
order_A=np.zeros((2*Bin+1,len(weight_max)))

exexavg_B=np.zeros((2*Bin+1,len(weight_max)))
inexavg_B=np.zeros((2*Bin+1,len(weight_max)))
ininavg_B=np.zeros((2*Bin+1,len(weight_max)))
exinavg_B=np.zeros((2*Bin+1,len(weight_max)))
order_B=np.zeros((2*Bin+1,len(weight_max)))


S_B_avg_ex=np.zeros((2*Bin+1,len(weight_max)))
S_B_avg_in=np.zeros((2*Bin+1,len(weight_max)))
S_A_avg_ex=np.zeros((2*Bin+1,len(weight_max)))
S_A_avg_in=np.zeros((2*Bin+1,len(weight_max)))

exex_A=[]
inex_A=[]
inin_A=[]
exin_A=[]
exex_B=[]
inex_B=[]
inin_B=[]
exin_B=[]

S_A_ex_all=[]
S_A_in_all=[]
S_B_ex_all=[]
S_B_in_all=[]


#  analyze weight distribution


for k in range(np.shape(best_models)[0]):
    for t in range(len(weight_max)):
        max_range=[0,max_dur+min_dur] # time range to choose max firing time
        nExc=nUnit-nInh #number of excitory units
        
        
        model=build_model(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha, seed1=seed1)
        # load weights
        checkpoint_filepath=best_models[k][t]
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
        
        
        
        
        
        
        # Sort units based on the time of maximum activity
        max_time_ex_A = np.argmax(act_avg_A[max_range[0]:max_range[1], :-nInh], axis=0)
        max_time_in_A = np.argmax(act_avg_A[max_range[0]:max_range[1], -nInh:], axis=0)
        sorted_units_ex_A = np.argsort(max_time_ex_A)
        sorted_units_in_A = np.argsort(max_time_in_A)+nExc
        sorted_units_A=np.concatenate((sorted_units_ex_A,sorted_units_in_A),axis=0)
        
        max_time_ex_B = np.argmax(act_avg_B[max_range[0]:max_range[1], :-nInh], axis=0)
        max_time_in_B = np.argmax(act_avg_B[max_range[0]:max_range[1], -nInh:], axis=0)
        sorted_units_ex_B = np.argsort(max_time_ex_B)
        sorted_units_in_B = np.argsort(max_time_in_B)+nExc
        sorted_units_B=np.concatenate((sorted_units_ex_B,sorted_units_in_B),axis=0)
        
        
        
        # sort recurrent weights, input weights, and output weights according to the time of maximum acitivity
        Wr_sort_A=Wr_A.copy()
        Wr_sort_A=Wr_sort_A[sorted_units_A,:]
        Wr_sort_A=Wr_sort_A[:,sorted_units_A]
        S_sort_B=S_B.copy()
        S_sort_B=S_sort_B[sorted_units_B,:]
        S_sort_B=S_sort_B[:,sorted_units_A]
        in_sort_A=in_A.copy()
        in_sort_A=in_sort_A[:,sorted_units_A]
        dense_sort_A=dense_kernel_A.copy()
        dense_sort_A=dense_sort_A[sorted_units_A,:]
        
        
        
        Wr_sort_B=Wr_B.copy()
        Wr_sort_B=Wr_sort_B[sorted_units_B,:]
        Wr_sort_B=Wr_sort_B[:,sorted_units_B]
        S_sort_A=S_A.copy()
        S_sort_A=S_sort_A[sorted_units_A,:]
        S_sort_A=S_sort_A[:,sorted_units_B]
        in_sort_B=in_B.copy()
        in_sort_B=in_sort_B[:,sorted_units_B]
        dense_sort_B=dense_kernel_B.copy()
        dense_sort_B=dense_sort_B[sorted_units_B,:]
        
        
        
        def weight_distribution(Bin,nExc,nInh,max_time_ex,max_time_in,weight_mat):
            # analyze weight distribution of Wr_A and Wr_B
            # get weights from prior and later units
            #Bin=40 #get +-Bin units
            exex=np.empty((2*Bin+1,nExc))
            exex[:]=np.nan
            inex=exex.copy()
            inin=np.empty((2*Bin+1,nInh))
            inin[:]=np.nan
            exin=inin.copy()
            
            # get the relationship between max firing rate for in and ex
            sort_units_time_ex=np.sort(max_time_ex)
            sort_units_time_in=np.sort(max_time_in)
            sort_units_time_ex=sort_units_time_ex.reshape((nExc,1))
            sort_units_time_in=sort_units_time_in.reshape((nInh,1))
            max_time_exin=sort_units_time_ex-np.transpose(sort_units_time_in)
            logical_max=~max_time_exin>0
            logical_max=logical_max*1
            In_ind=np.sum(logical_max,axis=0)
            
            max_time_exin=np.transpose(sort_units_time_ex)-sort_units_time_in
            logical_max=max_time_exin>0
            logical_max=logical_max*1
            Ex_ind=np.sum(logical_max,axis=0)
            
            for i in range(nExc):
                exex[max(0,Bin-i):min(Bin+nExc-i,2*Bin+1),i]=weight_mat[max(0,i-Bin):min(nExc,i+Bin+1),i]
                inex[max(0,Bin-Ex_ind[i]):min(Bin+nInh-Ex_ind[i],2*Bin+1),i]=weight_mat[nExc+max(0,Ex_ind[i]-Bin):nExc+min(nInh,Ex_ind[i]+Bin+1),i]
                
            
            for i in range(nInh):
                inin[max(0,Bin-i):min(Bin+nInh-i,2*Bin+1),i]=weight_mat[nExc+max(0,i-Bin):nExc+min(nInh,i+Bin+1),nExc+i]
                exin[max(0,Bin-In_ind[i]):min(Bin+nExc-In_ind[i],2*Bin+1),i]=weight_mat[max(0,In_ind[i]-Bin):min(nExc,In_ind[i]+Bin+1),nExc+i]
            
            exexavg=np.nanmean(exex,axis=1)
            inexavg=np.nanmean(inex,axis=1)
            ininavg=np.nanmean(inin,axis=1)
            exinavg=np.nanmean(exin,axis=1)
            order=np.arange(-Bin,Bin+1)
            return exexavg, inexavg, ininavg, exinavg, order
        
        
        
        fig, axs = plt.subplots(2,2,sharex=True)
        a1, a2, a3, a4, order_A=weight_distribution(Bin,nExc,nInh,max_time_ex_A,max_time_in_A,Wr_sort_A)
        exexavg_A[:,t]=a1
        inexavg_A[:,t]=a2
        ininavg_A[:,t]=a3
        exinavg_A[:,t]=a4
        
        # plot weights distribution
        axs[0,0].plot(np.delete(order_A,Bin),np.delete(exexavg_A[:,t],Bin),label='Ex Ex')
        axs[0,0].plot(order_A,exinavg_A[:,t],label='Ex In')
        axs[0,0].title.set_text(f'A')
        axs[0,0].legend(loc='lower center')
        axs[0,1].plot(np.delete(order_A,Bin),np.delete(ininavg_A[:,t],Bin),label='In In')
        axs[0,1].plot(order_A,inexavg_A[:,t],label='In Ex')
        axs[0,1].title.set_text(f'A')
        axs[0,1].legend(loc='lower center')
            
            
        b1, b2, b3, b4, order_B=weight_distribution(Bin,nExc,nInh,max_time_ex_B,max_time_in_B,Wr_sort_B)
        exexavg_B[:,t]=b1
        inexavg_B[:,t]=b2
        ininavg_B[:,t]=b3
        exinavg_B[:,t]=b4
        
        
        # plot weights distribution
        axs[1,0].plot(np.delete(order_B,Bin),np.delete(exexavg_B[:,t],Bin),label='Ex Ex')
        axs[1,0].plot(order_A,exinavg_B[:,t],label='Ex In')
        axs[1,0].title.set_text(f'B')
        axs[1,0].legend(loc='lower center')
        axs[1,1].plot(np.delete(order_B,Bin),np.delete(ininavg_B[:,t],Bin),label='In In')
        axs[1,1].plot(order_A,inexavg_B[:,t],label='In Ex')
        axs[1,1].title.set_text(f'B')
        axs[1,1].legend(loc='lower center')
    
        
        
        
        # analyze weights of the S_B and S_A
        S_B_nan=S_sort_B.copy()
        S_B_nan[S_B_nan==0]=np.nan
        S_A_nan=S_sort_A.copy()
        S_A_nan[S_A_nan==0]=np.nan
        
        maxtime_ex_A=np.sort(max_time_ex_A)
        maxtime_in_A=np.sort(max_time_in_A)
        maxtime_ex_B=np.sort(max_time_ex_B)
        maxtime_in_B=np.sort(max_time_in_B)
        
        maxtime_ex_A=maxtime_ex_A.reshape((nExc,1))
        maxtime_in_A=maxtime_in_A.reshape((nInh,1))
        maxtime_ex_B=maxtime_ex_B.reshape((nExc,1))
        maxtime_in_B=maxtime_in_B.reshape((nInh,1))
        
        
        
        
        
        maxtime_BA=maxtime_ex_B-np.transpose(np.concatenate((maxtime_ex_A,maxtime_in_A),axis=0))
        logical_max=maxtime_BA<0
        logical_max=logical_max*1
        In_ind=np.sum(logical_max,axis=0)
        maxtime_AB=maxtime_ex_A-np.transpose(np.concatenate((maxtime_ex_B,maxtime_in_B),axis=0))
        logical_max=maxtime_AB<0
        logical_max=logical_max*1
        Ex_ind=np.sum(logical_max,axis=0)
        
        
        
        S_B_bin=np.empty((2*Bin+1,nUnit))
        S_B_bin[:]=np.nan
        S_A_bin=np.empty((2*Bin+1,nUnit))
        S_A_bin[:]=np.nan
        for i in np.arange(nUnit):
            S_B_bin[max(0,Bin-In_ind[i]):min(Bin+nExc-In_ind[i],2*Bin+1),i]=S_B_nan[max(0,In_ind[i]-Bin):min(nExc,In_ind[i]+Bin+1),i]
            S_A_bin[max(0,Bin-Ex_ind[i]):min(Bin+nExc-Ex_ind[i],2*Bin+1),i]=S_A_nan[max(0,Ex_ind[i]-Bin):min(nExc,Ex_ind[i]+Bin+1),i]
            #S_B_bin[max(0,Bin-In_ind[i]):min(Bin+nExc-In_ind[i],2*Bin+1),i]=S_sort_B[max(0,In_ind[i]-Bin):min(nExc,In_ind[i]+Bin+1),i]
            #S_A_bin[max(0,Bin-Ex_ind[i]):min(Bin+nExc-Ex_ind[i],2*Bin+1),i]=S_sort_A[max(0,Ex_ind[i]-Bin):min(nExc,Ex_ind[i]+Bin+1),i]
            
            
        order=np.arange(-Bin,Bin+1)
        S_B_avg_ex[:,t]=np.nanmean(S_B_bin[:,:nExc],axis=1)
        S_B_avg_in[:,t]=np.nanmean(S_B_bin[:,nExc:],axis=1)
        S_A_avg_ex[:,t]=np.nanmean(S_A_bin[:,:nExc],axis=1)
        S_A_avg_in[:,t]=np.nanmean(S_A_bin[:,nExc:],axis=1)
        
        plt.figure()
        plt.plot(order,S_B_avg_ex[:,t],label='ExB ExA')
        plt.plot(order,S_B_avg_in[:,t],label='ExB InA')
        plt.plot(order,S_A_avg_ex[:,t],label='ExA ExB')
        plt.plot(order,S_A_avg_in[:,t],label='ExA InB')
        plt.legend(loc='lower right')
    
        
        
        fig, axs = plt.subplots(2, 2)  # Adjust figure size if needed
        # Plotting the first subplot
        axs[0, 0].plot(order, S_B_avg_ex[:,t], label='ExB ExA')
        axs[0, 0].set_title('ExB ExA')
        #axs[0, 0].legend(loc='upper right')
        
        # Plotting the second subplot
        axs[0, 1].plot(order, S_B_avg_in[:,t], label='ExB InA')
        axs[0, 1].set_title('ExB InA')
        #axs[0, 1].legend(loc='lower right')
        
        # Plotting the third subplot
        axs[1, 0].plot(order, S_A_avg_ex[:,t], label='ExA ExB')
        axs[1, 0].set_title('ExA ExB')
        #axs[1, 0].legend(loc='lower right')
        
        # Plotting the fourth subplot
        axs[1, 1].plot(order, S_A_avg_in[:,t], label='ExA InB')
        axs[1, 1].set_title('ExA InB')
        #axs[1, 1].legend(loc='lower right')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        #if save:
            #plt.savefig(os.path.join(savepath,f"weight_distribution_each_other_4_{minInd}"),transparent=True,dpi=600)
            #np.savez(f'weight_distribution_{k+1}.npz', S_A_avg_ex=S_A_avg_ex, S_A_avg_in=S_A_avg_in, S_B_avg_ex=S_B_avg_ex, S_B_avg_in=S_B_avg_in,
                     #exexavg_A=exexavg_A, inexavg_A=inexavg_A, ininavg_A=ininavg_A, exinavg_A=exinavg_A, order_A=order_A,
                     #exexavg_B=exexavg_B, inexavg_B=inexavg_B, ininavg_B=ininavg_B, exinavg_B=exinavg_B, order_B=order_B)
    
        # Show the plot
        plt.show()
    exex_A.append(exexavg_A) # exexavg_A is of shape(time,files)
    inex_A.append(inexavg_A)
    inin_A.append(ininavg_A)
    exin_A.append(exinavg_A)
    exex_B.append(exexavg_B)
    inex_B.append(inexavg_B)
    inin_B.append(ininavg_B)
    exin_B.append(exinavg_B)
    S_A_ex_all.append(S_A_avg_ex)
    S_A_in_all.append(S_A_avg_in)
    S_B_ex_all.append(S_B_avg_ex)
    S_B_in_all.append(S_B_avg_in)

# take mean
exex_A=np.mean(np.array(exex_A),axis=0) # exex_A is of shape(6files, time, 11files)->shape(files, time, 11files)
inex_A=np.mean(np.array(inex_A),axis=0)
exin_A=np.mean(np.array(exin_A),axis=0) 
inin_A=np.mean(np.array(inin_A),axis=0)
exex_B=np.mean(np.array(exex_B),axis=0) # exex_Bis of shape(6files, time, 11files)
inex_B=np.mean(np.array(inex_B),axis=0)
exin_B=np.mean(np.array(exin_B),axis=0) 
inin_B=np.mean(np.array(inin_B),axis=0)

S_A_ex_all=np.nanmean(np.array(S_A_ex_all),axis=0)
S_A_in_all=np.nanmean(np.array(S_A_in_all),axis=0)
S_B_ex_all=np.nanmean(np.array(S_B_ex_all),axis=0)
S_B_in_all=np.nanmean(np.array(S_B_in_all),axis=0)



from matplotlib.ticker import MaxNLocator
for t in range(len(weight_max)):
    # plot
    fig, axs = plt.subplots(2,2,sharex=True, sharey='row')
    fig.suptitle(f"Connection probability {conProbability[t]}")
    # plot weights distribution
    axs[0,0].plot(np.delete(order_A,Bin),np.delete(exex_A[:,t],Bin),label='Ex→Ex')
    axs[0,0].plot(order_A,exin_A[:,t],label='Ex→In')
    axs[0,0].title.set_text(f'A')
    axs[0,0].spines['right'].set_visible(False)
    axs[0,0].spines['top'].set_visible(False)
    #axs[0,0].legend(loc='lower center')
    axs[1,0].plot(np.delete(order_A,Bin),np.delete(inin_A[:,t],Bin),label='In→In')
    axs[1,0].plot(order_A,inex_A[:,t],label='In→Ex')
    axs[1,0].spines['right'].set_visible(False)
    axs[1,0].spines['top'].set_visible(False)
    #axs[1,0].title.set_text(f'A')
    #axs[1,0].legend(loc='lower center')
    
    # Limit the number of y-axis ticks to 3
    #axs[0, 0].yaxis.set_major_locator(MaxNLocator(3))
    axs[1, 0].yaxis.set_major_locator(MaxNLocator(3))
        
    axs[0,1].plot(np.delete(order_B,Bin),np.delete(exex_B[:,t],Bin),label='Ex→Ex')
    axs[0,1].plot(order_A,exin_B[:,t],label='Ex→In')
    axs[0,1].title.set_text(f'B')
    axs[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    axs[0,1].spines['right'].set_visible(False)
    axs[0,1].spines['top'].set_visible(False)
    axs[0,1].spines['left'].set_visible(False)
    axs[0, 1].tick_params(left=False, labelleft=False)  # Remove left y-ticks and labels
    axs[1,1].plot(np.delete(order_B,Bin),np.delete(inin_B[:,t],Bin),label='In→In')
    axs[1,1].plot(order_A,inex_B[:,t],label='In→Ex')
    #axs[1,1].title.set_text(f'B')
    axs[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    axs[1,1].spines['right'].set_visible(False)
    axs[1,1].spines['top'].set_visible(False)
    axs[1,1].spines['left'].set_visible(False)
    axs[1, 1].tick_params(left=False, labelleft=False)  # Remove left y-ticks and labels
    if save:
        #plt.savefig(os.path.join(r"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4ramp_probvar",f"weight_distribution_within_4"),transparent=True,dpi=600)
        plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
        plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
        # Your plotting code
        plt.savefig(os.path.join(figure_folders,f"W_distribution_conprob_{conProbability[t]}.svg"), format='svg')
    
    fig, axs = plt.subplots(2, 2)  # Adjust figure size if needed
    fig.suptitle(f"Connection probability {conProbability[t]}")
    # Plotting the first subplot
    axs[0, 0].plot(order, S_B_ex_all[:,t], label='ExB ExA')
    axs[0, 0].set_title('ExB ExA')
    #axs[0, 0].legend(loc='upper right')
    
    # Plotting the second subplot
    axs[0, 1].plot(order, S_B_in_all[:,t], label='ExB InA')
    axs[0, 1].set_title('ExB InA')
    #axs[0, 1].legend(loc='lower right')
    
    # Plotting the third subplot
    axs[1, 0].plot(order, S_A_ex_all[:,t], label='ExA ExB')
    axs[1, 0].set_title('ExA ExB')
    #axs[1, 0].legend(loc='lower right')
    
    # Plotting the fourth subplot
    axs[1, 1].plot(order, S_A_in_all[:,t], label='ExA InB')
    axs[1, 1].set_title('ExA InB')
    
    if save:
        #plt.savefig(os.path.join(r"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4ramp_probvar",f"weight_distribution_within_4"),transparent=True,dpi=600)
        plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
        plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
        # Your plotting code
        plt.savefig(os.path.join(figure_folders,f"S_distribution_conprob_{conProbability[t]}.svg"), format='svg')

#%%
# show average result
exexavg_A_avg=np.mean(exexavg_A,1)
inexavg_A_avg=np.mean(inexavg_A,1)
ininavg_A_avg=np.mean(ininavg_A,1)
exinavg_A_avg=np.mean(exinavg_A,1)


exexavg_B_avg=np.mean(exexavg_B,1)
inexavg_B_avg=np.mean(inexavg_B,1)
ininavg_B_avg=np.mean(ininavg_B,1)
exinavg_B_avg=np.mean(exinavg_B,1)


S_B_avg_ex_avg=np.nanmean(S_B_avg_ex[:,-4:],1)
S_B_avg_in_avg=np.nanmean(S_B_avg_in[:,-4:],1)
S_A_avg_ex_avg=np.nanmean(S_A_avg_ex[:,-4:],1)
S_A_avg_in_avg=np.nanmean(S_A_avg_in[:,-4:],1)






# plot
fig, axs = plt.subplots(2,2,sharex=True, sharey='row')
# plot weights distribution
axs[0,0].plot(np.delete(order_A,Bin),np.delete(exexavg_A_avg,Bin),label='Ex→Ex')
axs[0,0].plot(order_A,exinavg_A_avg,label='Ex→In')
axs[0,0].title.set_text(f'A')
axs[0,0].spines['right'].set_visible(False)
axs[0,0].spines['top'].set_visible(False)
#axs[0,0].legend(loc='lower center')
axs[1,0].plot(np.delete(order_A,Bin),np.delete(ininavg_A_avg,Bin),label='In→In')
axs[1,0].plot(order_A,inexavg_A_avg,label='In→Ex')
axs[1,0].spines['right'].set_visible(False)
axs[1,0].spines['top'].set_visible(False)
#axs[1,0].title.set_text(f'A')
#axs[1,0].legend(loc='lower center')

# Limit the number of y-axis ticks to 3
#axs[0, 0].yaxis.set_major_locator(MaxNLocator(3))
axs[1, 0].yaxis.set_major_locator(MaxNLocator(3))
    
axs[0,1].plot(np.delete(order_B,Bin),np.delete(exexavg_B_avg,Bin),label='Ex→Ex')
axs[0,1].plot(order_A,exinavg_B_avg,label='Ex→In')
axs[0,1].title.set_text(f'B')
axs[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
axs[0,1].spines['right'].set_visible(False)
axs[0,1].spines['top'].set_visible(False)
axs[0,1].spines['left'].set_visible(False)
axs[0, 1].tick_params(left=False, labelleft=False)  # Remove left y-ticks and labels
axs[1,1].plot(np.delete(order_B,Bin),np.delete(ininavg_B_avg,Bin),label='In→In')
axs[1,1].plot(order_A,inexavg_B_avg,label='In→Ex')
#axs[1,1].title.set_text(f'B')
axs[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
axs[1,1].spines['right'].set_visible(False)
axs[1,1].spines['top'].set_visible(False)
axs[1,1].spines['left'].set_visible(False)
axs[1, 1].tick_params(left=False, labelleft=False)  # Remove left y-ticks and labels
if save:
    #plt.savefig(os.path.join(r"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4ramp_probvar",f"weight_distribution_within_4"),transparent=True,dpi=600)
    plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    
    # Your plotting code
    plt.savefig(f"Perturb_decode_weight_distribution.svg", format='svg')

fig, axs = plt.subplots(2, 2)  # Adjust figure size if needed
# Plotting the first subplot
axs[0, 0].plot(order, S_B_avg_ex_avg, label='ExB ExA')
axs[0, 0].set_title('ExB ExA')
#axs[0, 0].legend(loc='upper right')

# Plotting the second subplot
axs[0, 1].plot(order, S_B_avg_in_avg, label='ExB InA')
axs[0, 1].set_title('ExB InA')
#axs[0, 1].legend(loc='lower right')

# Plotting the third subplot
axs[1, 0].plot(order, S_A_avg_ex_avg, label='ExA ExB')
axs[1, 0].set_title('ExA ExB')
#axs[1, 0].legend(loc='lower right')

# Plotting the fourth subplot
axs[1, 1].plot(order, S_A_avg_in_avg, label='ExA InB')
axs[1, 1].set_title('ExA InB')


# create boxplot to show distribution
Wr_A_exc = Wr_A[:-nInh, :]
Wr_A_inh = Wr_A[-nInh:, :]
Wr_B_exc = Wr_B[:-nInh, :]
Wr_B_inh = Wr_B[-nInh:, :]

# Flattening the arrays for boxplot input
Wr_A_exc_flat = Wr_A_exc.flatten()
Wr_A_inh_flat = Wr_A_inh.flatten()
Wr_B_exc_flat = Wr_B_exc.flatten()
Wr_B_inh_flat = Wr_B_inh.flatten()

# Extracting non-zero values from S_A and S_B
S_A_nonzero = S_A[S_A != 0]
S_B_nonzero = S_B[S_B != 0]

plt.figure(figsize=(10, 6))
scale_fac=(nUnit-nInh)/nInh
plt.boxplot([Wr_A_exc_flat, Wr_A_inh_flat/scale_fac, Wr_B_exc_flat, Wr_B_inh_flat/scale_fac, S_A_nonzero, S_B_nonzero], 
            labels=['Wr_A_exc', 'Wr_A_inh', 'Wr_B_exc', 'Wr_B_inh', 'S_A_nonzero', 'S_B_nonzero'])
plt.title(f'Distribution Comparison conprob{conProbability[t]}')
plt.ylabel('Values')
plt.grid(True)
# Get current y-ticks
yticks = plt.gca().get_yticks()

# Modify the negative ticks by scaling them by 4
scaled_yticks = [y * scale_fac if y < 0 else y for y in yticks]

# Re-label the y-tick labels with the scaled negative values
plt.gca().set_yticklabels([f'{y:.2f}' for y in scaled_yticks])

plt.show()


# visualize weight distribution using histogram
fig, axs = plt.subplots(2, 1, sharex='col',sharey=True)  # Adjust figure size if needed
axs[0].hist(np.ravel(Wr_A[:-nInh,:]),rwidth=1,color='blue',bins=100)
axs[1].hist(np.ravel(Wr_B[:-nInh,:]),rwidth=1,color='blue',bins=100)
axs[0].hist(np.ravel(Wr_A[-nInh:,:]),rwidth=1,color='red',bins=100)
axs[1].hist(np.ravel(Wr_B[-nInh:,:]),rwidth=1,color='red',bins=100)




#%% visualize relationship between interRNN connections and max firing time

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







def analyze_connections(S_matrix, peak_times_source, peak_times_target, title, is_S_A=True, use_phase=False, top_percent=100):
    """
    Analyze connections and plot the relationship between connection strength and time difference of peak activity.
    
    Parameters:
        S_matrix (np.ndarray): Connection matrix.
        peak_times_source (np.ndarray): Peak times of source neurons.
        peak_times_target (np.ndarray): Peak times of target neurons.
        title (str): Title for the plot.
        is_S_A (bool): If True, analyzing S_A; if False, analyzing S_B.
        use_phase (bool): If True, calculate time differences considering cyclic activity.
        top_percent (float): Percentage of top connections to analyze (between 0 and 100).
    """
    # S_matrix[i, j]: Connection from neuron i in source to neuron j in target
    indices = np.nonzero(S_matrix)
    i_indices = indices[0]
    j_indices = indices[1]
    S_values = S_matrix[i_indices, j_indices]
    
    if is_S_A:
        # For S_A, only excitatory neurons in A have outgoing connections
        valid_indices = i_indices < nExc
    else:
        # For S_B, only excitatory neurons in B have outgoing connections
        valid_indices = i_indices < nExc  # Adjust if necessary
    
    i_valid = i_indices[valid_indices]
    j_valid = j_indices[valid_indices]
    S_nonzero = S_matrix[i_valid, j_valid]
    
    # Calculate time differences
    t_source = peak_times_source[i_valid]
    t_target = peak_times_target[j_valid]
    
    if use_phase:
        # Calculate minimal time difference considering cyclic activity
        time_diffs = np.minimum((t_source - t_target) % T, (t_target - t_source) % T)
    else:
        # Calculate absolute time differences
        time_diffs = np.abs(t_source - t_target)
    
    # Select top x% of connections
    if top_percent < 100:
        num_top = int(len(S_nonzero) * top_percent / 100)
        if num_top < 1:
            num_top = 1  # Ensure at least one connection is selected
        # Get indices of top connections
        sorted_indices = np.argsort(-S_nonzero)  # Negative sign for descending sort
        top_indices = sorted_indices[:num_top]
        # Filter the data
        S_nonzero = S_nonzero[top_indices]
        time_diffs = time_diffs[top_indices]
    
    # Plotting
    plt.figure()
    plt.scatter(time_diffs, S_nonzero,s=1, alpha=0.5, label='Data points')
    plt.ylabel('Connection Strength')
    plt.xlabel('Time Difference of Peak Activity')
    if use_phase:
        plt.xlabel('Minimal Time Difference (Cyclic)')
    plt.title(title)
    
    # Linear regression
    if len(time_diffs) > 1:
        coefficients = np.polyfit(time_diffs, S_nonzero, 1)
        slope = coefficients[0]
        intercept = coefficients[1]
        
        # Generate x values for the line
        x_fit = np.linspace(np.min(time_diffs), np.max(time_diffs), 100)
        y_fit = slope * x_fit + intercept
        
        # Plot the best fit line
        plt.plot(x_fit, y_fit, color='red', label=f'Best fit line (slope = {slope:.2E})')
    else:
        slope = np.nan  # Not enough points to compute slope
        plt.text(0.5, 0.5, 'Not enough data for regression', transform=plt.gca().transAxes, ha='center')
    
    # Add legend
    plt.legend()
    
    plt.show()
    return time_diffs, S_nonzero

def plot_time_of_max_activity(S_matrix, peak_times_source, peak_times_target, title, is_S_A=True, use_alpha=False,top_percent=100):
    """
    Plot a scatter plot where each point represents a non-zero connection,
    and its position corresponds to the peak activity times of the source and target neurons.
    
    Parameters:
        S_matrix (np.ndarray): Connection matrix.
        peak_times_source (np.ndarray): Peak times of source neurons.
        peak_times_target (np.ndarray): Peak times of target neurons.
        title (str): Title for the plot.
        is_S_A (bool): If True, analyzing S_A; if False, analyzing S_B.
        use_alpha (bool): If True, use alpha to represent connection strength; otherwise, use color.
    """
    # S_matrix[i, j]: Connection from neuron i in source to neuron j in target
    indices = np.nonzero(S_matrix)
    i_indices = indices[0]
    j_indices = indices[1]
    S_values = S_matrix[i_indices, j_indices]
    
    if is_S_A:
        # For S_A, only excitatory neurons in A have outgoing connections
        valid_indices = i_indices < nExc
    else:
        # For S_B, only excitatory neurons in B have outgoing connections
        valid_indices = i_indices < nExc  # Adjust if necessary
    
    i_valid = i_indices[valid_indices]
    j_valid = j_indices[valid_indices]
    S_nonzero = S_matrix[i_valid, j_valid]
    
    # Get peak times
    t_source = peak_times_source[i_valid]
    t_target = peak_times_target[j_valid]
    

    # get only top percent of the connection
    # Select top x% of connections
    if top_percent < 100:
        num_top = int(len(S_nonzero) * top_percent / 100)
        if num_top < 1:
            num_top = 1  # Ensure at least one connection is selected
        # Get indices of top connections
        sorted_indices = np.argsort(-S_nonzero)  # Negative sign for descending sort
        top_indices = sorted_indices[:num_top]
        # Filter the data
        S_nonzero = S_nonzero[top_indices]  
        t_source =  t_source[top_indices]  
        t_target =  t_target[top_indices]  

    # Normalize connection strengths for color or alpha mapping
    S_min = np.min(S_nonzero)
    S_max = np.max(S_nonzero)
    S_norm = (S_nonzero - S_min) / (S_max - S_min + 1e-8)  # Avoid division by zero
    
    plt.figure(figsize=(8, 6))
    
    if use_alpha:
        colorvec=np.zeros((np.shape(S_norm)[0],4))
        colorvec[:,2]=1
        colorvec[:,3]=S_norm
        # Use alpha to represent connection strength
        plt.scatter(t_source+np.random.normal(0,0.3,size=np.shape(t_source)), t_target+np.random.normal(0,0.3,size=np.shape(t_source)),s=1,color=colorvec)
    else:
        # Use color to represent connection strength
        plt.scatter(t_source+np.random.normal(0,0.3,size=np.shape(t_source)), t_target,s=1+np.random.normal(0,0.3,size=np.shape(t_source)), c=S_norm, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label('Normalized Connection Strength')
    
    plt.xlabel('Peak Time of Source Neuron')
    plt.ylabel('Peak Time of Target Neuron')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_sorted_connection_matrix(S_matrix, peak_times_source, peak_times_target, title, is_S_A=True, top_percent=100, cmap='viridis',vmin=0, vmax=0.05):
    """
    Plot a sorted connection matrix using imshow, where neurons are ordered based on their peak activity times.

    Parameters:
        S_matrix (np.ndarray): Connection matrix (N x N).
        peak_times_source (np.ndarray): Peak times of source neurons (length N).
        peak_times_target (np.ndarray): Peak times of target neurons (length N).
        title (str): Title for the plot.
        is_S_A (bool): If True, analyzing S_A; if False, analyzing S_B.
        top_percent (float): Percentage of top connections to display (0-100). If <100, only top x% connections are shown.
        cmap (str): Colormap for imshow.
    """
    # Validate top_percent
    if not (0 < top_percent <= 100):
        raise ValueError("top_percent must be between 0 and 100.")
    
    
    # remove inh-> connections because they are all 0
    peak_times_source=peak_times_source[:nExc]
    # Sort source neurons based on peak times
    source_sorted_indices = np.argsort(peak_times_source)
    target_sorted_indices = np.argsort(peak_times_target)
    
    # Reorder the connection matrix
    S_sorted = S_matrix[source_sorted_indices, :][:, target_sorted_indices]
    
    # If top_percent < 100, mask out connections outside the top x%
    if top_percent < 100:
        # Flatten the matrix and get top x% values
        S_nonzero = S_sorted[S_sorted > 0]
        if len(S_nonzero) == 0:
            print("No non-zero connections to display.")
            return
        threshold = np.percentile(S_nonzero, 100 - top_percent)
        # Create a mask for connections below the threshold
        mask = S_sorted < threshold
        # Apply mask
        S_display = np.copy(S_sorted)
        S_display[mask] = 0
    else:
        S_display = S_sorted.copy()
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(np.log(S_display), aspect='auto', cmap=cmap, interpolation='none',vmin=vmin,vmax=vmax)
    plt.colorbar(im, label='Connection Strength')
    
    plt.xlabel('Target Neurons (Sorted by Peak Time)')
    plt.ylabel('Source Neurons (Sorted by Peak Time)')
    plt.title(title)
    
    # Optionally, add grid lines or ticks
    plt.tight_layout()
    plt.show()
    return S_display
    
from scipy import stats
def plotdistribution(S_nonzero_A, time_diffs_A, minthreash, maxthreash, binoption='Default'):
    """
    Plots the distribution of connection strengths for two subsets of data based on time differences.
    Additionally, it plots the mean values for each distribution and assesses the statistical significance
    of the difference between the means using an independent t-test and Mann-Whitney U test.

    Parameters:
        S_nonzero_A (np.ndarray): Array of non-zero connection strengths.
        time_diffs_A (np.ndarray): Array of time differences corresponding to connection strengths.
        minthreash (float): Minimum threshold for the first subset (time_diffs_A < minthreash).
        maxthreash (float): Maximum threshold for the second subset (time_diffs_A > maxthreash).
    """
    # Subset the data based on thresholds
    subset_low = S_nonzero_A[time_diffs_A < minthreash]
    subset_high = S_nonzero_A[time_diffs_A > maxthreash]

    # Ensure there are elements in each subset
    if len(subset_low) == 0 or len(subset_high) == 0:
        raise ValueError("One of the subsets is empty. Please check your data or thresholds.")

    # Combine subsets to determine common bin edges
    combined_data = np.concatenate((subset_low, subset_high))
    num_bins = 100  # Number of bins
    # Define bin edges based on the combined data
    if binoption=='Default':
        bins = np.linspace(combined_data.min(), combined_data.max(), num_bins)
    else:
        bins = np.linspace(binoption[0], binoption[1], num_bins)
        
    
    # Create the plot
    plt.figure(figsize=(12, 7))

    # Plot the first subset (time_diffs_A < minthreash)
    counts_low, bins_low, patches_low = plt.hist(
        subset_low, bins=bins, density=True, histtype='step', linewidth=2, 
        label=f'time_diffs_A < {minthreash}'
    )

    # Plot the second subset (time_diffs_A > maxthreash)
    counts_high, bins_high, patches_high = plt.hist(
        subset_high, bins=bins, density=True, histtype='step', linewidth=2, 
        label=f'time_diffs_A > {maxthreash}'
    )

    # Calculate means
    mean_low = np.mean(subset_low)
    mean_high = np.mean(subset_high)

    # Plot vertical lines for means
    plt.axvline(mean_low, color='blue', linestyle='dashed', linewidth=2, 
                label=f'Mean < {minthreash}: {mean_low:.2f}')
    plt.axvline(mean_high, color='orange', linestyle='dashed', linewidth=2, 
                label=f'Mean > {maxthreash}: {mean_high:.2f}')

    # Perform statistical tests
    # 1. Independent t-test
    t_stat, p_value_t = stats.ttest_ind(subset_low, subset_high, equal_var=False)

    # 2. Mann-Whitney U test
    u_stat, p_value_u = stats.mannwhitneyu(subset_low, subset_high, alternative='two-sided')

    # Determine significance
    alpha = 0.05
    significance_t = 'Significant' if p_value_t < alpha else 'Not Significant'
    significance_u = 'Significant' if p_value_u < alpha else 'Not Significant'

    # Annotate the plot with test results
    plt.text(0.95, 0.15, f'T-test p-value: {p_value_t:.3e} ({significance_t})',
             horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
             fontsize=12, color='blue')

    plt.text(0.95, 0.10, f'Mann-Whitney U p-value: {p_value_u:.3e} ({significance_u})',
             horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
             fontsize=12, color='orange')

    # Customize the plot
    plt.xlabel('Connection Strength', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Distribution of Connection Strengths for Different Time Differences', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Display the plot
    plt.tight_layout()
    plt.show()
    return subset_low, subset_high, bins
    
def z_score_with_zero_handling(A,dim=0):
    # Calculate mean and standard deviation along the first dimension (row-wise)
    mean_A = np.mean(A, axis=dim)
    std_A = np.std(A, axis=dim)
    # Avoid division by zero: if std_A is zero, set it to 1 to prevent invalid division
    std_A[std_A == 0] = 1
    # Calculate z-score
    z_scores = (A - mean_A) / std_A
    return z_scores    

def extract_average_weights_no_bins(A, e, f, p):
    m, n = A.shape
    B = np.zeros(2 * p + 1)
    counts = np.zeros(2 * p + 1)
    
    # Initialize a list to collect weights for each offset
    offset_weights = [[] for _ in range(2 * p + 1)]

    for k in range(n):
        # Compute time differences for column k
        t = e - f[k]  # Shape: (m,)
        # Get indices sorted by absolute time difference
        sorted_indices = np.argsort(np.abs(t))
        # Loop over offsets from 0 to 2p
        for o in range(min(2 * p + 1, m)):
            i = sorted_indices[o]
            weight = A[i, k]
            offset_weights[o].append(weight)
    
    # Compute average weights for each offset
    for o in range(2 * p + 1):
        if offset_weights[o]:
            B[o] = np.mean(offset_weights[o])
            counts[o] = len(offset_weights[o])
        else:
            B[o] = np.nan  # No data for this offset
    
    return B

def extract_average_weights(A, e, f, p):
    m, n = A.shape
    B = np.zeros(2 * p + 1)
    counts = np.zeros(2 * p + 1)  # To keep track of the number of weights in each bin

    # Compute the differences in firing times (m x n matrix)
    delta_t = e[:, np.newaxis] - f[np.newaxis, :]  # Shape: (m, n)

    # Define bin edges from -p - 0.5 to p + 0.5 to capture all differences
    bin_edges = np.linspace(-p - 0.5, p + 0.5, 2 * p + 2)

    # Flatten the differences and corresponding weights
    delta_t_flat = delta_t.flatten()
    weights_flat = A.flatten()

    # Assign each difference to a bin
    bin_indices = np.digitize(delta_t_flat, bins=bin_edges) - 1  # Subtract 1 to get bin indices from 0 to 2p

    # Filter out differences that fall outside our bins
    valid_indices = np.where((bin_indices >= 0) & (bin_indices < 2 * p + 1))

    bin_indices = bin_indices[valid_indices]
    weights_valid = weights_flat[valid_indices]

    # Initialize a list to collect weights for each bin
    bin_weights = [[] for _ in range(2 * p + 1)]

    # Collect weights into bins
    for idx, bin_idx in enumerate(bin_indices):
        bin_weights[bin_idx].append(weights_valid[idx])

    # Compute the average weights for each bin
    for b in range(2 * p + 1):
        if bin_weights[b]:
            B[b] = np.mean(bin_weights[b])
            counts[b] = len(bin_weights[b])
        else:
            B[b] = np.nan  # No data for this bin

    return B

def analyze_connection_matrix(S, act_avg_X, act_avg_Y, ax, title):
    """
    Analyze the connection matrix S between two groups (X and Y) and plot correlation vs. connection strength.

    Parameters
    ----------
    S : np.ndarray
        An (n x n) matrix where S[i, j] is the connection strength.
    act_avg_X : np.ndarray
        An (T x n) matrix of activity for group X, where T is the number of timepoints.
    act_avg_Y : np.ndarray
        An (T x n) matrix of activity for group Y, where T is the number of timepoints.
    ax : matplotlib.axes.Axes
        Axis object to plot on.
    title : str
        Title of the subplot.

    Returns
    -------
    None
    """
    # Compute the full correlation matrix between all units in X and Y
    corr_matrix = np.corrcoef(act_avg_X.T, act_avg_Y.T)[:S.shape[0], S.shape[0]:]

    # Mask for S > 0
    mask = np.logical_and(S > 0, ~np.isnan(corr_matrix))

    # Extract correlations and connection strengths where S > 0
    correlations = corr_matrix[mask]
    strengths = S[mask]

    # Create a scatter plot
    ax.scatter(correlations, strengths, s=0.1, alpha=0.7, label='Data Points')

    # Fit a linear regression line: y = m*x + b
    slope, intercept = np.polyfit(correlations, strengths, 1)
    x_fit = np.linspace(min(correlations), max(correlations), 100)
    y_fit = slope * x_fit + intercept

    # Plot the best-fit line
    ax.plot(x_fit, y_fit, 'r--', label=f'Best Fit Line (slope = {slope:.5f})')

    # Labeling the axes and setting the title
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Connection Strength')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return correlations, strengths

#load models
tind=np.array([10])
S_A_sorted=[]
S_B_sorted=[]
subset_lowA=[]
subset_highA=[]
subset_lowB=[]
subset_highB=[]
corr_A=[]
corr_B=[]
stre_A=[]
stre_B=[]

for k in range(np.shape(best_models)[0]):
    for t in tind:
            #  analyze weight distribution
        
        max_range=[0,max_dur+min_dur] # time range to choose max firing time
        nExc=nUnit-nInh #number of excitory units
        
        
        model=build_model(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha, seed1=seed1)
        # load weights
        checkpoint_filepath=best_models[k][t]
        model.load_weights(checkpoint_filepath)
        
        # the weights for this RNN
        RNN_input_kernel=model.layers[1].get_weights()[0]
        RNN_layer_Recurrent_kernel=model.layers[1].get_weights()[1]
        dense_kernel_A=model.layers[2].get_weights()[0]
        dense_bias_A=model.layers[2].get_weights()[1]
        dense_kernel_B=model.layers[3].get_weights()[0]
        dense_bias_B=model.layers[3].get_weights()[1]
        
        in_A,in_B=np.split(RNN_input_kernel,2, axis=1)
        Wr_A, Wr_B, S_A, S_B=np.split(RNN_layer_Recurrent_kernel,4, axis=1)    
        
        
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
        # take 12s interval
        #act_avg_A=act_avg_A[-max_dur:,:]
        #act_avg_B=act_avg_B[-max_dur:,:]
        act_avg_A=act_avg_A
        act_avg_B=act_avg_B
        
        # Compute the time of maximum activity for each neuron
        peak_times_A = np.argmax(act_avg_A, axis=0)  # Shape: (N,)
        peak_times_B = np.argmax(act_avg_B, axis=0)  # Shape: (N,)
        
        T=np.shape(act_avg_A)[0]
        nExc = nUnit - nInh  # Number of excitatory neurons
        
        top_percent=100
        # Analyze S_A connections (from A to B) using absolute time difference
        _,_=analyze_connections(S_A, peak_times_A, peak_times_B, 'S_A Connection Strength vs Time Difference of Peak Activity (Absolute Time)', is_S_A=True, use_phase=False,top_percent=top_percent)
        
        # Analyze S_A connections (from A to B) using phase (cyclic time difference)
        time_diffs_A, S_nonzero_A=analyze_connections(S_A, peak_times_A, peak_times_B, 'S_A Connection Strength vs Time Difference of Peak Activity (Cyclic)', is_S_A=True, use_phase=True,top_percent=top_percent)
        
        # Analyze S_B connections (from B to A) using absolute time difference
        _,_=analyze_connections(S_B, peak_times_B, peak_times_A, 'S_B Connection Strength vs Time Difference of Peak Activity (Absolute Time)', is_S_A=False, use_phase=False,top_percent=top_percent)
        
        # Analyze S_B connections (from B to A) using phase (cyclic time difference)
        time_diffs_B, S_nonzero_B=analyze_connections(S_B, peak_times_B, peak_times_A, 'S_B Connection Strength vs Time Difference of Peak Activity (Cyclic)', is_S_A=False, use_phase=True,top_percent=top_percent)
    
        # Plot using color to represent connection strength
        plot_time_of_max_activity(S_A, peak_times_A, peak_times_B, 
                                  'S_A Connections: Peak Times Scatter Plot (Color)', 
                                  is_S_A=True, use_alpha=False,top_percent=top_percent)
        
        # Plot using alpha to represent connection strength
        plot_time_of_max_activity(S_A, peak_times_A, peak_times_B, 
                                  'S_A Connections: Peak Times Scatter Plot (Alpha)', 
                                  is_S_A=True, use_alpha=True,top_percent=top_percent)
        
        
        
        
        
        S_A_sorted_sub=plot_sorted_connection_matrix(
            S_matrix=S_A,
            peak_times_source=peak_times_A,
            peak_times_target=peak_times_B,
            title='S_A: Sorted Connection Matrix',
            is_S_A=True,
            top_percent=100,  # Display all connections
            cmap='plasma',
            vmin=-4,
            vmax=-4.7      
        )
        S_A_sorted.append(S_A_sorted_sub)
        
        # Analyze and plot S_B
        S_B_sorted_sub=plot_sorted_connection_matrix(
            S_matrix=S_B,
            peak_times_source=peak_times_B,
            peak_times_target=peak_times_A,
            title='S_B: Sorted Connection Matrix',
            is_S_A=False,
            top_percent=100,  # Display all connections
            cmap='plasma',
            vmin=-4,
            vmax=-4.7
        )
        S_B_sorted.append(S_B_sorted_sub)
        
        # plot weight distribution  for close pairs or far pairs
        #plotdistribution(S_nonzero_A,time_diffs_A,50,500)
        #plotdistribution(S_nonzero_B,time_diffs_B,50,500)
        subset_low, subset_high, bins=plotdistribution(S_nonzero_A,time_diffs_A,50,500,binoption=np.array([0,0.15]))
        subset_lowA.append(subset_low)
        subset_highA.append(subset_high)
        subset_low, subset_high, bins=plotdistribution(S_nonzero_B,time_diffs_B,50,500,binoption=np.array([0,0.15]))
        subset_lowB.append(subset_low)
        subset_highB.append(subset_high)
    
        # plot average weight distribution
        binw=200
        dist_A=extract_average_weights_no_bins(S_A_sorted_sub, peak_times_A[:nExc],peak_times_B, binw)
        dist_B=extract_average_weights_no_bins(S_B_sorted_sub, peak_times_B[:nExc],peak_times_A,binw)
        dist_A=extract_average_weights_no_bins(z_score_with_zero_handling(S_A_sorted_sub,dim=0), peak_times_A[:nExc],peak_times_B, binw)
        dist_B=extract_average_weights_no_bins(z_score_with_zero_handling(S_B_sorted_sub,dim=0), peak_times_B[:nExc],peak_times_A,binw)
        x=np.arange(-binw,binw+1)
        fig,axs=plt.subplots(1,2)
        axs[0].plot(x,dist_A)
        axs[1].plot(x,dist_B)
        
        
        def running_mean(x, N):
            cumsum = np.cumsum(np.insert(x, 0, 0)) 
            return (cumsum[N:] - cumsum[:-N]) / float(N)
        window=10
        fig,axs=plt.subplots(1,2)
        axs[0].plot(running_mean(x,window),running_mean(dist_A,window))
        axs[1].plot(running_mean(x,window),running_mean(dist_B,window))
        
        # Example usage
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Call the function for S_A and S_B
        correlations,strengths=analyze_connection_matrix(S_A, act_avg_A, act_avg_B, axes[0], 'S_A: Correlation vs. Connection Strength')
        corr_A.append(correlations)
        stre_A.append(strengths)
        correlations,strengths=analyze_connection_matrix(S_B, act_avg_B, act_avg_A, axes[1], 'S_B: Correlation vs. Connection Strength')
        corr_B.append(correlations)
        stre_B.append(strengths)        
        
        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()        


S_A_sorted_avg=np.mean(np.array(S_A_sorted),axis=0)  
S_B_sorted_avg=np.mean(np.array(S_B_sorted),axis=0)

vmin=-4
vmax=-4.7 
# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

# Plot for S_A_sorted
im1 = axes[0].imshow(np.log(S_A_sorted_avg), aspect='auto', cmap='plasma', interpolation='none', vmin=vmin, vmax=vmax)
axes[0].set_title('S_A: Sorted Connection Matrix')
axes[0].set_xlabel('Target Neurons (Sorted by Peak Time)')
axes[0].set_ylabel('Source Neurons (Sorted by Peak Time)')
fig.colorbar(im1, ax=axes[0], label='Connection Strength')

# Plot for S_B_sorted
im2 = axes[1].imshow(np.log(S_B_sorted_avg), aspect='auto', cmap='plasma', interpolation='none', vmin=vmin, vmax=vmax)
axes[1].set_title('S_B: Sorted Connection Matrix')
axes[1].set_xlabel('Target Neurons (Sorted by Peak Time)')
axes[1].set_ylabel('Source Neurons (Sorted by Peak Time)')
fig.colorbar(im2, ax=axes[1], label='Connection Strength')

# Adjust layout
plt.tight_layout()

if save:
    #plt.savefig(os.path.join(r"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4ramp_probvar",f"weight_distribution_within_4"),transparent=True,dpi=600)
    plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    # Your plotting code
    plt.savefig(os.path.join(figure_folders,f"S_sorted_log_conprob_{conProbability[t]}.svg"), format='svg')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

# Plot for S_A_sorted
im1 = axes[0].imshow((S_A_sorted_avg), aspect='auto', cmap='plasma', interpolation='none', vmin=0.005, vmax=0.025)
axes[0].set_title('S_A: Sorted Connection Matrix')
axes[0].set_xlabel('Target Neurons (Sorted by Peak Time)')
axes[0].set_ylabel('Source Neurons (Sorted by Peak Time)')
fig.colorbar(im1, ax=axes[0], label='Connection Strength')

# Plot for S_B_sorted
im2 = axes[1].imshow((S_B_sorted_avg), aspect='auto', cmap='plasma', interpolation='none', vmin=0.005, vmax=0.025)
axes[1].set_title('S_B: Sorted Connection Matrix')
axes[1].set_xlabel('Target Neurons (Sorted by Peak Time)')
axes[1].set_ylabel('Source Neurons (Sorted by Peak Time)')
fig.colorbar(im2, ax=axes[1], label='Connection Strength')

# Adjust layout
plt.tight_layout()

if save:
    #plt.savefig(os.path.join(r"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4ramp_probvar",f"weight_distribution_within_4"),transparent=True,dpi=600)
    plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    # Your plotting code
    plt.savefig(os.path.join(figure_folders,f"S_sorted_conprob_{conProbability[t]}.svg"), format='svg')


lowA = np.concatenate(subset_lowA)
highA = np.concatenate(subset_highA)
lowB = np.concatenate(subset_lowB)
highB = np.concatenate(subset_highB)



def plotdist2_with_subplots(data_pairs, bins):
    """
    Plot multiple subsets in subplots.
    
    Parameters:
        data_pairs (list of tuples): List of (subset_low, subset_high) data pairs.
        bins (int or sequence): Bins for the histogram.
        minthreash (float): Minimum threshold value for annotation.
        maxthreash (float): Maximum threshold value for annotation.
    """
    num_plots = len(data_pairs)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 7 * num_plots), sharex=True)
    
    if num_plots == 1:
        axes = [axes]  # Make axes iterable if there's only one subplot.
    
    for i, (subset_low, subset_high) in enumerate(data_pairs):
        ax = axes[i]
        
        # Plot the first subset (time_diffs_A < minthreash)
        counts_low, bins_low, patches_low = ax.hist(
            subset_low, bins=bins, density=True, histtype='step', linewidth=2, 
            label=f'time_diffs_A < {minthreash}'
        )
        
        # Plot the second subset (time_diffs_A > maxthreash)
        counts_high, bins_high, patches_high = ax.hist(
            subset_high, bins=bins, density=True, histtype='step', linewidth=2, 
            label=f'time_diffs_A > {maxthreash}'
        )
        
        # Calculate means
        mean_low = np.mean(subset_low)
        mean_high = np.mean(subset_high)
        
        # Plot vertical lines for means
        ax.axvline(mean_low, color='blue', linestyle='dashed', linewidth=2, 
                   label=f'Mean < {minthreash}: {mean_low:.2f}')
        ax.axvline(mean_high, color='orange', linestyle='dashed', linewidth=2, 
                   label=f'Mean > {maxthreash}: {mean_high:.2f}')
        
        # Perform statistical tests
        t_stat, p_value_t = stats.ttest_ind(subset_low, subset_high, equal_var=False)
        u_stat, p_value_u = stats.mannwhitneyu(subset_low, subset_high, alternative='two-sided')
        
        # Determine significance
        alpha = 0.05
        significance_t = 'Significant' if p_value_t < alpha else 'Not Significant'
        significance_u = 'Significant' if p_value_u < alpha else 'Not Significant'
        
        # Annotate the plot with test results
        ax.text(0.95, 0.15, f'T-test p-value: {p_value_t:.3e} ({significance_t})',
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes,
                fontsize=12, color='blue')
        
        ax.text(0.95, 0.10, f'Mann-Whitney U p-value: {p_value_u:.3e} ({significance_u})',
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes,
                fontsize=12, color='orange')
        
        # Customize the subplot
        ax.set_xlabel('Connection Strength', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        if i==0:
            ax.set_title(f'Distribution A: Connection Strengths for Different Time Differences', fontsize=16)
        else:
            ax.set_title(f'Distribution B: Connection Strengths for Different Time Differences', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust layout
    plt.tight_layout()
    #plt.show()

# Example usage
data_pairs = [(lowA, highA), (lowB, highB)]
minthreash=50
maxthreash=500
plotdist2_with_subplots(data_pairs, bins=100)

if save:
    #plt.savefig(os.path.join(r"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4ramp_probvar",f"weight_distribution_within_4"),transparent=True,dpi=600)
    plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    # Your plotting code
    plt.savefig(os.path.join(figure_folders,f"S_dist_comparison_conprob_{conProbability[t]}.svg"), format='svg')
    
    
    
corr_A=np.concatenate(corr_A)
stre_A=np.concatenate(stre_A)
corr_B=np.concatenate(corr_B)
stre_B=np.concatenate(stre_B)



fig, ax = plt.subplots(1, 2, figsize=(14, 6))



ax[0].scatter(corr_A, stre_A, s=0.05, alpha=0.7, label='Data Points')
# Fit a linear regression line: y = m*x + b
slope, intercept = np.polyfit(corr_A, stre_A, 1)
x_fit = np.linspace(min(corr_A), max(corr_A), 100)
y_fit = slope * x_fit + intercept
# Plot the best-fit line
ax[0].plot(x_fit, y_fit, 'r--', label=f'Best Fit Line (slope = {slope:.5f})')

# Labeling the axes and setting the title
ax[0].set_xlabel('Correlation')
ax[0].set_ylabel('Connection Strength')
ax[0].set_title('S_A: Correlation vs. Connection Strength')
ax[0].legend()
ax[0].grid(True)


ax[1].scatter(corr_B, stre_B, s=0.05, alpha=0.7, label='Data Points')
# Fit a linear regression line: y = m*x + b
slope, intercept = np.polyfit(corr_B, stre_B, 1)
x_fit = np.linspace(min(corr_B), max(corr_B), 100)
y_fit = slope * x_fit + intercept
# Plot the best-fit line
ax[1].plot(x_fit, y_fit, 'r--', label=f'Best Fit Line (slope = {slope:.5f})')

# Labeling the axes and setting the title
ax[1].set_xlabel('Correlation')
ax[1].set_ylabel('Connection Strength')
ax[1].set_title('S_A: Correlation vs. Connection Strength')
ax[1].legend()
ax[1].grid(True)
#%% plot last graph with hist2d denstiy map
fig, ax = plt.subplots(1, 2, figsize=(14, 6),sharex=True,sharey=True)

vmin=0000
vmax=4000

hist = ax[0].hist2d(corr_A, stre_A, bins=50, cmap='viridis', cmin=vmin,cmax=vmax)

# Add a colorbar for the density
cbar = plt.colorbar(hist[3], ax=ax)
cbar.set_label('Counts')
# Fit a linear regression line: y = m*x + b
slope, intercept = np.polyfit(corr_A, stre_A, 1)
x_fit = np.linspace(min(corr_A), max(corr_A), 100)
y_fit = slope * x_fit + intercept
# Plot the best-fit line
ax[0].plot(x_fit, y_fit, 'r--', label=f'Best Fit Line (slope = {slope:.5f})')

# Labeling the axes and setting the title
ax[0].set_xlabel('Correlation')
ax[0].set_ylabel('Connection Strength')
ax[0].set_title('S_A: Correlation vs. Connection Strength')
ax[0].legend()
ax[0].grid(True)
ax[0].set_ylim(0,0.1)


# Create a 2D histogram (density map)
hist = ax[1].hist2d(corr_B, stre_B, bins=50, cmap='viridis', cmin=vmin,cmax=vmax)

# Fit a linear regression line: y = m*x + b
slope, intercept = np.polyfit(corr_B, stre_B, 1)
x_fit = np.linspace(min(corr_B), max(corr_B), 100)
y_fit = slope * x_fit + intercept
# Plot the best-fit line
ax[1].plot(x_fit, y_fit, 'r--', label=f'Best Fit Line (slope = {slope:.5f})')

# Labeling the axes and setting the title
ax[1].set_xlabel('Correlation')
ax[1].set_ylabel('Connection Strength')
ax[1].set_title('S_A: Correlation vs. Connection Strength')
ax[1].legend()
ax[1].grid(True)
ax[1].set_ylim(0,0.1)
#%% visualize perturbation experiments

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
    noise_range=1/20 #level of temporal noise with respect to duration
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
def make_pertind(In_ons,trial1,ind1,trial2,ind2):
    addvec=np.array([int(ind1/dt),int(ind2/dt)])
    pert_ind=In_ons[:,[trial1,trial2]]+addvec
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

def perturb_and_decode(trial1,ind1,trial2,ind2,pert_state,order,pca_A,pca_B,pca_C,clf_A,clf_B,clf_C):
    pert_state=pert_state
    x, y, In_ons=makeInOut_sameint(sample_size,trial_num,inputdur,nInput,min_dur,max_dur,dt,order)
    pert_ind=make_pertind(In_ons,trial1,ind1,trial2,ind2)

    #pert_ind=np.zeros((4,2))
    #model2=build_model_perturb(pert_ind, pert_state)
    model2=build_model_perturb(nUnit=nUnit, 
                               nInh=nInh, 
                               nInput=nInput, 
                               con_prob=con_prob, 
                               maxval=maxval ,
                               ReLUalpha=ReLUalpha,
                               pert_ind=pert_ind, 
                               pert_state=pert_state,
                               seed1=seed1)
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
    
    int_diff=int(np.round((In_ons[0,trial2]-In_ons[0,trial1])/min_dur)*min_dur)
    actpart_A=np.zeros((np.shape(activities_A)[0],min_dur+max_dur+200+int_diff,np.shape(activities_A)[2]))
    actpart_B=np.zeros((np.shape(activities_B)[0],min_dur+max_dur+200+int_diff,np.shape(activities_B)[2]))
    predictions2=np.zeros((np.shape(predictions)[0],min_dur+max_dur+200+int_diff,np.shape(predictions)[2]))
    # take predictions at the time of perturbation
    for i in np.arange(np.shape(In_ons)[0]):
        actpart_A[i,:,:]=activities_A[i,In_ons[i,trial2]-int_diff:In_ons[i,trial2]+min_dur+max_dur+200,:]
        actpart_B[i,:,:]=activities_B[i,In_ons[i,trial2]-int_diff:In_ons[i,trial2]+min_dur+max_dur+200,:]
        predictions2[i,:,:]=predictions[i,In_ons[i,trial2]-int_diff:In_ons[i,trial2]+(min_dur+max_dur)+200,:]
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
    
sample_size=6
trial_num=8
pert_state=0 # 0 to perturb RNN A and 1 to perturb RNN B
# take circular mean of the prediction
option=0 #0 for circular, 1 for mean
trial1=3
trial2=3
time_1=1000
time_2=5000
order=1# order 0 starts all trials with max_dur, 1 starts with min_dur

tind=[0,3,4,5,8,9]
tind=[1,2]
tind=[0,3,4,5,10]
for t in tind:
#for t in aa:
    maxval=weight_max[t]
    con_prob=conProbability[t]



    #build models
    model=build_model(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha, seed1=seed1)
    # load weights
    checkpoint_filepath=os.path.join(foldername[t], f"epoch_{model_index[t]:05d}.ckpt")
    model.load_weights(checkpoint_filepath)

    # create classifier for decoding
    outputs = [layer.output for layer in model.layers[1:]]  # Exclude the input layer
    activity_model = Model(inputs=model.input, outputs=outputs)
    trial_num=8
    # Get the output and activities of all layers for the new input data
    x, y, In_ons=x, y, In_ons=makeInOut(2,8,inputdur,nInput,min_dur,max_dur,dt)
    #xnew=sum(xnew,axis=2)
    output_and_activities = activity_model.predict(x)
    activities_A = output_and_activities[0]  # Activities of all intermediate layers
    activities_B=output_and_activities[1]
    act_avg_A=avgAct(activities_A,In_ons,min_dur,max_dur)
    act_avg_B=avgAct(activities_B,In_ons,min_dur,max_dur)
    act_stack_A=Act_2dsort(activities_A,In_ons,min_dur,max_dur)#(time,nUnit) time is multiple of mindur+maxdur
    act_stack_B=Act_2dsort(activities_B,In_ons,min_dur,max_dur)
    
    # make classifying classes
    Class_per_sec=2
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
    pred_A, pred_B, pred_C, A_predictions2, pert_ind_3, In_ons_a=perturb_and_decode(trial1,time_1,trial2,time_2,pert_state,order,pca_A,pca_B,pca_C,clf_A,clf_B,clf_C)
    predictions2.append(A_predictions2)
    pert_ind_2.append(pert_ind_3)
    In_ons_2.append(In_ons_a)
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
    
    #perturb B
    pert_state=1
    pred_A, pred_B, pred_C, B_predictions2, pert_ind_3,In_ons_a=perturb_and_decode(trial1,time_1,trial2,time_2,pert_state,order,pca_A,pca_B,pca_C,clf_A,clf_B,clf_C)
    predictions2.append(B_predictions2)
    pert_ind_2.append(pert_ind_3)
    In_ons_2.append(In_ons_a)
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
    
    
    fig, axs = plt.subplots(2,2,sharex=True,sharey='row',figsize=(10, 6))
    Line=[None]*4
    pert_target=['A', 'B']
    for ss in range(2):
        #axs[ss].plot(y[ss, :, 0], color='blue',label='Target 0', alpha=0.5)
        #x=np.arange(In_ons_2[ss][0,0],In_ons_2[ss][0,3])
        axs[0,ss].plot(predictions2[ss][0, :, 0], color='green',label='Prediction_A 0',alpha=0.5)
        axs[0,ss].plot(predictions2[ss][0, :, 2], color='turquoise',label='Prediction_B 0',alpha=0.5)
        #axs[ss].plot(y[ss, :, 1], color='red',label='Target 1',alpha=0.5)
        axs[0,ss].plot(predictions2[ss][0, :, 1], color='orangered',label='Prediction_A 1',alpha=0.5)
        axs[0,ss].plot(predictions2[ss][0, :, 3], color='gold',label='Prediction_B 1',alpha=0.5)
        axs[0,ss].axvline(np.mean(pert_ind_2[ss],0)[0], ymin=0.7, ymax=1)
        axs[0,ss].axvline(np.mean(pert_ind_2[ss],0)[1], ymin=0.7, ymax=1)
        axs[0,ss].set_title(f'Perturbation on {pert_target[ss]}')
        axs[0, ss].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')  # Place legend outside
        axs[0, ss].set_ylim(bottom=-0.6,top=0.6)
    
        axs[1,ss].plot(pred_Aall[ss],color='#ff7f0e',alpha=0.1)
        axs[1,ss].plot(pred_Ball[ss],color='#1f77b4',alpha=0.1)
        axs[1,ss].plot(predavg_A[ss],color='#ff7f0e',label='RNN A')
        axs[1,ss].plot(predavg_B[ss],color='#1f77b4',label='RNN B')
        #axs[1,ss].plot(predavg_C[ss],color='#2ca02c',label='RNN A&B')
        
        axs[1,ss].set_title(f'Decoded results')
        axs[1, ss].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')  # Place legend outside
        axs[1, ss].set_yticks(Class_per_sec*np.array([0,6,12,18]))
        axs[1,ss].set_yticklabels([0,6,12,18])
        xticks = axs[1,ss].get_xticks()
        scaled_xticks = np.round(xticks * 0.01)
        # Set new tick labels
        axs[1,ss].set_xticklabels(scaled_xticks)
    
    plt.suptitle(f'Connection Probability {conProbability[t]}')
    print(f"Loop {t+1} out of {len(tind)}")
    # Adjusting layout
    plt.tight_layout()
    if save:
        #plt.savefig(os.path.join(savepath,f"Perturb_decode_{time_1}_{time_2}_{order}_{option}"),transparent=True,dpi=400)
        plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
        plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
        
        # Your plotting code
        plt.savefig(f"Perturb_decode_{time_1}_{time_2}_{order}_{option}_conprob_{conProbability[t]}.svg", format='svg')
    plt.show()
    
    
    
    
#%% visualize perturbation experiment with noise

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
    noise_range=1/20 #level of temporal noise with respect to duration
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

def perturb_and_decode_noise(trial1,ind1,pert_state,order,pca_A,pca_B,pca_C,clf_A,clf_B,clf_C,pert_noisesd,stop):
    pert_state=pert_state
    x, y, In_ons=makeInOut_sameint(sample_size,trial_num,inputdur,nInput,min_dur,max_dur,dt,order)
        
    pert_ind=make_pertind(In_ons,trial1,ind1)
    if stop:
        x,In_ons2=makeInput(x,In_ons,pert_ind)

    #pert_ind=np.zeros((4,2))
    #model2=build_model_perturb(pert_ind, pert_state)
    model2=build_model_perturb_noise(nUnit=nUnit, 
                               nInh=nInh, 
                               nInput=nInput, 
                               con_prob=con_prob, 
                               maxval=maxval ,
                               ReLUalpha=ReLUalpha,
                               pert_ind=pert_ind, 
                               pert_state=pert_state,
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
    actpart_A=np.zeros((np.shape(activities_A)[0],min_dur+max_dur+200+int_diff,np.shape(activities_A)[2]))
    actpart_B=np.zeros((np.shape(activities_B)[0],min_dur+max_dur+200+int_diff,np.shape(activities_B)[2]))
    predictions2=np.zeros((np.shape(predictions)[0],min_dur+max_dur+200+int_diff,np.shape(predictions)[2]))
    # take predictions at the time of perturbation
    for i in np.arange(np.shape(In_ons)[0]):
        actpart_A[i,:,:]=activities_A[i,In_ons[i,trial2]-int_diff:In_ons[i,trial2]+min_dur+max_dur+200,:]
        actpart_B[i,:,:]=activities_B[i,In_ons[i,trial2]-int_diff:In_ons[i,trial2]+min_dur+max_dur+200,:]
        predictions2[i,:,:]=predictions[i,In_ons[i,trial2]-int_diff:In_ons[i,trial2]+(min_dur+max_dur)+200,:]
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
    
sample_size=6
trial_num=8
pert_state=0 # 0 to perturb RNN A and 1 to perturb RNN B
pert_noisesd=5.0
stop=False
# take circular mean of the prediction
option=0 #0 for circular, 1 for mean
trial1=3
time_1=2000
order=1# order 0 starts all trials with max_dur, 1 starts with min_dur
tind=[0,3,4,5,11]
#tind=[0,4,11]
#tind=[11]
tind=np.array([11])
for t in tind:
#for t in aa:
    maxval=weight_max[t]
    con_prob=conProbability[t]



    #build models
    model=build_model(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha, seed1=seed1)
    # load weights
    checkpoint_filepath=os.path.join(foldername[t], f"epoch_{model_index[t]:05d}.ckpt")
    model.load_weights(checkpoint_filepath)

    # create classifier for decoding
    outputs = [layer.output for layer in model.layers[1:]]  # Exclude the input layer
    activity_model = Model(inputs=model.input, outputs=outputs)
    trial_num=8
    # Get the output and activities of all layers for the new input data
    x, y, In_ons=x, y, In_ons=makeInOut(2,8,inputdur,nInput,min_dur,max_dur,dt)
    #xnew=sum(xnew,axis=2)
    output_and_activities = activity_model.predict(x)
    activities_A = output_and_activities[0]  # Activities of all intermediate layers
    activities_B=output_and_activities[1]
    act_avg_A=avgAct(activities_A,In_ons,min_dur,max_dur)
    act_avg_B=avgAct(activities_B,In_ons,min_dur,max_dur)
    act_stack_A=Act_2dsort(activities_A,In_ons,min_dur,max_dur)#(time,nUnit) time is multiple of mindur+maxdur
    act_stack_B=Act_2dsort(activities_B,In_ons,min_dur,max_dur)
    
    # make classifying classes
    Class_per_sec=2
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
    pred_A, pred_B, pred_C, A_predictions2, pert_ind_3, In_ons_a=perturb_and_decode_noise(trial1,time_1,pert_state,order,pca_A,pca_B,pca_C,clf_A,clf_B,clf_C,pert_noisesd,stop)
    predictions2.append(A_predictions2)
    pert_ind_2.append(pert_ind_3)
    In_ons_2.append(In_ons_a)
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
    
    #perturb B
    pert_state=1
    pred_A, pred_B, pred_C, B_predictions2, pert_ind_3,In_ons_a=perturb_and_decode_noise(trial1,time_1,pert_state,order,pca_A,pca_B,pca_C,clf_A,clf_B,clf_C,pert_noisesd,stop)
    predictions2.append(B_predictions2)
    pert_ind_2.append(pert_ind_3)
    In_ons_2.append(In_ons_a)
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
    
    
    fig, axs = plt.subplots(2,2,sharex=True,sharey='row',figsize=(10, 6))
    Line=[None]*4
    pert_target=['A', 'B']
    for ss in range(2):
        #axs[ss].plot(y[ss, :, 0], color='blue',label='Target 0', alpha=0.5)
        #x=np.arange(In_ons_2[ss][0,0],In_ons_2[ss][0,3])
        axs[0,ss].plot(predictions2[ss][0, :, 0], color='green',label='Prediction_A 0',alpha=0.5)
        axs[0,ss].plot(predictions2[ss][0, :, 2], color='turquoise',label='Prediction_B 0',alpha=0.5)
        #axs[ss].plot(y[ss, :, 1], color='red',label='Target 1',alpha=0.5)
        axs[0,ss].plot(predictions2[ss][0, :, 1], color='orangered',label='Prediction_A 1',alpha=0.5)
        axs[0,ss].plot(predictions2[ss][0, :, 3], color='gold',label='Prediction_B 1',alpha=0.5)
        axs[0,ss].axvline(np.mean(pert_ind_2[ss],0)[0], ymin=0.7, ymax=1)
        axs[0,ss].set_title(f'Perturbation on {pert_target[ss]}')
        axs[0, ss].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')  # Place legend outside
        axs[0, ss].set_ylim(bottom=-0.6,top=0.6)
    
        axs[1,ss].plot(pred_Aall[ss],color='#ff7f0e',alpha=0.1)
        axs[1,ss].plot(pred_Ball[ss],color='#1f77b4',alpha=0.1)
        axs[1,ss].plot(predavg_A[ss],color='#ff7f0e',label='RNN A')
        axs[1,ss].plot(predavg_B[ss],color='#1f77b4',label='RNN B')
        #axs[1,ss].plot(predavg_C[ss],color='#2ca02c',label='RNN A&B')
        
        axs[1,ss].set_title(f'Decoded results')
        axs[1, ss].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')  # Place legend outside
        axs[1, ss].set_yticks(Class_per_sec*np.array([0,6,12,18]))
        axs[1,ss].set_yticklabels([0,6,12,18])
        xticks = axs[1,ss].get_xticks()
        scaled_xticks = np.round(xticks * 0.01)
        # Set new tick labels
        axs[1,ss].set_xticklabels(scaled_xticks)
    
    plt.suptitle(f'Connection Probability {conProbability[t]}')
    print(f"Loop {t+1} out of {len(tind)}")
    # Adjusting layout
    plt.tight_layout()
    if save:
        #plt.savefig(os.path.join(savepath,f"Perturb_decode_{time_1}_{time_2}_{order}_{option}"),transparent=True,dpi=400)
        plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
        plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
        
        # Your plotting code
        plt.savefig(f"Perturb_decode_{time_1}_{time_2}_{order}_{option}_conprob_{conProbability[t]}.svg", format='svg')
    plt.show()
    
    
    
#%% perturbation with many noise ->  then decode

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
    
sample_size=12
trial_num=8
pert_state=0 # 0 to perturb RNN A and 1 to perturb RNN B
pert_noisesd=0.8#1.5 original 0.8
stop=False
# take circular mean of the prediction
option=0 #0 for circular, 1 for mean
trial1=2


pert_prob=1/100 #1/100 # probability of perturbation original 1/200
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


# Initialize a dictionary to store everything
Allinfo = {
    't_index':       [],
    'Confmat_A_ave': [],
    'Confmat_B_ave': [],
    'Offset_mat_ave':[],
    'temp_error_ave':[],
    'cat_error_ave': []
}


tind=np.array([0,3,4,5,11])
#tind=[0,4,11]
#tind=[11]
#tind=[0]

tind=np.arange(len(conProbability))
#tind=np.array([0,2,4,6,9])
#tind=np.array([4])
#tind=np.arange(12)
loopind=0
pred_diff_all=np.zeros(len(tind))
for t in tind:
    confA_list   = []
    confB_list   = []
    offset_list  = []
    terror_list  = []
    caterr_list  = []
    pred_diff2=[]
    
    for k in range(np.shape(best_models)[0]):
    #for t in aa:
        loopind+=1
        maxval=weight_max[t]
        con_prob=conProbability[t]
    
        #build models
        model=build_model(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha, seed1=seed1)
        # load weights
        checkpoint_filepath=best_models[k][t]
        model.load_weights(checkpoint_filepath)
    
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
        pred_diff_all[t]=np.linalg.norm(pred_diff)
        pred_diff2.append(pred_diff_all[t])
    
        
        
        fig, axs = plt.subplots(2,1,sharex=True,sharey='row',figsize=(10, 6))
        Line=[None]*4
        pert_target=['A', 'B']
        ss=0
        axs[0].plot(predictions2[ss][0, :, 0], color='green',label='Prediction_A 0',alpha=0.5)
        axs[0].plot(predictions2[ss][0, :, 2], color='turquoise',label='Prediction_B 0',alpha=0.5)
        #axs[ss].plot(y[ss, :, 1], color='red',label='Target 1',alpha=0.5)
        axs[0].plot(predictions2[ss][0, :, 1], color='orangered',label='Prediction_A 1',alpha=0.5)
        axs[0].plot(predictions2[ss][0, :, 3], color='gold',label='Prediction_B 1',alpha=0.5)
        axs[0].axvline(np.mean(pert_ind_2[ss],0)[0], ymin=0.7, ymax=1)
        axs[0].set_title(f'Perturbation on {pert_target[ss]}')
        axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')  # Place legend outside
        axs[0].set_ylim(bottom=-0.6,top=0.6)
    
        axs[1].plot(pred_Aall[ss],color='#ff7f0e',alpha=0.1)
        axs[1].plot(pred_Ball[ss],color='#1f77b4',alpha=0.1)
        axs[1].plot(predavg_A[ss],color='#ff7f0e',label='RNN A')
        axs[1].plot(predavg_B[ss],color='#1f77b4',label='RNN B')
        #axs[1,ss].plot(predavg_C[ss],color='#2ca02c',label='RNN A&B')
        
        axs[1].set_title(f'Decoded results')
        axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')  # Place legend outside
        axs[1].set_yticks(Class_per_sec*np.array([0,6,12,18]))
        axs[1].set_yticklabels([0,6,12,18])
        xticks = axs[1].get_xticks()
        scaled_xticks = np.round(xticks * 0.01)
        # Set new tick labels
        axs[1].set_xticklabels(scaled_xticks)
        
        plt.suptitle(f'{k}:Connection Probability {conProbability[t]}')
        #print(f"Loop {t+1} out of {len(tind)}")
        # Adjusting layout
        plt.tight_layout()
        if save:
            #plt.savefig(os.path.join(savepath,f"Perturb_decode_{time_1}_{time_2}_{order}_{option}"),transparent=True,dpi=400)
            plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
            plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
            
            # Your plotting code
            #plt.savefig(f"Perturb_decode_{time_1}_{time_2}_{order}_{option}_conprob_{conProbability[t]}.svg", format='svg')
        plt.show()
        
        
        
        # show confusion matrix
        from Confmatrix import confmat, confscore
        #pred_Aall_2=np.reshape(pred_Aall[0],(-1,1))
        pred_Aall_2 = pred_Aall[0].flatten(order='F').reshape(-1, 1)
        #pred_Ball_2=np.reshape(pred_Ball[0],(-1,1))
        pred_Ball_2 = pred_Ball[0].flatten(order='F').reshape(-1, 1)
        class_all=np.tile(class_all,np.shape(pred_Aall)[-1])
        class_all=np.reshape(class_all,(-1,1))
        confmat_A=confmat(class_all,pred_Aall_2)
        confmat_B=confmat(class_all,pred_Ball_2)
    
        
        plt.figure()
        fig, axs = plt.subplots(1,2,figsize=(10, 6))
        im0=axs[0].imshow(confmat_A,aspect='auto', cmap=parula, interpolation='none',vmin=0,vmax=400)
        im1=axs[1].imshow(confmat_B,aspect='auto', cmap=parula, interpolation='none',vmin=0,vmax=400)
        axs[0].set_box_aspect(1)
        axs[1].set_box_aspect(1)
        #axs[1].colorbar()
        #cbar = fig.colorbar(im0, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
        cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
        plt.suptitle(f'Connection Probability {conProbability[t]}')
        plt.show()
        
        
        # show temporal error and categorical error
        # get offset from the actual time
        from get_phase import get_phase
        diff_A=get_phase(pred_Aall_2-class_all,class_per_trial,'int')
        diff_B=get_phase(pred_Ball_2-class_all,class_per_trial,'int')
        
        #create 2d histogram
        offset_mat, xedges, yedges = np.histogram2d(diff_A.ravel(), diff_B.ravel(), bins=class_per_trial)
        
        plt.figure(figsize=(10, 6))
        
        # Set extent based on the bin edges for proper axis scaling
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        allpop = np.sum(offset_mat)
        # Use imshow to plot the data
        im0 = plt.imshow((offset_mat/allpop).T, aspect='auto', cmap=parula, interpolation='none', vmin=0, vmax=250/allpop, extent=extent, origin='lower')
        # Set the aspect of the axis to be equal
        plt.gca().set_aspect('equal')
        # Add color bar
        cbar = plt.colorbar(im0, orientation='vertical', fraction=0.02, pad=0.04)
        # Set axis labels
        plt.xlabel('A offset')
        plt.ylabel('B offset')
        # Set the title
        plt.title(f'{k}:Connection Probability {conProbability[t]}')
        # Show the plot
        plt.show()
      
        
        
        #show categorical error
        from error_analysis import get_cat_error, get_temp_error
        group_bound=int(class_per_trial*(min_dur/(min_dur+max_dur)))-1
        cat_error_rate=get_cat_error(pred_Aall_2, class_all, pred_Ball_2, class_all, group_bound)
    
        catlabel = np.array(["M2 mis, PPC mis", "M2 mis, PPC cor", "M2 cor, PPC mis", "M2 cor, PPC cor"])
        plt.figure()
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        axs[0].bar(catlabel,cat_error_rate)
        
        # show temporal error
        temp_error_rate=get_temp_error(pred_Aall_2, class_all, pred_Ball_2, class_all, class_per_trial,3)
        offset_temp=np.arange(-(np.ceil(class_per_trial/2)-1),np.floor(class_per_trial/2)+1)
        axs[1].plot(offset_temp,temp_error_rate)
        plt.suptitle(f'{k}:Connection Probability {conProbability[t]}')
        plt.show()
        #print(f"{k}: loop {loopind} out of {len(tind)}")
        
        confA_list.append(confmat_A)
        confB_list.append(confmat_B)
        offset_list.append(offset_mat)
        terror_list.append(temp_error_rate)
        caterr_list.append(cat_error_rate)
        
        print(f"[t={t}, k={k}] loopind={loopind}")
        
    confA_ave    = np.mean(confA_list,   axis=0)
    confB_ave    = np.mean(confB_list,   axis=0)
    offset_ave   = np.mean(offset_list,  axis=0)
    temp_err_ave = np.mean(terror_list,  axis=0)
    cat_err_ave  = np.mean(caterr_list,  axis=0)
    pred_diff_sub= np.mean(pred_diff2,  axis=0)
        
    Allinfo['t_index'].append(t)
    Allinfo['Confmat_A_ave'].append(confA_ave)
    Allinfo['Confmat_B_ave'].append(confB_ave)
    Allinfo['Offset_mat_ave'].append(offset_ave)
    Allinfo['temp_error_ave'].append(temp_err_ave)
    Allinfo['cat_error_ave'].append(cat_err_ave)
    Allinfo['pred_diff_sub'].append(cat_err_ave)
    
        
vmax=0.012
vmin=0.001
vmax_neu=0.01
vmin_neu=0.002
# plot results
for t in tind:
    offset_mat=Allinfo['Offset_mat_ave'][t]
    # plot offset matrix
    plt.figure(figsize=(10, 6))
    
    # Set extent based on the bin edges for proper axis scaling
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    allpop = np.sum(offset_mat)
    # Use imshow to plot the data
    im0 = plt.imshow((offset_mat/allpop).T, aspect='auto', cmap=parula, interpolation='none', vmin=vmin, vmax=vmax, extent=extent, origin='lower')
    # Set the aspect of the axis to be equal
    plt.gca().set_aspect('equal')
    # Add color bar
    cbar = plt.colorbar(im0, orientation='vertical', fraction=0.02, pad=0.04)
    # Set axis labels
    plt.xlabel('A offset')
    plt.ylabel('B offset')
    # Set the title
    plt.title(f'Connection Probability {conProbability[t]}')
    # Show the plot
    plt.show()



    catlabel = np.array(["M2 mis, PPC mis", "M2 mis, PPC cor", "M2 cor, PPC mis", "M2 cor, PPC cor"])
    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].bar(catlabel, Allinfo['cat_error_ave'][t])
    
    # show temporal error
    axs[1].plot(offset_temp,Allinfo['temp_error_ave'][t])
    plt.suptitle(f'Connection Probability {conProbability[t]}')
    plt.show()


# load neural data
import scipy.io
# Load the .mat file
data = scipy.io.loadmat(r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\t4stateful\alldata2.mat")
#save('alldata2.mat','caterror','caterror_avg','errorand_rate','errorand_avg','population2','predall1','predall2','classall1','classall2','confmatrix1','confmatrix2')


confmat_diff_M2=np.zeros((len(tind),2))
confmat_diff_PPC=np.zeros((len(tind),2))
offset_diff=np.zeros(len(tind))
cat_diff=np.zeros(len(tind))
temp_diff=np.zeros(len(tind))
off_norm=data['population2']/np.sum(data['population2'])

leng1=np.shape(off_norm)[0]
mid_ind=int(np.ceil(leng1/2)-1)
marg=0
good_indx=np.round(np.concatenate((np.arange(0,mid_ind-marg),np.arange(mid_ind+marg+1,leng1)),axis=0))
good_indy=np.round(np.concatenate((np.arange(0,mid_ind-marg+1),np.arange(mid_ind+marg+2,leng1)),axis=0))

off_norm=data['population2'][np.ix_(good_indy,good_indx)].astype(np.float64)
off_norm/=np.sum(off_norm)



off_norm=np.array(data['population2'])
off_norm=off_norm/np.sum(off_norm)
off_norm=np.flip(off_norm,0)

#plot neural data
plt.figure(figsize=(10, 6))
im0 = plt.imshow(off_norm.T, aspect='auto', cmap=parula, interpolation='none', origin='lower',vmin=vmin_neu, vmax=vmax_neu)
plt.gca().set_aspect('equal')
plt.title(f'Neural data')
# Add color bar
cbar = plt.colorbar(im0, orientation='vertical', fraction=0.02, pad=0.04)


from scipy.special import rel_entr

for i in range(len(tind)):
    confmat_diff_M2[i,0]=np.linalg.norm(Allinfo['Confmat_A_ave'][i]-data['confmatrix1'])
    confmat_diff_M2[i,1]=np.linalg.norm(Allinfo['Confmat_B_ave'][i]-data['confmatrix1'])
    confmat_diff_PPC[i,0]=np.linalg.norm(Allinfo['Confmat_A_ave'][i]-data['confmatrix2'])
    confmat_diff_PPC[i,1]=np.linalg.norm(Allinfo['Confmat_B_ave'][i]-data['confmatrix2'])
    
    # normalize two offset matrices
    RNN_offset=Allinfo['Offset_mat_ave'][i]
    #RNN_offset=RNN_offset[np.ix_(good_indy,good_indx)]
    RNN_offset/=np.sum(RNN_offset)
    offset_diff[i]=np.linalg.norm(RNN_offset-off_norm)#+np.linalg.norm(Allinfo['Offset_mat'][:,:,i]-np.transpose(data['population2']))
    #offset_diff[i]=sum(rel_entr(RNN_offset.flatten(), off_norm.flatten()))
    cat_diff[i]=np.linalg.norm(Allinfo['cat_error_ave'][i]-data['caterror_avg'])
    temp_diff[i]=np.linalg.norm(Allinfo['temp_error_ave'][i]-data['errorand_avg'])
    
npprob=np. array(conProbability)
x=npprob[np.round(tind)]

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8))

# Plot for RNN_offsets
axs[0].plot(x, offset_diff)
axs[0].set_title('Decoding offset')
axs[0].set_xscale('log')
axs[0].tick_params(axis='x', labelbottom=False)  # Remove x-axis labels

# Plot for cat_diff
axs[1].plot(x, cat_diff)
axs[1].set_title('Categorical errors')
axs[1].set_xscale('log')
axs[1].tick_params(axis='x', labelbottom=False)  # Remove x-axis labels

# Plot for temp_diff
axs[2].plot(x, temp_diff)
axs[2].set_title('Temporal errors')
axs[2].set_xscale('log')

# Add labels and show the plot
plt.xlabel('Con prob (log scale)')
plt.tight_layout()  # Adjust subplots to fit into figure
plt.show()
   
fig,axs=plt.subplots(1,1)
axs.plot(x,np.array(Allinfo['pred_diff_sub']))
axs.set_xscale('log')
axs.set_title('Decoded difference over connection probability')
    

# plot errors using colormap
import matplotlib.cm as cm
catlabel = np.array(["M2 mis, PPC mis", "M2 mis, PPC cor", "M2 cor, PPC mis", "M2 cor, PPC cor"])
plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(10, 6))

# Generate colormap based on t
colors = cm.viridis(np.linspace(0, 1, len(conProbability)))
# Iterate over t
for t in range(len(conProbability)):

    color = colors[t]

    # Plot the categorical data as a line plot
    axs[0].plot(catlabel, Allinfo['cat_error_ave'][t], marker='o', color=color, label=f'Con Prob {conProbability[t]:.2f}')
    axs[0].set_title('Categorical Error')
    axs[0].legend()

    # Plot temporal error data
    axs[1].plot(offset_temp, Allinfo['temp_error_ave'][t], color=color, label=f'Con Prob {conProbability[t]:.2f}')
    axs[1].set_title('Temporal Error')
    axs[1].legend()

# Add additional data to each subplot in red
axs[0].plot(catlabel, np.squeeze(data['caterror_avg']) , marker='o', color='red', label='Neural Data')
axs[1].plot(offset_temp, np.squeeze(data['errorand_avg']) , color='red', label='Neural Data')

# Update legends to include "Neural Data"
axs[0].legend()
axs[1].legend()

plt.suptitle(f'Connection Probability {conProbability[t]:.2f}')
plt.tight_layout()
plt.show()


import json
Allinfo['pert_prob']=pert_prob
Allinfo['pert_noisesd']=pert_noisesd
Allinfo['xedges']=xedges
Allinfo['yedges']=yedges
Allinfo['offset_temp']=offset_temp
Allinfo['tind']=tind
#%%

# Convert dictionary to Python-native types
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
json_file_path = os.path.join(analysis_folder, f"Allinfo_pert_noisesd_{pert_noisesd}_pert_prob_{pert_prob}.json")
with open(json_file_path, 'w') as json_file:
    json.dump(Allinfo2, json_file)



#%% load data from json file and display graphs
import json
def load_json_to_variables(filename):
    with open(filename, "r") as file:
        data = json.load(file)  # Load JSON data as a Python dictionary

    # Convert dictionary keys into variables
    for key, value in data.items():
        globals()[key] = value  # Create variables with the same names as dictionary keys

    return data  # Return the loaded dictionary in case it's needed

# Usage
filename=r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\Analysis_folder\Allinfo_pert_noisesd_0.8_pert_prob_0.01.json"
Allinfo = load_json_to_variables(filename)


#%% make decoding classifier and display the neural trajectory
tind=np.array([0,3,4,5,10])
from Confmatrix import confscore, confmat
import matplotlib.cm as cm
def plot_matrix_3d(matrix, ax, k, m):
    hsv = cm.get_cmap('hsv', m)  # Colormap with length m
    for i in range(k):
        # Extract segment (m, 3) from matrix
        segment = matrix[i*m:(i+1)*m, :3]
        # Generate colors for each point in this segment
        colors = hsv(np.linspace(0, 1, m))
        # Scatter the points in 3D space with assigned colors
        ax.scatter(segment[:, 0], segment[:, 1], segment[:, 2], c=colors, marker='o')

    # Set labels
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    return

def plot_matrix_2d(matrix, ax, k, m):
    hsv = cm.get_cmap('hsv', m)  # Colormap with length m
    for i in range(k):
        # Extract segment (m, 3) from matrix
        segment = matrix[i*m:(i+1)*m, :3]
        # Generate colors for each point in this segment
        colors = hsv(np.linspace(0, 1, m))
        # Scatter the points in 3D space with assigned colors
        ax.scatter(segment[:, 0], segment[:, 1], c=colors, marker='o')

    # Set labels
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    return

def z_score_with_zero_handling(A,dim=0):
    # Calculate mean and standard deviation along the first dimension (row-wise)
    mean_A = np.mean(A, axis=dim)
    std_A = np.std(A, axis=dim)
    # Avoid division by zero: if std_A is zero, set it to 1 to prevent invalid division
    std_A[std_A == 0] = 1
    # Calculate z-score
    z_scores = (A - mean_A) / std_A
    return z_scores

zscore_option=True
tind=np.arange(len(conProbability))
#tind=np.arange(12)
loopind=0
tind=np.array([0])
tind=np.array([0,3,4,5,10])
for t in tind:
#for t in aa:

    loopind+=1
    maxval=weight_max[t]
    con_prob=conProbability[t]

    #build models
    model=build_model(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha, seed1=seed1)
    # load weights
    checkpoint_filepath=os.path.join(foldername[t], f"epoch_{model_index[t]:05d}.ckpt")
    model.load_weights(checkpoint_filepath)

    # create classifier for decoding
    outputs = [layer.output for layer in model.layers[1:]]  # Exclude the input layer
    activity_model = Model(inputs=model.input, outputs=outputs)
    trial_num=8
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
    if zscore_option:
        act_avg_A=z_score_with_zero_handling(act_avg_A)
        act_avg_B=z_score_with_zero_handling(act_avg_B)
        act_stack_A=z_score_with_zero_handling(act_stack_A)
        act_stack_B=z_score_with_zero_handling(act_stack_B)


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
    Dim=3
    #for RNN A
    clf_A=RandomForestClassifier(n_estimators=100,bootstrap=True,n_jobs=-1)
    clf_A.fit(proj_A_train[:,:Dim],class_A_train)
    #for RNN B
    clf_B=RandomForestClassifier(n_estimators=100,bootstrap=True,n_jobs=-1)
    clf_B.fit(proj_B_train[:,:Dim],class_B_train)
    #combine both RNN A and B
    clf_C=RandomForestClassifier(n_estimators=100,bootstrap=True,n_jobs=-1)
    clf_C.fit(proj_C_train[:,:Dim],class_A_train)    
    
    
    # test the classifier and make confusion matrices
    # create testing data
    x, y, In_ons=makeInOut(1,10,inputdur,nInput,min_dur,max_dur,dt)
    #xnew=sum(xnew,axis=2)
    output_and_activities = activity_model.predict(x)
    activities_A = output_and_activities[0]  # Activities of all intermediate layers
    activities_B=output_and_activities[1]
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
    class_A_test=np.tile(class_A,(trial_rep_A))
    trial_rep_B=int(np.shape(act_stack_B)[0]/(min_dur+max_dur))
    class_B_test=np.tile(class_A,(trial_rep_B))
    
    if zscore_option:
        act_avg_A=z_score_with_zero_handling(act_avg_A)
        act_avg_B=z_score_with_zero_handling(act_avg_B)
        act_stack_A=z_score_with_zero_handling(act_stack_A)
        act_stack_B=z_score_with_zero_handling(act_stack_B)
        
    # reduce dimension of the testing data
    proj_A_test=pca_A.transform(act_stack_A)
    proj_B_test=pca_B.transform(act_stack_B)
    act_avg_C=np.concatenate((act_avg_A,act_avg_B),axis=1)
    act_stack_C=np.concatenate((act_stack_A,act_stack_B),axis=1)
    pca_C = PCA()
    pca_C.fit(act_avg_C)
    proj_C_test=pca_C.transform(act_stack_C)   
    
    # predict on testing data
    pred_A=clf_A.predict(proj_A_test[:,:Dim])
    pred_B=clf_B.predict(proj_B_test[:,:Dim])
    pred_C=clf_C.predict(proj_C_test[:,:Dim])

    # plot confusion matrix
    confmat_A=confmat(class_A_test,pred_A)
    confmat_B=confmat(class_B_test,pred_B)
    plt.figure()
    fig, axs = plt.subplots(1,2,figsize=(10, 6))
    im0=axs[0].imshow(confmat_A,aspect='auto', cmap=parula, interpolation='none',vmin=0,vmax=400)
    im1=axs[1].imshow(confmat_B,aspect='auto', cmap=parula, interpolation='none',vmin=0,vmax=400)
    axs[0].set_box_aspect(1)
    axs[1].set_box_aspect(1)
    #axs[1].colorbar()
    #cbar = fig.colorbar(im0, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    plt.suptitle(f'Connection Probability {conProbability[t]}')
    plt.show()
    
    
    # plot neural trajectory
    k=int(np.shape(pred_A)[0]/(min_dur+max_dur))# number of trials
    m=min_dur+max_dur
    
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    # Plot matrix A and B
    plot_matrix_3d(proj_A_test, ax1, k, m)
    ax1.set_title('RNN A')
    plot_matrix_3d(proj_B_test, ax2, k, m)
    ax2.set_title('RNN B')
    fig.suptitle(f'conProb {conProbability[t]}')
    plt.tight_layout()
    plt.show()
    
    # plot variance explained
    fig, axes = plt.subplots(1,2)
    axes[0].bar(np.arange(10),np.cumsum(pca_A.explained_variance_ratio_[:10]))
    axes[0].set_xlabel('Principal components')
    axes[0].set_ylabel('Cumulative variance explained')   

    axes[1].bar(np.arange(10),np.cumsum(pca_B.explained_variance_ratio_[:10]))
    axes[1].set_xlabel('Principal components')
    axes[1].set_ylabel('Cumulative variance explained') 
    
 
fig, axs = plt.subplots(2,len(conProbability))
for t in range(len(conProbability)):
#for t in aa:

    loopind+=1
    maxval=weight_max[t]
    con_prob=conProbability[t]

    #build models
    model=build_model(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha, seed1=seed1)
    # load weights
    checkpoint_filepath=os.path.join(foldername[t], f"epoch_{model_index[t]:05d}.ckpt")
    model.load_weights(checkpoint_filepath)

    # create classifier for decoding
    outputs = [layer.output for layer in model.layers[1:]]  # Exclude the input layer
    activity_model = Model(inputs=model.input, outputs=outputs)
    trial_num=8
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
    


        
    # reduce dimensions with pca
    if zscore_option:
        act_avg_A=z_score_with_zero_handling(act_avg_A)
        act_avg_B=z_score_with_zero_handling(act_avg_B)
        act_stack_A=z_score_with_zero_handling(act_stack_A)
        act_stack_B=z_score_with_zero_handling(act_stack_B)


    pca_A = PCA()
    pca_A.fit(act_avg_A)
    proj_A_train=pca_A.transform(act_stack_A)
    pca_B = PCA()
    pca_B.fit(act_avg_B)
    proj_B_train=pca_B.transform(act_stack_B)
     
    
    
    # test the classifier and make confusion matrices
    # create testing data
    x, y, In_ons=makeInOut(1,10,inputdur,nInput,min_dur,max_dur,dt)
    #xnew=sum(xnew,axis=2)
    output_and_activities = activity_model.predict(x)
    activities_A = output_and_activities[0]  # Activities of all intermediate layers
    activities_B=output_and_activities[1]
    act_avg_A=avgAct(activities_A,In_ons,min_dur,max_dur)
    act_avg_B=avgAct(activities_B,In_ons,min_dur,max_dur)
    act_stack_A=Act_2dsort(activities_A,In_ons,min_dur,max_dur)#(time,nUnit) time is multiple of mindur+maxdur
    act_stack_B=Act_2dsort(activities_B,In_ons,min_dur,max_dur)

    
    if zscore_option:
        act_avg_A=z_score_with_zero_handling(act_avg_A)
        act_avg_B=z_score_with_zero_handling(act_avg_B)
        act_stack_A=z_score_with_zero_handling(act_stack_A)
        act_stack_B=z_score_with_zero_handling(act_stack_B)
        
    # reduce dimension of the testing data
    proj_A_test=pca_A.transform(act_stack_A)
    proj_B_test=pca_B.transform(act_stack_B)

  
    

    
    
    # plot neural trajectory
    k=int(np.shape(proj_A_test)[0]/(min_dur+max_dur))# number of trials
    m=min_dur+max_dur
    # Plot matrix A and B
    plot_matrix_2d(proj_A_test, axs[0,t], k, m)
    plot_matrix_2d(proj_B_test, axs[1,t], k, m)
    axs[0,t].set_title(f'conProb {conProbability[t]}')

axs[0,0].set_ylabel('RNN A')    
axs[1,0].set_ylabel('RNN B')    
plt.tight_layout()
plt.show()
#%% decoding option 2: ridge rigression
from sklearn.linear_model import Ridge
Dim=50
PCA_option=False
Lin_reduc='pca'
w=0.8
x, y, In_ons=makeInOut(1,10,inputdur,nInput,min_dur,max_dur,dt)
#xnew=sum(xnew,axis=2)
output_and_activities = activity_model.predict(x)
activities_A = output_and_activities[0]  # Activities of all intermediate layers
activities_B=output_and_activities[1]
act_avg_A=avgAct(activities_A,In_ons,min_dur,max_dur)
act_avg_B=avgAct(activities_B,In_ons,min_dur,max_dur)
act_stack_A=Act_2dsort(activities_A,In_ons,min_dur,max_dur)#(time,nUnit) time is multiple of mindur+maxdur
act_stack_B=Act_2dsort(activities_B,In_ons,min_dur,max_dur)
if zscore_option:
    act_avg_A=z_score_with_zero_handling(act_avg_A)
    act_avg_B=z_score_with_zero_handling(act_avg_B)
    act_stack_A=z_score_with_zero_handling(act_stack_A)
    act_stack_B=z_score_with_zero_handling(act_stack_B)
    
# perform PCA
if Lin_reduc.lower()=='pca':
    pca_A = PCA()
    pca_A.fit(act_avg_A)
    proj_A_train=pca_A.transform(act_stack_A)
    pca_B = PCA()
    pca_B.fit(act_avg_B)
    proj_B_train=pca_B.transform(act_stack_B)
elif Lin_reduc.lower()=='nfda':
    from nFDA import nFDA
    N=min_dur+max_dur
    Coeff_A, Score_A,_,_,_,_,_=nFDA(act_stack_A.reshape(N,nUnit,-1),'nFDA',w=0.8)
    Coeff_B, Score_B,_,_,_,_,_=nFDA(act_stack_B.reshape(N,nUnit,-1),'nFDA',w=0.8)
    Score_A=np.matmul(act_stack_A,Coeff_A)
    Score_B=np.matmul(act_stack_B,Coeff_B)


# make outputs
Out1=np.cos(2*np.pi*np.arange(np.shape(act_stack_A)[0])/(min_dur+max_dur))
Out2=np.sin(2*np.pi*np.arange(np.shape(act_stack_A)[0])/(min_dur+max_dur))
Out_all=np.column_stack((Out1, Out2))
alpha=100
A_ridge=Ridge(alpha=alpha)
B_ridge=Ridge(alpha=alpha)

if Lin_reduc.lower()=='pca':
    A_ridge.fit(proj_A_train[:,:Dim],Out_all)
    B_ridge.fit(proj_B_train[:,:Dim],Out_all)
elif Lin_reduc.lower()=='nfda':
    A_ridge.fit(Score_A[:,:Dim],Out_all)
    B_ridge.fit(Score_B[:,:Dim],Out_all)   
else:
    A_ridge.fit(act_stack_A,Out_all)
    B_ridge.fit(act_stack_B,Out_all)    

x, y, In_ons=makeInOut(1,10,inputdur,nInput,min_dur,max_dur,dt)
#xnew=sum(xnew,axis=2)
output_and_activities = activity_model.predict(x)
activities_A = output_and_activities[0]  # Activities of all intermediate layers
activities_B=output_and_activities[1]
act_avg_A=avgAct(activities_A,In_ons,min_dur,max_dur)
act_avg_B=avgAct(activities_B,In_ons,min_dur,max_dur)
act_stack_A=Act_2dsort(activities_A,In_ons,min_dur,max_dur)#(time,nUnit) time is multiple of mindur+maxdur
act_stack_B=Act_2dsort(activities_B,In_ons,min_dur,max_dur)
if zscore_option:
    act_avg_A=z_score_with_zero_handling(act_avg_A)
    act_avg_B=z_score_with_zero_handling(act_avg_B)
    act_stack_A=z_score_with_zero_handling(act_stack_A)
    act_stack_B=z_score_with_zero_handling(act_stack_B)

# perform PCA
if Lin_reduc.lower()=='pca':
    proj_A_train=pca_A.transform(act_stack_A)
    proj_B_train=pca_B.transform(act_stack_B)
    pred_A=A_ridge.predict(proj_A_train[:,:Dim])
    pred_B=B_ridge.predict(proj_B_train[:,:Dim])
elif Lin_reduc.lower()=='nfda':
    proj_A_train=np.matmul(act_stack_A,Coeff_A)
    proj_B_train=np.matmul(act_stack_B,Coeff_B)
    pred_A=A_ridge.predict(proj_A_train[:,:Dim])
    pred_B=B_ridge.predict(proj_B_train[:,:Dim])    
# test regression
else:
    pred_A=A_ridge.predict(act_stack_A)
    pred_B=B_ridge.predict(act_stack_B)



"""#plot prediction
fig, axes = plt.subplots(1,2)
axes[0].plot(pred_A[:,0],label='Pred 0')
axes[0].plot(pred_A[:,1],label='Pred 1')
axes[0].plot(Out1,label='Class 0')
axes[0].plot(Out2,label='Class 1')
axes[0].set_xlabel('Time')
  

axes[1].plot(pred_B[:,0],label='Pred 0')
axes[1].plot(pred_B[:,1],label='Pred 1')
axes[1].plot(Out1,label='Class 0')
axes[1].plot(Out2,label='Class 1')
axes[1].set_xlabel('Time')"""

"""#plot prediction on circular map
AB_decode_difffig, axes = plt.subplots(1,2)
# Plot for pred_A
axes[0].plot(pred_A[:, 0], pred_A[:, 1], label='Predictions')
axes[0].plot(Out1, Out2, label='Class')
axes[0].scatter(pred_A[600::1800, 0], pred_A[600::1800, 1], color='red', s=100, label='Reward')
axes[0].scatter(pred_A[1800::1800, 0], pred_A[1800::1800, 1], color='red', s=100, label='')
axes[0].set_aspect('equal', 'box')
axes[0].legend()
axes[0].set_title('Pred_A with Red Dot')

# Plot for pred_B
axes[1].plot(pred_B[:, 0], pred_B[:, 1], label='Pred 0')
axes[1].plot(Out1, Out2, label='Class 0')
axes[1].scatter(pred_B[600::1800, 0], pred_B[600::1800, 1], color='red', s=100, label='Reward')
axes[1].scatter(pred_B[1800::1800, 0], pred_B[1800::1800, 1], color='red', s=100, label='')
axes[1].set_aspect('equal', 'box')
axes[1].legend()
axes[1].set_title('Pred_B with Red Dot')

plt.tight_layout()
plt.show()"""


AB_colorfig, axes_color = plt.subplots(1, 2, figsize=(12, 6))

# For pred_A with HSV colormap
n_A = len(pred_A)
indices_A = np.arange(n_A)
phase_A = (indices_A % 1800) / 1800  # Normalize to [0, 1]
scatter_A = axes_color[0].scatter(pred_A[:, 0], pred_A[:, 1], c=phase_A, cmap='hsv', s=5)
axes_color[0].scatter(pred_A[600::1800, 0], pred_A[600::1800, 1], color='black', s=100, label='Reward')
axes_color[0].scatter(pred_A[1800::1800, 0], pred_A[1800::1800, 1], color='black', s=100, label='')
axes_color[0].set_aspect('equal', 'box')
axes_color[0].set_title('Pred_A with HSV Colormap')
plt.colorbar(scatter_A, ax=axes_color[0], label='Cycle Phase')

# For pred_B with HSV colormap
n_B = len(pred_B)
indices_B = np.arange(n_B)
phase_B = (indices_B % 1800) / 1800  # Normalize to [0, 1]

scatter_B = axes_color[1].scatter(pred_B[:, 0], pred_B[:, 1], c=phase_B, cmap='hsv', s=5)
axes_color[1].scatter(pred_B[600::1800, 0], pred_B[600::1800, 1], color='black', s=100, label='Reward')
axes_color[1].scatter(pred_B[1800::1800, 0], pred_B[1800::1800, 1], color='black', s=100, label='')
axes_color[1].set_aspect('equal', 'box')
axes_color[1].set_title('Pred_B with HSV Colormap')
plt.colorbar(scatter_B, ax=axes_color[1], label='Cycle Phase')
plt.tight_layout()
plt.show()



# plot distribution of weights
fig, axes = plt.subplots(2,1,sharex=True)
axes[0].hist(A_ridge.coef_.flatten(),bins=100)
axes[1].hist(B_ridge.coef_.flatten(),bins=100)

#%% load weights and make it more sparse, and perturb to see the synchronicity
save=False
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


# Example usage:
# Wr_A is your original sparse matrix (CSR format)
# m is the desired number of non-zero elements

# Wr_A_reduced = reduce_nonzero_elements(Wr_A, m)

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


nonzero_num_prob=np.array([1,0.1,0.03,0.01,0.003,0.001,0.0003,0.0001,0.00003,0.00001,1e-10])
nonzero_num=np.round(nUnit*nUnit*nonzero_num_prob).astype(np.int32)  

pred_diff_cat=[]
pred_diff_nan=np.empty((len(nonzero_num),len(conProbability)))
pred_diff_nan[:]=np.nan
for ss in range(len(conProbability)):

    modelind=ss
    conprobmodel=conProbability[modelind]
    pred_diff_all=np.zeros(len(nonzero_num))
    for t in range(len(nonzero_num)):# the last index is for case when there is no change added to S_A and S_B
            #  analyze weight distribution
        
        max_range=[0,max_dur+min_dur] # time range to choose max firing time
        nExc=nUnit-nInh #number of excitory units
        
        
        model=build_model(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha, seed1=seed1)
        # load weights
        checkpoint_filepath=os.path.join(foldername[modelind], f"epoch_{model_index[modelind]:05d}.ckpt")
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
        if nonzero_num_prob[t]!=conProbability[modelind]:
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
        fig, axs = plt.subplots(1, 2, figsize=(12,12))
        axs[0].imshow(act_avg_A_sort.T, aspect='auto', cmap=parula,interpolation='none', vmin=0, vmax=1)
        axs[0].set_title('A ')
        axs[1].imshow(act_avg_B_sort.T, aspect='auto', cmap=parula,interpolation='none', vmin=0, vmax=1)
        axs[1].set_title('B ')  
        plt.show()

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
        pred_diff_all[t]=np.linalg.norm(pred_diff)
        
    
    x=nonzero_num
    fig,axs=plt.subplots(1,1)
    axs.plot(x,pred_diff_all)
    axs.set_xscale('log')
    axs.set_title('Decoded difference over connection probability')
    axs.vlines(conprobmodel*nUnit*nUnit,0,np.max(pred_diff_all),'r')
    
    pred_diff_cat.append(pred_diff_all)
    
    cutind=np.argwhere(nonzero_num_prob==conProbability[modelind])
    # 
    pred_diff_nan[cutind[0][0]:,ss]=pred_diff_all[cutind[0][0]:]
    

pred_diff_cat=np.array(pred_diff_cat)
pred_diff_avg=np.mean(pred_diff_cat,0)
fig,axs=plt.subplots(1,1)
axs.plot(x,pred_diff_avg,'r')
axs.plot(x,pred_diff_cat.T,'b',alpha=0.1)
axs.set_xscale('log')

# take mean while omitting nans
meanall=np.nanmean(pred_diff_nan,1)
fig,axs=plt.subplots(1,1)
axs.plot(x,meanall,'r')
axs.plot(x,pred_diff_nan,'b',alpha=0.1)
axs.set_xscale('log')


#%% remove a portion of inter rnn connections and study their synchronicity
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

data = np.load(r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\Analysis_folder\012_pertdata.npz")
confavg_A_all = np.ndarray.tolist(data['confavg_A_all'])
confavg_B_all = np.ndarray.tolist(data['confavg_B_all'])
loaded_pred_diff_all2 = np.ndarray.tolist(data['pred_diff_all2'])


kloop=np.arange(3,np.shape(best_models)[0])
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
                if k==0:
                    fig, axs = plt.subplots(1, 2, figsize=(12,12))
                    axs[0].imshow(act_avg_A_sort.T, aspect='auto', cmap=parula,interpolation='none', vmin=0, vmax=1)
                    axs[0].set_title(f'A Con_prob={conProbability[ss]} ')
                    axs[1].imshow(act_avg_B_sort.T, aspect='auto', cmap=parula,interpolation='none', vmin=0, vmax=1)
                    axs[1].set_title(f'B Ratio={nonzero_num_ratio[t]}')  
                    plt.show()
        
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


    file_name = f"deletepert_k_{k_ind}.npz"
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


import matplotlib.cm as cm
# Create a colormap
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(conProbability)))

# Create the plot
fig, ax = plt.subplots()

for idx, cp in enumerate(conProbability):
    color = colors[idx]
    ax.plot(nonzero_num_ratio, pred_diff_all[:, idx], color=color)

# Create a ScalarMappable for the colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(conProbability), vmax=max(conProbability)))
sm.set_array([])  # Dummy array for the ScalarMappable

# Add the colorbar
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('conProbability')

# Set the ticks and labels for the colorbar
tick_locs = np.linspace(min(conProbability), max(conProbability), len(conProbability))
cbar.set_ticks(tick_locs)
cbar.set_ticklabels([f'{cp:.2e}' for cp in conProbability])

# Set labels and title
ax.set_xlabel('Nonzero con. ratio')
ax.set_ylabel('Prediction difference')
ax.set_title('Plot of pred_diff_all vs nonzero_num_ratio')

#save figure
if save:
    #plt.savefig(os.path.join(savepath,f"Perturb_decode_{time_1}_{time_2}_{order}_{option}"),transparent=True,dpi=400)
    plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    
    # Your plotting code
    plt.savefig(f"Delete_interRNN_portion{pert_noisesd}_{pert_prob}.svg", format='svg')
    plt.savefig(f"Delete_interRNN_portion{pert_noisesd}_{pert_prob}.png", transparent=True,dpi=500)
plt.show()




# plot neutral conditions (no conditions omitted)
fig, ax = plt.subplots()
ax.plot(conProbability,pred_diff_all[0,:])
ax.set_xscale('log')
ax.set_xlabel('Connection probability')
ax.set_ylabel('Prediction difference')
if save:
    #plt.savefig(os.path.join(savepath,f"Perturb_decode_{time_1}_{time_2}_{order}_{option}"),transparent=True,dpi=400)
    plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    
    # Your plotting code
    plt.savefig(f"pred_diff_over_prob{pert_noisesd}_{pert_prob}.svg", format='svg')
    plt.savefig(f"pred_diff_over_prob{pert_noisesd}_{pert_prob}.png", transparent=True,dpi=500)
plt.show()



# plot decoding accuracy A
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(conProbability)))

# Create the plot
fig, ax = plt.subplots()

for idx, cp in enumerate(conProbability):
    color = colors[idx]
    ax.plot(nonzero_num_ratio, confavg_A[:, idx], color=color)

# Create a ScalarMappable for the colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(conProbability), vmax=max(conProbability)))
sm.set_array([])  # Dummy array for the ScalarMappable

# Add the colorbar
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('conProbability')

# Set the ticks and labels for the colorbar
tick_locs = np.linspace(min(conProbability), max(conProbability), len(conProbability))
cbar.set_ticks(tick_locs)
cbar.set_ticklabels([f'{cp:.2e}' for cp in conProbability])

# Set labels and title
ax.set_xlabel('Nonzero con. ratio')
ax.set_ylabel('Decoding accuracy')
ax.set_title('A: decoding accuracy')



# plot decoding accuracy B
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(conProbability)))

# Create the plot
fig, ax = plt.subplots()

for idx, cp in enumerate(conProbability):
    color = colors[idx]
    ax.plot(nonzero_num_ratio, confavg_B[:, idx], color=color)

# Create a ScalarMappable for the colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(conProbability), vmax=max(conProbability)))
sm.set_array([])  # Dummy array for the ScalarMappable
    
# Add the colorbar
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('conProbability')

# Set the ticks and labels for the colorbar
tick_locs = np.linspace(min(conProbability), max(conProbability), len(conProbability))
cbar.set_ticks(tick_locs)
cbar.set_ticklabels([f'{cp:.2e}' for cp in conProbability])

# Set labels and title
ax.set_xlabel('Nonzero con. ratio')
ax.set_ylabel('Decoding accuracy')
ax.set_title('B: decoding accuracy')
#%% save data
np.savez('delete_decode_{pert_noisesd}_{pert_prob}.npz',conProbability=conProbability,nonzero_num_ratio=nonzero_num_ratio,pred_diff_ALL=pred_diff_ALL,pred_diff_all=pred_diff_all,pert_noisesd=pert_noisesd,pert_prob=pert_prob,repnum=repnum)


#%% load saved data
data = np.load(r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\Analysis_folder\012_pertdata.npz")
confavg_A_all = data['confavg_A_all']
confavg_B_all = data['confavg_B_all']
pred_diff_all2 = data['pred_diff_all2']

for i in [3,4,5]:
    data=np.load(fr"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\Analysis_folder\deletepert_k_{i}.npz")
    confavg_A = np.expand_dims(data['confavg_A'], axis=0)
    confavg_B = np.expand_dims(data['confavg_B'], axis=0)
    loaded_pred_diff_all = np.expand_dims(data['pred_diff_all'], axis=0)
    
    
    confavg_A_all=np.concatenate((confavg_A_all, confavg_A), axis=0)
    confavg_B_all=np.concatenate((confavg_B_all, confavg_B), axis=0)
    pred_diff_all2=np.concatenate((pred_diff_all2, loaded_pred_diff_all), axis=0)



#%% show L2norm of decoding offset matrices if necessary

i=0
fig, axs = plt.subplots(1, 2, sharex=True, figsize=(8, 8))
axs[0].imshow(data['population2']/np.sum(data['population2']),vmin=0,vmax=0.01)
RNN_offset=np.flip(Allinfo['Offset_mat'][:,:,i],axis=0)
RNN_offset/=np.sum(RNN_offset)
axs[1].imshow(RNN_offset,vmin=0,vmax=0.01)   
plt.show()


#%% code I used to determine phase and class is almost the same (class is integer, phase is continuous)
x, y, In_ons, phase=makeInOutphase(sample_size,6,inputdur,nInput,min_dur,max_dur,dt)
#convert phase information (-pi,pi) to (-18,18)
phase=class_per_trial*(phase+np.pi)/(2*np.pi)
phase=np.concatenate((phase,phase),axis=2)

phase_sort=Act_2dsort(phase,In_ons,min_dur,max_dur)
Class_per_sec=2
classleng=int(1000/(dt*Class_per_sec))   #amount of step equaling 1 class
class_per_trial=int((min_dur+max_dur)/classleng)
class_A=np.arange(0,class_per_trial)
class_A=np.repeat(class_A,classleng) #(time,nUnit)
trial_rep_A=int(np.shape(phase_sort)[0]/(min_dur+max_dur))
class_A_train=np.tile(class_A,(trial_rep_A))
plt.figure()
plt.plot(phase_sort[0:2000,0])
plt.plot(class_A_train[0:2000])

#%% Determine the number of slices
num_slices = errorMat1.shape[2]

# Find the min and max values of the array A
vmin = np.min(errorMat1)
vmax = np.max(errorMat1)

# Create subplots with 2 rows
fig, axes = plt.subplots(2, (num_slices + 1) // 2, figsize=(12, 6))

# Flatten the axes array for easy indexing
axes = axes.flatten()

for i in range(num_slices):
    ax = axes[i]
    im = ax.imshow(errorMat1[:, :, i], vmin=vmin, vmax=vmax, cmap='viridis')
    ax.set_title(f'max weight {weight_max[i]}')
    fig.colorbar(im, ax=ax)

# Remove any unused subplots
for j in range(num_slices, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()


plt.savefig(os.path.join(savedirectory,f"erroMat1"),transparent=True,dpi=200)

plt.show()

#%%
np.savez('arrays.npz',errorMat1=errorMat1, errorMat2=errorMat2, weight_max=weight_max, conProbability=conProbability,model_index=model_index)


#%%
loaded = np.load('errormatrix_9dims.npz')
errorMat1=loaded['errorMat1']
errorMat2=loaded['errorMat2']
errorMat1_0=errorMat1
errorMat2_0=errorMat2
errorMat1=np.insert(errorMat1_0,[1],errorMat1_2,axis=2)
errorMat2=np.insert(errorMat2_0,[1],errorMat2_2,axis=2)
weight_max=[0.003,0.005,0.006,0.008,0.01,0.012,0.015,0.025,0.05,2]


errorMat1_2=errorMat1_2, errorMat2_2=errorMat2_2
