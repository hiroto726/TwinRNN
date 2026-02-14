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
sys.path.append(r'C:\Users\ifumi\anaconda3\envs\myRNN1\Python_scripts\RNN2')
from RNNcustom_2_fix2 import RNNCustom2Fix2
from CustomConstraintWithMax import IEWeightandLim, IEWeightOut
from WInitial_3 import OrthoCustom3
#from GaussianNoiseCustom import GaussianNoiseAdd
from tensorflow.keras.regularizers import l1, l2
from parula_colormap import parula

# set settings for saving svg images so that text are saved as texts
# Apply your configurations 
plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text in SVG 
plt.rcParams['text.usetex'] = False


savedirectory=r'C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4ramp_probvar'
os.chdir(savedirectory)

# save options, if set to true, a new folder will be created and the weights and script will be saved in the folder

weight_max=[0.003,0.006,0.008,0.01,0.012,0.015,0.025,0.05,2]
weight_max=[0.2,0.2,0.2,0.2,0.2,0.2,0.2]
conProbability=[0.0001,0.001,0.01,0.1,0.5,0.6,1.0]
model_index=[48765,49671,49573,49858,11680,48473,49364]
max_iteration=50000

#20240827
weight_max=[0.2,0.2,0.2,0.2,0.2]
conProbability=[0.00001,0.00003,0.0003,0.003,0.03]
model_index=[49796,45539,36486,49796,48701]

#20240829
weight_max=[0.2]
conProbability=[0.01]
model_index=[47735]


#20240829
weight_max=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
conProbability=[1.e-05, 3.e-05, 1.e-04, 3.e-04, 1.e-03, 3.e-03, 1.e-02, 3.e-02, 1.e-01, 5.e-01, 6.e-01, 1.e+00]
model_index=[49796, 45539, 48765, 36486, 49671, 49796, 47735, 48701, 49858, 11680, 48473, 49364]



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
for i in range(len(weight_max)):
    #weight_name=get_numbers_after_period(i)
    #foldername.append(fr'C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\2sameInputs_prob005_weightmax{weight_name}_fix3')

    weight_name=get_numbers_after_period(weight_max[i])
    conpro_name=get_numbers_after_period(conProbability[i])
    #foldername.append(fr'C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\2sameInputs__halfpower4_tau100a2_prob{conpro_name}_weightmax{weight_name}_fix3')
    foldername.append(fr"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4ramp_probvar\prob{conpro_name}_weightmax{weight_name}_fix3")
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


#loaded = np.load(r"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4ramp_probvar\errormat_first4.npz")
#errorMat1A=loaded['errorMat1A']
#errorMat2A=loaded['errorMat2A']
#errorMat1B=loaded['errorMat1B']
#errorMat2B=loaded['errorMat2B']


for t in range(len(weight_max)):
#for t in aa:
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
savepath=r"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4ramp_probvar"
np.savez(os.path.join(savepath,'errormat_001.npz'),errorMat1A=errorMat1A, errorMat2A=errorMat2A, errorMat1B=errorMat1B, errorMat2B=errorMat2B, weight_max=weight_max, conProbability=conProbability,model_index=model_index)


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



#aa=[4,5,6,7]
#loaded = np.load('arrays.npz')
#errorMat1=loaded['errorMat1']
#errorMat2=loaded['errorMat2']


#loaded = np.load(r"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4ramp_probvar\errormat_first4.npz")
#errorMat1A=loaded['errorMat1A']
#errorMat2A=loaded['errorMat2A']
#errorMat1B=loaded['errorMat1B']
#errorMat2B=loaded['errorMat2B']



from Confmatrix import confmat, confscore
for t in range(len(weight_max)):
#for t in aa:
    maxval=weight_max[t]
    con_prob=conProbability[t]



    #build models
    model=build_model(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha, seed1=seed1)
    # load weights
    checkpoint_filepath=os.path.join(foldername[t], f"epoch_{model_index[t]:05d}.ckpt")
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
    fig.suptitle(f'MSE after perturbation {conProbability[t]}', fontsize=16)
    
    # Display the plots
    plt.tight_layout()
    plt.show()

error_mean1=np.mean(errorMat1, axis=(0, 1)) # shape (len(weight_max),2)
error_mean2=np.mean(errorMat2, axis=(0, 1))
AB_decode_diff_mean=np.mean(AB_decode_diff, axis=(0, 1))
A_decode_diff_mean=np.mean(A_decode_diff, axis=(0, 1))
B_decode_diff_mean=np.mean(B_decode_diff, axis=(0, 1))
C_decode_diff_mean=np.mean(C_decode_diff, axis=(0, 1))



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
#if save:
#    plt.savefig(os.path.join(savepath,f"weight_distribution_each_other_{minInd}"),transparent=True,dpi=600)
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
#if save:
#    plt.savefig(os.path.join(savepath,f"weight_distribution_each_other_{minInd}"),transparent=True,dpi=600)
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
#if save:
#    plt.savefig(os.path.join(savepath,f"weight_distribution_each_other_{minInd}"),transparent=True,dpi=600)
# Show the plot
plt.show()


#%%
savepath=r"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4ramp_probvar"
np.savez(os.path.join(savepath,'perturb_decode_1to7.npz'),errorMat1=errorMat1,errorMat2=errorMat2,
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


for t in range(len(weight_max)):
        #  analyze weight distribution
    
    max_range=[0,max_dur+min_dur] # time range to choose max firing time
    nExc=nUnit-nInh #number of excitory units
    
    
    model=build_model(nUnit=nUnit, nInh=nInh, nInput=nInput, con_prob=con_prob, maxval=maxval ,ReLUalpha=ReLUalpha, seed1=seed1)
    # load weights
    checkpoint_filepath=os.path.join(foldername[t], f"epoch_{model_index[t]:05d}.ckpt")
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





from matplotlib.ticker import MaxNLocator
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
tind=[0,3,4,5,11]
tind=[0,4,11]
#tind=[0]
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
#tind=[0]
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
    
    
    
#%% perturbation with many noise

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
    
sample_size=6
trial_num=8
pert_state=0 # 0 to perturb RNN A and 1 to perturb RNN B
pert_noisesd=1.5
stop=False
# take circular mean of the prediction
option=0 #0 for circular, 1 for mean
trial1=2


pert_prob=1/200 # probability of perturbation
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



tind=[0,3,4,5,11]
#tind=[0,4,11]
#tind=[11]
#tind=[0]
tind=np.array([4])
#tind=np.arange(12)
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
    plt.title(f'Connection Probability {conProbability[t]}')
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
    temp_error_rate=get_temp_error(pred_Aall_2, class_all, pred_Ball_2, class_all, class_per_trial)
    offset_temp=np.arange(-(np.ceil(class_per_trial/2)-1),np.floor(class_per_trial/2)+1)
    axs[1].plot(offset_temp,temp_error_rate)
    plt.suptitle(f'Connection Probability {conProbability[t]}')
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
