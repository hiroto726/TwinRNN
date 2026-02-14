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
#from GaussianNoiseCustom import GaussianNoiseAdd
from tensorflow.keras.regularizers import l1, l2


savedirectory=r'C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models'
os.chdir(savedirectory)

# save options, if set to true, a new folder will be created and the weights and script will be saved in the folder

weight_max=[0.003,0.006,0.008,0.01,0.012,0.015,0.025,0.05,2]
weight_max=[0.2,0.2,0.2,0.2,0.2,0.2]
conProbability=[0.0001,0.001,0.01,0.1,0.5,1]
model_index=[40447,38209,41381,35129,41022,36847]
max_iteration=20000



def get_numbers_after_period(number):
  # Convert the number to a string
  number_str = str(number)
  
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
    foldername.append(fr'C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\2sameInputs__halframp_tau100a2_prob{conpro_name}_weightmax{weight_name}_fix3')
 
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
            y[i,in_start:vecbf,0]=np.linspace(0,1,num=vecbf-in_start)-0.5  # relative timing 1       
            y[i,vec:vecbf,1]=np.arange(-0.5,-0.5+(Dur/max_dur)-1e-10,1/max_dur) # aboslute timing 1

            
    x+=np.random.normal(loc=0.0, scale=0.01, size=np.shape(x))
    y=np.tile(y,(1,1,2))    
    x=x[:,:total_time_orig,:]
    y=y[:,:total_time_orig,:]
    return x, y, In_ons





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
        print(f"RNN{t} out of {len(weight_max)}, {i} out of {step}")
    
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
    fig.suptitle(f'MSE after perturbation {weight_max[t]}', fontsize=16)
    # Display the plots
    plt.tight_layout()
    plt.show()

error_mean1=np.mean(errorMat1, axis=(0, 1))
error_mean2=np.mean(errorMat2, axis=(0, 1))

plt.figure(figsize=(10, 6))

plt.plot(weight_max, error_mean1, marker='o', linestyle='-', label='Prediction 1')
plt.plot(weight_max, error_mean2, marker='o', linestyle='-', label='Prediction 2')
plt.xscale('log')
plt.title('Squared differences of prediction after perturbation')
plt.xlabel('max value')
plt.ylabel('squared difference')
plt.legend()

# Show the plot
plt.show()



#%%


# Determine the number of slices
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
