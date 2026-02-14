# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:18:05 2024

@author: Hiroto
"""
# use the code for training models at C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4rampprobvarfix with noise

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
from RNNCustom2FixPerturb_noise_prob_train import RNNCustom2FixPerturb_noise_prob_train
from CustomConstraintWithMax import IEWeightandLim, IEWeightOut
from WInitial_3 import OrthoCustom3
#from GaussianNoiseCustom import GaussianNoiseAdd
from tensorflow.keras.regularizers import l1, l2
from parula_colormap import parula


# create same experiment as "Multiplexing working memory and time in the trajectories of neural networks"
# save options, if set to true, a new folder will be created and the weights and script will be saved in the folder

conProb=[0.0001,0.001,0.01,0.1,0.6,1.0,0.00001]
weight_max=[0.2,0.2,0.2,0.2,0.2,0.2,0.2]

conProb=[0.00001,0.00003,0.0003,0.003,0.03]
weight_max=[0.2,0.2,0.2,0.2,0.2]
#conProb=[0.5]
#weight_max=[0.2]

conProb=[0.01]
weight_max=[0.2]


max_iteration=50000
conProb=[1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.5, 1.0]
weight_max=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

conProb=[3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.5, 1.0]
weight_max=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]


# get foldername of previously trained model without noise -> use the weight to train with noise
#20240829
weight_max=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
conProbability=[1.e-05, 3.e-05, 1.e-04, 3.e-04, 1.e-03, 3.e-03, 1.e-02, 3.e-02, 1.e-01, 5.e-01, 6.e-01, 1.e+00]
model_index=[49796, 45539, 48765, 36486, 49671, 49796, 47735, 48701, 49858, 11680, 48473, 49364]
conProb=conProbability

#20240928
max_iteration=15000
conProbability=[0.1]
weight_max=[0.2]
model_index=[14314]

# for relualpha=1
max_iteration=50000
conProbability=[0.01]
weight_max=[0.2]
model_index=[17125]


weight_max=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
conProbability=[1.e-05, 3.e-05, 1.e-04, 3.e-04, 1.e-03, 3.e-03, 1.e-02, 3.e-02, 1.e-01,1.e+00]
model_index=[48835,14334,14967,46296,14967,10117,12246,12710,14314,15526]



conProbability=np.array(conProbability)
conProb=conProbability
weight_max=np.array(weight_max)
model_index=np.array(model_index)

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
    foldername.append(fr"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\t4rampprobvarfix\2RNNs_prob{conpro_name}_weightmax{weight_name}_fix3")
    #foldername.append(fr"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\2RNNs_alpha1_prob{conpro_name}_weightmax{weight_name}_fix3")

def makepertind(sample_size,x,pert_prob,pert_A_prob):
    pert_prob_all=np.random.uniform(size=np.shape(x))<pert_prob # all of the index to perturb is True
    pert_assign=np.random.uniform(size=np.shape(x))<pert_A_prob # all index to perturb A is True
    
    pert_A=np.logical_and(pert_prob_all,pert_assign)
    pert_B=np.logical_and(pert_prob_all,np.invert(pert_assign))
    
    x_final=np.concatenate((x,pert_A,pert_B),axis=2)

    return x_final

def makeInOut(sample_size,trial_num,inputdur,nInput,min_dur,max_dur,dt,pert_prob,pert_A_prob):
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
            y[i,vec:vecbf,0]=np.power(np.linspace(0,1,num=Dur),4)-0.5 
            y[i,vec:vecbf,1]=np.arange(-0.5,-0.5+(Dur/max_dur)-1e-10,1/max_dur) # aboslute timing 1

            
    x+=np.random.normal(loc=0.0, scale=0.01, size=np.shape(x))
    y=np.tile(y,(1,1,2))    
    x=x[:,:total_time_orig,:]
    y=y[:,:total_time_orig,:]
    x=makepertind(sample_size,x,pert_prob,pert_A_prob)
    return x, y, In_ons

def makeInOut_sameint(sample_size,trial_num,inputdur,nInput,min_dur,max_dur,dt,interval):
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

def makeInOutTest(int_list,inputdur,nInput,min_dur,max_dur,dt):
    # make inputs and outputs
    # int list is a binary list specifying min and max_dur
    dur=np.array([min_dur,max_dur])
    dur_list=dur[np.array(int_list)]
    total_time=int(np.sum(dur_list,axis=0)+300/dt)
    sample_size=1
    x=np.zeros((sample_size,total_time,nInput)) # x is the input
    y=-0.5*np.ones((sample_size,total_time,2)) # y is the ouput
    minratio=min_dur/(max_dur+min_dur)
    
    trial_num=np.shape(int_list)[0]
    # make random input for all samples
    In_ons=np.zeros((sample_size,int(trial_num)),dtype=np.int64)
    binvec=[0,1]
    for i in range(sample_size):
        vec=total_time
        for j in np.arange(trial_num):
            vecbf=vec           
            vec-=dur_list[-j-1]
            In_ons[i,int(-j-1)]=vec
            Dur=vecbf-vec
            x[i,vec:vec+inputdur,:]=1
            in_start=vec
            y[i,in_start:vecbf,0]=np.power(np.linspace(0,1,num=vecbf-in_start),4)-0.5  # relative timing 1       
            y[i,vec:vecbf,1]=np.arange(-0.5,-0.5+(Dur/max_dur)-1e-10,1/max_dur) # aboslute timing 1
    y=np.tile(y,(1,1,2)) 
    x+=np.random.normal(loc=0.0, scale=0.01, size=np.shape(x))
    return x, y, In_ons

def makeInOutTest_exp(int_list,sample_size,inputdur,nInput,min_dur,max_dur,dt):
    # make inputs and outputs
    # int list is a binary list specifying min and max_dur
    dur_list=int_list
    total_time=int(np.sum(int_list,axis=None))
    x=np.zeros((sample_size,total_time,nInput)) # x is the input
    y=-0.5*np.ones((sample_size,total_time,2)) # y is the ouput
    
    trial_num=np.shape(int_list)[0]
    # make random input for all samples
    In_ons=np.zeros((sample_size,int(trial_num)),dtype=np.int64)
    binvec=[0,1]
    for i in range(sample_size):
        vec=total_time
        for j in np.arange(trial_num):
            vecbf=vec           
            vec-=dur_list[-j-1]
            In_ons[i,int(-j-1)]=vec
            Dur=vecbf-vec
            x[i,vec:vec+inputdur,:]=1
            y[i,vec:vecbf,0]=np.linspace(-0.5,0.5,num=Dur)        
            y[i,vec:vecbf,1]=np.arange(-0.5,-0.5+(Dur/max_dur)-1e-10,1/max_dur)
    x+=np.random.normal(loc=0.0, scale=0.01, size=np.shape(x))
    return x, y, In_ons

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



def avgPredOut(y,pred,In_ons,min_dur,max_dur):
    dura=[min_dur,max_dur]
    dur0=In_ons[0,1]-In_ons[0,0]
    ind1=np.argmin(dur0-dura)
    dur0_dur=dura[ind1]
    kk=0
    y_avg=np.zeros((min_dur+max_dur,np.shape(y)[2]))
    pred_avg=np.zeros((min_dur+max_dur,np.shape(pred)[2]))
    for i in np.arange(np.shape(y)[0]):
        In_time=In_ons[i,1+((ind1+i+1) % 2):-2:2]
        for j in In_time:
            if j+min_dur+max_dur<np.shape(y)[1]:
                y_avg+=np.squeeze(y[i,j:j+min_dur+max_dur,:])
                pred_avg+=np.squeeze(pred[i,j:j+min_dur+max_dur,:])
                kk=kk+1   
    y_avg/=kk
    pred_avg/=kk
    return y_avg, pred_avg
            
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
    
def avgAct_flexible(activities,In_ons,min_dur,max_dur):
    avg_min=np.zeros((min_dur,1))
    avg_max=np.zeros((max_dur,1))
    dura=[min_dur,max_dur]
    dur0=In_ons[0,1]-In_ons[0,0]
    ind1=np.argmin(dur0-dura)
    dur0_dur=dura[ind1]
    threashold=0.5*(min_dur+max_dur)
    kkmin=0
    kkmax=0
    act_avg=np.zeros((min_dur+max_dur,np.shape(activities)[2]))
    for i in np.arange(np.shape(activities)[0]):
        In_time=In_ons[i,1+((ind1+i+1) % 2):-1]
        for j in np.arange(np.shape(In_ons)[1]):
            if In_ons[i,j+1]-In_ons[i,j]<threashold:
                avg_min+=np.squeeze(activities[i,j:j+min_dur,:])
                kkmin=kkmin+1
                
            else:
                avg_max+=np.squeeze(activities[i,j:j+max_dur,:])
                kkmax=kkmax+1
    avg_min/=kkmin
    avg_max/=kkmax
    return avg_min, avg_max


def avgAct_lastfew(activities,In_ons):
    # activities is sample_size*timepoints*nUnit, and In_ons is array of 3 elements specifying duration for each trial
    act_avg=np.mean(activities,axis=0)
    avg_1=act_avg[-1-(np.sum(In_ons,axis=None)):-(In_ons[1]+In_ons[2]),:]
    avg_2=act_avg[-1-(In_ons[1]+In_ons[2]):-(In_ons[2]),:]
    avg_3=act_avg[-1-(In_ons[2]):,:]
    return avg_1, avg_2, avg_3



def build_masks(nUnit,nInh, con_prob,seed):
    random_matrix = tf.random.uniform([nUnit-nInh,nUnit], minval=0, maxval=1,seed=seed)
    # Apply threshold to generate binary values
    mask_A_1 = tf.cast(tf.random.uniform([nUnit-nInh,nUnit], minval=0, maxval=1)< con_prob, dtype=tf.int32)
    mask_A=tf.concat([mask_A_1,tf.zeros([nInh,nUnit],dtype=tf.int32)],0)
    return mask_A


def build_model():
    # A_mask and B_mask will be created from already trained connections
    #A_mask=build_masks(nUnit,nInh, con_prob,seed=seed1)
    #B_mask=build_masks(nUnit,nInh, con_prob,seed=seed1)
    visible = Input(shape=(None,3)) 
    #vis_noise=GaussianNoiseAdd(stddev=0.01, seed=seed1)(visible)# used to be 0.01*np.sqrt(tau*2)
    #hidden1 = SimpleRNN(nUnit,activation='tanh', use_bias=False, batch_size=batch_sz, stateful=False, input_shape=(None, 1), return_sequences=True)(vis_noise)

    
    # the code below incorporated options to train input kernel within RNN layer
    hidden1=RNN(RNNCustom2FixPerturb_noise_prob_train(nUnit, 
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
                          pert_noisesd=2.0), # used to be 0.05*np.sqrt(tau*2)
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

# add custom weightening to loss function
def weightMat_forLoss(y_true,In_ons):
    weight=np.ones(np.shape(y_true))
    
    for i in np.arange(np.shape(y_true)[0]):
        k=In_ons[i,1]
        weight[i,:k,:]=0
        weight[i,k:,:]=1
        
    return weight

# loss function
def custom_mse_loss(y_true, y_pred, weights_batch):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    weights_batch = tf.cast(weights_batch, tf.float32) 
    squared_diff = tf.square(y_true - y_pred)
    weighted_squared_diff = squared_diff * weights_batch
    return tf.reduce_mean(weighted_squared_diff)



# choose which model to choose from
tind=np.array([11])
tind=np.array([3,4,5,6])
tind=np.arange(len(weight_max))
tind=np.array([4,5,6,7,8,9,0,1,2,3])
for weight_ind in tind:
    save=True
    if save:
        # set name of the folder
        weight_name=get_numbers_after_period(weight_max[weight_ind])
        conprobability=get_numbers_after_period(conProb[weight_ind])
        #directory = f"2RNNs_prob{conprobability}_weightmax{weight_name}_fix3__noise2"      
        # Parent Directory path 
        #parent_dir = r"C:\Users\ifumi\anaconda3\envs\myRNN1\RNNModels\t4rampprobvarfix"   
        # Path 
        #savepath = os.path.join(parent_dir, directory)    
        savepath=foldername[weight_ind]+"_noise2"
        
        if not os.path.exists(savepath):
            os.mkdir(savepath)
            print("Directory '% s' created" % savepath) 
            # get the directory of current python script
            current_script_dir = os.path.abspath(__file__)
            # set name of the copied python script
            destination_filename = "copy_of_script.py"
            
            #save the copied python script to the new path
            shutil.copyfile(current_script_dir, os.path.join(savepath, destination_filename))    
        else:
            print("Directory already exists. Save option oborted.")
            save=False
                
 
    # training function
    @tf.function
    def train_step2(model, optimizer, x_batch, y_batch, weights_batch):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)# ypred may return nan incase of exploding activity. It may be helpful to replace nan with some big values like 5
            loss = custom_mse_loss(y_batch, y_pred, weights_batch)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    # make 2  inputs with some radom interval and output is slowly increasing sequence with that duration
    # parameters
    seed=30
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
    con_prob=conProb[weight_ind] # probability of connection between neurons
    maxval=weight_max[weight_ind]
    ReLUalpha=0.2
    pert_noisesd=2.0 # also change the actual parameter at build_model()
    pert_prob=1/200 # probability of perturbation
    pert_A_prob=0.5 # probability of perturbing RNNA
    
    # scale time parameters by dt
    min_dur=int(min_dur/dt) # minimum duration
    max_dur=int(max_dur/dt) # maximum duration
    # make input function
    inputdur=int(inputdur/dt) # duration of input in ms
    tau=tau/dt
    

    
    
    # plot inputs and outputs
    x, y, In_ons=makeInOut(sample_size,trial_num,inputdur,nInput,min_dur,max_dur,dt,pert_prob,pert_A_prob)
    plt.imshow(x[:,:,0],aspect='auto',interpolation='none')
    plt.colorbar()
    plt.show()
    plt.imshow(y[:,:,0],aspect='auto',interpolation='none')
    plt.show()
    
    k=0
    xax=range(np.shape(x)[1])
    plt.plot(xax,x[0,:,0],label='Input 1')
    plt.plot(xax,y[0,:,0],label='A output 1')
    plt.plot(xax,y[0,:,1],label='A output 2')
    plt.plot(xax,y[0,:,2],label='A output 3')
    plt.plot(xax,y[0,:,3],label='A output 4')
    
    
    

    
    
    #build models
    A_mask=tf.zeros((nUnit,nUnit))
    B_mask=tf.zeros((nUnit,nUnit))
    model=build_model()
    # summarize layers3
    print(model.summary())
    
    modelmin=build_model()
    model_prev=build_model()
    model_prev_2=build_model()
    model_prev_3=build_model()

    
    # define optimizers
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,clipvalue=1)
    optimizer_smallstep=tf.keras.optimizers.Adam(learning_rate=0.0002,clipvalue=0.0001)
    optimizer_smallstep=tf.keras.optimizers.Adam(learning_rate=0.00005,clipvalue=0.00002)
    optimizer_smallstep=tf.keras.optimizers.Adam(learning_rate=0.00002,clipvalue=0.00002)
    optimizer_smallstep=tf.keras.optimizers.Adam(learning_rate=0.00001,clipvalue=0.00001)
    optimizer_smallstep=tf.keras.optimizers.Adam(learning_rate=0.00005,clipvalue=0.00005)
    if conProbability[weight_ind]<1e-3:
        optimizer_smallstep=tf.keras.optimizers.Adam(learning_rate=0.00001,clipvalue=0.00001)
    
    # load optimizer parameters
    # make zero gradient to help optimizer understand the shape of the model paramters
    grad_vars = model.trainable_weights
    zero_grads = [tf.zeros_like(w) for w in grad_vars]
    # Apply gradients which don't do anything with Adam
    optimizer_smallstep.apply_gradients(zip(zero_grads, grad_vars))
    optimizer_savepath=os.path.join(foldername[weight_ind],f"epoch_opt_{model_index[weight_ind]:05d}.npy")
    optimizer_weights=np.load(optimizer_savepath,allow_pickle=True)
    optimizer_smallstep.set_weights(optimizer_weights) 
    
    
    # load model weights to extract A_mask and B_mask
    checkpoint_filepath=os.path.join(foldername[weight_ind], f"epoch_{model_index[weight_ind]:05d}.ckpt")
    model.load_weights(checkpoint_filepath)
    all_weights=model.get_weights()
    Wr_A, Wr_B, S_A, S_B=tf.split(all_weights[1],num_or_size_splits=4, axis=1)
    A_mask=tf.cast(S_A>0,dtype=tf.float32)
    B_mask=tf.cast(S_B>0,dtype=tf.float32)
    
    # build model again with new A_mask and B_mask and load weights again
    model=build_model()
    modelmin=build_model()
    model.load_weights(checkpoint_filepath)
    modelmin.load_weights(checkpoint_filepath)
  

    
    
    # training loop
    Loss_threashold=0.0001
    mean_loss_per_epoch=[]
    loss_sublist=[]
    i=0
    minLoss=100
    indic=0
    minInd=0
    minLoss = float('inf')
    #for i in range(epoch_size):
    i=0
    while (minLoss>Loss_threashold or i % 100!=1) and i<max_iteration and indic==0 :
        x, y, In_ons=makeInOut(sample_size,trial_num,inputdur,nInput,min_dur,max_dur,dt,pert_prob,pert_A_prob)
        # Assuming calculate_weights can work with batch data
        weights_batch =weightMat_forLoss(y,In_ons)
        # Perform the training step
        batch_loss = train_step2(model, optimizer_smallstep, x, y, weights_batch)
        
    
        
        if not np.isnan(batch_loss):
            model_prev_3.set_weights(model_prev_2.get_weights())
            model_prev_2.set_weights(model_prev.get_weights())
            model_prev.set_weights(model.get_weights())
            if batch_loss <minLoss:
                minLoss = batch_loss
                # Transfer weights from model to modelmin
                modelmin.set_weights(model.get_weights())
                print("model minimum updated")
                minInd=i
                indic=0
        else:
                print("Loss bacame Nan")
                indic=1
                model.set_weights(modelmin.get_weights())
                
        if batch_loss>=0.2 and i>100:
            model.load_weights(unique_checkpoint_path_bf)
            allweights=model.get_weights()
            newweights=[w+tf.random.normal(tf.shape(w),mean=0,stddev=1e-10) for w in allweights]
            model.set_weights(newweights)
            optimizer_weights=np.load(optimizer_savepath_bf,allow_pickle=True)
            optimizer_smallstep.set_weights(optimizer_weights)    
       
        loss_sublist.append(batch_loss)
              
            
        #save best model per 50 epochs
        if i % 100==0 and minInd>=i-100 and i>0 and indic==0:
            if save and indic==0:
                if minInd>100:
                    unique_checkpoint_path_bf=unique_checkpoint_path
                    optimizer_savepath_bf=optimizer_savepath
                
                unique_checkpoint_path = os.path.join(savepath, f"epoch_{minInd+1:05d}.ckpt")
                modelmin.save_weights(unique_checkpoint_path)
                optimizer_savepath=os.path.join(savepath,f"epoch_opt_{minInd+1:05d}.npy")
                np.save(optimizer_savepath,optimizer_smallstep.get_weights())
                
                if minInd<=100:
                    unique_checkpoint_path_bf=unique_checkpoint_path
                    optimizer_savepath_bf=optimizer_savepath
            if minLoss<=Loss_threashold:
                # if loss is small enough and the model is saved
                print("required loss achieved")
                if save:
                    modelmin.load_weights(unique_checkpoint_path)
                x, y, In_ons=makeInOut(2,trial_num,inputdur,nInput,min_dur,max_dur,dt)
                predictions1 = modelmin.predict(x)
                
                
                fig, axs = plt.subplots(2, 1, figsize=(14, 8))
                
                # Plotting data for each subplot
                for k in range(2):
                    axs[k].plot(y[k, :, 0], color='blue')
                    axs[k].plot(predictions1[k, :, 0], color='green')
                    axs[k].plot(y[k, :, 1], color='red')
                    axs[k].plot(predictions1[k, :, 1], color='orange')                
                    axs[k].set_title(f'Plot {i+1}')
                # Adjusting layout
                plt.tight_layout()
                plt.show()               
                   
        
        # plot loss over epochs and prediction
        if i % 100==0:
    
            plt.plot(loss_sublist[:i])
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale("log") 
            plt.xlim(0,i)
            plt.show()        
                
            x, y, In_ons=makeInOut(2,trial_num,inputdur,nInput,min_dur,max_dur,dt,pert_prob,pert_A_prob)
            predictions = modelmin.predict(x)
            
            
            # Plotting data for each subplot
            fig, axs = plt.subplots(4, 1, figsize=(14, 8))
            
            # Plotting data for each subplot
            for k in range(4):
                axs[k].plot(y[0, :, k], color='blue')
                axs[k].plot(predictions[0, :, k], color='green')            
                #axs[k].set_title(f'Plot {i+1}')
            # Adjusting layout
            plt.tight_layout()
            plt.show()
    
        print(f"epoch: {i}  loss: {batch_loss:.3g}" )
        i=i+1


    
    #analysis of the model
    # load the best model from training   
    sample_size=4
    k=minInd
    checkpoint_filepath=os.path.join(savepath, f"epoch_{k+1:05d}.ckpt")
    model.load_weights(checkpoint_filepath)
    
    x, y, In_ons=makeInOut(sample_size,trial_num+4,inputdur,nInput,min_dur,max_dur,dt,pert_prob,pert_A_prob)
    predictions = model.predict(x)
    
    fig, axs = plt.subplots(2, 1, figsize=(14, 8))
    Line=[None]*4
    for ss in range(2):
        axs[ss].plot(y[ss,:,0],color='blue',label='Target 0',alpha=0.5)
        axs[ss].plot(y[ss,:,1],color='red',label='Target 1',alpha=0.5)
        axs[ss].plot(y[ss,:,2],color='purple',label='Target 2',alpha=0.5)
        axs[ss].plot(y[ss,:,3],color='yellow',label='Target 3',alpha=0.5)
        axs[ss].plot(predictions[ss,:,0],color='green',label='Prediction_A 0',alpha=0.5)
        axs[ss].plot(predictions[ss,:,1],color='orangered',label='Prediction_A 1',alpha=0.5)
        axs[ss].plot(predictions[ss,:,2],color='turquoise',label='Prediction_B 0',alpha=0.5)
        axs[ss].plot(predictions[ss,:,3],color='gold',label='Prediction_B 1',alpha=0.5)
        axs[ss].set_title(f'Plot {ss+1}')
    
    # Adjusting layout
    handles, labels = axs[0,].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    #plt.show()
    if save:
        plt.savefig(os.path.join(savepath,f"Figure_result_{k+1}_{minInd}"),transparent=True,dpi=200)
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(14, 8),sharex=True)
    Line=[None]*4
    ind1=In_ons[0,3]-100
    ind2=In_ons[0,5]+100
    x=np.arange(-100,ind2-ind1-100)
    #x=np.arange(ind1,ind2)
    axs[0].plot(x,y[0,ind1:ind2,0],color='black',label='_nolegend_')
    axs[0].plot(x,y[0,ind1:ind2,1],color='black',label='_nolegend_')
    axs[0].plot(x,predictions[0,ind1:ind2,0],color='green',label='Output 0',)
    axs[0].plot(x,predictions[0,ind1:ind2,1],color='red',label='Output 1')
    
    axs[0].set_ylabel(f'RNN A')
    xticks = axs[0].get_xticks()
    scaled_xticks = np.round(xticks * 0.01)
    # Set new tick labels
    axs[0].set_xticklabels(scaled_xticks)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].tick_params(bottom=False, labelbottom=False)
    
    axs[1].plot(x,y[0,ind1:ind2,0],color='black',label='_nolegend_')
    axs[1].plot(x,y[0,ind1:ind2,1],color='black',label='_nolegend_')
    axs[1].plot(x,predictions[0,ind1:ind2,2],color='green',label='Output 0')
    axs[1].plot(x,predictions[0,ind1:ind2,3],color='red',label='Output 1')
    
    axs[1].set_ylabel(f'RNN B')
    axs[1].set_xlabel(f'Time (s)')
    axs[1].set_xticks((0,600,1200,1800))
    axs[1].set_xticklabels((0,6,12,18))
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    # Adjusting layout
    handles, labels = axs[0,].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    if save:
         plt.savefig(os.path.join(savepath,f"output_1trial_{k+1}"),transparent=True,dpi=200)
    plt.show()
    
    # show average output
    y_avg, pred_avg=avgPredOut(y,predictions,In_ons,min_dur,max_dur)
    plt.plot(y_avg,label='Target')
    plt.plot(pred_avg,label='Prediction')
    if save:
        plt.savefig(os.path.join(savepath,f"Figure_result_{k+1}_avg_{minInd}"),transparent=True,dpi=200)
    plt.show()
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    for ss in range(4):
        axs[ss].plot(y_avg[:,ss])
        axs[ss].plot(pred_avg[:,ss])

    if save:
        plt.savefig(os.path.join(savepath,f"Figure_result_{k+1}_avg2_{minInd}"),transparent=True,dpi=200)
    plt.show()

    """# response to different intervals
    fig, axs = plt.subplots(4, 1,sharex=True)
    int_list=np.array([[1,0,1,0,1],[0,1,0,1,0],[0,0,0,0,0],[1,1,1,1,1]])
    for i in range(4):
        x,y,In_ons=makeInOutTest(int_list[i],inputdur,nInput,min_dur,max_dur,dt)
        predictions = model.predict(x)
        axs[i].plot(y[0,:,0],color='blue',label='Target 0',alpha=0.5)
        axs[i].plot(y[0,:,1],color='red',label='Target 1',alpha=0.5)
        axs[i].plot(y[0,:,2],color='purple',label='Target 2',alpha=0.5)
        axs[i].plot(y[0,:,3],color='yellow',label='Target 3',alpha=0.5)
        axs[i].plot(predictions[0,:,0],color='green',label='Prediction_A 0',alpha=0.5)
        axs[i].plot(predictions[0,:,1],color='orangered',label='Prediction_A 1',alpha=0.5)
        axs[i].plot(predictions[0,:,2],color='turquoise',label='Prediction_B 0',alpha=0.5)
        axs[i].plot(predictions[0,:,3],color='gold',label='Prediction_B 1',alpha=0.5)
        xticks = axs[i].get_xticks()
        # Rescale tick labels
        scaled_xticks = np.round(xticks * 0.01)
        # Set new tick labels
        axs[i].set_xticklabels(scaled_xticks)
    
    handles, labels = axs[0,].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(savepath,f"Output_4types"),transparent=True,dpi=200)
    plt.show()"""
    """# response to different intervals
    fig, axs = plt.subplots(4, 1,sharex=True)
    int_list=np.array([[1,0,1,0,1],[0,1,0,1,0],[0,0,0,0,0],[1,1,1,1,1]])
    for i in range(4):
        x,y,In_ons=makeInOutTest(int_list[i],inputdur,nInput,min_dur,max_dur,dt)
        predictions = model.predict(x)
        axs[i].plot(y[0,:,0],color='blue',label='Target 0',alpha=0.5)
        axs[i].plot(y[0,:,1],color='red',label='Target 1',alpha=0.5)
        axs[i].plot(y[0,:,2],color='purple',label='Target 2',alpha=0.5)
        axs[i].plot(y[0,:,3],color='yellow',label='Target 3',alpha=0.5)
        axs[i].plot(predictions[0,:,0],color='green',label='Prediction_A 0',alpha=0.5)
        axs[i].plot(predictions[0,:,1],color='orangered',label='Prediction_A 1',alpha=0.5)
        axs[i].plot(predictions[0,:,2],color='turquoise',label='Prediction_B 0',alpha=0.5)
        axs[i].plot(predictions[0,:,3],color='gold',label='Prediction_B 1',alpha=0.5)
        xticks = axs[i].get_xticks()
        # Rescale tick labels
        scaled_xticks = np.round(xticks * 0.01)
        # Set new tick labels
        axs[i].set_xticklabels(scaled_xticks)
    
    handles, labels = axs[0,].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(savepath,f"Output_4types"),transparent=True,dpi=200)
    plt.show()"""
    
    """
    testdur=[1500,3000,4500,7500,9000,10500,13500,15000,18000]
    testdur=[1500,3000,4500,5000,5500,6000,7000,7500,8000,8500,11000,11500,12000,12500,13000,13500,13500,15000,18000]
    fig, axs = plt.subplots(len(testdur), 2,sharex=True,sharey=True)
    diffout=np.zeros((len(testdur),2))
    for i in range(len(testdur)):
        int_list=[max_dur,min_dur,max_dur,int(testdur[i]/dt),max_dur]
        x,y,In_ons=makeInOutTest_exp(int_list,1,inputdur,nInput,min_dur,max_dur,dt)
        predictions = model.predict(x)
        
        diffout[i,0]=predictions[0,-int(5000/dt),0]-y[0,-int(5000/dt),0]
        axs[i,0].plot(y[0,max_dur+min_dur:,1],color='red',label='Target 1')
        axs[i,0].plot(predictions[0,max_dur+min_dur:,1],color='orange',label='Prediction 1')
        axs[i,0].plot(y[0,max_dur+min_dur:,0],color='blue',label='Target 0')
        axs[i,0].plot(predictions[0,max_dur+min_dur:,0],color='green',label='Prediction 0')
        axs[i,0].set(ylabel=f"{testdur[i]}")
        #axs[i,0].set_xticklabels([])
        #axs[i,0].set_yticklabels([])
        
        int_list=[min_dur,max_dur,min_dur,int(testdur[i]/dt),max_dur]
        x,y,In_ons=makeInOutTest_exp(int_list,1,inputdur,nInput,min_dur,max_dur,dt)
        predictions = model.predict(x)
        diffout[i,1]=predictions[0,-int(5000/dt),0]-y[0,-int(5000/dt),0]
        
        axs[i,1].plot(y[0,max_dur+min_dur:,1],color='red',label='Target 1')
        axs[i,1].plot(predictions[0,max_dur+min_dur:,1],color='orange',label='Prediction 1') 
        axs[i,1].plot(y[0,max_dur+min_dur:,0],color='blue',label='Target 0')
        axs[i,1].plot(predictions[0,max_dur+min_dur:,0],color='green',label='Prediction 0')
        #axs[i,1].set_xticklabels([])
        #axs[i,1].set_yticklabels([])
    fig.subplots_adjust(hspace=0,wspace=0)    
    #plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(os.path.join(savepath,f"variousInput_{int_list}"),transparent=True,dpi=200)
    plt.show()
    
    fig=plt.figure()
    fig.plot(diffout[:,0])
    fig.plot(diffout[:,1])"""
    
    
    
    
    # get activity of all units 
    # Assuming you have a trained model named 'model'
    # Define a new model that outputs both the output and the activities of intermediate layers
    outputs = [layer.output for layer in model.layers[1:]]  # Exclude the input layer
    activity_model = Model(inputs=model.input, outputs=outputs)
    
    trial_num=4
    # Get the output and activities of all layers for the new input data
    x, y, In_ons=makeInOut(sample_size,trial_num+4,inputdur,nInput,min_dur,max_dur,dt,0,pert_A_prob)
    #xnew=sum(xnew,axis=2)
    output_and_activities = activity_model.predict(x)
    activities_A = output_and_activities[0]  # Activities of all intermediate layers
    activities_B=output_and_activities[1]
    act_avg_A=avgAct(activities_A,In_ons,min_dur,max_dur)
    act_avg_B=avgAct(activities_B,In_ons,min_dur,max_dur)
    
    
    # load data from which to show activity
    x, y, In_ons=makeInOut(sample_size,trial_num+4,inputdur,nInput,min_dur,max_dur,dt,0,pert_A_prob)
    #xnew=sum(xnew,axis=2)
    output_and_activities = activity_model.predict(x)
    activities_A = output_and_activities[0]  # Activities of all intermediate layers
    activities_B=output_and_activities[1]
    act_avg_A_2=avgAct(activities_A,In_ons,min_dur,max_dur)
    act_avg_B_2=avgAct(activities_B,In_ons,min_dur,max_dur)
    
    
    
    
    max_range=[min_dur,max_dur+min_dur] # time range to choose max firing time
    # visualize activity of units
    act_avg_A_2=act_avg_A_2/(np.max(act_avg_A_2[max_range[0]:max_range[1], :],axis=0)+1e-10)
    # Reorder the activities based on sorted units
    maxtime_A=np.argmax(act_avg_A[max_range[0]:max_range[1], :],axis=0)
    sort_filtered_units_A=np.argsort(maxtime_A)
    Cellact_A = act_avg_A_2[:, sort_filtered_units_A]
    
    act_avg_B_2=act_avg_B_2/(np.max(act_avg_B_2[max_range[0]:max_range[1], :],axis=0)+1e-10)
    # Reorder the activities based on sorted units
    maxtime_B=np.argmax(act_avg_B[max_range[0]:max_range[1], :],axis=0)
    sort_filtered_units_B=np.argsort(maxtime_B)
    Cellact_B= act_avg_B_2[:, sort_filtered_units_B]
    
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(Cellact_A.T, aspect='auto', cmap=parula,interpolation='none', vmin=0, vmax=1)
    axs[0].set_title(f'A')
    axs[0].set_xticks([0,600,1200,1800])
    axs[0].set_xticklabels([0,6,12,18])
    axs[1].imshow(Cellact_B.T, aspect='auto', cmap=parula,interpolation='none', vmin=0, vmax=1)
    axs[1].set_title(f'B')
    axs[1].spines['left'].set_visible(False)
    axs[1].tick_params(left=False, labelleft=False)
    axs[1].set_xticks([0,600,1200,1800])
    axs[1].set_xticklabels([0,6,12,18])
    if save:
        plt.savefig(os.path.join(savepath,f"units_activity_crossval_{minInd}"),transparent=True,dpi=600)
        np.savez(f'avg_activity_{k+1}.npz',x=x, y=y, In_ons=In_ons, activities_A=activities_A, activities_B=activities_B, act_avg_A_2=act_avg_A_2, act_avg_B_2=act_avg_B_2)
    plt.show
    
    
    
    
    
    
    """fig=plt.figure()
    Unitind=100
    for i in range(2):
        plt.plot(Cellact[Unitind,:,i],label=f'Output {i}')
    plt.legend()   
    plt.title(f"Cell {sort_filtered_units[Unitind]}")
    if save:
        fig.savefig(os.path.join(savepath,f"Activity of Unit {sort_filtered_units[Unitind]}"),transparent=True,dpi=600)
    plt.show"""
    
    # save model
    #model.save(r"C:\Users\Fumiya\anaconda3\envs\myRNN2\RNN_Models/custom_model_ver6.keras")
    
    
    
    # Sequentiality index functions
    def xlogx(p):
        a = np.zeros_like(p, dtype=float)
        a[p < 0] = np.nan
        a[p == 0] = 0
        b = p * np.log(p)
        a[p > 0] = b[p > 0]
        return a
    
    def Seq_ind(act_avg_A, mean_range):
        """
        Computes Peak Entropy, Temporal Sparsity, and Sequentiality Index.
    
        Parameters:
            act_avg_A (np.ndarray): Activity data.
            mean_range (int): Range to compute mean over.
    
        Returns:
            tuple: Peak Entropy, Temporal Sparsity, Sequentiality Index.
        """
        # Assuming min_dur is defined or set it as needed
        data6 = act_avg_A[:min_dur, :]
        data12 = act_avg_A[min_dur:, :]
        ncell = data6.shape[1]
    
        # Reduce bins
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
    
        peak_ent = np.zeros(2)
        temp_spar = np.zeros(2)
        Sql = np.zeros(2)
        peak_ent[0] = np.sum(-xlogx(p6)) / np.log(data6.shape[0])
        peak_ent[1] = np.sum(-xlogx(p12)) / np.log(data12.shape[0])
    
        data6_norm = data6 / np.sum(data6, axis=1, keepdims=True)
        data12_norm = data12 / np.sum(data12, axis=1, keepdims=True)
        temp_spar[0] = 1 - np.mean(np.sum(-xlogx(data6_norm), axis=1)) / np.log(ncell)
        temp_spar[1] = 1 - np.mean(np.sum(-xlogx(data12_norm), axis=1)) / np.log(ncell)
    
        Sql[0] = np.sqrt(peak_ent[0] * temp_spar[0])
        Sql[1] = np.sqrt(peak_ent[1] * temp_spar[1])
    
        return peak_ent, temp_spar, Sql
    
    mean_range=50;
    PE_A, TS_A, Sql_A=Seq_ind(act_avg_A,mean_range)
    PE_B, TS_B, Sql_B=Seq_ind(act_avg_B,mean_range)
    
    A_min=[PE_A[0], TS_A[0],Sql_A[0]]
    A_max=[PE_A[1], TS_A[1],Sql_A[1]]
    B_min=[PE_B[0], TS_B[0],Sql_B[0]]
    B_max=[PE_B[1], TS_B[1],Sql_B[1]]
    
    # Define the width of the bars
    bar_width = 0.2
    
    categories = ['Peak entropy', 'Temporal sparsity', 'Sequentieality index']
    # Define the position of the bars on the x-axis
    r1 = np.arange(len(categories))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    
    # Create the bars
    plt.figure()
    plt.bar(r1, A_min, color='blue', width=bar_width, edgecolor='grey', label='A min')
    plt.bar(r2, B_min, color='green', width=bar_width, edgecolor='grey', label='B min')
    plt.bar(r3, A_max, color='orange', width=bar_width, edgecolor='grey', label='A max')
    plt.bar(r4, B_max, color='red', width=bar_width, edgecolor='grey', label='B max')
    
    # Add labels to the x-axis and y-axis
    plt.xlabel('Categories', fontweight='bold')
    plt.ylabel('Values', fontweight='bold')
    plt.xticks([r + 1.5*bar_width for r in range(len(categories))], categories,fontsize=7)
    #plt.xticks([r + 1.5 * bar_width for r in range(len(categories))], categories, rotation=45, ha='right')
    
    # Add a title and legend
    plt.title('Comparison of Values Across Categories')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    # Show the plot
    if save:
        plt.savefig(os.path.join(savepath,f"Sequentiality_Index_{minInd}"),transparent=True,dpi=600)
        np.savez(f'sequentiality_index_{k+1}.npz', A_min=A_min, A_max=A_max, B_min=B_min, B_max=B_max)

    plt.show
    
    
    
    #  analyze weight distribution
    
    max_range=[0,max_dur+min_dur] # time range to choose max firing time
    nExc=nUnit-nInh #number of excitory units
    
    
    # the weights for this RNN
    RNN_input_kernel=model.layers[1].get_weights()[0]
    RNN_layer_Recurrent_kernel=model.layers[1].get_weights()[1]
    dense_kernel_A=model.layers[2].get_weights()[0]
    dense_bias_A=model.layers[2].get_weights()[1]
    dense_kernel_B=model.layers[3].get_weights()[0]
    dense_bias_B=model.layers[3].get_weights()[1]
    
    in_A,in_B=np.split(RNN_input_kernel,2, axis=1)
    Wr_A, Wr_B, S_A, S_B=np.split(RNN_layer_Recurrent_kernel,4, axis=1)
    
    from scipy.io import savemat
    mdict={"input_kernel": RNN_input_kernel,
          "recurrent_kernel": RNN_layer_Recurrent_kernel,
          "dense_kernel_A": dense_kernel_A,
          "dense_bias_A":dense_bias_A,
          "dense_kernel_B": dense_kernel_B,
          "dense_bias_B":dense_bias_B,
          "input":x,
          "output":y,
          "In_ons":In_ons,
          "tau":tau,
          "ReLUalpha":ReLUalpha}
    if save:
        os.chdir(savepath)
        savemat(f"model_{minInd}_weights_matlab",mdict)
    
    
    
    
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
    
    
    Bin=100
    fig, axs = plt.subplots(2,2,sharex=True)
    exexavg_A, inexavg_A, ininavg_A, exinavg_A, order_A=weight_distribution(Bin,nExc,nInh,max_time_ex_A,max_time_in_A,Wr_sort_A)
    # plot weights distribution
    axs[0,0].plot(np.delete(order_A,Bin),np.delete(exexavg_A,Bin),label='Ex Ex')
    axs[0,0].plot(order_A,exinavg_A,label='Ex In')
    axs[0,0].title.set_text(f'A')
    axs[0,0].legend(loc='lower center')
    axs[0,1].plot(np.delete(order_A,Bin),np.delete(ininavg_A,Bin),label='In In')
    axs[0,1].plot(order_A,inexavg_A,label='In Ex')
    axs[0,1].title.set_text(f'A')
    axs[0,1].legend(loc='lower center')
        
        
    exexavg_B, inexavg_B, ininavg_B, exinavg_B, order_B=weight_distribution(Bin,nExc,nInh,max_time_ex_B,max_time_in_B,Wr_sort_B)
    # plot weights distribution
    axs[1,0].plot(np.delete(order_B,Bin),np.delete(exexavg_B,Bin),label='Ex Ex')
    axs[1,0].plot(order_A,exinavg_B,label='Ex In')
    axs[1,0].title.set_text(f'B')
    axs[1,0].legend(loc='lower center')
    axs[1,1].plot(np.delete(order_B,Bin),np.delete(ininavg_B,Bin),label='In In')
    axs[1,1].plot(order_A,inexavg_B,label='In Ex')
    axs[1,1].title.set_text(f'B')
    axs[1,1].legend(loc='lower center')
    if save:
        plt.savefig(os.path.join(savepath,f"weight_distribution_sort_by_18s_{minInd}"),transparent=True,dpi=600)
    
    
    
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
    S_B_avg_ex=np.nanmean(S_B_bin[:,:nExc],axis=1)
    S_B_avg_in=np.nanmean(S_B_bin[:,nExc:],axis=1)
    S_A_avg_ex=np.nanmean(S_A_bin[:,:nExc],axis=1)
    S_A_avg_in=np.nanmean(S_A_bin[:,nExc:],axis=1)
    
    plt.figure()
    plt.plot(order,S_B_avg_ex,label='ExB ExA')
    plt.plot(order,S_B_avg_in,label='ExB InA')
    plt.plot(order,S_A_avg_ex,label='ExA ExB')
    plt.plot(order,S_A_avg_in,label='ExA InB')
    plt.legend(loc='lower right')
    if save:
        plt.savefig(os.path.join(savepath,f"weight_distribution_each_other_{minInd}"),transparent=True,dpi=600)
    
    
    fig, axs = plt.subplots(2, 2)  # Adjust figure size if needed
    # Plotting the first subplot
    axs[0, 0].plot(order, S_B_avg_ex, label='ExB ExA')
    axs[0, 0].set_title('ExB ExA')
    #axs[0, 0].legend(loc='upper right')
    
    # Plotting the second subplot
    axs[0, 1].plot(order, S_B_avg_in, label='ExB InA')
    axs[0, 1].set_title('ExB InA')
    #axs[0, 1].legend(loc='lower right')
    
    # Plotting the third subplot
    axs[1, 0].plot(order, S_A_avg_ex, label='ExA ExB')
    axs[1, 0].set_title('ExA ExB')
    #axs[1, 0].legend(loc='lower right')
    
    # Plotting the fourth subplot
    axs[1, 1].plot(order, S_A_avg_in, label='ExA InB')
    axs[1, 1].set_title('ExA InB')
    #axs[1, 1].legend(loc='lower right')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(savepath,f"weight_distribution_each_other_4_{minInd}"),transparent=True,dpi=600)
        np.savez(f'weight_distribution_{k+1}.npz', S_A_avg_ex=S_A_avg_ex, S_A_avg_in=S_A_avg_in, S_B_avg_ex=S_B_avg_ex, S_B_avg_in=S_B_avg_in,
                 exexavg_A=exexavg_A, inexavg_A=inexavg_A, ininavg_A=ininavg_A, exinavg_A=exinavg_A, order_A=order_A,
                 exexavg_B=exexavg_B, inexavg_B=inexavg_B, ininavg_B=ininavg_B, exinavg_B=exinavg_B, order_B=order_B)
    # Show the plot
    plt.show()
    
    
    # show distribution of weights
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
    plt.title('Distribution Comparison')
    plt.ylabel('Values')
    plt.grid(True)
    # Get current y-ticks
    yticks = plt.gca().get_yticks()
    # Modify the negative ticks by scaling them by 4
    scaled_yticks = [y * scale_fac if y < 0 else y for y in yticks]
    # Re-label the y-tick labels with the scaled negative values
    plt.gca().set_yticklabels([f'{y:.2f}' for y in scaled_yticks])
    if save:
        plt.savefig(os.path.join(savepath,f"weight_distribution_boxplot {minInd} prob{con_prob}.png"),transparent=True,dpi=600)
    plt.show()
    
    
    # weight distribution histogram
    fig, axs = plt.subplots(2, 1, sharex='col',sharey=True)  # Adjust figure size if needed
    axs[0].hist(np.ravel(Wr_A[:-nInh,:]),rwidth=1,color='blue',bins=100)
    axs[1].hist(np.ravel(Wr_B[:-nInh,:]),rwidth=1,color='blue',bins=100)
    axs[0].hist(np.ravel(Wr_A[-nInh:,:]),rwidth=1,color='red',bins=100)
    axs[1].hist(np.ravel(Wr_B[-nInh:,:]),rwidth=1,color='red',bins=100)

    if save:
        plt.savefig(os.path.join(savepath,f"weight_distribution_histogram_{minInd}"),transparent=True,dpi=600)
    plt.show()

    # end of training
    
    
    # comparison between before training and after training
    model_before=build_model() # model before training
    # load model weights to extract A_mask and B_mask
    checkpoint_filepath=os.path.join(foldername[weight_ind], f"epoch_{model_index[weight_ind]:05d}.ckpt")
    model_before.load_weights(checkpoint_filepath)
    
    max_range=[0,max_dur+min_dur] # time range to choose max firing time
    nExc=nUnit-nInh #number of excitory units
    
    
    # the weights for trained RNN
    RNN_input_kernel=model.layers[1].get_weights()[0]
    RNN_layer_Recurrent_kernel=model.layers[1].get_weights()[1]
    dense_kernel_A=model.layers[2].get_weights()[0]
    dense_bias_A=model.layers[2].get_weights()[1]
    dense_kernel_B=model.layers[3].get_weights()[0]
    dense_bias_B=model.layers[3].get_weights()[1]
    in_A,in_B=np.split(RNN_input_kernel,2, axis=1)
    Wr_A, Wr_B, S_A, S_B=np.split(RNN_layer_Recurrent_kernel,4, axis=1)
    
    
    # the weights for pretrained RNN
    RNN_input_kernel=model_before.layers[1].get_weights()[0]
    RNN_layer_Recurrent_kernel=model_before.layers[1].get_weights()[1]
    dense_kernel_A=model_before.layers[2].get_weights()[0]
    dense_bias_A=model_before.layers[2].get_weights()[1]
    dense_kernel_B=model_before.layers[3].get_weights()[0]
    dense_bias_B=model_before.layers[3].get_weights()[1]
    in_A_before,in_B_before=np.split(RNN_input_kernel,2, axis=1)
    Wr_A_before, Wr_B_before, S_A_before, S_B_before=np.split(RNN_layer_Recurrent_kernel,4, axis=1)
    
    
    def analyze_connections(S_matrix, peak_times_source, peak_times_target, is_S_A=True, use_phase=False, top_percent=100, ax=None):
        """
        Analyze connections and plot the relationship between connection strength and time difference of peak activity.
    
        Parameters:
            S_matrix (np.ndarray): Connection matrix.
            peak_times_source (np.ndarray): Peak times of source neurons.
            peak_times_target (np.ndarray): Peak times of target neurons.
            is_S_A (bool): If True, analyzing S_A; if False, analyzing S_B.
            use_phase (bool): If True, calculate time differences considering cyclic activity.
            top_percent (float): Percentage of top connections to analyze (between 0 and 100).
            ax (matplotlib.axes.Axes): Axes object to draw the plot onto, optional.
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
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(time_diffs, S_nonzero, s=1, alpha=0.5, label='Data points')
        ax.set_ylabel('Connection Strength')
        if use_phase:
            ax.set_xlabel('Minimal Time Difference (Cyclic)')
        else:
            ax.set_xlabel('Time Difference of Peak Activity')
        
        # Linear regression
        if len(time_diffs) > 1:
            coefficients = np.polyfit(time_diffs, S_nonzero, 1)
            slope = coefficients[0]
            intercept = coefficients[1]
            
            # Generate x values for the line
            x_fit = np.linspace(np.min(time_diffs), np.max(time_diffs), 100)
            y_fit = slope * x_fit + intercept
            
            # Plot the best fit line
            ax.plot(x_fit, y_fit, color='red', label=f'Best fit line (slope = {slope:.2E})')
        else:
            slope = np.nan  # Not enough points to compute slope
            ax.text(0.5, 0.5, 'Not enough data for regression', transform=ax.transAxes, ha='center')
        
        # Add legend
        ax.legend()
        
        # Return results if needed
        return time_diffs, S_nonzero, slope, x_fit, intercept
    
    
    
    # get average activity for trained model
    outputs = [layer.output for layer in model.layers[1:]]  # Exclude the input layer
    activity_model = Model(inputs=model.input, outputs=outputs)
    trial_num=4
    # Get the output and activities of all layers for the new input data
    x, y, In_ons=makeInOut(sample_size,trial_num+4,inputdur,nInput,min_dur,max_dur,dt,0,pert_A_prob)
    #xnew=sum(xnew,axis=2)
    output_and_activities = activity_model.predict(x)
    activities_A = output_and_activities[0]  # Activities of all intermediate layers
    activities_B=output_and_activities[1]
    act_avg_A=avgAct(activities_A,In_ons,min_dur,max_dur)
    act_avg_B=avgAct(activities_B,In_ons,min_dur,max_dur)
    # Compute the time of maximum activity for each neuron
    peak_times_A = np.argmax(act_avg_A, axis=0)  # Shape: (N,)
    peak_times_B = np.argmax(act_avg_B, axis=0)  # Shape: (N,)
    
    # get average activity for pretrained model
    outputs = [layer.output for layer in model_before.layers[1:]]  # Exclude the input layer
    activity_model_before = Model(inputs=model_before.input, outputs=outputs)
    trial_num=4
    # Get the output and activities of all layers for the new input data
    x, y, In_ons=makeInOut(sample_size,trial_num+4,inputdur,nInput,min_dur,max_dur,dt,0,pert_A_prob)
    #xnew=sum(xnew,axis=2)
    output_and_activities = activity_model_before.predict(x)
    activities_A = output_and_activities[0]  # Activities of all intermediate layers
    activities_B=output_and_activities[1]
    act_avg_A_before=avgAct(activities_A,In_ons,min_dur,max_dur)
    act_avg_B_before=avgAct(activities_B,In_ons,min_dur,max_dur)
    # Compute the time of maximum activity for each neuron
    peak_times_A_before = np.argmax(act_avg_A_before, axis=0)  # Shape: (N,)
    peak_times_B_before = np.argmax(act_avg_B_before, axis=0)  # Shape: (N,)
    
    
    # make figure
    T=np.shape(act_avg_A)[0]
    nExc = nUnit - nInh  # Number of excitatory neurons
    top_percent=100
    
    
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot each dataset in its respective subplot
    time_diffs_A, S_nonzero_A, slope_A, x_fit_A, intercept_A=analyze_connections(S_A, peak_times_A, peak_times_B, is_S_A=True, use_phase=True, top_percent=100, ax=axs[1, 0])
    axs[1, 0].set_title('A: After traning A')
    
    time_diffs_B, S_nonzero_B, slope_B, x_fit_B, intercept_B=analyze_connections(S_B, peak_times_B, peak_times_A, is_S_A=False, use_phase=True, top_percent=100, ax=axs[1, 1])
    axs[1, 1].set_title('B: After traning')
    
    time_diffs_A_before, S_nonzero_A_before, slope_A_before, x_fit_A_before, intercept_A_before=analyze_connections(S_A_before, peak_times_A_before, peak_times_B_before, is_S_A=True, use_phase=True, top_percent=100, ax=axs[0, 0])
    axs[0, 0].set_title('A: Before training')
    
    time_diffs_B_before, S_nonzero_B_before, slope_B_before, x_fit_B_before, intercept_B_before=analyze_connections(S_B_before, peak_times_B_before, peak_times_A_before, is_S_A=False, use_phase=True, top_percent=100, ax=axs[0, 1])
    axs[0, 1].set_title('B: Before training')
    
    # Adjust layout and display the plot
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(savepath,f"weight_distribution_scatter_{minInd}"),transparent=True,dpi=600)
    plt.show()
    plt.show()
    
    
    
    # get connection that increased its value
    nonzero_ind_A=np.logical_or(S_A>0, S_A_before>0)
    nonzero_ind_B=np.logical_or(S_B>0, S_B_before>0)
    fig, ax = plt.subplots(1,2,figsize=(10,10))
    ax[0].scatter(S_A_before[nonzero_ind_A],S_A[nonzero_ind_A])
    ax[0].set_xlabel('Before training')
    ax[0].set_ylabel('After training')
    lims = [
        np.min([ax[0].get_xlim(), ax[0].get_ylim()]),  # min of both axes
        np.max([ax[0].get_xlim(), ax[0].get_ylim()]),  # max of both axes
    ]
    # now plot both limits against eachother
    ax[0].plot(lims, lims, 'k-', alpha=1)
    ax[0].set_aspect('equal')
    ax[0].set_xlim(lims)
    ax[0].set_ylim(lims)
    
    ax[1].scatter(S_B_before[nonzero_ind_B],S_B[nonzero_ind_B])
    ax[1].set_xlabel('Before training')
    ax[1].set_ylabel('After training')
    lims = [
        np.min([ax[1].get_xlim(), ax[1].get_ylim()]),  # min of both axes
        np.max([ax[1].get_xlim(), ax[1].get_ylim()]),  # max of both axes
    ]
    # now plot both limits against eachother
    ax[1].plot(lims, lims, 'k-', alpha=1)
    ax[1].set_aspect('equal')
    ax[1].set_xlim(lims)
    ax[1].set_ylim(lims)

    if save:
        plt.savefig(os.path.join(savepath,f"weight_distribution_bforeafter_scatter_{minInd}"),transparent=True,dpi=600)
    plt.show()
    
    
    
    from scipy import stats
    def plotdistribution(S_nonzero_A, time_diffs_A, minthreash, maxthreash, ax=None):
        """
        Plots the distribution of connection strengths for two subsets of data based on time differences.
        Additionally, it plots the mean values for each distribution and assesses the statistical significance
        of the difference between the means using an independent t-test and Mann-Whitney U test.
    
        Parameters:
            S_nonzero_A (np.ndarray): Array of non-zero connection strengths.
            time_diffs_A (np.ndarray): Array of time differences corresponding to connection strengths.
            minthreash (float): Minimum threshold for the first subset (time_diffs_A < minthreash).
            maxthreash (float): Maximum threshold for the second subset (time_diffs_A > maxthreash).
            ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to plot on.
                If None, a new figure and axes will be created.
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
        bins = np.linspace(combined_data.min(), combined_data.max(), num_bins)
    
        # Create the plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))
    
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
        # 1. Independent t-test
        t_stat, p_value_t = stats.ttest_ind(subset_low, subset_high, equal_var=False)
    
        # 2. Mann-Whitney U test
        u_stat, p_value_u = stats.mannwhitneyu(subset_low, subset_high, alternative='two-sided')
    
        # Determine significance
        alpha = 0.05
        significance_t = 'Significant' if p_value_t < alpha else 'Not Significant'
        significance_u = 'Significant' if p_value_u < alpha else 'Not Significant'
    
        # Annotate the plot with test results
        ax.text(0.95, 0.55, f'T-test p-value: {p_value_t:.3e} ({significance_t})',
                 horizontalalignment='right', verticalalignment='top', transform=ax.transAxes,
                 fontsize=12, color='blue')
    
        ax.text(0.95, 0.50, f'Mann-Whitney U p-value: {p_value_u:.3e} ({significance_u})',
                 horizontalalignment='right', verticalalignment='top', transform=ax.transAxes,
                 fontsize=12, color='orange')
    
        # Customize the plot
        ax.set_xlabel('Connection Strength', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        #ax.set_title('Distribution of Connection Strengths for Different Time Differences', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    min1=100
    max1=800
    # Plot on each subplot
    plotdistribution(S_nonzero_A_before, time_diffs_A_before, min1, max1, ax=axs[0, 0])
    axs[0,0].set_title('S_A before training')
    plotdistribution(S_nonzero_B_before, time_diffs_B_before, min1, max1, ax=axs[0, 1])
    axs[0,1].set_title('S_B before training')
    plotdistribution(S_nonzero_A, time_diffs_A, min1, max1, ax=axs[1, 0])
    axs[1,0].set_title('S_A after training')
    plotdistribution(S_nonzero_B, time_diffs_B, min1, max1, ax=axs[1, 1])
    axs[1,1].set_title('S_B after training')
    
    # Adjust layout and display the plot
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(savepath,f"weight_distribution_density_{minInd}"),transparent=True,dpi=600)
    plt.show()
    
    
    
    
    
    # compare sequentiality index before filtering cells
    mean_range = 10
    datasets = ['A Before', 'A After', 'B Before', 'B After']
    categories = ['Peak Entropy', 'Temporal Sparsity', 'Sequentiality Index']
    indices = ['6s', '12s']
    
    # Compute indices for the first half (6s interval)
    PE_A_before, TS_A_before, Sql_A_before = Seq_ind(act_avg_A_before, mean_range)
    PE_A, TS_A, Sql_A = Seq_ind(act_avg_A, mean_range)
    PE_B_before, TS_B_before, Sql_B_before = Seq_ind(act_avg_B_before, mean_range)
    PE_B, TS_B, Sql_B = Seq_ind(act_avg_B, mean_range)
    
    data_first_half = np.array([
        [PE_A_before[0], TS_A_before[0], Sql_A_before[0]],
        [PE_A[0], TS_A[0], Sql_A[0]],
        [PE_B_before[0], TS_B_before[0], Sql_B_before[0]],
        [PE_B[0], TS_B[0], Sql_B[0]],
    ])
    
    data_second_half = np.array([
        [PE_A_before[1], TS_A_before[1], Sql_A_before[1]],
        [PE_A[1], TS_A[1], Sql_A[1]],
        [PE_B_before[1], TS_B_before[1], Sql_B_before[1]],
        [PE_B[1], TS_B[1], Sql_B[1]],
    ])
    
    # Prepare data for plotting
    data_category_half = {}
    for i, category in enumerate(categories):
        data_category_half[(category, '6s')] = data_first_half[:, i]
        data_category_half[(category, '12s')] = data_second_half[:, i]
    
    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 10),sharey='col')
    bar_width = 0.35
    
    for i, category in enumerate(categories):
        for j, half in enumerate(indices):
            ax = axs[j, i]
            data = data_category_half[(category, half)]
            x = np.arange(2)  # Positions for 'A' and 'B'
            x_before = x - bar_width / 2
            x_after = x + bar_width / 2
    
            # Extract data for 'A' and 'B', 'Before' and 'After'
            data_before = [data[0], data[2]]  # 'A Before', 'B Before'
            data_after = [data[1], data[3]]   # 'A After', 'B After'
    
            # Plot bars
            ax.bar(x_before, data_before, width=bar_width, label='Before')
            ax.bar(x_after, data_after, width=bar_width, label='After')
    
            # Set labels and titles
            ax.set_xticks(x)
            ax.set_xticklabels(['A', 'B'])
            ax.set_title(f'{category} - {half}')
            ax.set_ylabel('Values')
    
    
            # Add legend only to the first subplot
            if i == 0 and j == 0:
                ax.legend()
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(savepath,f"sequentiality_index_comparison_nofilter_{minInd}"),transparent=True,dpi=600)
    plt.show()
    
    
    
    # filter cells before calulating sequentiality index
    
    def z_score_with_zero_handling(A,dim=0):
        # Calculate mean and standard deviation along the first dimension (row-wise)
        mean_A = np.mean(A, axis=dim)
        std_A = np.std(A, axis=dim)
        # Avoid division by zero: if std_A is zero, set it to 1 to prevent invalid division
        std_A[std_A == 0] = 1
        # Calculate z-score
        z_scores = (A - mean_A) / std_A
        return z_scores
    
    def circular_selectivity(act_avg_A,start,stop):
        # act_avg_A is matrix with size (time, nCells)
        # this function calculates circular selectivity at the interval start to stop
        # make circular kernel
        act_avg=z_score_with_zero_handling(act_avg_A,dim=0)
        circker=np.exp(1j*np.linspace(0,2*np.pi,stop-start))/(stop-start)
        circker=circker[np.newaxis,:]
        selec=np.abs(np.matmul(circker,act_avg[start:stop,:]))
        return np.squeeze(selec)
    
    
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
    
    A_selec6=circular_selectivity(act_avg_A,0,min_dur)
    B_selec6=circular_selectivity(act_avg_B,0,min_dur)
    A_bselec6=circular_selectivity(act_avg_A_before,0,min_dur)
    B_bselec6=circular_selectivity(act_avg_B_before,0,min_dur)
    
    A_selec12=circular_selectivity(act_avg_A,min_dur,min_dur+max_dur)
    B_selec12=circular_selectivity(act_avg_B,min_dur,min_dur+max_dur)
    A_bselec12=circular_selectivity(act_avg_A_before,min_dur,min_dur+max_dur)
    B_bselec12=circular_selectivity(act_avg_B_before,min_dur,min_dur+max_dur)
    
    fig, axs = plt.subplots(2, 2, figsize=(10,10),sharex=True,sharey=True)
    axs[0,0].scatter(A_bselec6, A_bselec12)
    axs[0,1].scatter(B_bselec6, B_bselec12)
    axs[1,0].scatter(A_selec6, A_selec12)
    axs[1,1].scatter(B_selec6, B_selec12)
    axs[0,0].set_ylabel('Before training \n 12s interval ')
    axs[1,0].set_ylabel('After training \n12s interval')
    axs[1,0].set_xlabel('6s interval')
    axs[1,1].set_xlabel('6s interval')
    if save:
        plt.savefig(os.path.join(savepath,f"temporal_selectivity_scatter_{minInd}"),transparent=True,dpi=600)

    
    # filter cells    
    def filtercells(act_avg_A_before,act_avg_B_before,act_avg_A,act_avg_B,
                    A_bselec12,B_bselec12,A_selec12,B_selec12,
                    A_bselec6,B_bselec6,A_selec6,B_selec6,
                    start,stop,bound):
        if start<min_dur:
            A_bfilter=np.logical_and(A_bselec12+A_bselec6>bound , A_bselec6>0.05)
            B_bfilter=np.logical_and(B_bselec12+B_bselec6>bound , B_bselec6>0.05)
            A_filter=np.logical_and(A_selec12+A_selec6>bound , A_selec6>0.05)
            B_filter=np.logical_and(B_selec12+B_selec6>bound , B_selec6>0.05)
        else:
            A_bfilter=np.logical_and(A_bselec12+A_bselec6>bound , A_bselec12>0.05)
            B_bfilter=np.logical_and(B_bselec12+B_bselec6>bound , B_bselec12>0.05)
            A_filter=np.logical_and(A_selec12+A_selec6>bound , A_selec12>0.05)
            B_filter=np.logical_and(B_selec12+B_selec6>bound , B_selec12>0.05)
            
     
        act_avg_Ab2=act_avg_A_before[:,A_bfilter]
        act_avg_Bb2=act_avg_B_before[:,B_bfilter]
        act_avg_A2=act_avg_A[:,A_filter]
        act_avg_B2=act_avg_B[:,B_filter]
    

        act_avg_Ab2_sort=sort_and_scale(act_avg_Ab2,start,stop)
        act_avg_Bb2_sort=sort_and_scale(act_avg_Bb2,start,stop)
        act_avg_A2_sort=sort_and_scale(act_avg_A2,start,stop)
        act_avg_B2_sort=sort_and_scale(act_avg_B2,start,stop)
        
        return act_avg_Ab2_sort, act_avg_Bb2_sort, act_avg_A2_sort, act_avg_B2_sort
    
    
    # filter cells for 12s interval
    start=min_dur
    stop=min_dur+max_dur
    bound=0.5
    act_avg_Ab2_sort12, act_avg_Bb2_sort12, act_avg_A2_sort12, act_avg_B2_sort12=filtercells(act_avg_A_before,act_avg_B_before,act_avg_A,act_avg_B,
                                                                                    A_bselec12,B_bselec12,A_selec12,B_selec12,
                                                                                    A_bselec6,B_bselec6,A_selec6,B_selec6,
                                                                                    start,stop,bound)
    # filter cells for 6s interval
    start=0
    stop=min_dur
    act_avg_Ab2_sort6, act_avg_Bb2_sort6, act_avg_A2_sort6, act_avg_B2_sort6=filtercells(act_avg_A_before,act_avg_B_before,act_avg_A,act_avg_B,
                                                                                    A_bselec12,B_bselec12,A_selec12,B_selec12,
                                                                                    A_bselec6,B_bselec6,A_selec6,B_selec6,
                                                                                    start,stop,bound)
    
    # plot activities
    fig, axs = plt.subplots(2, 2, figsize=(12,12))
    axs[0,0].imshow(act_avg_Ab2_sort12.T, aspect='auto', cmap=parula,interpolation='none', vmin=0, vmax=1)
    axs[0,0].set_title('A before training')
    axs[0,1].imshow(act_avg_Bb2_sort12.T, aspect='auto', cmap=parula,interpolation='none', vmin=0, vmax=1)
    axs[0,1].set_title('B before training')
    axs[1,0].imshow(act_avg_A2_sort12.T, aspect='auto', cmap=parula,interpolation='none', vmin=0, vmax=1)
    axs[1,0].set_title('A after training')
    axs[1,1].imshow(act_avg_B2_sort12.T, aspect='auto', cmap=parula,interpolation='none', vmin=0, vmax=1)
    axs[1,1].set_title('B after training')
    if save:
        plt.savefig(os.path.join(savepath,f"sequentiality_comparison_imshow_maxdur_{minInd}"),transparent=True,dpi=600)
    
    
    fig, axs = plt.subplots(2, 2, figsize=(12,12))
    axs[0,0].imshow(act_avg_Ab2_sort6.T, aspect='auto', cmap=parula,interpolation='none', vmin=0, vmax=1)
    axs[0,0].set_title('A before training')
    axs[0,1].imshow(act_avg_Bb2_sort6.T, aspect='auto', cmap=parula,interpolation='none', vmin=0, vmax=1)
    axs[0,1].set_title('B before training')
    axs[1,0].imshow(act_avg_A2_sort6.T, aspect='auto', cmap=parula,interpolation='none', vmin=0, vmax=1)
    axs[1,0].set_title('A after training')
    axs[1,1].imshow(act_avg_B2_sort6.T, aspect='auto', cmap=parula,interpolation='none', vmin=0, vmax=1)
    axs[1,1].set_title('B after training')    
    if save:
        plt.savefig(os.path.join(savepath,f"sequentiality_comparison_imshow_mindur_{minInd}"),transparent=True,dpi=600)
    

    

    mean_range = 10
    datasets = ['A Before', 'A After', 'B Before', 'B After']
    categories = ['Peak Entropy', 'Temporal Sparsity', 'Sequentiality Index']
    indices = ['6s', '12s']
    
    # Compute indices for the first half (6s interval)
    PE_A_before, TS_A_before, Sql_A_before = Seq_ind(act_avg_Ab2_sort6, mean_range)
    PE_A, TS_A, Sql_A = Seq_ind(act_avg_A2_sort6, mean_range)
    PE_B_before, TS_B_before, Sql_B_before = Seq_ind(act_avg_Bb2_sort6, mean_range)
    PE_B, TS_B, Sql_B = Seq_ind(act_avg_B2_sort6, mean_range)
    
    data_first_half = np.array([
        [PE_A_before[0], TS_A_before[0], Sql_A_before[0]],
        [PE_A[0], TS_A[0], Sql_A[0]],
        [PE_B_before[0], TS_B_before[0], Sql_B_before[0]],
        [PE_B[0], TS_B[0], Sql_B[0]],
    ])
    
    # Compute indices for the second half (12s interval)
    PE_A_before, TS_A_before, Sql_A_before = Seq_ind(act_avg_Ab2_sort12, mean_range)
    PE_A, TS_A, Sql_A = Seq_ind(act_avg_A2_sort12, mean_range)
    PE_B_before, TS_B_before, Sql_B_before = Seq_ind(act_avg_Bb2_sort12, mean_range)
    PE_B, TS_B, Sql_B = Seq_ind(act_avg_B2_sort12, mean_range)
    
    data_second_half = np.array([
        [PE_A_before[1], TS_A_before[1], Sql_A_before[1]],
        [PE_A[1], TS_A[1], Sql_A[1]],
        [PE_B_before[1], TS_B_before[1], Sql_B_before[1]],
        [PE_B[1], TS_B[1], Sql_B[1]],
    ])
    
    # Prepare data for plotting
    data_category_half = {}
    for i, category in enumerate(categories):
        data_category_half[(category, '6s')] = data_first_half[:, i]
        data_category_half[(category, '12s')] = data_second_half[:, i]
    
    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 10),sharey='col')
    bar_width = 0.35
    
    for i, category in enumerate(categories):
        for j, half in enumerate(indices):
            ax = axs[j, i]
            data = data_category_half[(category, half)]
            x = np.arange(2)  # Positions for 'A' and 'B'
            x_before = x - bar_width / 2
            x_after = x + bar_width / 2
    
            # Extract data for 'A' and 'B', 'Before' and 'After'
            data_before = [data[0], data[2]]  # 'A Before', 'B Before'
            data_after = [data[1], data[3]]   # 'A After', 'B After'
    
            # Plot bars
            ax.bar(x_before, data_before, width=bar_width, label='Before')
            ax.bar(x_after, data_after, width=bar_width, label='After')
    
            # Set labels and titles
            ax.set_xticks(x)
            ax.set_xticklabels(['A', 'B'])
            ax.set_title(f'{category} - {half}')
            ax.set_ylabel('Values')
    
    
            # Add legend only to the first subplot
            if i == 0 and j == 0:
                ax.legend()
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(savepath,f"sequentiality_index_comparison_filter_{minInd}"),transparent=True,dpi=600)
    
    plt.show()
    
    
    
    # compare sequential activity with no cell filtering
    datasets = ['A Before', 'A After', 'B Before', 'B After']
    categories = ['Peak Entropy', 'Temporal Sparsity', 'Sequentiality Index']
    indices = ['6s', '12s']
    
    # Compute indices for the first half (6s interval)
    start=0
    stop=min_dur
    PE_A_before, TS_A_before, Sql_A_before = Seq_ind(sort_and_scale(act_avg_A_before,start,stop), mean_range)
    PE_A, TS_A, Sql_A = Seq_ind(sort_and_scale(act_avg_A,start,stop), mean_range)
    PE_B_before, TS_B_before, Sql_B_before = Seq_ind(sort_and_scale(act_avg_B_before,start,stop), mean_range)
    PE_B, TS_B, Sql_B = Seq_ind(sort_and_scale(act_avg_B,start,stop), mean_range)
    
    data_first_half = np.array([
        [PE_A_before[0], TS_A_before[0], Sql_A_before[0]],
        [PE_A[0], TS_A[0], Sql_A[0]],
        [PE_B_before[0], TS_B_before[0], Sql_B_before[0]],
        [PE_B[0], TS_B[0], Sql_B[0]],
    ])
    
    # Compute indices for the second half (12s interval)
    start=min_dur
    stop=min_dur+max_dur
    PE_A_before, TS_A_before, Sql_A_before = Seq_ind(sort_and_scale(act_avg_A_before,start,stop), mean_range)
    PE_A, TS_A, Sql_A = Seq_ind(sort_and_scale(act_avg_A,start,stop), mean_range)
    PE_B_before, TS_B_before, Sql_B_before = Seq_ind(sort_and_scale(act_avg_B_before,start,stop), mean_range)
    PE_B, TS_B, Sql_B = Seq_ind(sort_and_scale(act_avg_B,start,stop), mean_range)
    
    data_second_half = np.array([
        [PE_A_before[1], TS_A_before[1], Sql_A_before[1]],
        [PE_A[1], TS_A[1], Sql_A[1]],
        [PE_B_before[1], TS_B_before[1], Sql_B_before[1]],
        [PE_B[1], TS_B[1], Sql_B[1]],
    ])
    
    # Prepare data for plotting
    data_category_half = {}
    for i, category in enumerate(categories):
        data_category_half[(category, '6s')] = data_first_half[:, i]
        data_category_half[(category, '12s')] = data_second_half[:, i]
    
    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 10),sharey='col')
    bar_width = 0.35
    
    for i, category in enumerate(categories):
        for j, half in enumerate(indices):
            ax = axs[j, i]
            data = data_category_half[(category, half)]
            x = np.arange(2)  # Positions for 'A' and 'B'
            x_before = x - bar_width / 2
            x_after = x + bar_width / 2
    
            # Extract data for 'A' and 'B', 'Before' and 'After'
            data_before = [data[0], data[2]]  # 'A Before', 'B Before'
            data_after = [data[1], data[3]]   # 'A After', 'B After'
    
            # Plot bars
            ax.bar(x_before, data_before, width=bar_width, label='Before')
            ax.bar(x_after, data_after, width=bar_width, label='After')
    
            # Set labels and titles
            ax.set_xticks(x)
            ax.set_xticklabels(['A', 'B'])
            ax.set_title(f'{category} - {half}')
            ax.set_ylabel('Values')
    
    
            # Add legend only to the first subplot
            if i == 0 and j == 0:
                ax.legend()
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(savepath,f"sequentiality_index_comparison_nofilter_scaled_{minInd}"),transparent=True,dpi=600)

    plt.show()
        
        
    
#%%

pert_time=np.array([[In_ons[0,1]+int(3000/dt),In_ons[0,2]+int(3000/dt)]
                    ,[In_ons[1,1]+int(3000/dt),In_ons[1,2]+int(3000/dt)]])
pert_time=np.array([[In_ons[0,2]+int(1000/dt),In_ons[0,2]+int(3000/dt)]
                    ,[In_ons[1,2]+int(1000/dt),In_ons[1,2]+int(3000/dt)]
                    ,[In_ons[2,4]+int(1000/dt),In_ons[2,4]+int(3000/dt)]
                    ,[In_ons[3,4]+int(1000/dt),In_ons[3,4]+int(3000/dt)]])
pert_time=np.array([[In_ons[0,2]+int(100/dt),In_ons[0,3]+int(100/dt)]
                    ,[In_ons[1,2]+int(100/dt),In_ons[1,3]+int(100/dt)]
                    ,[In_ons[2,4]+int(100/dt),In_ons[2,5]+int(100/dt)]
                    ,[In_ons[3,4]+int(100/dt),In_ons[3,5]+int(100/dt)]])
pert_time=np.array([[In_ons[0,2]-int(3000/dt),In_ons[0,3]-int(3000/dt)]
                    ,[In_ons[1,2]-int(3000/dt),In_ons[1,3]-int(3000/dt)]
                    ,[In_ons[2,4]-int(3000/dt),In_ons[2,5]-int(3000/dt)]
                    ,[In_ons[3,4]-int(3000/dt),In_ons[3,5]-int(3000/dt)]])
pert_time=np.array([[In_ons[0,1]+int(3000/dt),In_ons[0,2]+int(3000/dt)]
                    ,[In_ons[1,1]+int(3000/dt),In_ons[1,2]+int(3000/dt)]
                    ,[In_ons[2,3]+int(3000/dt),In_ons[2,4]+int(3000/dt)]
                    ,[In_ons[3,3]+int(3000/dt),In_ons[3,4]+int(3000/dt)]])
#pert_time=np.array([[0,0],[0,0]])



# %% decode time from neural data
from Confmatrix import confmat, confscore
outputs = [layer.output for layer in model.layers[1:]]  # Exclude the input layer
activity_model = Model(inputs=model.input, outputs=outputs)
trial_num=8
# Get the output and activities of all layers for the new input data
x, y, In_ons=makeInOut(sample_size,trial_num,inputdur,nInput,min_dur,max_dur,dt)
#xnew=sum(xnew,axis=2)
output_and_activities = activity_model.predict(x)
activities_A = output_and_activities[0]  # Activities of all intermediate layers
activities_B=output_and_activities[1]
act_avg_A=avgAct(activities_A,In_ons,min_dur,max_dur)
act_avg_B=avgAct(activities_B,In_ons,min_dur,max_dur)
act_stack_A=Act_2dsort(activities_A,In_ons,min_dur,max_dur)
act_stack_B=Act_2dsort(activities_B,In_ons,min_dur,max_dur)



# make classifying classes
Class_per_sec=2
classleng=int(1000/(dt*Class_per_sec))   #amount of step equaling 1 class
class_per_trial=int((min_dur+max_dur)/classleng)
class_A=np.arange(1,1+class_per_trial)
class_A=np.repeat(class_A,classleng)
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


# make testing data
x, y, In_ons=makeInOut(sample_size,trial_num,inputdur,nInput,min_dur,max_dur,dt)
output_and_activities = activity_model.predict(x)
activities_A = output_and_activities[0]  # Activities of all intermediate layers
activities_B=output_and_activities[1]
act_stack_A=Act_2dsort(activities_A,In_ons,min_dur,max_dur)
act_stack_B=Act_2dsort(activities_B,In_ons,min_dur,max_dur)
#act_stack_A=act_stack_A+np.random.normal(loc=0,scale=0.00,size=np.shape(act_stack_A))# add noise if necessary
proj_A_test=pca_A.transform(act_stack_A)
proj_B_test=pca_B.transform(act_stack_B)
trial_rep_A=int(np.shape(act_stack_A)[0]/(min_dur+max_dur))
class_A_test=np.tile(class_A,(trial_rep_A))
trial_rep_B=int(np.shape(act_stack_B)[0]/(min_dur+max_dur))
class_B_test=np.tile(class_A,(trial_rep_B))


conf_A=np.zeros((class_per_trial,class_per_trial,1))
conf_B=np.zeros((class_per_trial,class_per_trial,1))
alldim=[3,5,10,15,20,50,100,200,512]
n_subplots = len(alldim)
for i  in range(n_subplots):
    Dim=alldim[i];
    # fit classifier with training data and test its accuracy
    #for RNN A
    clf_A=RandomForestClassifier(n_estimators=100,bootstrap=True,n_jobs=-1)
    clf_A.fit(proj_A_train[:,:Dim],class_A_train)
    pred_A=clf_A.predict(proj_A_test[:,:Dim])
    confmatA=confmat(class_A_test,pred_A)
    confmatA=confmatA[:,:,np.newaxis]
    conf_A=np.concatenate((conf_A,confmatA),axis=2)
    
    #for RNN B
    clf_B=RandomForestClassifier(n_estimators=100,bootstrap=True,n_jobs=-1)
    clf_B.fit(proj_B_train[:,:Dim],class_B_train)
    pred_B=clf_B.predict(proj_B_test[:,:Dim])
    confmatB=confmat(class_B_test,pred_B)
    confmatB=confmatB[:,:,np.newaxis]
    conf_B=np.concatenate((conf_B,confmatB),axis=2)
    print(f'iteration {i} out of {n_subplots-1}')



# figure output
conf_1=conf_A[:,:,1:]
Score_A=confscore(conf_A[:,:,1:], 1)
conf_2=conf_B[:,:,1:]
Score_B=confscore(conf_B[:,:,1:], 1)


fig, axs = plt.subplots(1,2)
axs[0].plot(alldim,Score_A)
axs[0].set_xscale('log')
axs[0].set_title('accuracy A')
axs[1].plot(alldim,Score_B)
axs[1].set_xscale('log')
axs[1].set_title('accuracy B')
if save:
    plt.savefig(os.path.join(savepath,f"Decoded_accuracy"),transparent=True,dpi=400)
plt.show



#confusion matrix for A
max_columns = 4
n_rows = (n_subplots + max_columns - 1) // max_columns  # Calculate the number of rows needed
fig, axs = plt.subplots(n_rows, max_columns, figsize=(15, 3 * n_rows))
# Flatten the axs array for easy iteration if it's 2D
axs = axs.flatten()
for i in range(n_subplots):
    axs[i].imshow(conf_1[:,:,i], aspect='auto', cmap='viridis', interpolation='none')
    axs[i].set_title(f'A: Dimension {alldim[i]}')
# Hide any unused subplots
for i in range(n_subplots, len(axs)):
    fig.delaxes(axs[i])
fig.suptitle('Decoded accuracy A')
plt.tight_layout()
if save:
    plt.savefig(os.path.join(savepath,f"Decoded_confmat_A"),transparent=True,dpi=400)
plt.show()


#confusion matrix for B
max_columns = 4
n_rows = (n_subplots + max_columns - 1) // max_columns  # Calculate the number of rows needed
fig, axs = plt.subplots(n_rows, max_columns, figsize=(15, 3 * n_rows))
# Flatten the axs array for easy iteration if it's 2D
axs = axs.flatten()
for i in range(n_subplots):
    axs[i].imshow(conf_2[:,:,i], aspect='auto', cmap='viridis', interpolation='none')
    axs[i].set_title(f'A: Dimension {alldim[i]}')
# Hide any unused subplots
for i in range(n_subplots, len(axs)):
    fig.delaxes(axs[i])
fig.suptitle('Decoded accuracy B')

plt.tight_layout()
if save:
    plt.savefig(os.path.join(savepath,f"Decoded_confmat_B"),transparent=True,dpi=400)
plt.show()

#%% pertrubation with dedicated model
from RNNcustom_2_perturb import RNNCustom2FixPerturb
def build_model_perturb(pert_ind, pert_state):
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




sample_size=4
x, y, In_ons=makeInOut(sample_size,trial_num,inputdur,nInput,min_dur,max_dur,dt)
pert_state=1 # 0 to perturb RNN A and 1 to perturb RNN B
pert_ind=np.array([[In_ons[0,2]+int(3000/dt),In_ons[0,3]+int(3000/dt)]
                    ,[In_ons[1,2]+int(3000/dt),In_ons[1,3]+int(3000/dt)]
                    ,[In_ons[2,4]+int(3000/dt),In_ons[2,5]+int(3000/dt)]
                    ,[In_ons[3,4]+int(3000/dt),In_ons[3,5]+int(3000/dt)]])
pert_ind=np.array([[In_ons[0,2]+int(1000/dt),In_ons[0,2]+int(3000/dt)]
                    ,[In_ons[1,2]+int(1000/dt),In_ons[1,2]+int(3000/dt)]
                    ,[In_ons[2,4]+int(1000/dt),In_ons[2,4]+int(3000/dt)]
                    ,[In_ons[3,4]+int(1000/dt),In_ons[3,4]+int(3000/dt)]])
model2=build_model_perturb(pert_ind, pert_state)
model2.set_weights(model.get_weights())
predictions = model2.predict(x)

fig, axs = plt.subplots(sample_size,1, figsize=(14, 8))
Line=[None]*4
for ss in range(sample_size):
    axs[ss].plot(y[ss, :, 0], color='blue',label='Target 0', alpha=0.5)
    axs[ss].plot(predictions[ss, :, 0], color='green',label='Prediction_A 0',alpha=0.5)
    axs[ss].plot(predictions[ss, :, 2], color='turquoise',label='Prediction_B 0',alpha=0.5)
    axs[ss].plot(y[ss, :, 1], color='red',label='Target 1',alpha=0.5)
    axs[ss].plot(predictions[ss, :, 1], color='orangered',label='Prediction_A 1',alpha=0.5)
    axs[ss].plot(predictions[ss, :, 3], color='gold',label='Prediction_B 1',alpha=0.5)
    axs[ss].axvline(pert_ind[ss, 0], ymin=0.7, ymax=1)
    axs[ss].axvline(pert_ind[ss, 1], ymin=0.7, ymax=1)
    axs[ss].set_title(f'Plot {i+1}')
plt.suptitle('')
# Adjusting layout
handles, labels = axs[0,].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left')
plt.tight_layout()

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
sample_size=4
step=18 # number of bins over min_dur+max_dur duration
bin_width=(min_dur+max_dur)//step
errorMat1=np.zeros([step,step]) # error of prediction 1
errorMat2=np.zeros([step,step])
count_mat=np.ones([step,step])

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
        model2=build_model_perturb(pert_ind, pert_state)
        model2.set_weights(model.get_weights())
        predictions = model2.predict(x)
        pred1, pred2=pred_diff(predictions,pert_ind)
        
        errorMat1[step11,step12]+=0.5*(pred1[0]+pred1[2])
        errorMat2[step11,step12]+=0.5*(pred2[0]+pred2[2])
        errorMat1[step21,step22]+=0.5*(pred1[1]+pred1[3])
        errorMat2[step21,step22]+=0.5*(pred2[1]+pred2[3])
        count_mat[step11,step12]+=1
        count_mat[step21,step22]+=1
    print(f"{i} out of {step}")

errorMat1/=count_mat
errorMat2/=count_mat


# display results
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im1 = axes[0].imshow(errorMat1, cmap='viridis')
axes[0].set_title('Prediction 1')
fig.colorbar(im1, ax=axes[0])
im2 = axes[1].imshow(errorMat2, cmap='viridis')
axes[1].set_title('Prediction 2')
fig.colorbar(im2, ax=axes[1])
fig.suptitle('MSE after perturbation', fontsize=16)
plt.tight_layout()
if save:
    plt.savefig(os.path.join(savepath,f"Perturbation_synchronicity"),transparent=True,dpi=400)
plt.show()




fig, axs = plt.subplots(sample_size,1, figsize=(14, 8))
Line=[None]*4
for ss in range(sample_size):
    axs[ss].plot(y[ss, :, 0], color='blue',label='Target 0', alpha=0.5)
    axs[ss].plot(predictions[ss, :, 0], color='green',label='Prediction_A 0',alpha=0.5)
    axs[ss].plot(predictions[ss, :, 2], color='turquoise',label='Prediction_B 0',alpha=0.5)
    axs[ss].plot(y[ss, :, 1], color='red',label='Target 1',alpha=0.5)
    axs[ss].plot(predictions[ss, :, 1], color='orangered',label='Prediction_A 1',alpha=0.5)
    axs[ss].plot(predictions[ss, :, 3], color='gold',label='Prediction_B 1',alpha=0.5)
    axs[ss].axvline(pert_ind[ss, 0], ymin=0.7, ymax=1)
    axs[ss].axvline(pert_ind[ss, 1], ymin=0.7, ymax=1)
    axs[ss].set_title(f'Plot {i+1}')
plt.suptitle('')
# Adjusting layout
handles, labels = axs[0,].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left')
plt.tight_layout()





#%% perturb and decode
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

sample_size=6
trial_num=8
pert_state=0 # 0 to perturb RNN A and 1 to perturb RNN B

def perturb_and_decode(trial1,ind1,trial2,ind2,pert_state,order):
    pert_state=pert_state
    x, y, In_ons=makeInOut_sameint(sample_size,trial_num,inputdur,nInput,min_dur,max_dur,dt,order)
    pert_ind=make_pertind(In_ons,trial1,ind1,trial2,ind2)

    #pert_ind=np.zeros((4,2))
    model2=build_model_perturb(pert_ind, pert_state)
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
    #create decoder
    Dim=15
    clf_A=RandomForestClassifier(n_estimators=100,bootstrap=True,n_jobs=-1)
    clf_A.fit(proj_A_train[:,:Dim],class_A_train)
    
    clf_B=RandomForestClassifier(n_estimators=100,bootstrap=True,n_jobs=-1)
    clf_B.fit(proj_B_train[:,:Dim],class_B_train)
    
    
    for i in np.arange(np.shape(actpart_A)[2]):
        #dimension reduction
        proj_A=pca_A.transform(actpart_A[:,:,i])
        proj_B=pca_B.transform(actpart_B[:,:,i])
        
        #decode    
        pred_A[:,i]=clf_A.predict(proj_A[:,:Dim])
        pred_B[:,i]=clf_B.predict(proj_B[:,:Dim])
    return pred_A, pred_B, predictions2, pert_ind_2, In_ons_2
    

# take circular mean of the prediction
option=0 #0 for circular, 1 for mean
trial1=3
trial2=4
time_1=100
time_2=100
order=1
predavg_A=[]
predavg_B=[]
pred_Aall=[]
pred_Ball=[]
predictions2=[]
pert_ind_2=[]
In_ons_2=[]

#perturB A
pert_state=0
pred_A, pred_B, A_predictions2, pert_ind_3,In_ons_a=perturb_and_decode(trial1,time_1,trial2,time_2,pert_state,order)
predictions2.append(A_predictions2)
pert_ind_2.append(pert_ind_3)
In_ons_2.append(In_ons_a)
if option==0:
    predavg_A.append(scipy.stats.circmean(pred_A,high=class_per_trial,low=1,axis=1))
    predavg_B.append(scipy.stats.circmean(pred_B,high=class_per_trial,low=1,axis=1))
else:
    predavg_A.append(np.mean(pred_A,axis=1))
    predavg_B.append(np.mean(pred_B,axis=1))
        
pred_Aall.append(pred_A)
pred_Ball.append(pred_B)

#perturb B
pert_state=1
pred_A, pred_B, B_predictions2, pert_ind_3,In_ons_a=perturb_and_decode(trial1,time_1,trial2,time_2,pert_state,order)
predictions2.append(B_predictions2)
pert_ind_2.append(pert_ind_3)
In_ons_2.append(In_ons_a)
if option==0:
    predavg_A.append(scipy.stats.circmean(pred_A,high=class_per_trial,low=1,axis=1))
    predavg_B.append(scipy.stats.circmean(pred_B,high=class_per_trial,low=1,axis=1))
else:
    predavg_A.append(np.mean(pred_A,axis=1))
    predavg_B.append(np.mean(pred_B,axis=1))
pred_Aall.append(pred_A)
pred_Ball.append(pred_B)


fig, axs = plt.subplots(2,2,sharex='col',figsize=(12, 8))
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
    axs[1,ss].plot(predavg_B[ss],color='#1f77b4',label='RNN_B')
    
    axs[1,ss].set_title(f'Decoded results')
    axs[1, ss].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')  # Place legend outside

plt.suptitle('')
# Adjusting layout
plt.tight_layout()
if save:
    plt.savefig(os.path.join(savepath,f"Perturb_decode_{time_1}_{time_2}_{order}_{option}"),transparent=True,dpi=400)
plt.show()

# %% visualize activity under various conditions
outputs = [layer.output for layer in model.layers[1:]]  # Exclude the input layer
activity_model = Model(inputs=model.input, outputs=outputs)


x, y, In_ons=makeInOut(sample_size,trial_num+2,inputdur,nInput,min_dur,max_dur,dt)
#xnew=sum(xnew,axis=2)
output_and_activities = activity_model.predict(x)
activities = output_and_activities[0]  # Activities of all intermediate layers
act_avg=avgAct(activities,In_ons,min_dur,max_dur)
max_range=[min_dur,max_dur+min_dur] # time range to choose max firing time
input_activities= act_avg.copy()
#input_activities=activities[0,In_ons[0,1]:In_ons[0,1]+min_dur+max_dur,:]
max_times = np.argmax(input_activities[max_range[0]:max_range[1], :], axis=0)
# Sort units based on the time of maximum activity
sorted_units = np.argsort(max_times)
act18s=input_activities[:,sorted_units]
act18s=scipy.stats.zscore(act18s, axis=0)


# perform pca
# Perform PCA on the first dataset
pca = PCA(n_components=3)
pca_score = pca.fit_transform(act18s)



# get mean and std for each unit so we can zscore each data identically
int_list=[min_dur,max_dur,min_dur,max_dur,int(testdur[i]/dt),max_dur]
x,y,In_ons=makeInOutTest_exp(int_list,8,inputdur,nInput,min_dur,max_dur,dt)
output_and_activities = activity_model.predict(x)
activities = output_and_activities[0]  # Activities of all intermediate layers
activities=activities[:,-1-np.sum(int_list[-3:],axis=None):,:]
avg_1,avg_2,avg_3=avgAct_lastfew(activities,int_list[-3:])
avg_all=np.concatenate((avg_1, avg_2, avg_3),axis=0)
Cellmean=np.mean(avg_all,axis=0)
Cellsd=np.std(avg_all,axis=0)



testdur=[500,1500,3000,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,10500,11000,11500,12000,13500,15000,18000]
#testdur=[1500,3000,4500,6000,7500,9000,10500,12000,13500,15000,18000]
#testdur=[3000,6000,9000,12000,18000]
#testdur=[3000,6000]
colormap=plt.cm.get_cmap('turbo',len(testdur))
colormap=plt.cm.get_cmap('viridis',len(testdur))
classification=np.zeros((len(testdur),2))
Cellact_A=[] # activity where RNN thought was in 12s interval
Cellact_B=[]
fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
#fig = plt.figure()
#ax = fig.add_subplot(1,2, projection='3d')
for i in range(len(testdur)):
    int_list=[min_dur,max_dur,min_dur,max_dur,int(testdur[i]/dt),max_dur]
    x,y,In_ons=makeInOutTest_exp(int_list,8,inputdur,nInput,min_dur,max_dur,dt)
    output_and_activities = activity_model.predict(x)
    activities = output_and_activities[0]  # Activities of all intermediate layers
    output1=np.mean(output_and_activities[1][:,:,0],axis=0)
    activities=activities[:,-1-np.sum(int_list[-3:],axis=None):,:]
    avg_1,avg_2,avg_3=avgAct_lastfew(activities,int_list[-3:])
    avg_all=np.concatenate((avg_1, avg_2, avg_3),axis=0)
    
    # zscore avg_all
    avg_all=np.divide((avg_all-Cellmean),Cellsd)
    
    

    
    Cellact=avg_all[:,sorted_units]
    #Cellact=scipy.stats.zscore(Cellact,axis=0)
    
    # perform pca
    proj_cell = pca.transform(Cellact)
    k_a=int_list[-3]-1
    k_b=int_list[-3]+int_list[-2]-1
    #ax.scatter(proj_cell[:, 0], proj_cell[:, 1], proj_cell[:, 2],
          # color=colormap(i), label=f'{testdur[i]}')
        
    if output1[-int(5000/dt)]-y[0,-int(5000/dt),0]<0.21:
        Cellact_A.append(Cellact)
        classification[i,:]=[testdur[i],0]
    else:
        Cellact_B.append(Cellact)
        classification[i,:]=[testdur[i],1]
        
    
    linewidth=1
    mintime=k_a
    maxtime=k_b
    if testdur[i]==6000:
        linewidth=2
       # ax.scatter(proj_cell[0, 0], proj_cell[0, 1], proj_cell[0, 2],
         #  color='green', s=100) 
    elif testdur[i]==12000:
        linewidth=2
    
    if classification[i,1]==0:
        Color='blue'
    else:
        Color='green'
    
    ax[0].plot(proj_cell[mintime:maxtime, 0], proj_cell[mintime:maxtime, 1], proj_cell[mintime:maxtime, 2],
           color=colormap(i),linewidth=linewidth, label=f'{testdur[i]}')
    #ax.scatter(proj_cell[0, 0], proj_cell[0, 1], proj_cell[0, 2],
           #color='green', s=100)    
    ax[0].scatter(proj_cell[k_a, 0], proj_cell[k_a, 1], proj_cell[k_a, 2],
           color='red', s=100)
    ax[0].scatter(proj_cell[k_b, 0], proj_cell[k_b, 1], proj_cell[k_b, 2],
           color=colormap(i), s=100)

    mintime=k_b
    maxtime=-1
    ax[1].plot(proj_cell[mintime:maxtime, 0], proj_cell[mintime:maxtime, 1], proj_cell[mintime:maxtime, 2],
           color=colormap(i),linewidth=linewidth, label=f'{testdur[i]}')
    #ax.scatter(proj_cell[0, 0], proj_cell[0, 1], proj_cell[0, 2],
           #color='green', s=100)    
    ax[1].scatter(proj_cell[k_b, 0], proj_cell[k_b, 1], proj_cell[k_b, 2],
           color=Color, s=100)
    
    

ax[1].legend()


# dimensional reduction usging nFDA
method='nFDA'
A3d=np.zeros((max_dur+1,nUnit,len(Cellact_A)))
B3d=np.zeros((max_dur+1,nUnit,len(Cellact_B)))
AtA=np.zeros((nUnit,nUnit))
BtB=np.zeros((nUnit,nUnit))
for i in range(len(Cellact_A)):
    a3=np.array(Cellact_A[i][-max_dur-1:,:])
    A3d[:,:,i]=a3.copy()
for i in range(len(Cellact_B)):
    b3=np.array(Cellact_B[i][-max_dur-1:,:])
    B3d[:,:,i]=b3.copy()

A3dt=np.transpose(A3d,(2,1,0))
B3dt=np.transpose(B3d,(2,1,0))
A3dt-=np.mean(A3dt,axis=0)
B3dt-=np.mean(B3dt,axis=0)
for i in range(np.shape(A3dt)[2]):
    a3=A3dt[:,:,i]-np.mean(A3dt[:,:,i],axis=0)
    AtA+=np.matmul(np.transpose(a3),a3)
for i in range(np.shape(B3dt)[2]):
    b3=B3dt[:,:,i]-np.mean(B3dt[:,:,i],axis=0)
    BtB+=np.matmul(np.transpose(b3),b3)


AtA/=np.trace(AtA)
BtB/=np.trace(BtB)
CtC=AtA+BtB
CtC/=np.trace(CtC)
D=np.mean(A3d[:min_dur,:,:],axis=2)-np.mean(B3d[:min_dur,:,:],axis=2)
D-=np.mean(D,axis=0)
D=np.matmul(np.transpose(D),D)
D/=np.trace(D)


if method=='nFDA':
    Cof=0.8
    E=(Cof)*D-(1-Cof)*CtC
    eigvalue,vecCoeff=np.linalg.eigh(0.5*(E+np.transpose(E)))
else:
    eigvalue,vecCoeff=scipy.linalg.eigh(0.5*(D+np.transpose(D)),0.5*(CtC+np.transpose(CtC)))

vecCoeff=vecCoeff[:,-1:-4:-1]


fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
#testdur=[5000,6000]
#testdur=[500,1500,3000,4000,4500,5000,5500,6000,7500,8000,8500,9000,9500,10000,10500,11000,11500,12000,13500,15000,18000]
colormap=plt.cm.get_cmap('viridis',len(testdur))
for i in range(len(testdur)):
    int_list=[min_dur,max_dur,min_dur,max_dur,int(testdur[i]/dt),max_dur]
    x,y,In_ons=makeInOutTest_exp(int_list,8,inputdur,nInput,min_dur,max_dur,dt)
    output_and_activities = activity_model.predict(x)
    activities = output_and_activities[0]  # Activities of all intermediate layers
    output1=np.mean(output_and_activities[1][:,:,0],axis=0)
    activities=activities[:,-1-np.sum(int_list[-3:],axis=None):,:]
    avg_1,avg_2,avg_3=avgAct_lastfew(activities,int_list[-3:])
    avg_all=np.concatenate((avg_1, avg_2, avg_3),axis=0)
    
    

    
    Cellact=avg_all[:,sorted_units]
    Cellact=scipy.stats.zscore(Cellact,axis=0)
    proj_cell = np.matmul(Cellact,vecCoeff)
    k_a=int_list[-3]-1
    k_b=int_list[-3]+int_list[-2]-1
    #ax.scatter(proj_cell[:, 0], proj_cell[:, 1], proj_cell[:, 2],
          # color=colormap(i), label=f'{testdur[i]}')
        
    
    linewidth=1
    mintime=k_a
    maxtime=k_b
    if testdur[i]==6000:
        linewidth=2
       # ax.scatter(proj_cell[0, 0], proj_cell[0, 1], proj_cell[0, 2],
         #  color='green', s=100) 
    elif testdur[i]==12000:
        linewidth=2
    
    if classification[i,1]==0:
        Color='blue'
    else:
        Color='green'
    
    ax[0].plot(proj_cell[mintime:maxtime, 0], proj_cell[mintime:maxtime, 1], proj_cell[mintime:maxtime, 2],
           color=colormap(i),linewidth=linewidth, label=f'{testdur[i]}')
    #ax.scatter(proj_cell[0, 0], proj_cell[0, 1], proj_cell[0, 2],
           #color='green', s=100)    
    ax[0].scatter(proj_cell[k_a, 0], proj_cell[k_a, 1], proj_cell[k_a, 2],
           color='red', s=100)
    ax[0].scatter(proj_cell[k_b, 0], proj_cell[k_b, 1], proj_cell[k_b, 2],
           color=colormap(i), s=100)

    mintime=k_b
    maxtime=-1
    ax[1].plot(proj_cell[mintime:maxtime, 0], proj_cell[mintime:maxtime, 1], proj_cell[mintime:maxtime, 2],
           color=colormap(i),linewidth=linewidth, label=f'{testdur[i]}')
    #ax.scatter(proj_cell[0, 0], proj_cell[0, 1], proj_cell[0, 2],
           #color='green', s=100)    
    ax[1].scatter(proj_cell[k_b, 0], proj_cell[k_b, 1], proj_cell[k_b, 2],
           color=Color, s=100)
    
#ax.set_xlabel('Principal Component 1')
#ax.set_ylabel('Principal Component 2')
#ax.set_zlabel('Principal Component 3')
#ax.set_title('nFDA 3D Plot')
ax[1].legend()






# slightly modified
method='nFDA'
A3d=np.zeros((max_dur+30,nUnit,len(Cellact_A)))
B3d=np.zeros((max_dur+30,nUnit,len(Cellact_B)))
AtA=np.zeros((nUnit,nUnit))
BtB=np.zeros((nUnit,nUnit))
for i in range(len(Cellact_A)):
    a3=np.array(Cellact_A[i][-max_dur-30:,:])
    A3d[:,:,i]=a3.copy()
for i in range(len(Cellact_B)):
    b3=np.array(Cellact_B[i][-max_dur-30:,:])
    B3d[:,:,i]=b3.copy()

A3dt=np.transpose(A3d[30:,:,:],(2,1,0))
B3dt=np.transpose(B3d[30:,:,:],(2,1,0))
A3dt-=np.mean(A3dt,axis=0)
B3dt-=np.mean(B3dt,axis=0)
for i in range(np.shape(A3dt)[2]):
    a3=A3dt[:,:,i]-np.mean(A3dt[:,:,i],axis=0)
    AtA+=np.matmul(np.transpose(a3),a3)
for i in range(np.shape(B3dt)[2]):
    b3=B3dt[:,:,i]-np.mean(B3dt[:,:,i],axis=0)
    BtB+=np.matmul(np.transpose(b3),b3)


AtA/=np.trace(AtA)
BtB/=np.trace(BtB)
CtC=AtA+BtB
CtC/=np.trace(CtC)
D=np.mean(A3d[:30,:,:],axis=2)-np.mean(B3d[:30,:,:],axis=2)
#D-=np.mean(D,axis=0)
D=np.matmul(np.transpose(D),D)
D/=np.trace(D)


if method=='nFDA':
    Cof=0.8
    E=(Cof)*D-(1-Cof)*CtC
    eigvalue,vecCoeff=np.linalg.eigh(0.5*(E+np.transpose(E)))
else:
    eigvalue,vecCoeff=scipy.linalg.eigh(0.5*(D+np.transpose(D)),0.5*(CtC+np.transpose(CtC)))

vecCoeff=vecCoeff[:,-1:-4:-1]


fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
#testdur=[5000,6000]
#testdur=[500,1500,3000,4000,4500,5000,5500,6000,7500,8000,8500,9000,9500,10000,10500,11000,11500,12000,13500,15000,18000]
colormap=plt.cm.get_cmap('viridis',len(testdur))
for i in range(len(testdur)):
    int_list=[min_dur,max_dur,min_dur,max_dur,int(testdur[i]/dt),max_dur]
    x,y,In_ons=makeInOutTest_exp(int_list,8,inputdur,nInput,min_dur,max_dur,dt)
    output_and_activities = activity_model.predict(x)
    activities = output_and_activities[0]  # Activities of all intermediate layers
    output1=np.mean(output_and_activities[1][:,:,0],axis=0)
    activities=activities[:,-1-np.sum(int_list[-3:],axis=None):,:]
    avg_1,avg_2,avg_3=avgAct_lastfew(activities,int_list[-3:])
    avg_all=np.concatenate((avg_1, avg_2, avg_3),axis=0)
    
    

    
    Cellact=avg_all[:,sorted_units]
    Cellact=scipy.stats.zscore(Cellact,axis=0)
    proj_cell = np.matmul(Cellact,vecCoeff)
    k_a=int_list[-3]-1
    k_b=int_list[-3]+int_list[-2]-1
    #ax.scatter(proj_cell[:, 0], proj_cell[:, 1], proj_cell[:, 2],
          # color=colormap(i), label=f'{testdur[i]}')
        
    
    linewidth=1
    mintime=k_a
    maxtime=k_b
    if testdur[i]==6000:
        linewidth=2
       # ax.scatter(proj_cell[0, 0], proj_cell[0, 1], proj_cell[0, 2],
         #  color='green', s=100) 
    elif testdur[i]==12000:
        linewidth=2
    
    if classification[i,1]==0:
        Color='blue'
    else:
        Color='green'
    
    ax[0].plot(proj_cell[mintime:maxtime, 0], proj_cell[mintime:maxtime, 1], proj_cell[mintime:maxtime, 2],
           color=colormap(i),linewidth=linewidth, label=f'{testdur[i]}')
    #ax.scatter(proj_cell[0, 0], proj_cell[0, 1], proj_cell[0, 2],
           #color='green', s=100)    
    ax[0].scatter(proj_cell[k_a, 0], proj_cell[k_a, 1], proj_cell[k_a, 2],
           color='red', s=100)
    ax[0].scatter(proj_cell[k_b, 0], proj_cell[k_b, 1], proj_cell[k_b, 2],
           color=colormap(i), s=100)

    mintime=k_b
    maxtime=-1
    ax[1].plot(proj_cell[mintime:maxtime, 0], proj_cell[mintime:maxtime, 1], proj_cell[mintime:maxtime, 2],
           color=colormap(i),linewidth=linewidth, label=f'{testdur[i]}')
    #ax.scatter(proj_cell[0, 0], proj_cell[0, 1], proj_cell[0, 2],
           #color='green', s=100)    
    ax[1].scatter(proj_cell[k_b, 0], proj_cell[k_b, 1], proj_cell[k_b, 2],
           color=Color, s=100)
    
#ax.set_xlabel('Principal Component 1')
#ax.set_ylabel('Principal Component 2')
#ax.set_zlabel('Principal Component 3')
#ax.set_title('nFDA 3D Plot')
ax[1].legend()
    





# display variance explained
comp=8
pca2 = PCA(n_components=comp)
pca2.fit(act18s)
# Calculate cumulative variance explained
cumulative_variance = np.cumsum(pca2.explained_variance_ratio_)
# Plotting
plt.figure()
plt.plot(np.arange(1, comp + 1), cumulative_variance, marker='o', linestyle='-')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained')
plt.ylim(0,1)
plt.grid(True)
plt.show()


#%% plot speed of neural trajectory

x, y, In_ons=makeInOut(sample_size,trial_num+2,inputdur,nInput,min_dur,max_dur,dt)
#xnew=sum(xnew,axis=2)
output_and_activities = activity_model.predict(x)
activities = output_and_activities[0]  # Activities of all intermediate layers
act_avg=avgAct(activities,In_ons,min_dur,max_dur)
max_range=[min_dur,max_dur+min_dur] # time range to choose max firing time
input_activities= act_avg.copy()
#input_activities=activities[0,In_ons[0,1]:In_ons[0,1]+min_dur+max_dur,:]
max_times = np.argmax(input_activities[max_range[0]:max_range[1], :], axis=0)
# Sort units based on the time of maximum activity
sorted_units = np.argsort(max_times)
act18s=input_activities[:,sorted_units]
act18s=scipy.stats.zscore(act18s, axis=0)
ncomp=10
pca = PCA(n_components=ncomp)
pca_score = pca.fit_transform(act18s)



mat=act18s
speed=np.sqrt(np.mean(np.square(np.diff(mat,axis=0)),axis=1))
plt.figure()
plt.plot(speed)

ymin=0
ymax=0.4
plt.ylim(top=0.4,bottom=0)
plt.vlines(x=min_dur,ymin=ymin,ymax=ymax,colors='Red')



durvec=[min_dur,max_dur]
for i in range(2):
    int_list=[min_dur,max_dur,min_dur,max_dur,durvec[i],max_dur]
    x,y,In_ons=makeInOutTest_exp(int_list,8,inputdur,nInput,min_dur,max_dur,dt)
    output_and_activities = activity_model.predict(x)
    activities = output_and_activities[0]  # Activities of all intermediate layers
    output1=np.mean(output_and_activities[1][:,:,0],axis=0)
    activities=activities[:,-1-np.sum(int_list[-3:],axis=None):,:]
    avg_1,avg_2,avg_3=avgAct_lastfew(activities,int_list[-3:])
    avg_all=np.concatenate((avg_1, avg_2, avg_3),axis=0)
    if i==0:
        avg_all0=avg_all[:,sorted_units]
    else:
        avg_all1=avg_all[:,sorted_units]


avg_all=np.concatenate((avg_all0,avg_all1),axis=0)
avg_all=scipy.stats.zscore(avg_all, axis=0)
avg_all0=avg_all[:np.shape(avg_all0)[0],:]
avg_all1=avg_all[np.shape(avg_all0)[0]:,:]



ncomp=10

pca_score=pca.transform(avg_all)
pca = PCA(n_components=ncomp)
pca_score = pca.fit_transform(avg_all)

mat=pca_score
speed0=np.sqrt(np.mean(np.square(np.diff(avg_all0,axis=0)),axis=1))
speed1=np.sqrt(np.mean(np.square(np.diff(avg_all1,axis=0)),axis=1))
plt.figure()
plt.plot(speed0)
plt.plot(speed1)
plt.ylim(top=0.4,bottom=0)
#plt.yscale('log')

# %%
# iterate RNN explicitly with specific conditions
x, y, In_ons=makeInOut(sample_size,trial_num+2,inputdur,nInput,min_dur,max_dur,dt)
time_length=np.shape(x)[1]
#time_length=2000
state=np.zeros((time_length,nUnit,2))
act2=np.transpose(activities,(1,2,0))
act2=act2[[0],:,[0]]
ac2=act2[:,:,np.newaxis]
state[0,:,0]=act2
state[0,:,1]=act2

#state[0,:,:]=np.random.random(size=(1,nUnit,2))
state[0,:,:]=np.random.normal(loc=0,scale=0.1,size=(1,nUnit,2))
input1=x[0:2,:,:]
input1=np.transpose(input1,(1,2,0))
output1=np.zeros((time_length,2,2))
for i in range(time_length-1):
    #for k in range(2):
        #state[i+1,:,k]=state[i,:,k]*RNN_layer_Recurrent_kernel+input1[i,:,k]*RNN_layer_kernel+np.random.normal(Loc=0,scale=0.1,size=(1,nUnit))
    for k in range(2):
        ii=np.min([i,np.shape(input1)[0]-1])
        hiddena=np.matmul(state[[i],:,k],RNN_layer_Recurrent_kernel)+np.matmul(input1[[ii],:,k],RNN_layer_kernel)+np.random.normal(loc=0,scale=0.08,size=(1,nUnit))
        state[i+1,:,k]=(1-1/tau)*state[[i],:,k]+(1/tau)*np.maximum(hiddena,0)
    
        #calculate output
        output1[i+1,:,k]=np.tanh(np.matmul(state[[i+1],:,k],dense_layer_kernel)+dense_layer_bias)
    

fig, axs = plt.subplots(1, 2, figsize=(14, 8))
Line=[None]*4
for ss in range(2):
    axs[ss].plot(y[ss, :, 0], color='blue',label='Target')
    axs[ss].plot(output1[:,0,ss], color='green',label='Prediction')
    axs[ss].plot(y[ss, :, 1], color='red')
    axs[ss].plot(output1[:,1,ss], color='orange')   
    axs[ss].set_title(f'Plot {i+1}')
plt.suptitle('')
# Adjusting layout
handles, labels = axs[0,].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
plt.show

plt.imshow(state[:,:,0].T,aspect='auto', cmap='viridis',interpolation='none')
plt.colorbar()
plt.show


plt.plot(np.mean(state[:,:,0],axis=1),label='mean1')
plt.plot(np.std(state[:,:,0],axis=1),label='std1')
plt.plot(np.mean(state[:,:,1],axis=1),label='mean2')
plt.plot(np.std(state[:,:,1],axis=1),label='std2')
#plt.axhline(y=0.1)
plt.yscale('log')
plt.legend()

#plt.hist(np.sum(state[:,:,0],axis=0)+np.sum(state[:,:,1],axis=0))
plt.scatter(np.sum(state[1:,:,0],axis=0),np.sum(state[1:,:,1],axis=0))




# filter low activity cells and show output
act6s=np.sum(state[1:,:,0],axis=0)
act12s=np.sum(state[1:,:,1],axis=0)
badCell=[i for i in range(nUnit) if act6s[i] < 5 and act12s[i]<5]
#badCell=[i for i in range(nUnit) if act6s[i] < -float('inf') and act12s[i]<-float('inf')]

RNN_Recurrent_kernel_2=RNN_layer_Recurrent_kernel.copy()
RNN_Recurrent_kernel_2[:,badCell]=0;
state2=np.zeros((time_length,nUnit,2))
state2[0,:,:]=np.random.normal(loc=0,scale=0.1,size=(1,nUnit,2))
state2[0,badCell,:]=0
input1=x[0:2,:,:]
input1=np.transpose(input1,(1,2,0))
output2=np.zeros((time_length,2,2))
for i in range(time_length-1):
    #for k in range(2):
        #state[i+1,:,k]=state[i,:,k]*RNN_layer_Recurrent_kernel+input1[i,:,k]*RNN_layer_kernel+np.random.normal(Loc=0,scale=0.1,size=(1,nUnit))
    state2[i,badCell,:]=0
    for k in range(2):
        ii=np.min([i,np.shape(input1)[0]-1])
        hiddena=np.matmul(state2[[i],:,k],RNN_Recurrent_kernel_2)+np.matmul(input1[[ii],:,k],RNN_layer_kernel)+np.random.normal(loc=0,scale=0.08,size=(1,nUnit))
        state2[i+1,:,k]=(1-1/tau)*state2[[i],:,k]+(1/tau)*np.maximum(hiddena,0)
        
        #calculate output
        output2[i+1,:,k]=np.tanh(np.matmul(state2[[i+1],:,k],dense_layer_kernel)+dense_layer_bias)
    

fig, axs = plt.subplots(1, 2, figsize=(14, 8))
Line=[None]*4
for ss in range(2):
    axs[ss].plot(y[ss, :, 0], color='blue',label='Target')
    axs[ss].plot(output2[:,0,ss], color='green',label='Prediction')
    axs[ss].plot(y[ss, :, 1], color='red')
    axs[ss].plot(output2[:,1,ss], color='orange')   
    axs[ss].set_title(f'Plot {i+1}')
plt.suptitle('')
# Adjusting layout
handles, labels = axs[0,].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
plt.show





n=100;
mean=np.zeros(n)
std=np.zeros(n)
unitact=np.random.rand(nUnit)-1/2

for i in range(n):
    mean[i]=np.mean(unitact)
    std[i]=np.std(unitact)
    unitact=(1-1/tau)*unitact+(1/tau)*np.maximum(RNN_layer_kernel*unitact,0)

plt.plot(mean)
plt.plot(std)

# make a model from saved weights
model1=model
checkpoint_filepath = os.path.join(savepath, f"training_2/cp-{i+1:05d}.ckpt")
model1.load_weights(checkpoint_filepath)


