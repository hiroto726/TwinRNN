#%%
import sys
sys.path.append(r'C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_scripts\my2RNN_fix2')
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Model
from decode_analyze import PerturbDecodeAnalyze
import os
import tensorflow as tf
from parula_colormap import parula
from get_folders import load_checkpoint_with_max_number, load_npy_with_max_number


weight_ind=0
# make 2  inputs with some radom interval and output is slowly increasing sequence with that duration
# parameters
seed=30#seed_num[weight_ind]
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
con_prob=0 # probability of connection between neurons
maxval=0
ReLUalpha=0.2
batch_size=32

# scale time parameters by dt
min_dur=int(min_dur/dt) # minimum duration
max_dur=int(max_dur/dt) # maximum duration
# make input function
inputdur=int(inputdur/dt) # duration of input in ms
tau=tau/dt



# Parameters for the analysis (make sure these are defined in your code)
sample_size = 12
trial_num = 8
pert_state = 0       # 0: perturb RNN A, 1: perturb RNN B
pert_noisesd = 1.0   # perturb noise standard deviation
stop = False
option = 0           # 0: use circular mean; 1: use arithmetic mean
trial1 = 2

pert_prob = 1/100
pert_A_prob = 0.5
order = 1  # order=1 means start with min_dur

# Generate perturbation time indices and a perturbation mask (pert_which)
max_ind = int(np.floor((min_dur + max_dur) * (np.floor((trial_num - trial1) / 2) * 19 / 20)))
pert_number = int(np.floor(max_ind * pert_prob))
vectors = []
for i in range(sample_size):
    time0 = np.random.randint(0, max_ind, pert_number)
    time0.sort()
    time0 = np.reshape(time0, (1, -1))
    vectors.append(time0)
time_1 = np.concatenate(vectors, axis=0)
pert_which = np.random.uniform(size=time_1.shape)
pert_which = pert_which < pert_A_prob


# Define the connection probability indices (assuming conProbability is defined)

loopind = 0
#pred_diff_all = np.zeros(len(tind))

# Choose the dimensionality reduction method ("pca", "cca", or "pls")
dim_method = "cca"  # or "cca" or "pls"
Dim=100
lin_method="act_stack"
dropout_num=np.concatenate(([None],np.arange(40)))
# Instantiate the analysis class.
analysis = PerturbDecodeAnalyze(min_dur, max_dur, dt, dim_method, Dim=Dim, lin_method=lin_method)

# Initialize a dictionary to store the analysis results
Allinfo = {
    't_index':       [],
    'Confmat_A_ave': [],
    'Confmat_B_ave': [],
    'Confscore_A_ave':[],
    'Confscore_B_ave':[],
    'Offset_mat_ave':[],
    'temp_error_ave':[],
    'cat_error_ave': [],
    'pred_diff_sub': [],
    'pred_diff_real':[],
    'dim_method': dim_method,
    'Dim': Dim,
    'lin_method': lin_method,
    'dropout_num':dropout_num,
    
}


rank=3
exp=1.0# 1.0 for brown noise
brown_scale=1




#%%
brown_scale=1
exp=1
# load model weights
foldername_1=r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\tsubame_models\rank2_noise1"
noise_weights=np.load(os.path.join(foldername_1,"noise_weights.npy"))
rank=np.shape(noise_weights)[0]
print(rank)

# Build the base model (assume build_model is defined elsewhere)
model = analysis.build_model_brown(nUnit=nUnit, nInh=nInh, nInput=nInput,
                    con_prob=con_prob, maxval=maxval, ReLUalpha=ReLUalpha, seed1=seed1, tau=tau,
                    rank=rank, exp=exp, noise_weights=noise_weights)
# Load weights from the checkpoint
checkpoint_filepath = os.path.join(foldername_1,"epoch_09748.ckpt")
ckeckpoint_filepath2,_ = load_checkpoint_with_max_number(foldername_1)
#checkpoint_filepath=os.path.join(r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\tsubame_models\brown_noise_taskid_fixed_6","epoch_09708.ckpt")
model.load_weights(ckeckpoint_filepath2)


# Create an "activity model" to output intermediate layer activations
activity_model = Model(inputs=model.input, outputs=[layer.output for layer in model.layers[1:]])

# Generate input, output, and onset times using the analysis class method.
#-> it may be better to set brown_scale to 0
x, y, In_ons = analysis.makeInOut_sameint_brown(sample_size, trial_num, inputdur, nInput, order,brown_scale=brown_scale, rank=rank, exp=exp)

input_noise=np.matmul(x[:,:,1:],noise_weights)# batch, time, unit

output_and_activities = activity_model.predict(x)
activities_A = output_and_activities[1]
activities_B = output_and_activities[2]
output_all=output_and_activities[5]

# Compute averaged and stacked activations.
act_avg_A = analysis.avgAct2(activities_A, In_ons)
act_avg_B = analysis.avgAct2(activities_B, In_ons)
act_stack_A = analysis.concatAct(activities_A, In_ons)
act_stack_B = analysis.concatAct(activities_B, In_ons)
noise_stack=analysis.concatAct(input_noise,In_ons)
noise_stack_raw=analysis.concatAct(x[:,:,1:],In_ons)# batch, time, rank



fig,axs=plt.subplots(4,1,figsize = (16, 8))
for i in range(4):
    axs[i].plot(y[0,:,i])
    axs[i].plot(output_all[0,:,i])




# %%
foldername_1=r"/gs/fs/tga-isomura/Hiroto/old_tf/RNN_models/models_1/2RNNs_prob01_weightmax2_fix3/rank3_noise1"

# Build the base model (assume build_model is defined elsewhere)
model = analysis.build_model_brown(nUnit=nUnit, nInh=nInh, nInput=nInput,
                    con_prob=con_prob, maxval=maxval, ReLUalpha=ReLUalpha, seed1=seed1, tau=tau,
                    rank=rank, exp=1, noise_weights=noise_weights)
# Load weights from the checkpoint
#checkpoint_filepath = os.path.join(foldername_1,"epoch_09748.ckpt")
checkpoint_filepath,_ = load_checkpoint_with_max_number(foldername_1)
#checkpoint_filepath=os.path.join(r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\tsubame_models\brown_noise_taskid_fixed_6","epoch_09708.ckpt")
model.load_weights(checkpoint_filepath)
print(checkpoint_filepath)

#%%
# Random sampling and metrics computation

# Parameters
mean_range = 100
sample_num = 100
repetition_num = 100
#min_dur = 60  # e.g., first 60 rows as in MATLAB code

# Assuming act_avg_A and act_avg_B are defined numpy arrays with shape (time, cells)
ncells = np.shape(act_avg_A)[1]

# Lists to store metric results for each repetition
PE_A = []
TS_A = []
Sql_A = []
PE_B = []
TS_B = []
Sql_B = []

for i in range(repetition_num):
    # Randomly select a subset of cells
    random_ind = np.random.permutation(ncells)
    random_sub_ind = random_ind[:sample_num]
    
    # Compute indices for both datasets (include min_dur parameter)
    PE_A_sub, TS_A_sub, Sql_A_sub = analysis.Seq_ind(act_avg_A[:, random_sub_ind], mean_range)
    PE_B_sub, TS_B_sub, Sql_B_sub = analysis.Seq_ind(act_avg_B[:, random_sub_ind], mean_range)
    
    PE_A.append(PE_A_sub)
    TS_A.append(TS_A_sub)
    Sql_A.append(Sql_A_sub)
    PE_B.append(PE_B_sub)
    TS_B.append(TS_B_sub)
    Sql_B.append(Sql_B_sub)

# Convert lists to arrays (shape: (repetition_num, 2))
PE_A = np.array(PE_A)
TS_A = np.array(TS_A)
Sql_A = np.array(Sql_A)
PE_B = np.array(PE_B)
TS_B = np.array(TS_B)
Sql_B = np.array(Sql_B)

# Compute means and SEM (standard error = std/sqrt(n))
mean_PE_A = np.mean(PE_A, axis=0)
sem_PE_A  = np.std(PE_A, axis=0, ddof=1) / np.sqrt(repetition_num)
mean_TS_A = np.mean(TS_A, axis=0)
sem_TS_A  = np.std(TS_A, axis=0, ddof=1) / np.sqrt(repetition_num)
mean_Sql_A = np.mean(Sql_A, axis=0)
sem_Sql_A  = np.std(Sql_A, axis=0, ddof=1) / np.sqrt(repetition_num)

mean_PE_B = np.mean(PE_B, axis=0)
sem_PE_B  = np.std(PE_B, axis=0, ddof=1) / np.sqrt(repetition_num)
mean_TS_B = np.mean(TS_B, axis=0)
sem_TS_B  = np.std(TS_B, axis=0, ddof=1) / np.sqrt(repetition_num)
mean_Sql_B = np.mean(Sql_B, axis=0)
sem_Sql_B  = np.std(Sql_B, axis=0, ddof=1) / np.sqrt(repetition_num)

# For display, we interpret the two elements returned by Seq_ind as:
# index 0 -> "min" (first segment) and index 1 -> "max" (second segment)
A_min_means = [mean_PE_A[0], mean_TS_A[0], mean_Sql_A[0]]
A_min_err   = [sem_PE_A[0],  sem_TS_A[0],  sem_Sql_A[0]]
A_max_means = [mean_PE_A[1], mean_TS_A[1], mean_Sql_A[1]]
A_max_err   = [sem_PE_A[1],  sem_TS_A[1],  sem_Sql_A[1]]

B_min_means = [mean_PE_B[0], mean_TS_B[0], mean_Sql_B[0]]
B_min_err   = [sem_PE_B[0],  sem_TS_B[0],  sem_Sql_B[0]]
B_max_means = [mean_PE_B[1], mean_TS_B[1], mean_Sql_B[1]]
B_max_err   = [sem_PE_B[1],  sem_TS_B[1],  sem_Sql_B[1]]

# -----------------------------
# Plotting using bar plots with error bars

# Define the width of each bar and positions on x-axis
bar_width = 0.2
categories = ['Peak entropy', 'Temporal sparsity', 'Sequentiality index']
r1 = np.arange(len(categories))
r2 = r1 + bar_width
r3 = r1 + 2 * bar_width
r4 = r1 + 3 * bar_width

plt.figure(figsize=(8,5))
plt.bar(r1, A_min_means, color='blue', width=bar_width, edgecolor='grey', 
        label='A min', yerr=A_min_err, capsize=5)
plt.bar(r2, B_min_means, color='green', width=bar_width, edgecolor='grey', 
        label='B min', yerr=B_min_err, capsize=5)
plt.bar(r3, A_max_means, color='orange', width=bar_width, edgecolor='grey', 
        label='A max', yerr=A_max_err, capsize=5)
plt.bar(r4, B_max_means, color='red', width=bar_width, edgecolor='grey', 
        label='B max', yerr=B_max_err, capsize=5)

plt.xlabel('Categories', fontweight='bold')
plt.ylabel('Values', fontweight='bold')
plt.xticks(r1 + 1.5 * bar_width, categories, fontsize=9)
plt.title('Comparison of Indices (Mean ± SEM)', fontweight='bold')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()


#%%
# change scale of the noise and see how the activities change 
import numpy as np
import matplotlib.pyplot as plt

# Predefine brown_scale_vec and other parameters
brown_scale_vec = np.arange(20)
# (Make sure that sample_size, trial_num, inputdur, nInput, order, rank, exp,
#  repetition_num, sample_num, mean_range, noise_weights, activity_model, and analysis
#  are defined in your workspace.)

# Preallocate arrays to store the metrics for each brown_scale value.
n_brown = len(brown_scale_vec)
# Each metric is a 2-element array (segment "min" and segment "max").
PE_A_mean = np.zeros((n_brown, 2))
PE_A_sem  = np.zeros((n_brown, 2))
TS_A_mean = np.zeros((n_brown, 2))
TS_A_sem  = np.zeros((n_brown, 2))
Sql_A_mean = np.zeros((n_brown, 2))
Sql_A_sem  = np.zeros((n_brown, 2))

PE_B_mean = np.zeros((n_brown, 2))
PE_B_sem  = np.zeros((n_brown, 2))
TS_B_mean = np.zeros((n_brown, 2))
TS_B_sem  = np.zeros((n_brown, 2))
Sql_B_mean = np.zeros((n_brown, 2))
Sql_B_sem  = np.zeros((n_brown, 2))

# Loop over each brown_scale value
for idx, b in enumerate(brown_scale_vec):
    # Generate input and output data using your function
    x, y, In_ons = analysis.makeInOut_sameint_brown(
        sample_size, trial_num, inputdur, nInput, order, brown_scale=b, rank=rank, exp=exp)
    
    input_noise = np.matmul(x[:, :, 1:], noise_weights)  # batch, time, unit
    output_and_activities = activity_model.predict(x)
    activities_A = output_and_activities[1]
    activities_B = output_and_activities[2]
    # output_all = output_and_activities[5]  # Not used here
    
    # Compute averaged activations (assumed to be 2D arrays: time x cells)
    act_avg_A = analysis.avgAct2(activities_A, In_ons)
    act_avg_B = analysis.avgAct2(activities_B, In_ons)
    
    # Determine number of cells
    ncells = np.shape(act_avg_A)[1]
    
    # Initialize temporary lists for storing metrics over repetitions
    PE_A_rep = []
    TS_A_rep = []
    Sql_A_rep = []
    PE_B_rep = []
    TS_B_rep = []
    Sql_B_rep = []
    
    # Loop over repetitions for random sampling
    for j in range(repetition_num):
        random_ind = np.random.permutation(ncells)
        random_sub_ind = random_ind[:sample_num]
        # Compute sequentiality indices. Make sure your analysis.Seq_ind accepts the necessary parameters.
        PE_A_sub, TS_A_sub, Sql_A_sub = analysis.Seq_ind(act_avg_A[:, random_sub_ind], mean_range)
        PE_B_sub, TS_B_sub, Sql_B_sub = analysis.Seq_ind(act_avg_B[:, random_sub_ind], mean_range)
        
        PE_A_rep.append(PE_A_sub)
        TS_A_rep.append(TS_A_sub)
        Sql_A_rep.append(Sql_A_sub)
        PE_B_rep.append(PE_B_sub)
        TS_B_rep.append(TS_B_sub)
        Sql_B_rep.append(Sql_B_sub)
    
    # Convert lists to arrays with shape (repetition_num, 2)
    PE_A_rep = np.array(PE_A_rep)
    TS_A_rep = np.array(TS_A_rep)
    Sql_A_rep = np.array(Sql_A_rep)
    PE_B_rep = np.array(PE_B_rep)
    TS_B_rep = np.array(TS_B_rep)
    Sql_B_rep = np.array(Sql_B_rep)
    
    # Compute mean and SEM (standard error = std/sqrt(n)) for each metric, separately for min (index 0) and max (index 1)
    PE_A_mean[idx, :] = np.mean(PE_A_rep, axis=0)
    PE_A_sem[idx, :]  = np.std(PE_A_rep, axis=0, ddof=1) / np.sqrt(repetition_num)
    TS_A_mean[idx, :] = np.mean(TS_A_rep, axis=0)
    TS_A_sem[idx, :]  = np.std(TS_A_rep, axis=0, ddof=1) / np.sqrt(repetition_num)
    Sql_A_mean[idx, :] = np.mean(Sql_A_rep, axis=0)
    Sql_A_sem[idx, :]  = np.std(Sql_A_rep, axis=0, ddof=1) / np.sqrt(repetition_num)
    
    PE_B_mean[idx, :] = np.mean(PE_B_rep, axis=0)
    PE_B_sem[idx, :]  = np.std(PE_B_rep, axis=0, ddof=1) / np.sqrt(repetition_num)
    TS_B_mean[idx, :] = np.mean(TS_B_rep, axis=0)
    TS_B_sem[idx, :]  = np.std(TS_B_rep, axis=0, ddof=1) / np.sqrt(repetition_num)
    Sql_B_mean[idx, :] = np.mean(Sql_B_rep, axis=0)
    Sql_B_sem[idx, :]  = np.std(Sql_B_rep, axis=0, ddof=1) / np.sqrt(repetition_num)

# Now, plot the results.
# We'll create three subplots (one per metric: PE, TS, and sequentiality index).
# Each subplot will contain four lines:
#   - Condition A, min segment (index 0)
#   - Condition A, max segment (index 1)
#   - Condition B, min segment (index 0)
#   - Condition B, max segment (index 1)
# and the SEM is shaded using plt.fill_between.

fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# Plot Peak Entropy
axs[0].plot(brown_scale_vec, PE_A_mean[:, 0], label='PE A min', color='blue')
axs[0].fill_between(brown_scale_vec, PE_A_mean[:, 0] - PE_A_sem[:, 0], 
                    PE_A_mean[:, 0] + PE_A_sem[:, 0], color='blue', alpha=0.3)
axs[0].plot(brown_scale_vec, PE_A_mean[:, 1], label='PE A max', color='orange')
axs[0].fill_between(brown_scale_vec, PE_A_mean[:, 1] - PE_A_sem[:, 1], 
                    PE_A_mean[:, 1] + PE_A_sem[:, 1], color='orange', alpha=0.3)
axs[0].plot(brown_scale_vec, PE_B_mean[:, 0], label='PE B min', color='green')
axs[0].fill_between(brown_scale_vec, PE_B_mean[:, 0] - PE_B_sem[:, 0], 
                    PE_B_mean[:, 0] + PE_B_sem[:, 0], color='green', alpha=0.3)
axs[0].plot(brown_scale_vec, PE_B_mean[:, 1], label='PE B max', color='red')
axs[0].fill_between(brown_scale_vec, PE_B_mean[:, 1] - PE_B_sem[:, 1], 
                    PE_B_mean[:, 1] + PE_B_sem[:, 1], color='red', alpha=0.3)
axs[0].set_ylabel('Peak Entropy')
axs[0].legend()

# Plot Temporal Sparsity
axs[1].plot(brown_scale_vec, TS_A_mean[:, 0], label='TS A min', color='blue')
axs[1].fill_between(brown_scale_vec, TS_A_mean[:, 0] - TS_A_sem[:, 0], 
                    TS_A_mean[:, 0] + TS_A_sem[:, 0], color='blue', alpha=0.3)
axs[1].plot(brown_scale_vec, TS_A_mean[:, 1], label='TS A max', color='orange')
axs[1].fill_between(brown_scale_vec, TS_A_mean[:, 1] - TS_A_sem[:, 1], 
                    TS_A_mean[:, 1] + TS_A_sem[:, 1], color='orange', alpha=0.3)
axs[1].plot(brown_scale_vec, TS_B_mean[:, 0], label='TS B min', color='green')
axs[1].fill_between(brown_scale_vec, TS_B_mean[:, 0] - TS_B_sem[:, 0], 
                    TS_B_mean[:, 0] + TS_B_sem[:, 0], color='green', alpha=0.3)
axs[1].plot(brown_scale_vec, TS_B_mean[:, 1], label='TS B max', color='red')
axs[1].fill_between(brown_scale_vec, TS_B_mean[:, 1] - TS_B_sem[:, 1], 
                    TS_B_mean[:, 1] + TS_B_sem[:, 1], color='red', alpha=0.3)
axs[1].set_ylabel('Temporal Sparsity')
axs[1].legend()

# Plot Sequentiality Index
axs[2].plot(brown_scale_vec, Sql_A_mean[:, 0], label='Sql A min', color='blue')
axs[2].fill_between(brown_scale_vec, Sql_A_mean[:, 0] - Sql_A_sem[:, 0], 
                    Sql_A_mean[:, 0] + Sql_A_sem[:, 0], color='blue', alpha=0.3)
axs[2].plot(brown_scale_vec, Sql_A_mean[:, 1], label='Sql A max', color='orange')
axs[2].fill_between(brown_scale_vec, Sql_A_mean[:, 1] - Sql_A_sem[:, 1], 
                    Sql_A_mean[:, 1] + Sql_A_sem[:, 1], color='orange', alpha=0.3)
axs[2].plot(brown_scale_vec, Sql_B_mean[:, 0], label='Sql B min', color='green')
axs[2].fill_between(brown_scale_vec, Sql_B_mean[:, 0] - Sql_B_sem[:, 0], 
                    Sql_B_mean[:, 0] + Sql_B_sem[:, 0], color='green', alpha=0.3)
axs[2].plot(brown_scale_vec, Sql_B_mean[:, 1], label='Sql B max', color='red')
axs[2].fill_between(brown_scale_vec, Sql_B_mean[:, 1] - Sql_B_sem[:, 1], 
                    Sql_B_mean[:, 1] + Sql_B_sem[:, 1], color='red', alpha=0.3)
axs[2].set_ylabel('Sequentiality Index')
axs[2].set_xlabel('Brown Scale')
axs[2].legend()

plt.tight_layout()
plt.show()

#%% do pca -> then cca and take the activities and avg activities
from column_corr import pairwise_corr

foldername_1=r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\tsubame_models\rank2_noise1"


# decode and analyze
# analyze single case
# Parameters for the analysis (make sure these are defined in your code)
sample_size = 8 #16
trial_num = 12 # 24
pert_state = 0       # 0: perturb RNN A, 1: perturb RNN B
pert_noisesd = 2.0   # perturb noise standard deviation
stop = False
option = 0           # 0: use circular mean; 1: use arithmetic mean
trial1 = 2

pert_prob = 1/100
pert_A_prob = 0.5
order = 1  # order=1 means start with min_dur

# Generate perturbation time indices and a perturbation mask (pert_which)
max_ind = int(np.floor((min_dur + max_dur) * (np.floor((trial_num - trial1) / 2) * 19 / 20)))
pert_number = int(np.floor(max_ind * pert_prob))
vectors = []
for i in range(sample_size):
    time0 = np.random.randint(0, max_ind, pert_number)
    time0.sort()
    time0 = np.reshape(time0, (1, -1))
    vectors.append(time0)
time_1 = np.concatenate(vectors, axis=0)
pert_which = np.random.uniform(size=time_1.shape)
pert_which = pert_which < pert_A_prob


# Define the connection probability indices (assuming conProbability is defined)


# Choose the dimensionality reduction method ("pca", "cca", or "pls")
dim_method = "cca"  # or "cca" or "pls"
pre_method="pca"
Dim=100
lin_method="act_stack"
max_drop_num=10
dropout_num=np.concatenate(([None],np.arange(max_drop_num)))
# Instantiate the analysis class.
analysis = PerturbDecodeAnalyze(min_dur, max_dur, dt, dim_method, Dim=Dim, lin_method=lin_method)

# Initialize a dictionary to store the analysis results
Allinfo = {
    't_index':       [],
    'Confmat_A_ave': [],
    'Confmat_B_ave': [],
    'Confscore_A_ave':[],
    'Confscore_B_ave':[],
    'Offset_mat_ave':[],
    'temp_error_ave':[],
    'cat_error_ave': [],
    'pred_diff_sub': [],
    'pred_diff_real':[],
    'dim_method': dim_method,
    'Dim': Dim,
    'lin_method': lin_method,
    'dropout_num':dropout_num,
    
}

exp=1
brown_scale=1.0

t=0
k=0
confA_list   = []
confB_list   = []
confsc_A=[]
confsc_B=[]
offset_list  = []
terror_list  = []
caterr_list  = []
pred_diff2   = []
pred_diff_real_sub = []
loopind += 1
maxval = 0
con_prob = 0

# load noise_weights and model weights
# load noise_weights
# load model weights
noise_weights=np.load(os.path.join(foldername_1,"noise_weights.npy"))
rank=np.shape(noise_weights)[0]


# Build the base model (assume build_model is defined elsewhere)
model = analysis.build_model_brown(nUnit=nUnit, nInh=nInh, nInput=nInput,
                    con_prob=con_prob, maxval=maxval, ReLUalpha=ReLUalpha, seed1=seed1, tau=tau,
                    rank=rank, exp=1, noise_weights=noise_weights)
# Load weights from the checkpoint
#checkpoint_filepath = os.path.join(foldername_1,"epoch_09748.ckpt")
ckeckpoint_filepath2,_ = load_checkpoint_with_max_number(foldername_1)
#checkpoint_filepath=os.path.join(r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\tsubame_models\brown_noise_taskid_fixed_6","epoch_09708.ckpt")
model.load_weights(ckeckpoint_filepath2)


# Create an "activity model" to output intermediate layer activations
activity_model = Model(inputs=model.input, outputs=[layer.output for layer in model.layers[1:]])

# Generate input, output, and onset times using the analysis class method.
#-> it may be better to set brown_scale to 0
x, y, In_ons = analysis.makeInOut_sameint_brown(sample_size, trial_num, inputdur, nInput, order,brown_scale=brown_scale, rank=rank, exp=exp)

input_noise=np.matmul(x[:,:,1:],noise_weights)# batch, time, unit

output_and_activities = activity_model.predict(x)
activities_A = output_and_activities[1]
activities_B = output_and_activities[2]

# Compute averaged and stacked activations.
act_avg_A = analysis.avgAct2(activities_A, In_ons)
act_avg_B = analysis.avgAct2(activities_B, In_ons)
act_stack_A = analysis.concatAct(activities_A, In_ons)
act_stack_B = analysis.concatAct(activities_B, In_ons)
noise_stack=analysis.concatAct(input_noise,In_ons)
noise_stack_raw=analysis.concatAct(x[:,:,1:],In_ons)# batch, time, rank

#remove inactive components
act_stack_A,act_stack_B=analysis.remove_inactive(act_stack_A,act_stack_B)
act_avg_A,act_avg_B=analysis.remove_inactive_transform(act_avg_A,act_avg_B,dim=1)


# Create classification labels.
Class_per_sec = 1
class_A_train, class_B_train = analysis.make_classying_classes(act_stack_A, act_stack_B, Class_per_sec)


# take first 100 pca components to avoid overfitting
act_stack_A, act_stack_B, act_avg_A, act_avg_B =analysis.reduce_dimension_pre(act_avg_A, act_avg_B, act_stack_A, act_stack_B, methodname=pre_method, Dim=Dim)


#plot outputs
num_outputs=np.shape(y)[2]
fig, axes = plt.subplots(num_outputs, 1, figsize=(6, 2 * num_outputs), sharex=True)

for out_ind in range(num_outputs):
    ax = axes[out_ind] if num_outputs > 1 else axes  # Ensure correct axis selection for single subplot case
    ax.plot(output_and_activities[5][0, :, out_ind], linestyle='-', color=f'C{out_ind}', label=f'Output {out_ind}')
    ax.plot(y[0, :, out_ind], linestyle='--', color=f'C{out_ind}', label=f'Ground Truth {out_ind}')
    
    ax.set_ylabel(f'Out {out_ind}')  # Label for each subplot
    ax.legend(loc='best', fontsize='small')

# Set common x-axis label
plt.xlabel("Time Steps")

# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()


# Reduce dimensions using the chosen method.
if analysis.dim_method.lower() == "pca":
    proj_A_train, proj_B_train, proj_C_train = analysis.reduce_dimension(act_avg_A, act_avg_B, act_stack_A, act_stack_B)
else:
    proj_A_train, proj_B_train = analysis.reduce_dimension(act_avg_A, act_avg_B, act_stack_A, act_stack_B)

#calculate the correlation with noise components
noise_corr_A=pairwise_corr(noise_stack_raw,proj_A_train)# rank, components
noise_corr_B=pairwise_corr(noise_stack_raw,proj_B_train)



# calculate mean of the data
A_train_mean=analysis.trial_avg(proj_A_train)# mindur+maxdur,units
B_train_mean=analysis.trial_avg(proj_B_train)
# plot mean of each axis
p=9# number of components to display
ncols = int(np.ceil(np.sqrt(p)))      # Number of columns in the subplot grid
nrows = int(np.ceil(p / ncols))         # Number of rows needed
fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(12, 8))
axs = axs.flatten()  # Flatten to easily iterate over axes
for i in range(p):
    axs[i].plot(A_train_mean[:, i])
    axs[i].set_title(f"A Component {i}")
for ax in axs[p:]:
    ax.remove()
plt.tight_layout()
plt.show()
fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(12, 8))
axs = axs.flatten()  # Flatten to easily iterate over axes
for i in range(p):
    axs[i].plot(B_train_mean[:, i])
    axs[i].set_title(f"B Component {i}")
for ax in axs[p:]:
    ax.remove()
plt.tight_layout()
plt.show()

# plot each axis
fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(12, 8))
axs = axs.flatten()  # Flatten to easily iterate over axes
n=np.shape(proj_A_train)[0]
for i in range(p):
    axs[i].plot(proj_A_train[:, i])
    axs[i].set_title(f"A Component {i}")
    for x in np.arange(0, n, 1800):
        axs[i].axvline(x=x, color='red', linestyle='--', linewidth=1)
for ax in axs[p:]:
    ax.remove()
plt.tight_layout()
plt.show()
fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(12, 8))
axs = axs.flatten()  # Flatten to easily iterate over axes
for i in range(p):
    axs[i].plot(proj_B_train[:, i])
    axs[i].set_title(f"B Component {i}")
    for x in np.arange(0, n, 1800):
        axs[i].axvline(x=x, color='red', linestyle='--', linewidth=1)
for ax in axs[p:]:
    ax.remove()
plt.tight_layout()
plt.show()

# plot intra trial variance/ inter trial variance
score_A=analysis.intra_inter_var(proj_A_train)
score_B=analysis.intra_inter_var(proj_B_train)
x=np.arange(1,len(score_A)+1)
plt.figure()
plt.plot(x,score_A,label="A: intra/inter var")
plt.plot(x,score_B,label="B: intra/inter var")
plt.legend()
plt.xlim([0,max_drop_num])




# plot correlation of noise with components
plt.figure()
x=np.arange(1,max_drop_num+1)
for i in range(rank):
    plt.plot(x,np.abs(noise_corr_A[i, :max_drop_num]), linestyle='-', color=f'C{i}', label=f'A {i}' if i < 10 else "_nolegend_")
    plt.plot(x,np.abs(noise_corr_B[i, :max_drop_num]), linestyle='-.', color=f'C{i}', label=f'B {i}' if i < 10 else "_nolegend_")

# Add labels and legend
plt.xlabel("Components")
plt.ylabel("Absolute Correlation Coefficient")
plt.legend(title="Rank Index", ncol=1, fontsize='small', loc='upper left', bbox_to_anchor=(1.05, 1))

plt.show() 


#%% do pca-> cca -> add NOISE ALONG CERTAIN COMPONENT AXIS
from column_corr import pairwise_corr
rep_num=1
brown_scale=1
foldername_1=r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\tsubame_models\rank2_noise1"

conProbability=[0,1.e-05, 3.e-05, 1.e-04, 3.e-04, 1.e-03, 3.e-03, 1.e-02, 3.e-02, 1.e-01,3.e-01,1.e+00]
weight_max = [0.2] * len(conProbability)
seed_num=[1010,1011,1012,1013,1014,1015]

# decode and analyze
# analyze single case
# Parameters for the analysis (make sure these are defined in your code)
sample_size = 16 #16
trial_num = 24 # 24
pert_state = 0       # 0: perturb RNN A, 1: perturb RNN B
pert_noisesd = 0.003  # perturb noise standard deviation
stop = False
option = 0           # 0: use circular mean; 1: use arithmetic mean
trial1 = 2

pert_prob = 1/100
pert_A_prob = 0.5
order = 1  # order=1 means start with min_dur

# Generate perturbation time indices and a perturbation mask (pert_which)
max_ind = int(np.floor((min_dur + max_dur) * (np.floor((trial_num - trial1) / 2) * 19 / 20)))
pert_number = int(np.floor(max_ind * pert_prob))
vectors = []
for i in range(sample_size):
    time0 = np.random.randint(0, max_ind, pert_number)
    time0.sort()
    time0 = np.reshape(time0, (1, -1))
    vectors.append(time0)
time_1 = np.concatenate(vectors, axis=0)
pert_which = np.random.uniform(size=time_1.shape)
pert_which = pert_which < pert_A_prob


# Define the connection probability indices (assuming conProbability is defined)


# Choose the dimensionality reduction method ("pca", "cca", or "pls")
dim_method = "cca"  # or "cca" or "pls"
pre_method="pca"
Dim=100
lin_method="act_stack"
max_drop_num=40
dropout_num=np.arange(max_drop_num)
# Instantiate the analysis class.
analysis = PerturbDecodeAnalyze(min_dur, max_dur, dt, dim_method, Dim=Dim, lin_method=lin_method)

# Initialize a dictionary to store the analysis results
Allinfo = {
    't_index':       [],
    'Confmat_A_ave': [],
    'Confmat_B_ave': [],
    'Confscore_A_ave':[],
    'Confscore_B_ave':[],
    'Offset_mat_ave':[],
    'temp_error_ave':[],
    'cat_error_ave': [],
    'pred_diff_sub': [],
    'pred_diff_real':[],
    'dim_method': dim_method,
    'Dim': Dim,
    'lin_method': lin_method,
    'dropout_num':dropout_num,
    'pre_method':pre_method,

}
print(Allinfo)
tind=np.arange(len(conProbability))

t=0
k=0

pred_acc=[]
pred_accA=[]
pred_accB=[]
pred_off=[]
pred_offA=[]
pred_offB=[]

noise_corr_A_all=[]
noise_corr_B_all=[]
confmat_A_all=[]
confmat_B_all=[]
intra_inter_var_A_all=[]
intra_inter_var_B_all=[]
offset_mat_all=[]
cat_error_rate_all=[]
temp_error_rate_all=[]


for rep_now in range(rep_num):

    confA_list   = []
    confB_list   = []
    confsc_A=[]
    confsc_B=[]
    offset_list  = []
    terror_list  = []
    caterr_list  = []
    pred_diff2   = []
    pred_diff_real_sub = []
    loopind += 1
    maxval = weight_max[t]
    con_prob = conProbability[t]

    # load noise_weights and model weights
    # load noise_weights
    foldername_1=r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\tsubame_models\rank2_noise1"
    noise_weights=np.load(os.path.join(foldername_1,"noise_weights.npy"))
    rank=np.shape(noise_weights)[0]


    # Build the base model (assume build_model is defined elsewhere)
    model = analysis.build_model_brown(nUnit=nUnit, nInh=nInh, nInput=nInput,
                        con_prob=con_prob, maxval=maxval, ReLUalpha=ReLUalpha, seed1=seed1, tau=tau,
                        rank=rank, exp=1, noise_weights=noise_weights)
    # Load weights from the checkpoint
    #checkpoint_filepath = os.path.join(foldername_1,"epoch_09748.ckpt")
    checkpoint_filepath,_ = load_checkpoint_with_max_number(foldername_1)
    
    model.load_weights(checkpoint_filepath)


    # Create an "activity model" to output intermediate layer activations
    activity_model = Model(inputs=model.input, outputs=[layer.output for layer in model.layers[1:]])

    # Generate input, output, and onset times using the analysis class method.
    #-> it may be better to set brown_scale to 0
    x, y, In_ons = analysis.makeInOut_sameint_brown(sample_size, trial_num, inputdur, nInput, order,brown_scale=brown_scale, rank=rank, exp=exp)
    
    print(f"brown_scale={brown_scale}, rank={rank}, repetition:{rep_now+1}, max_drop: {dropout_num[-1]} ")
    
    input_noise=np.matmul(x[:,:,1:],noise_weights)# batch, time, unit

    output_and_activities = activity_model.predict(x)
    activities_A = output_and_activities[1]
    activities_B = output_and_activities[2]

    # Compute averaged and stacked activations.
    act_avg_A = analysis.avgAct2(activities_A, In_ons)
    act_avg_B = analysis.avgAct2(activities_B, In_ons)
    act_stack_A = analysis.concatAct(activities_A, In_ons)
    act_stack_B = analysis.concatAct(activities_B, In_ons)
    noise_stack=analysis.concatAct(input_noise,In_ons)
    noise_stack_raw=analysis.concatAct(x[:,:,1:],In_ons)# batch, time, rank

    #remove inactive components
    act_stack_A,act_stack_B=analysis.remove_inactive(act_stack_A,act_stack_B)
    act_avg_A,act_avg_B=analysis.remove_inactive_transform(act_avg_A,act_avg_B,dim=1)


    # Create classification labels.
    Class_per_sec = 1
    class_A_train, class_B_train = analysis.make_classying_classes(act_stack_A, act_stack_B, Class_per_sec)
    
    # take first 100 pca components to avoid overfitting
    act_stack_A, act_stack_B, act_avg_A, act_avg_B =analysis.reduce_dimension_pre(act_avg_A, act_avg_B, act_stack_A, act_stack_B, methodname=pre_method, Dim=Dim+1)



    #plot outputs
    num_outputs=np.shape(y)[2]




    # Reduce dimensions using the chosen method.
    if analysis.dim_method.lower() == "pca":
        proj_A_train, proj_B_train, proj_C_train = analysis.reduce_dimension(act_avg_A, act_avg_B, act_stack_A, act_stack_B)
    else:
        proj_A_train, proj_B_train = analysis.reduce_dimension(act_avg_A, act_avg_B, act_stack_A, act_stack_B)


    # get transformation matrices
    trans_A_sub, trans_B_sub=analysis.get_transfomation_matrix()
    trans_A, trans_B=analysis.make_entire_trans_mat(nUnit,trans_A_sub,trans_B_sub, norm=False)
    
    norm_A=np.linalg.norm(trans_A[0], axis=0, keepdims=True)
    norm_B=np.linalg.norm(trans_B[0], axis=0, keepdims=True)
    trans_A_norm=np.divide(trans_A[0], norm_A, where=norm_A!=0)
    trans_B_norm=np.divide(trans_B[0], norm_B, where=norm_B!=0)
    
    # get vectors along each cca coordinates
    comp_dir_A=analysis.get_ortho_mat(trans_A_norm)
    comp_dir_B=analysis.get_ortho_mat(trans_B_norm)
    
    # normalize comp_dir so that it becomes 1 after dot product
    comp_dir_A/=np.diag(np.matmul(trans_A[0].T, comp_dir_A))
    comp_dir_B/=np.diag(np.matmul(trans_B[0].T, comp_dir_B))
    
    """
    # test whether they are orthogonal
    A_dot=trans_A_norm.T @ comp_dir_A
    B_dot=trans_B_norm.T @ comp_dir_B
    fig,axs=plt.subplots(1,2)
    im0=axs[0].imshow(A_dot)
    im1=axs[1].imshow(B_dot)
    fig.colorbar(im1, ax=axs[1])
    plt.show()
    
    # get matrix that returns orthogonal vector
    ind=0
    mat=analysis.get_ortho_vec(trans_A[0],ind)
    ortho_vec=mat @ np.random.rand(nUnit,1)
    ortho_vec/=np.linalg.norm(ortho_vec, axis=0)
    
    dot=ortho_vec.T @ trans_A[0]
    plt.figure()
    plt.plot(dot.T)
    
    # plot correlation with noise weights
    plt.figure()
    plt.plot(np.abs(noise_weights[[0],:]@comp_dir_A).T)
    plt.plot(np.abs(noise_weights[[1],:]@comp_dir_A).T)
    
    
    """
    
    
    # test
    """
    trans_A_test=trans_A.T
    trans_B_test=trans_B.T
    trans_A_test,trans_B_test=analysis.remove_inactive_transform(trans_A_test,trans_B_test)
    trans_A_test,trans_B_test=analysis.transform_stack_pre(trans_A_test,trans_B_test)
    trans_A_test, trans_B_test=analysis.reduce_dim_transform(np.squeeze(trans_A_test), np.squeeze(trans_B_test))
    """

    #calculate the correlation with noise components
    noise_corr_A=pairwise_corr(noise_stack_raw,proj_A_train)# rank, components
    noise_corr_B=pairwise_corr(noise_stack_raw,proj_B_train)



    # calculate mean of the data
    A_train_mean=analysis.trial_avg(proj_A_train)# mindur+maxdur,units
    B_train_mean=analysis.trial_avg(proj_B_train)



    # plot intra trial variance/ inter trial variance
    score_A=analysis.intra_inter_var(proj_A_train)
    score_B=analysis.intra_inter_var(proj_B_train)

    intra_inter_var_A_all.append(np.array(score_A))
    intra_inter_var_B_all.append(np.array(score_B))





    noise_corr_A_all.append((np.abs(noise_corr_A[:, :max_drop_num])))
    noise_corr_B_all.append((np.abs(noise_corr_B[:, :max_drop_num])))



    # Create classifiers for decoding.
    if analysis.dim_method.lower() == "pca":
        train_data=(proj_A_train, proj_B_train, proj_C_train)
        class_data=(class_A_train, class_B_train)

        clf_A, clf_B, clf_C = analysis.create_train_classifier(analysis.remove_component_stack(train_data, None), class_data)

    else:
        train_data=(proj_A_train, proj_B_train)
        class_data=(class_A_train, class_B_train)
        clf_A, clf_B= analysis.create_train_classifier(analysis.remove_component_stack(train_data, None), class_data)
               


    # add noise along a cca direction
    pred_A=[]
    pred_B=[]
    pred_C=[]
    for var,comp in enumerate(dropout_num):
        actpart_A, actpart_B = analysis.perturb_and_decode_noise_prob2_brown_dir(
            trial1, time_1, pert_which, order,
            pert_noisesd, stop,
            sample_size, trial_num, inputdur, nInput,
            nUnit, nInh, con_prob, maxval, ReLUalpha, seed1, tau, model,
            brown_scale, rank,exp=exp,noise_weights=noise_weights,
            noise_vec=(comp_dir_A[:,var].reshape(1,-1),comp_dir_B[:,var].reshape(1,-1)),
            sync_noise=False,
        )

        # Perform perturbation and decoding.
        if analysis.dim_method.lower() == "pca":
    
            # Decode time predictions (returns pred_A, pred_B, pred_C)
            # create classes
            total_time_len=np.shape(actpart_A)[0]*np.shape(actpart_A)[2]
            right_class_A,right_class_B=analysis.make_classying_classes_2(total_time_len,total_time_len,Class_per_sec)
    
            # remove inactive axis
            actpart_A,actpart_B=analysis.remove_inactive_transform(actpart_A,actpart_B,dim=1)
            actpart_A, actpart_B=analysis.transform_stack_pre(actpart_A, actpart_B)
            data_all=(actpart_A, actpart_B)
    
    
            pred_A_sub, pred_B_sub, pred_C_sub = analysis.decode_time(data_all,
                                                          clf_A=clf_A,
                                                          clf_B=clf_B,
                                                          clf_C=clf_C,
                                                          remove_ind=None,)
            pred_A.append(pred_A_sub)
            pred_B.append(pred_B_sub)
            pred_C.append(pred_C_sub)
        else:
    
            total_time_len=np.shape(actpart_A)[0]*np.shape(actpart_A)[2]
            right_class_A,right_class_B=analysis.make_classying_classes_2(total_time_len,total_time_len,Class_per_sec)
    
            # Decode time predictions (returns pred_A, pred_B, pred_C)
            actpart_A,actpart_B=analysis.remove_inactive_transform(actpart_A,actpart_B,dim=1)
            actpart_A, actpart_B=analysis.transform_stack_pre(actpart_A, actpart_B)
            data_all=(actpart_A, actpart_B)


            pred_A_sub, pred_B_sub = analysis.decode_time(data_all,
                                                          clf_A=clf_A,
                                                          clf_B=clf_B,
                                                          remove_ind=None,)
            pred_A.append(pred_A_sub)
            pred_B.append(pred_B_sub)

    """
    # plot predictions
    plt.figure()
    plt.plot(pred_A_sub[:,0])
    plt.plot(right_class_A[:np.shape(pred_A_sub)[0]])
    """



    # Compute the average of the decoded predictions.
    classleng = int(1000 / (dt * Class_per_sec))
    class_per_trial = int((min_dur + max_dur) / classleng)

    predavg_A=[]
    predavg_B=[]
    predavg_C=[]
    pred_diff2_sub=[]
    from get_phase import get_phase

    for ind in range(len(dropout_num)):
        if option == 0:
            predavg_A.append(scipy.stats.circmean(pred_A[ind], high=class_per_trial, low=1, axis=1))
            predavg_B.append(scipy.stats.circmean(pred_B[ind], high=class_per_trial, low=1, axis=1))
            if analysis.dim_method.lower() == "pca":
                predavg_C.append(scipy.stats.circmean(pred_C, high=class_per_trial, low=1, axis=1))
        else:
            predavg_A.append(np.mean(pred_A[ind], axis=1))
            predavg_B.append(np.mean(pred_B[ind], axis=1))
            if analysis.dim_method.lower() == "pca":
                predavg_C.append(np.mean(pred_C, axis=1))

        # Compute prediction differences using get_phase function.
        pred_diff = np.round(get_phase(pred_A[ind] - pred_B[ind], class_per_trial, 'int'))
        pred_diff2_sub.append(np.linalg.norm(pred_diff))

    pred_diff2.append(np.array(pred_diff2_sub)) # each list is length dropout_num

    """
    # plot predictions
    show_ind=0
    for i in dropout_num:
        fig, axs = plt.subplots(1, 1, sharex=True, sharey='row', figsize=(10, 6))
        
        axs.plot(pred_A[i], color='#ff7f0e', alpha=0.1)
        axs.plot(pred_B[i], color='#1f77b4', alpha=0.1)
        axs.plot(predavg_A[i], color='#ff7f0e', label='RNN A')
        axs.plot(predavg_B[i], color='#1f77b4', label='RNN B')
        axs.set_title('Decoded results')
        axs.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
        axs.set_yticks(Class_per_sec * np.array([0, 6, 12, 18]))
        axs.set_yticklabels([0, 6, 12, 18])
        xticks = axs.get_xticks()
        scaled_xticks = np.round(xticks * 0.01)
        axs.set_xticklabels(scaled_xticks)
        #plt.suptitle(f'{k}: Connection Probability {conProbability[t]}')
        plt.title(f'Drop comp: {i}')
        plt.tight_layout()
        plt.show()
    
    """




    # show confusion matrix
    from Confmatrix import confmat, confscore
    from error_analysis import get_cat_error, get_temp_error
    pred_Aall_2=[None]*len(dropout_num)
    pred_Ball_2=[None]*len(dropout_num)
    pred_Call_2=[None]*len(dropout_num)
    confmat_A=[None]*len(dropout_num)
    confmat_B=[None]*len(dropout_num)
    confscore_A=[None]*len(dropout_num)
    confscore_B=[None]*len(dropout_num)  
    pred_diff_real_sub2=[]
    offset_mat=[None]*len(dropout_num) 
    cat_error_rate=[None]*len(dropout_num) 
    temp_error_rate=[None]*len(dropout_num)

    for ind in range(len(dropout_num)):
        pred_Aall_2[ind] = pred_A[ind].flatten(order='F').reshape(-1, 1)
        pred_Ball_2[ind] = pred_B[ind].flatten(order='F').reshape(-1, 1)
        if analysis.dim_method.lower() == "pca":
            pred_Call_2[ind] = pred_C[ind].flatten(order='F').reshape(-1, 1)


        class_all=right_class_A.copy().reshape(-1, 1)
        confmat_A[ind]=confmat(class_all,pred_Aall_2[ind])
        confmat_B[ind]=confmat(class_all,pred_Ball_2[ind])
        confscore_A[ind]=confscore(confmat_A[ind],1)
        confscore_B[ind]=confscore(confmat_B[ind],1)


        # get offset from the actual time
        diff_A=get_phase(pred_Aall_2[ind]-class_all,class_per_trial,'int')
        diff_B=get_phase(pred_Ball_2[ind]-class_all,class_per_trial,'int')
        if analysis.dim_method.lower() == "pca":
            diff_C=get_phase(pred_Call_2[ind]-class_all,class_per_trial,'int')
            # get offset from the actual decoded results
            pred_diff_real_sub2.append([np.linalg.norm(diff_A),np.linalg.norm(diff_B),np.linalg.norm(diff_C)])
        else:
            pred_diff_real_sub2.append([np.linalg.norm(diff_A),np.linalg.norm(diff_B)])
        #create 2d histogram
        min_val=-np.ceil(class_per_trial/2)+1
        max_val=np.floor(class_per_trial/2)
        bin_edges = np.arange(min_val-0.5,max_val+1.5,1)
        offset_mat[ind], xedges, yedges = np.histogram2d(diff_A.ravel(), diff_B.ravel(), bins=[bin_edges, bin_edges])


        #show categorical error
        group_bound=int(class_per_trial*(min_dur/(min_dur+max_dur)))-1
        cat_error_rate[ind]=get_cat_error(pred_Aall_2[ind], class_all, pred_Ball_2[ind], class_all, group_bound)

        # show temporal error
        temp_error_rate[ind]=get_temp_error(pred_Aall_2[ind], class_all, pred_Ball_2[ind], class_all, class_per_trial,3)



    # Convert lists to NumPy arrays
    pred_Aall_2 = np.array(pred_Aall_2)
    pred_Ball_2 = np.array(pred_Ball_2)
    pred_Call_2 = np.array(pred_Call_2)# if any(pred_Call_2) else None  # Handle case where PCA isn't used
    confmat_A = np.array(confmat_A)
    confmat_B = np.array(confmat_B)
    confscore_A = np.array(confscore_A)
    confscore_B = np.array(confscore_B)    
    pred_diff_real_sub.append(np.array(pred_diff_real_sub2))
    offset_mat=np.array(offset_mat)
    cat_error_rate=np.array(cat_error_rate) # dropout_num, 4
    temp_error_rate=np.array(temp_error_rate)


    confmat_A_all.append(np.array(confmat_A))
    confmat_B_all.append(np.array(confmat_B))
    offset_mat_all.append(offset_mat)
    cat_error_rate_all.append(cat_error_rate)
    temp_error_rate_all.append(temp_error_rate)

    """
    # plot confmat and decoding offset
    show_ind=0
    plt.figure()
    fig, axs = plt.subplots(1,2,figsize=(10, 6))
    im0=axs[0].imshow(confmat_A[show_ind],aspect='auto', cmap=parula, interpolation='none',vmin=0,vmax=400)
    im1=axs[1].imshow(confmat_B[show_ind],aspect='auto', cmap=parula, interpolation='none',vmin=0,vmax=400)
    axs[0].set_box_aspect(1)
    axs[1].set_box_aspect(1)
    #axs[1].colorbar()
    #cbar = fig.colorbar(im0, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    #plt.suptitle(f'Connection Probability {conProbability[t]}')
    plt.show()
    
    
    
    for i in dropout_num:
        plt.figure(figsize=(10, 6))
        
        # Set extent based on the bin edges for proper axis scaling
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        allpop = np.sum(offset_mat[i])
        # Use imshow to plot the data
        #im0 = plt.imshow((offset_mat[i]/allpop).T, aspect='auto', cmap=parula, interpolation='none', vmin=0, vmax=250/allpop, extent=extent, origin='lower')
        im0=plt.imshow((offset_mat[i]/allpop).T, origin='lower', 
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                       aspect='auto',cmap=parula, interpolation='none', vmin=0, vmax=250/allpop)     
        # Set the aspect of the axis to be equal
        plt.gca().set_aspect('equal')
        # Add color bar
        cbar = plt.colorbar(im0, orientation='vertical', fraction=0.02, pad=0.04)
        # Set axis labels
        plt.xlabel('A offset')
        plt.ylabel('B offset')
        # Set the title
        plt.title(f'Drop comp: {i}')
        # Show the plot
        plt.show()   
    
    
    plt.figure()
    plt.plot(confscore_A)
    plt.plot(confscore_B)
    plt.xlabel("Components")
    plt.ylabel("Accuracy")
    
    """



    # Set extent based on the bin edges for proper axis scaling
    show_ind=0
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    allpop = np.sum(offset_mat[show_ind])

    #show categorical error
    from error_analysis import get_cat_error, get_temp_error
    group_bound=int(class_per_trial*(min_dur/(min_dur+max_dur)))-1
    cat_error_rate=get_cat_error(pred_Aall_2[show_ind], class_all, pred_Ball_2, class_all, group_bound)

    catlabel = np.array(["M2 mis, PPC mis", "M2 mis, PPC cor", "M2 cor, PPC mis", "M2 cor, PPC cor"])


    # show temporal error
    temp_error_rate=get_temp_error(pred_Aall_2[show_ind], class_all, pred_Ball_2, class_all, class_per_trial,3)
    offset_temp=np.arange(-(np.ceil(class_per_trial/2)-1),np.floor(class_per_trial/2)+1)

    #print(f"{k}: loop {loopind} out of {len(tind)}")


    x_ax=np.arange(np.shape(confscore_A)[0])



    x_ax=np.arange(np.shape(confscore_A)[0]-1)+1


    pred_accA.append(confscore_A)
    pred_accB.append(confscore_B)
    pred_offA.append(confscore_A[0]-confscore_A[1:])
    pred_offB.append(confscore_B[0]-confscore_B[1:])


noise_corr_A_all=np.array(noise_corr_A_all)
noise_corr_B_all=np.array(noise_corr_B_all)
confmat_A_all=np.array(confmat_A_all)
confmat_B_all=np.array(confmat_B_all)
intra_inter_var_A_all=np.array(intra_inter_var_A_all)
intra_inter_var_B_all=np.array(intra_inter_var_B_all)
offset_mat_all=np.array(offset_mat_all)
cat_error_rate_all=np.array(cat_error_rate_all)
temp_error_rate_all=np.array(temp_error_rate_all)
pred_accA=np.array(pred_accA)
pred_accB=np.array(pred_accB)
pred_offA=np.array(pred_offA)
pred_offB=np.array(pred_offB)
pred_diff2=np.array(pred_diff2)


#%%

trans_A,trans_B=analysis.get_transfomation_matrix()


#%% compare reward input with brown noise

brown_scale=1
x, y, In_ons = analysis.makeInOut_sameint_brown(sample_size, trial_num, inputdur, nInput, order,brown_scale=brown_scale, rank=rank, exp=exp)

input_noise=np.matmul(x[:,:,1:],noise_weights)# batch, time, unit
input_units= np.matmul(x[:,:,[0]], model.weights[0].numpy().reshape(1, 1, 1024))# batch, time, unit
input_noise_all=np.concatenate((input_noise,input_noise),axis=2)

fig,axs=plt.subplots(1,2)
# Compute the global min and max values across both matrices
vmin = min(input_units[0].min(), input_noise_all[0].min())
vmax = max(input_units[0].max(), input_noise_all[0].max())

vmin=-0.5
vmax=0.5
im0 = axs[0].imshow(input_units[0], aspect='auto', cmap=parula, interpolation='none', vmin=vmin, vmax=vmax)
im1 = axs[1].imshow(input_noise_all[0],aspect='auto', cmap=parula, interpolation='none',  vmin=vmin, vmax=vmax)

fig.colorbar(im0, ax=axs, orientation='horizontal', fraction=0.05, pad=0.1)
plt.show()


# compare max inputs
max_unit_input=np.max(np.abs(input_units[0]),axis=0)
max_unit_noise=np.max(np.abs(input_noise_all[0]),axis=0)
plt.figure()
plt.scatter(max_unit_input,max_unit_noise,2)
plt.xlabel('input')
plt.ylabel('noise')
plt.gca().set_aspect('equal')

#%% do pca_> cca with different brown_scale and see avg activity

from column_corr import pairwise_corr
from scipy.signal import welch

foldername_1=r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\tsubame_models\rank2_noise1"


# decode and analyze
# analyze single case
# Parameters for the analysis (make sure these are defined in your code)
sample_size = 8 #16
trial_num = 12 # 24
pert_state = 0       # 0: perturb RNN A, 1: perturb RNN B
pert_noisesd = 2.0   # perturb noise standard deviation
stop = False
option = 0           # 0: use circular mean; 1: use arithmetic mean
trial1 = 2

pert_prob = 1/100
pert_A_prob = 0.5
order = 1  # order=1 means start with min_dur

# Generate perturbation time indices and a perturbation mask (pert_which)
max_ind = int(np.floor((min_dur + max_dur) * (np.floor((trial_num - trial1) / 2) * 19 / 20)))
pert_number = int(np.floor(max_ind * pert_prob))
vectors = []
for i in range(sample_size):
    time0 = np.random.randint(0, max_ind, pert_number)
    time0.sort()
    time0 = np.reshape(time0, (1, -1))
    vectors.append(time0)
time_1 = np.concatenate(vectors, axis=0)
pert_which = np.random.uniform(size=time_1.shape)
pert_which = pert_which < pert_A_prob


# Define the connection probability indices (assuming conProbability is defined)


# Choose the dimensionality reduction method ("pca", "cca", or "pls")
dim_method = "pls"  # or "cca" or "pls"
pre_method="pca"
Dim=100
lin_method="act_stack"
max_drop_num=10
dropout_num=np.concatenate(([None],np.arange(max_drop_num)))
# Instantiate the analysis class.
analysis = PerturbDecodeAnalyze(min_dur, max_dur, dt, dim_method, Dim=Dim, lin_method=lin_method)

# Initialize a dictionary to store the analysis results
Allinfo = {
    't_index':       [],
    'Confmat_A_ave': [],
    'Confmat_B_ave': [],
    'Confscore_A_ave':[],
    'Confscore_B_ave':[],
    'Offset_mat_ave':[],
    'temp_error_ave':[],
    'cat_error_ave': [],
    'pred_diff_sub': [],
    'pred_diff_real':[],
    'dim_method': dim_method,
    'Dim': Dim,
    'lin_method': lin_method,
    'dropout_num':dropout_num,
    
}

exp=1
brown_scale=np.linspace(0,2,41)



t=0
k=0
confA_list   = []
confB_list   = []
confsc_A=[]
confsc_B=[]
offset_list  = []
terror_list  = []
caterr_list  = []
pred_diff2   = []
pred_diff_real_sub = []
loopind += 1
maxval = 0
con_prob = 0

# load noise_weights and model weights
# load noise_weights
# load model weights
noise_weights=np.load(os.path.join(foldername_1,"noise_weights.npy"))
rank=np.shape(noise_weights)[0]


# Build the base model (assume build_model is defined elsewhere)
model = analysis.build_model_brown(nUnit=nUnit, nInh=nInh, nInput=nInput,
                    con_prob=con_prob, maxval=maxval, ReLUalpha=ReLUalpha, seed1=seed1, tau=tau,
                    rank=rank, exp=1, noise_weights=noise_weights)
# Load weights from the checkpoint
#checkpoint_filepath = os.path.join(foldername_1,"epoch_09748.ckpt")
ckeckpoint_filepath2,_ = load_checkpoint_with_max_number(foldername_1)
#checkpoint_filepath=os.path.join(r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\tsubame_models\brown_noise_taskid_fixed_6","epoch_09708.ckpt")
model.load_weights(ckeckpoint_filepath2)


# Create an "activity model" to output intermediate layer activations
activity_model = Model(inputs=model.input, outputs=[layer.output for layer in model.layers[1:]])

# Generate input, output, and onset times using the analysis class method.
#-> it may be better to set brown_scale to 0
A_train_all=[]
B_train_all=[]
noise_corr_A_all=[]
noise_corr_B_all=[]
PSD_A=[]
PSD_B=[]
for brown in brown_scale:
    x, y, In_ons = analysis.makeInOut_sameint_brown(sample_size, trial_num, inputdur, nInput, order,brown_scale=brown, rank=rank, exp=exp)
    
    input_noise=np.matmul(x[:,:,1:],noise_weights)# batch, time, unit
    
    output_and_activities = activity_model.predict(x)
    activities_A = output_and_activities[1]
    activities_B = output_and_activities[2]
    
    # Compute averaged and stacked activations.
    act_avg_A = analysis.avgAct2(activities_A, In_ons)
    act_avg_B = analysis.avgAct2(activities_B, In_ons)
    act_stack_A = analysis.concatAct(activities_A, In_ons)
    act_stack_B = analysis.concatAct(activities_B, In_ons)
    noise_stack=analysis.concatAct(input_noise,In_ons)
    noise_stack_raw=analysis.concatAct(x[:,:,1:],In_ons)# batch, time, rank
    
    #remove inactive components
    act_stack_A,act_stack_B=analysis.remove_inactive(act_stack_A,act_stack_B)
    act_avg_A,act_avg_B=analysis.remove_inactive_transform(act_avg_A,act_avg_B,dim=1)
    
    
    # Create classification labels.
    Class_per_sec = 1
    class_A_train, class_B_train = analysis.make_classying_classes(act_stack_A, act_stack_B, Class_per_sec)
    
    
    # take first 100 pca components to avoid overfitting
    act_stack_A, act_stack_B, act_avg_A, act_avg_B =analysis.reduce_dimension_pre(act_avg_A, act_avg_B, act_stack_A, act_stack_B, methodname=pre_method, Dim=Dim+1)
    
    
    #plot outputs
    num_outputs=np.shape(y)[2]
    
    
    
    # Reduce dimensions using the chosen method.
    if analysis.dim_method.lower() == "pca":
        proj_A_train, proj_B_train, proj_C_train = analysis.reduce_dimension(act_avg_A, act_avg_B, act_stack_A, act_stack_B)
    else:
        proj_A_train, proj_B_train = analysis.reduce_dimension(act_avg_A, act_avg_B, act_stack_A, act_stack_B)
    
    #calculate the correlation with noise components
    noise_corr_A=pairwise_corr(noise_stack_raw,proj_A_train)# rank, components
    noise_corr_B=pairwise_corr(noise_stack_raw,proj_B_train)
    noise_corr_A_all.append(noise_corr_A)
    noise_corr_B_all.append(noise_corr_B)
    
    
    
    # calculate mean of the data
    A_train_mean=analysis.trial_avg(proj_A_train)# mindur+maxdur,units
    B_train_mean=analysis.trial_avg(proj_B_train)
    
    A_train_all.append(A_train_mean)
    B_train_all.append(B_train_mean)
    
    
    T = 32 # seconds
    fs = 100
    nperseg = int(fs * T)  # e.g., 16000
    window = "hamming"
    nfft = np.power(2,16)#2 ** int(np.ceil(np.log2(nperseg)))  # e.g., 16384 or 32768
    noverlap = nperseg // 2
    
    #spectral analysis
    PSD_A_sub=welch(proj_A_train,
                         fs=fs,
                         window=window, 
                         nperseg=nperseg, 
                         noverlap=noverlap, 
                         nfft=nfft, 
                         scaling='density',
                         axis=0,
                         detrend=False)     # More freq bins (optional but helps)
    
    PSD_B_sub=welch(proj_B_train,
                         fs=fs,
                         window=window, 
                         nperseg=nperseg, 
                         noverlap=noverlap, 
                         nfft=nfft, 
                         scaling='density',
                         axis=0,
                         detrend=False)
    
    # Create subplots with shared axes
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    
    # Plot for PSD_A_sub
    period_A = 1 / PSD_A_sub[0][1:]
    num_cols = 6
    cmap= plt.get_cmap("viridis", num_cols)  # Use the 'viridis' colormap
    
    for i in range(num_cols):
        axs[0].plot(period_A, PSD_A_sub[1][1:, i], color=cmap(i))
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    
    # Plot for PSD_B_sub
    period_B = 1 / PSD_B_sub[0][1:]
    
    
    for i in range(num_cols):
        axs[1].plot(period_B, PSD_B_sub[1][1:, i], color=cmap(i))
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    
    # Add vertical dashed lines at x = 3, 6, 12, and 18 to each subplot
    for ax in axs:
        for x_val in [3, 6, 9, 12, 18]:
            ax.axvline(x=x_val, color='gray', linestyle='--')
    
    axs[0].set_title("RNN A")
    axs[1].set_title("RNN_B")
    fig.suptitle(f"Power Spectral Density (PSD), Brown scale={brown}")
    axs[0].set_xlabel("Period (s)")
    axs[0].set_ylabel("Power")  
    
    plt.show()
        
#plot first components
comp_show=0
A_train_all=np.array(A_train_all)#(num_brown, 1800, 400)
B_train_all=np.array(B_train_all)
noise_corr_A_all=np.abs(np.array(noise_corr_A_all))#(num_brown, rank, 400)
noise_corr_B_all=np.abs(np.array(noise_corr_B_all))


A_sub=A_train_all[:,:,comp_show]
B_sub=B_train_all[:,:,comp_show]


# Determine subplot grid size
rows = int(np.ceil(len(brown_scale) / 2 )) # Number of rows
cols = 2        # Always 2 columns
fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 3), sharex=True)
# Flatten the 2D axes array for easy iteration
axes = axes.flatten()

# Plot each column in a separate subplot
for i in range(len(brown_scale)):
    axes[i].plot(A_sub[i,:])  # Plot i-th column
    axes[i].plot(B_sub[i,:])  # Plot i-th column
    axes[i].set_title(f'Brown scale= {brown_scale[i]}')
    axes[i].grid(True)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

# show correlation with noise 
noise_corr_A_sub=noise_corr_A_all[:,:,comp_show]
noise_corr_B_sub=noise_corr_B_all[:,:,comp_show]

comp_show_num=10
fig, axs=plt.subplots(comp_show_num,2, figsize=(10, comp_show_num * 3), sharex=True)
for k in range(comp_show_num):
    for i in range(np.shape(noise_corr_A_sub)[1]):
        axs[k,0].plot(brown_scale,np.abs(noise_corr_A_all[:,i,k]))
        axs[k,1].plot(brown_scale,np.abs(noise_corr_B_all[:,i,k]))
    axs[k,0].set_ylabel(f"comp {k}")
    axs[k,0].set_ylim([0,1])
    axs[k,1].set_ylim([0,1])
        
axs[0,0].set_title("RNN A")
axs[0,1].set_title("RNN B")
axs[-1,0].set_xlabel("Brown scale")
axs[-1,1].set_xlabel("Brown scale")




# plot all correlation 
n_cols = noise_corr_A_all.shape[1]
n_rows = 2  # First row for noise_corr_A_all and second for noise_corr_B_all

fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(4*n_cols, 8))
comp_show_num = 20
vmin = 0
vmax = 1

# Plot images in a loop for each column.
# Save a reference to one of the images for the colorbar.
for col in range(n_cols):
    # First row: RNN A
    imA = axs[0, col].imshow(noise_corr_A_all[:, col, :comp_show_num],
                              aspect='auto', cmap=parula, interpolation='none',
                              vmin=vmin, vmax=vmax)
    # Second row: RNN B
    imB = axs[1, col].imshow(noise_corr_B_all[:, col, :comp_show_num],
                              aspect='auto', cmap=parula, interpolation='none',
                              vmin=vmin, vmax=vmax)
    # Optionally add a title for each column if desired
    axs[0, col].set_title(f"Noise component {col+1}")

# Set y-axis labels for the first column of each row
axs[0, 0].set_ylabel("RNN A, brown scale", fontsize=12)
axs[1, 0].set_ylabel("RNN B, brown scale", fontsize=12)
# Set x-axis labels for the bottom row only
for col in range(n_cols):
    axs[1, col].set_xlabel("Components", fontsize=12)

# Reduce the number of y ticks (example: 5 ticks) and format them with two decimals.
n_ticks = 5
y_positions = np.linspace(0, len(brown_scale) - 1, n_ticks)
y_labels = [f"{brown_scale[int(round(pos))]:.2f}" for pos in y_positions]

# Apply the reduced/formatted y ticks to all subplots.
for ax in axs.flat:
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)

# Add an overall title.
plt.suptitle("Correlation between components and brown noise", fontsize=16)

# Adjust subplot layout to leave space on the right for the colorbar.
plt.subplots_adjust(right=0.8, top=0.9)

# Create a new axes for the global colorbar.
cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
# Use the first image as a reference for the colorbar.
cbar = fig.colorbar(imA, cax=cbar_ax)
cbar.set_label("Absolute Pearson correlation", fontsize=12)

plt.show()




# plot maximum correlation across ranks
max_image_A = np.max(noise_corr_A_all[:, :, :comp_show_num], axis=1)  # shape: (n, comp_show_num)
max_image_B = np.max(noise_corr_B_all[:, :, :comp_show_num], axis=1)  # shape: (n, comp_show_num)

# Create a new figure with 2 rows, 1 column
fig2, axs2 = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(6, 8))

# Display the maximum images for each RNN using imshow
imA_max = axs2[0].imshow(max_image_A, aspect='auto', cmap=parula,
                          interpolation='none', vmin=vmin, vmax=vmax)
imB_max = axs2[1].imshow(max_image_B, aspect='auto', cmap=parula,
                          interpolation='none', vmin=vmin, vmax=vmax)

# Set y-axis labels for each row
axs2[0].set_ylabel("RNN A, brown scale", fontsize=12)
axs2[1].set_ylabel("RNN B, brown scale", fontsize=12)
axs2[1].set_xlabel("Components", fontsize=12)

# Reduce the number of y ticks (for example, 5 ticks) and format them with two decimals.
n_ticks = 5
y_positions = np.linspace(0, len(brown_scale) - 1, n_ticks)
y_labels = [f"{brown_scale[int(round(pos))]:.2f}" for pos in y_positions]

for ax in axs2:
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)

# Add an overall title for the new figure.
plt.suptitle("Maximum correlation across noise components", fontsize=16)

# Adjust the subplot layout to leave space for the colorbar on the right.
plt.subplots_adjust(right=0.8, top=0.9)

# Create a new axes on the right side for the global colorbar.
cbar_ax2 = fig2.add_axes([0.83, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
# Use the first image as a reference for the colorbar.
cbar = fig2.colorbar(imA_max, cax=cbar_ax2)
cbar.set_label("Absolute Pearson correlation", fontsize=12)

plt.show()
#%% add brown noise to RNNs with no brown noise training and see cca 1st component
from column_corr import pairwise_corr

foldername_1=r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\t4stateful\2RNNs_prob01_weightmax2_fix3"


# decode and analyze
# analyze single case
# Parameters for the analysis (make sure these are defined in your code)
sample_size = 16 #16
trial_num = 24 # 24
pert_state = 0       # 0: perturb RNN A, 1: perturb RNN B
pert_noisesd = 2.0   # perturb noise standard deviation
stop = False
option = 0           # 0: use circular mean; 1: use arithmetic mean
trial1 = 2

pert_prob = 1/100
pert_A_prob = 0.5
order = 1  # order=1 means start with min_dur

# Generate perturbation time indices and a perturbation mask (pert_which)
max_ind = int(np.floor((min_dur + max_dur) * (np.floor((trial_num - trial1) / 2) * 19 / 20)))
pert_number = int(np.floor(max_ind * pert_prob))
vectors = []
for i in range(sample_size):
    time0 = np.random.randint(0, max_ind, pert_number)
    time0.sort()
    time0 = np.reshape(time0, (1, -1))
    vectors.append(time0)
time_1 = np.concatenate(vectors, axis=0)
pert_which = np.random.uniform(size=time_1.shape)
pert_which = pert_which < pert_A_prob


# Define the connection probability indices (assuming conProbability is defined)


# Choose the dimensionality reduction method ("pca", "cca", or "pls")
dim_method = "cca"  # or "cca" or "pls"
pre_method="pca"
Dim=100
lin_method="act_stack"
max_drop_num=10
dropout_num=np.concatenate(([None],np.arange(max_drop_num)))
# Instantiate the analysis class.
analysis = PerturbDecodeAnalyze(min_dur, max_dur, dt, dim_method, Dim=Dim, lin_method=lin_method)

# Initialize a dictionary to store the analysis results
Allinfo = {
    't_index':       [],
    'Confmat_A_ave': [],
    'Confmat_B_ave': [],
    'Confscore_A_ave':[],
    'Confscore_B_ave':[],
    'Offset_mat_ave':[],
    'temp_error_ave':[],
    'cat_error_ave': [],
    'pred_diff_sub': [],
    'pred_diff_real':[],
    'dim_method': dim_method,
    'Dim': Dim,
    'lin_method': lin_method,
    'dropout_num':dropout_num,
    
}

exp=1
brown_scale=np.linspace(0,2,41)



t=0
k=0
confA_list   = []
confB_list   = []
confsc_A=[]
confsc_B=[]
offset_list  = []
terror_list  = []
caterr_list  = []
pred_diff2   = []
pred_diff_real_sub = []
loopind += 1
maxval = 0
con_prob = 0

# load noise_weights and model weights
# load noise_weights
# load model weights
noise_weights=np.load(os.path.join(r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\tsubame_models\rank2_noise1","noise_weights.npy"))
rank=np.shape(noise_weights)[0]
#noise_weights=np.zeros((rank,nUnit))

# Build the base model (assume build_model is defined elsewhere)
model = analysis.build_model_brown(nUnit=nUnit, nInh=nInh, nInput=nInput,
                    con_prob=con_prob, maxval=maxval, ReLUalpha=ReLUalpha, seed1=seed1, tau=tau,
                    rank=rank, exp=1, noise_weights=noise_weights)
# Load weights from the checkpoint
#checkpoint_filepath = os.path.join(foldername_1,"epoch_09748.ckpt")
ckeckpoint_filepath2,_ = load_checkpoint_with_max_number(foldername_1)
#checkpoint_filepath=os.path.join(r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\tsubame_models\brown_noise_taskid_fixed_6","epoch_09708.ckpt")
model.load_weights(ckeckpoint_filepath2)


# Create an "activity model" to output intermediate layer activations
activity_model = Model(inputs=model.input, outputs=[layer.output for layer in model.layers[1:]])

# Generate input, output, and onset times using the analysis class method.
#-> it may be better to set brown_scale to 0
A_train_all=[]
B_train_all=[]
noise_corr_A_all=[]
noise_corr_B_all=[]
for brown in brown_scale:
    x, y, In_ons = analysis.makeInOut_sameint_brown(sample_size, trial_num, inputdur, nInput, order,brown_scale=brown, rank=rank, exp=exp)
    
    input_noise=np.matmul(x[:,:,1:],noise_weights)# batch, time, unit
    
    output_and_activities = activity_model.predict(x)
    activities_A = output_and_activities[1]
    activities_B = output_and_activities[2]
    
    # Compute averaged and stacked activations.
    act_avg_A = analysis.avgAct2(activities_A, In_ons)
    act_avg_B = analysis.avgAct2(activities_B, In_ons)
    act_stack_A = analysis.concatAct(activities_A, In_ons)
    act_stack_B = analysis.concatAct(activities_B, In_ons)
    noise_stack=analysis.concatAct(input_noise,In_ons)
    noise_stack_raw=analysis.concatAct(x[:,:,1:],In_ons)# batch, time, rank
    
    #remove inactive components
    act_stack_A,act_stack_B=analysis.remove_inactive(act_stack_A,act_stack_B)
    act_avg_A,act_avg_B=analysis.remove_inactive_transform(act_avg_A,act_avg_B,dim=1)
    
    
    # Create classification labels.
    Class_per_sec = 1
    class_A_train, class_B_train = analysis.make_classying_classes(act_stack_A, act_stack_B, Class_per_sec)
    
    
    # take first 100 pca components to avoid overfitting
    act_stack_A, act_stack_B, act_avg_A, act_avg_B =analysis.reduce_dimension_pre(act_avg_A, act_avg_B, act_stack_A, act_stack_B, methodname=pre_method, Dim=Dim+1)
    
    
    #plot outputs
    num_outputs=np.shape(y)[2]
    
    
    
    # Reduce dimensions using the chosen method.
    if analysis.dim_method.lower() == "pca":
        proj_A_train, proj_B_train, proj_C_train = analysis.reduce_dimension(act_avg_A, act_avg_B, act_stack_A, act_stack_B)
    else:
        proj_A_train, proj_B_train = analysis.reduce_dimension(act_avg_A, act_avg_B, act_stack_A, act_stack_B)
    
    #calculate the correlation with noise components
    noise_corr_A=pairwise_corr(noise_stack_raw,proj_A_train)# rank, components
    noise_corr_B=pairwise_corr(noise_stack_raw,proj_B_train)
    noise_corr_A_all.append(noise_corr_A)
    noise_corr_B_all.append(noise_corr_B)
    
    
    
    # calculate mean of the data
    A_train_mean=analysis.trial_avg(proj_A_train)# mindur+maxdur,units
    B_train_mean=analysis.trial_avg(proj_B_train)
    
    A_train_all.append(A_train_mean)
    B_train_all.append(B_train_mean)


#plot first components
comp_show=0
A_train_all=np.array(A_train_all)#(num_brown, 1800, 400)
B_train_all=np.array(B_train_all)
noise_corr_A_all=np.abs(np.array(noise_corr_A_all))#(num_brown, rank, 400)
noise_corr_B_all=np.abs(np.array(noise_corr_B_all))


A_sub=A_train_all[:,:,comp_show]
B_sub=B_train_all[:,:,comp_show]


# Determine subplot grid size
rows = int(np.ceil(len(brown_scale) / 2 )) # Number of rows
cols = 2        # Always 2 columns
fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 3), sharex=True)
# Flatten the 2D axes array for easy iteration
axes = axes.flatten()

# Plot each column in a separate subplot
for i in range(len(brown_scale)):
    axes[i].plot(A_sub[i,:])  # Plot i-th column
    axes[i].plot(B_sub[i,:])  # Plot i-th column
    axes[i].set_title(f'Brown scale= {brown_scale[i]}')
    axes[i].grid(True)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

# show correlation with noise 
noise_corr_A_sub=noise_corr_A_all[:,:,comp_show]
noise_corr_B_sub=noise_corr_B_all[:,:,comp_show]

comp_show_num=10
fig, axs=plt.subplots(comp_show_num,2, figsize=(10, comp_show_num * 3), sharex=True)
for k in range(comp_show_num):
    for i in range(np.shape(noise_corr_A_sub)[1]):
        axs[k,0].plot(brown_scale,np.abs(noise_corr_A_all[:,i,k]))
        axs[k,1].plot(brown_scale,np.abs(noise_corr_B_all[:,i,k]))
    axs[k,0].set_ylabel(f"comp {k}")
    axs[k,0].set_ylim([0,1])
    axs[k,1].set_ylim([0,1])
        
axs[0,0].set_title("RNN A")
axs[0,1].set_title("RNN B")
axs[-1,0].set_xlabel("Brown scale")
axs[-1,1].set_xlabel("Brown scale")




# plot all correlation 
n_cols = noise_corr_A_all.shape[1]
n_rows = 2  # First row for noise_corr_A_all and second for noise_corr_B_all

fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(4*n_cols, 8))
comp_show_num = 20
vmin = 0
vmax = 1

# Plot images in a loop for each column.
# Save a reference to one of the images for the colorbar.
for col in range(n_cols):
    # First row: RNN A
    imA = axs[0, col].imshow(noise_corr_A_all[:, col, :comp_show_num],
                              aspect='auto', cmap=parula, interpolation='none',
                              vmin=vmin, vmax=vmax)
    # Second row: RNN B
    imB = axs[1, col].imshow(noise_corr_B_all[:, col, :comp_show_num],
                              aspect='auto', cmap=parula, interpolation='none',
                              vmin=vmin, vmax=vmax)
    # Optionally add a title for each column if desired
    axs[0, col].set_title(f"Noise component {col+1}")

# Set y-axis labels for the first column of each row
axs[0, 0].set_ylabel("RNN A, brown scale", fontsize=12)
axs[1, 0].set_ylabel("RNN B, brown scale", fontsize=12)
# Set x-axis labels for the bottom row only
for col in range(n_cols):
    axs[1, col].set_xlabel("Components", fontsize=12)

# Reduce the number of y ticks (example: 5 ticks) and format them with two decimals.
n_ticks = 5
y_positions = np.linspace(0, len(brown_scale) - 1, n_ticks)
y_labels = [f"{brown_scale[int(round(pos))]:.2f}" for pos in y_positions]

# Apply the reduced/formatted y ticks to all subplots.
for ax in axs.flat:
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)

# Add an overall title.
plt.suptitle("Correlation between components and brown noise", fontsize=16)

# Adjust subplot layout to leave space on the right for the colorbar.
plt.subplots_adjust(right=0.8, top=0.9)

# Create a new axes for the global colorbar.
cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
# Use the first image as a reference for the colorbar.
cbar = fig.colorbar(imA, cax=cbar_ax)
cbar.set_label("Absolute Pearson correlation", fontsize=12)

plt.show()



#%%
# analyze single case: do pca 100 before fitting to prevent overfitting
from column_corr import pairwise_corr

foldername_1=r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\tsubame_models\rank2_noise1"


# decode and analyze
# analyze single case
# Parameters for the analysis (make sure these are defined in your code)
sample_size = 8 #16
trial_num = 12 # 24
pert_state = 0       # 0: perturb RNN A, 1: perturb RNN B
pert_noisesd = 2.0   # perturb noise standard deviation
stop = False
option = 0           # 0: use circular mean; 1: use arithmetic mean
trial1 = 2

pert_prob = 1/100
pert_A_prob = 0.5
order = 1  # order=1 means start with min_dur

# Generate perturbation time indices and a perturbation mask (pert_which)
max_ind = int(np.floor((min_dur + max_dur) * (np.floor((trial_num - trial1) / 2) * 19 / 20)))
pert_number = int(np.floor(max_ind * pert_prob))
vectors = []
for i in range(sample_size):
    time0 = np.random.randint(0, max_ind, pert_number)
    time0.sort()
    time0 = np.reshape(time0, (1, -1))
    vectors.append(time0)
time_1 = np.concatenate(vectors, axis=0)
pert_which = np.random.uniform(size=time_1.shape)
pert_which = pert_which < pert_A_prob


# Define the connection probability indices (assuming conProbability is defined)


# Choose the dimensionality reduction method ("pca", "cca", or "pls")
dim_method = "pls"  # or "cca" or "pls"
pre_method="pca"
Dim=100
lin_method="act_stack"
max_drop_num=10
dropout_num=np.concatenate(([None],np.arange(max_drop_num)))
# Instantiate the analysis class.
analysis = PerturbDecodeAnalyze(min_dur, max_dur, dt, dim_method, Dim=Dim, lin_method=lin_method)

# Initialize a dictionary to store the analysis results
Allinfo = {
    't_index':       [],
    'Confmat_A_ave': [],
    'Confmat_B_ave': [],
    'Confscore_A_ave':[],
    'Confscore_B_ave':[],
    'Offset_mat_ave':[],
    'temp_error_ave':[],
    'cat_error_ave': [],
    'pred_diff_sub': [],
    'pred_diff_real':[],
    'dim_method': dim_method,
    'Dim': Dim,
    'lin_method': lin_method,
    'dropout_num':dropout_num,
    
}

exp=1
brown_scale=1

t=0
k=0
confA_list   = []
confB_list   = []
confsc_A=[]
confsc_B=[]
offset_list  = []
terror_list  = []
caterr_list  = []
pred_diff2   = []
pred_diff_real_sub = []
loopind += 1
maxval = 0
con_prob = 0

# load noise_weights and model weights
# load noise_weights
# load model weights
noise_weights=np.load(os.path.join(foldername_1,"noise_weights.npy"))
rank=np.shape(noise_weights)[0]


# Build the base model (assume build_model is defined elsewhere)
model = analysis.build_model_brown(nUnit=nUnit, nInh=nInh, nInput=nInput,
                    con_prob=con_prob, maxval=maxval, ReLUalpha=ReLUalpha, seed1=seed1, tau=tau,
                    rank=rank, exp=1, noise_weights=noise_weights)
# Load weights from the checkpoint
#checkpoint_filepath = os.path.join(foldername_1,"epoch_09748.ckpt")
ckeckpoint_filepath2,_ = load_checkpoint_with_max_number(foldername_1)
#checkpoint_filepath=os.path.join(r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\tsubame_models\brown_noise_taskid_fixed_6","epoch_09708.ckpt")
model.load_weights(ckeckpoint_filepath2)


# Create an "activity model" to output intermediate layer activations
activity_model = Model(inputs=model.input, outputs=[layer.output for layer in model.layers[1:]])

# Generate input, output, and onset times using the analysis class method.
#-> it may be better to set brown_scale to 0
x, y, In_ons = analysis.makeInOut_sameint_brown(sample_size, trial_num, inputdur, nInput, order,brown_scale=brown_scale, rank=rank, exp=exp)

input_noise=np.matmul(x[:,:,1:],noise_weights)# batch, time, unit

output_and_activities = activity_model.predict(x)
activities_A = output_and_activities[1]
activities_B = output_and_activities[2]

# Compute averaged and stacked activations.
act_avg_A = analysis.avgAct2(activities_A, In_ons)
act_avg_B = analysis.avgAct2(activities_B, In_ons)
act_stack_A = analysis.concatAct(activities_A, In_ons)
act_stack_B = analysis.concatAct(activities_B, In_ons)
noise_stack=analysis.concatAct(input_noise,In_ons)
noise_stack_raw=analysis.concatAct(x[:,:,1:],In_ons)# batch, time, rank

#remove inactive components
act_stack_A,act_stack_B=analysis.remove_inactive(act_stack_A,act_stack_B)
act_avg_A,act_avg_B=analysis.remove_inactive_transform(act_avg_A,act_avg_B,dim=1)


# Create classification labels.
Class_per_sec = 1
class_A_train, class_B_train = analysis.make_classying_classes(act_stack_A, act_stack_B, Class_per_sec)


# take first 100 pca components to avoid overfitting
act_stack_A, act_stack_B, act_avg_A, act_avg_B =analysis.reduce_dimension_pre(act_avg_A, act_avg_B, act_stack_A, act_stack_B, methodname=pre_method, Dim=Dim)


#plot outputs
num_outputs=np.shape(y)[2]
fig, axes = plt.subplots(num_outputs, 1, figsize=(6, 2 * num_outputs), sharex=True)

for out_ind in range(num_outputs):
    ax = axes[out_ind] if num_outputs > 1 else axes  # Ensure correct axis selection for single subplot case
    ax.plot(output_and_activities[5][0, :, out_ind], linestyle='-', color=f'C{out_ind}', label=f'Output {out_ind}')
    ax.plot(y[0, :, out_ind], linestyle='--', color=f'C{out_ind}', label=f'Ground Truth {out_ind}')
    
    ax.set_ylabel(f'Out {out_ind}')  # Label for each subplot
    ax.legend(loc='best', fontsize='small')

# Set common x-axis label
plt.xlabel("Time Steps")

# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()


# Reduce dimensions using the chosen method.
if analysis.dim_method.lower() == "pca":
    proj_A_train, proj_B_train, proj_C_train = analysis.reduce_dimension(act_avg_A, act_avg_B, act_stack_A, act_stack_B)
else:
    proj_A_train, proj_B_train = analysis.reduce_dimension(act_avg_A, act_avg_B, act_stack_A, act_stack_B)

#calculate the correlation with noise components
noise_corr_A=pairwise_corr(noise_stack_raw,proj_A_train)# rank, components
noise_corr_B=pairwise_corr(noise_stack_raw,proj_B_train)



# calculate mean of the data
A_train_mean=analysis.trial_avg(proj_A_train)# mindur+maxdur,units
B_train_mean=analysis.trial_avg(proj_B_train)
# plot mean of each axis
p=9# number of components to display
ncols = int(np.ceil(np.sqrt(p)))      # Number of columns in the subplot grid
nrows = int(np.ceil(p / ncols))         # Number of rows needed
fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(12, 8))
axs = axs.flatten()  # Flatten to easily iterate over axes
for i in range(p):
    axs[i].plot(A_train_mean[:, i])
    axs[i].set_title(f"A Component {i}")
for ax in axs[p:]:
    ax.remove()
plt.tight_layout()
plt.show()
fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(12, 8))
axs = axs.flatten()  # Flatten to easily iterate over axes
for i in range(p):
    axs[i].plot(B_train_mean[:, i])
    axs[i].set_title(f"B Component {i}")
for ax in axs[p:]:
    ax.remove()
plt.tight_layout()
plt.show()

# plot each axis
fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(12, 8))
axs = axs.flatten()  # Flatten to easily iterate over axes
n=np.shape(proj_A_train)[0]
for i in range(p):
    axs[i].plot(proj_A_train[:, i])
    axs[i].set_title(f"A Component {i}")
    for x in np.arange(0, n, 1800):
        axs[i].axvline(x=x, color='red', linestyle='--', linewidth=1)
for ax in axs[p:]:
    ax.remove()
plt.tight_layout()
plt.show()
fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(12, 8))
axs = axs.flatten()  # Flatten to easily iterate over axes
for i in range(p):
    axs[i].plot(proj_B_train[:, i])
    axs[i].set_title(f"B Component {i}")
    for x in np.arange(0, n, 1800):
        axs[i].axvline(x=x, color='red', linestyle='--', linewidth=1)
for ax in axs[p:]:
    ax.remove()
plt.tight_layout()
plt.show()

# plot intra trial variance/ inter trial variance
score_A=analysis.intra_inter_var(proj_A_train)
score_B=analysis.intra_inter_var(proj_B_train)
plt.figure()
plt.plot(score_A,label="A: intra/inter var")
plt.plot(score_B,label="B: intra/inter var")
plt.legend()
plt.xlim([0,max_drop_num])




# plot correlation of noise with components
plt.figure()
for i in range(rank):
    plt.plot(np.abs(noise_corr_A[i, :max_drop_num]), linestyle='-', color=f'C{i}', label=f'A {i}' if i < 10 else "_nolegend_")
    plt.plot(np.abs(noise_corr_B[i, :max_drop_num]), linestyle='-.', color=f'C{i}', label=f'B {i}' if i < 10 else "_nolegend_")

# Add labels and legend
plt.xlabel("Components")
plt.ylabel("Absolute Correlation Coefficient")
plt.legend(title="Rank Index", ncol=1, fontsize='small', loc='upper left', bbox_to_anchor=(1.05, 1))

plt.show() 


# Create classifiers for decoding.
if analysis.dim_method.lower() == "pca":
    train_data=(proj_A_train, proj_B_train, proj_C_train)
    class_data=(class_A_train, class_B_train)
    clf_A=[]
    clf_B=[]
    clf_C=[]
    for ind in dropout_num:
        sub_A, sub_B, sub_C = analysis.create_train_classifier(analysis.remove_component_stack(train_data, ind), class_data)
        clf_A.append(sub_A)
        clf_B.append(sub_B)
        clf_C.append(sub_C)
else:
    train_data=(proj_A_train, proj_B_train)
    class_data=(class_A_train, class_B_train)
    clf_A=[]
    clf_B=[]
    for ind in dropout_num:
        sub_A, sub_B= analysis.create_train_classifier(analysis.remove_component_stack(train_data, ind), class_data)
        clf_A.append(sub_A)
        clf_B.append(sub_B)            



actpart_A, actpart_B = analysis.perturb_and_decode_noise_prob2_brown(
    trial1, time_1, pert_which, order,
    pert_noisesd, stop,
    sample_size, trial_num, inputdur, nInput,
    nUnit, nInh, con_prob, maxval, ReLUalpha, seed1, tau, model,
    brown_scale, rank,exp=exp,noise_weights=noise_weights
)

# Perform perturbation and decoding.
if analysis.dim_method.lower() == "pca":

    # Decode time predictions (returns pred_A, pred_B, pred_C)
    # create classes
    total_time_len=np.shape(actpart_A)[0]*np.shape(actpart_A)[2]
    right_class_A,right_class_B=analysis.make_classying_classes_2(total_time_len,total_time_len,Class_per_sec)

    # remove inactive axis
    actpart_A,actpart_B=analysis.remove_inactive_transform(actpart_A,actpart_B,dim=1)
    actpart_A, actpart_B=analysis.transform_stack_pre(actpart_A, actpart_B)
    data_all=(actpart_A, actpart_B)
    pred_A=[]
    pred_B=[]
    pred_C=[]
    for var,comp in enumerate(dropout_num):
        pred_A_sub, pred_B_sub, pred_C_sub = analysis.decode_time(data_all,
                                                      clf_A=clf_A[var],
                                                      clf_B=clf_B[var],
                                                      clf_C=clf_C[var],
                                                      remove_ind=comp,)
        pred_A.append(pred_A_sub)
        pred_B.append(pred_B_sub)
        pred_C.append(pred_C_sub)
else:

    total_time_len=np.shape(actpart_A)[0]*np.shape(actpart_A)[2]
    right_class_A,right_class_B=analysis.make_classying_classes_2(total_time_len,total_time_len,Class_per_sec)


    # Decode time predictions (returns pred_A, pred_B, pred_C)
    actpart_A,actpart_B=analysis.remove_inactive_transform(actpart_A,actpart_B,dim=1)
    actpart_A, actpart_B=analysis.transform_stack_pre(actpart_A, actpart_B)
    data_all=(actpart_A, actpart_B)
    pred_A=[]
    pred_B=[]
    for var,comp in enumerate(dropout_num):
        pred_A_sub, pred_B_sub = analysis.decode_time(data_all,
                                                      clf_A=clf_A[var],
                                                      clf_B=clf_B[var],
                                                      remove_ind=comp,)
        pred_A.append(pred_A_sub)
        pred_B.append(pred_B_sub)



# Compute the average of the decoded predictions.
classleng = int(1000 / (dt * Class_per_sec))
class_per_trial = int((min_dur + max_dur) / classleng)

predavg_A=[]
predavg_B=[]
predavg_C=[]
pred_diff2_sub=[]
from get_phase import get_phase

for ind in range(len(dropout_num)):
    if option == 0:
        predavg_A.append(scipy.stats.circmean(pred_A[ind], high=class_per_trial, low=1, axis=1))
        predavg_B.append(scipy.stats.circmean(pred_B[ind], high=class_per_trial, low=1, axis=1))
        if analysis.dim_method.lower() == "pca":
            predavg_C.append(scipy.stats.circmean(pred_C, high=class_per_trial, low=1, axis=1))
    else:
        predavg_A.append(np.mean(pred_A[ind], axis=1))
        predavg_B.append(np.mean(pred_B[ind], axis=1))
        if analysis.dim_method.lower() == "pca":
            predavg_C.append(np.mean(pred_C, axis=1))

    # Compute prediction differences using your get_phase function.
    pred_diff = np.round(get_phase(pred_A[ind] - pred_B[ind], class_per_trial, 'int'))
    pred_diff2_sub.append(np.linalg.norm(pred_diff))

pred_diff2.append(np.array(pred_diff2_sub)) # each list is length dropout_num


fig, axs = plt.subplots(1, 1, sharex=True, sharey='row', figsize=(10, 6))

axs.plot(pred_A[0], color='#ff7f0e', alpha=0.1)
axs.plot(pred_B[0], color='#1f77b4', alpha=0.1)
axs.plot(predavg_A[0], color='#ff7f0e', label='RNN A')
axs.plot(predavg_B[0], color='#1f77b4', label='RNN B')
axs.set_title('Decoded results')
axs.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
axs.set_yticks(Class_per_sec * np.array([0, 6, 12, 18]))
axs.set_yticklabels([0, 6, 12, 18])
xticks = axs.get_xticks()
scaled_xticks = np.round(xticks * 0.01)
axs.set_xticklabels(scaled_xticks)
#plt.suptitle(f'{k}: Connection Probability {conProbability[t]}')
plt.tight_layout()
plt.show()



# show confusion matrix
from Confmatrix import confmat, confscore
from error_analysis import get_cat_error, get_temp_error
pred_Aall_2=[None]*len(dropout_num)
pred_Ball_2=[None]*len(dropout_num)
pred_Call_2=[None]*len(dropout_num)
confmat_A=[None]*len(dropout_num)
confmat_B=[None]*len(dropout_num)
confscore_A=[None]*len(dropout_num)
confscore_B=[None]*len(dropout_num)  
pred_diff_real_sub2=[]
offset_mat=[None]*len(dropout_num) 
cat_error_rate=[None]*len(dropout_num) 
temp_error_rate=[None]*len(dropout_num)

for ind in range(len(dropout_num)):
    pred_Aall_2[ind] = pred_A[ind].flatten(order='F').reshape(-1, 1)
    pred_Ball_2[ind] = pred_B[ind].flatten(order='F').reshape(-1, 1)
    if analysis.dim_method.lower() == "pca":
        pred_Call_2[ind] = pred_C[ind].flatten(order='F').reshape(-1, 1)


    class_all=right_class_A.copy().reshape(-1, 1)
    confmat_A[ind]=confmat(class_all,pred_Aall_2[ind])
    confmat_B[ind]=confmat(class_all,pred_Ball_2[ind])
    confscore_A[ind]=confscore(confmat_A[ind],1)
    confscore_B[ind]=confscore(confmat_B[ind],1)


    # get offset from the actual time
    diff_A=get_phase(pred_Aall_2[ind]-class_all,class_per_trial,'int')
    diff_B=get_phase(pred_Ball_2[ind]-class_all,class_per_trial,'int')
    if analysis.dim_method.lower() == "pca":
        diff_C=get_phase(pred_Call_2[ind]-class_all,class_per_trial,'int')
        # get offset from the actual decoded results
        pred_diff_real_sub2.append([np.linalg.norm(diff_A),np.linalg.norm(diff_B),np.linalg.norm(diff_C)])
    else:
        pred_diff_real_sub2.append([np.linalg.norm(diff_A),np.linalg.norm(diff_B)])
    #create 2d histogram
    offset_mat[ind], xedges, yedges = np.histogram2d(diff_A.ravel(), diff_B.ravel(), bins=class_per_trial)


    #show categorical error
    group_bound=int(class_per_trial*(min_dur/(min_dur+max_dur)))-1
    cat_error_rate[ind]=get_cat_error(pred_Aall_2[ind], class_all, pred_Ball_2[ind], class_all, group_bound)

    # show temporal error
    temp_error_rate[ind]=get_temp_error(pred_Aall_2[ind], class_all, pred_Ball_2[ind], class_all, class_per_trial,3)


# Convert lists to NumPy arrays
pred_Aall_2 = np.array(pred_Aall_2)
pred_Ball_2 = np.array(pred_Ball_2)
pred_Call_2 = np.array(pred_Call_2)# if any(pred_Call_2) else None  # Handle case where PCA isn't used
confmat_A = np.array(confmat_A)
confmat_B = np.array(confmat_B)
confscore_A = np.array(confscore_A)
confscore_B = np.array(confscore_B)    
pred_diff_real_sub.append(np.array(pred_diff_real_sub2))
offset_mat=np.array(offset_mat)
cat_error_rate=np.array(cat_error_rate) # dropout_num, 4
temp_error_rate=np.array(temp_error_rate)




show_ind=0
plt.figure()
fig, axs = plt.subplots(1,2,figsize=(10, 6))
im0=axs[0].imshow(confmat_A[show_ind],aspect='auto', cmap=parula, interpolation='none',vmin=0,vmax=400)
im1=axs[1].imshow(confmat_B[show_ind],aspect='auto', cmap=parula, interpolation='none',vmin=0,vmax=400)
axs[0].set_box_aspect(1)
axs[1].set_box_aspect(1)
#axs[1].colorbar()
#cbar = fig.colorbar(im0, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
#plt.suptitle(f'Connection Probability {conProbability[t]}')
plt.show()



plt.figure(figsize=(10, 6))

# Set extent based on the bin edges for proper axis scaling
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
allpop = np.sum(offset_mat[show_ind])
# Use imshow to plot the data
im0 = plt.imshow((offset_mat[show_ind]/allpop).T, aspect='auto', cmap=parula, interpolation='none', vmin=0, vmax=250/allpop, extent=extent, origin='lower')
# Set the aspect of the axis to be equal
plt.gca().set_aspect('equal')
# Add color bar
cbar = plt.colorbar(im0, orientation='vertical', fraction=0.02, pad=0.04)
# Set axis labels
plt.xlabel('A offset')
plt.ylabel('B offset')
# Set the title
#plt.title(f'{k}:Connection Probability {conProbability[t]}')
# Show the plot
plt.show()



#show categorical error
from error_analysis import get_cat_error, get_temp_error
group_bound=int(class_per_trial*(min_dur/(min_dur+max_dur)))-1
cat_error_rate=get_cat_error(pred_Aall_2[show_ind], class_all, pred_Ball_2, class_all, group_bound)

catlabel = np.array(["M2 mis, PPC mis", "M2 mis, PPC cor", "M2 cor, PPC mis", "M2 cor, PPC cor"])
plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(10, 6))
axs[0].bar(catlabel,cat_error_rate)

# show temporal error
temp_error_rate=get_temp_error(pred_Aall_2[show_ind], class_all, pred_Ball_2, class_all, class_per_trial,3)
offset_temp=np.arange(-(np.ceil(class_per_trial/2)-1),np.floor(class_per_trial/2)+1)
axs[1].plot(offset_temp,temp_error_rate)
#plt.suptitle(f'{k}:Connection Probability {conProbability[t]}')
plt.show()
#print(f"{k}: loop {loopind} out of {len(tind)}")

plt.figure()
x_ax=np.arange(np.shape(confscore_A)[0])
plt.plot(x_ax,confscore_A)
plt.plot(x_ax,confscore_B)
#plt.xlim([0,max_drop_num])
plt.xlabel("components")
plt.ylabel("Accuracy")


plt.figure()
x_ax=np.arange(np.shape(confscore_A)[0]-1)+1
plt.plot(x_ax,confscore_A[0]-confscore_A[1:])
plt.plot(x_ax,confscore_B[0]-confscore_B[1:])
plt.xlim([0,max_drop_num])
plt.xlabel("components")
plt.ylabel("Delta accuracy")


#%% perturb and decode-> get lyapunov spectrum along certain axis

from column_corr import pairwise_corr
rep_num=1
brown_scale=1
exp=1
foldername_1=r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\tsubame_models\rank2_noise1"

conProbability=[0,1.e-05, 3.e-05, 1.e-04, 3.e-04, 1.e-03, 3.e-03, 1.e-02, 3.e-02, 1.e-01,3.e-01,1.e+00]
weight_max = [0.2] * len(conProbability)
seed_num=[1010,1011,1012,1013,1014,1015]

# decode and analyze
# analyze single case
# Parameters for the analysis (make sure these are defined in your code)
sample_size = 1 #16
trial_num = 5# 24
pert_state = 0       # 0: perturb RNN A, 1: perturb RNN B
pert_noisesd = 0# 0.003  # perturb noise standard deviation
stop = False
option = 0           # 0: use circular mean; 1: use arithmetic mean
trial1 = 2

pert_prob = 1/100
pert_A_prob = 0.5
order = 0  # order=1 means start with min_dur

dim_method = "cca"  # or "cca" or "pls"
pre_method="pca"
Dim=100
lin_method="act_stack"
max_drop_num=40

# for analyzing lyapunov
stateful=True
batch_size=10
start=None
option="norm_scale" # lyapunov, norm_scale
update_state=False


# Generate perturbation time indices and a perturbation mask (pert_which)
max_ind = int(np.floor((min_dur + max_dur) * (np.floor((trial_num - trial1) / 2) * 19 / 20)))
pert_number = int(np.floor(max_ind * pert_prob))
vectors = []
for i in range(sample_size):
    time0 = np.random.randint(0, max_ind, pert_number)
    time0.sort()
    time0 = np.reshape(time0, (1, -1))
    vectors.append(time0)
time_1 = np.concatenate(vectors, axis=0)
pert_which = np.random.uniform(size=time_1.shape)
pert_which = pert_which < pert_A_prob


# Choose the dimensionality reduction method ("pca", "cca", or "pls")

dropout_num=np.arange(max_drop_num)
# Instantiate the analysis class.
analysis = PerturbDecodeAnalyze(min_dur, max_dur, dt, dim_method, Dim=Dim, lin_method=lin_method)

# Initialize a dictionary to store the analysis results
Allinfo = {
    't_index':       [],
    'Confmat_A_ave': [],
    'Confmat_B_ave': [],
    'Confscore_A_ave':[],
    'Confscore_B_ave':[],
    'Offset_mat_ave':[],
    'temp_error_ave':[],
    'cat_error_ave': [],
    'pred_diff_sub': [],
    'pred_diff_real':[],
    'dim_method': dim_method,
    'Dim': Dim,
    'lin_method': lin_method,
    'dropout_num':dropout_num,
    'pre_method':pre_method,

}
print(Allinfo)
tind=np.arange(len(conProbability))

t=0
k=0

pred_acc=[]
pred_accA=[]
pred_accB=[]
pred_off=[]
pred_offA=[]
pred_offB=[]

noise_corr_A_all=[]
noise_corr_B_all=[]
confmat_A_all=[]
confmat_B_all=[]
intra_inter_var_A_all=[]
intra_inter_var_B_all=[]
offset_mat_all=[]
cat_error_rate_all=[]
temp_error_rate_all=[]


for rep_now in range(rep_num):

    confA_list   = []
    confB_list   = []
    confsc_A=[]
    confsc_B=[]
    offset_list  = []
    terror_list  = []
    caterr_list  = []
    pred_diff2   = []
    pred_diff_real_sub = []
    loopind += 1
    maxval = weight_max[t]
    con_prob = conProbability[t]

    # load noise_weights and model weights
    # load noise_weights
    foldername_1=r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\tsubame_models\rank2_noise1"
    noise_weights=np.load(os.path.join(foldername_1,"noise_weights.npy"))
    rank=np.shape(noise_weights)[0]


    # Build the base model (assume build_model is defined elsewhere)
    model = analysis.build_model_brown(nUnit=nUnit, nInh=nInh, nInput=nInput,
                        con_prob=con_prob, maxval=maxval, ReLUalpha=ReLUalpha, seed1=seed1, tau=tau,
                        rank=rank, exp=1, noise_weights=noise_weights)
    # Load weights from the checkpoint
    #checkpoint_filepath = os.path.join(foldername_1,"epoch_09748.ckpt")
    checkpoint_filepath,_ = load_checkpoint_with_max_number(foldername_1)
    
    model.load_weights(checkpoint_filepath)


    # Create an "activity model" to output intermediate layer activations
    activity_model = Model(inputs=model.input, outputs=[layer.output for layer in model.layers[1:]])

    # Generate input, output, and onset times using the analysis class method.
    #-> it may be better to set brown_scale to 0
    sample_size_temp=16
    trial_num_temp=24
    x, y, In_ons = analysis.makeInOut_sameint_brown(sample_size_temp, trial_num_temp, inputdur, nInput, order,brown_scale=brown_scale, rank=rank, exp=exp)
    
    print(f"brown_scale={brown_scale}, rank={rank}, repetition:{rep_now+1}, max_drop: {dropout_num[-1]} ")
    
    input_noise=np.matmul(x[:,:,1:],noise_weights)# batch, time, unit

    output_and_activities = activity_model.predict(x)
    activities_A = output_and_activities[1]
    activities_B = output_and_activities[2]

    # Compute averaged and stacked activations.
    act_avg_A = analysis.avgAct2(activities_A, In_ons)
    act_avg_B = analysis.avgAct2(activities_B, In_ons)
    act_stack_A = analysis.concatAct(activities_A, In_ons)
    act_stack_B = analysis.concatAct(activities_B, In_ons)
    noise_stack=analysis.concatAct(input_noise,In_ons)
    noise_stack_raw=analysis.concatAct(x[:,:,1:],In_ons)# batch, time, rank

    #remove inactive components
    act_stack_A,act_stack_B=analysis.remove_inactive(act_stack_A,act_stack_B)
    act_avg_A,act_avg_B=analysis.remove_inactive_transform(act_avg_A,act_avg_B,dim=1)


    # Create classification labels.
    Class_per_sec = 1
    class_A_train, class_B_train = analysis.make_classying_classes(act_stack_A, act_stack_B, Class_per_sec)
    
    # take first 100 pca components to avoid overfitting
    act_stack_A, act_stack_B, act_avg_A, act_avg_B =analysis.reduce_dimension_pre(act_avg_A, act_avg_B, act_stack_A, act_stack_B, methodname=pre_method, Dim=Dim+1)



    #plot outputs
    num_outputs=np.shape(y)[2]




    # Reduce dimensions using the chosen method.
    if analysis.dim_method.lower() == "pca":
        proj_A_train, proj_B_train, proj_C_train = analysis.reduce_dimension(act_avg_A, act_avg_B, act_stack_A, act_stack_B)
    else:
        proj_A_train, proj_B_train = analysis.reduce_dimension(act_avg_A, act_avg_B, act_stack_A, act_stack_B)


    # get transformation matrices
    trans_A_sub, trans_B_sub=analysis.get_transfomation_matrix()
    trans_A, trans_B=analysis.make_entire_trans_mat(nUnit,trans_A_sub,trans_B_sub, norm=False)
    
    norm_A=np.linalg.norm(trans_A[0], axis=0, keepdims=True)
    norm_B=np.linalg.norm(trans_B[0], axis=0, keepdims=True)
    trans_A_norm=np.divide(trans_A[0], norm_A, where=norm_A!=0)
    trans_B_norm=np.divide(trans_B[0], norm_B, where=norm_B!=0)
    
    # get vectors along each cca coordinates
    comp_dir_A=analysis.get_ortho_mat(trans_A_norm)
    comp_dir_B=analysis.get_ortho_mat(trans_B_norm)
    
    # normalize comp_dir so that it becomes 1 after dot product
    comp_dir_A/=np.diag(np.matmul(trans_A[0].T, comp_dir_A))
    comp_dir_B/=np.diag(np.matmul(trans_B[0].T, comp_dir_B))
    
    mat_AB=np.concatenate((comp_dir_A,comp_dir_B),axis=0)
    
    """
    # test whether they are orthogonal
    A_dot=trans_A_norm.T @ comp_dir_A
    B_dot=trans_B_norm.T @ comp_dir_B
    fig,axs=plt.subplots(1,2)
    im0=axs[0].imshow(A_dot)
    im1=axs[1].imshow(B_dot)
    fig.colorbar(im1, ax=axs[1])
    plt.show()
    
    # get matrix that returns orthogonal vector
    ind=0
    mat=analysis.get_ortho_vec(trans_A[0],ind)
    ortho_vec=mat @ np.random.rand(nUnit,1)
    ortho_vec/=np.linalg.norm(ortho_vec, axis=0)
    
    dot=ortho_vec.T @ trans_A[0]
    plt.figure()
    plt.plot(dot.T)
    
    # plot correlation with noise weights
    plt.figure()
    plt.plot(np.abs(noise_weights[[0],:]@comp_dir_A).T)
    plt.plot(np.abs(noise_weights[[1],:]@comp_dir_A).T)
    
    
    """
    
    
    # test
    """
    trans_A_test=trans_A.T
    trans_B_test=trans_B.T
    trans_A_test,trans_B_test=analysis.remove_inactive_transform(trans_A_test,trans_B_test)
    trans_A_test,trans_B_test=analysis.transform_stack_pre(trans_A_test,trans_B_test)
    trans_A_test, trans_B_test=analysis.reduce_dim_transform(np.squeeze(trans_A_test), np.squeeze(trans_B_test))
    """

    #calculate the correlation with noise components
    noise_corr_A=pairwise_corr(noise_stack_raw,proj_A_train)# rank, components
    noise_corr_B=pairwise_corr(noise_stack_raw,proj_B_train)



    # calculate mean of the data
    A_train_mean=analysis.trial_avg(proj_A_train)# mindur+maxdur,units
    B_train_mean=analysis.trial_avg(proj_B_train)



    # plot intra trial variance/ inter trial variance
    score_A=analysis.intra_inter_var(proj_A_train)
    score_B=analysis.intra_inter_var(proj_B_train)

    intra_inter_var_A_all.append(np.array(score_A))
    intra_inter_var_B_all.append(np.array(score_B))





    noise_corr_A_all.append((np.abs(noise_corr_A[:, :max_drop_num])))
    noise_corr_B_all.append((np.abs(noise_corr_B[:, :max_drop_num])))




    # run the rnn and get activities
    ind1=time_1
    outputs=analysis.run_model_with_jacobian(trial1, ind1, pert_which, order, pert_noisesd, stop,
                                    sample_size, trial_num, inputdur, nInput,
                                    nUnit, nInh, con_prob, maxval, ReLUalpha, seed1, tau, model,
                                    brown_scale, rank, exp=exp, noise_weights=noise_weights,
                                    stateful=stateful, start=start,
                                    option=option,test_mat=mat_AB, batch_size=batch_size,
                                    update_state=update_state)    
    outputs, Rmat=outputs
    
    if option.lower()=="lyapunov":
        Rmat2=analysis.get_lyapunov(Rmat,start) # (time, output_dim=2*nUnit)
        sort_ind=tf.argsort(Rmat2[-1,:],direction="DESCENDING")
        Rmat2 = tf.gather(Rmat2, sort_ind, axis=1)  # gather columns in sorted order
        plt.figure()
        plt.plot(Rmat2[-1,:])
        plt.xlabel("axis")
        plt.ylabel(f"lyapunov exp")
        plt.title(f"lyap spectrum")
    elif option.lower()=="norm_scale":
        Rmat2=analysis.get_lyapunov_along_axis(Rmat,start)#(time, ncol)
        Rmat2_mean=tf.math.reduce_mean(Rmat2,axis=0) # (ncol)
        plt.figure()
        plt.plot(Rmat2_mean)
        plt.xlabel("CCA components")
        plt.ylabel(f"lyapunov exp")
        plt.title(f"lyap exp along cca comp, update: {update_state}")
        
        plt.figure()
        plt.imshow(tf.transpose(Rmat2),aspect='auto', cmap=parula, interpolation='none')
        plt.colorbar()
        plt.ylabel("CCA components")
        plt.xlabel("Time (0.1s)")
        plt.title(f"lyap exp along cca comp, update state: {update_state}")
            

#%%# load data and analyze
import os
import numpy as np
dim_method='pls'
savepath=r'C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\tsubame_models\rank2_noise1'
#data = np.load(os.path.join(savepath, f'small_pls_bs7_pls.npz'), allow_pickle=True)
data= np.load(r"C:\Users\RHIRAsimulation\AppData\Local\anaconda3\envs\HirotoRNN3\RNN_models\tsubame_models\rank2_noise1\small_pls_bs1_pls.npz", allow_pickle=True)
noise_weights=np.load(os.path.join(savepath,"noise_weights.npy"))
rank=np.shape(noise_weights)[0]
# Extract variables dynamically
globals().update(data)  # Automatically assign variables with saved names


print(data.files)

dt=10
Class_per_sec = 1
classleng = int(1000 / (dt * Class_per_sec))
class_per_trial = int((min_dur + max_dur) / classleng)


mean_accA = np.mean(pred_accA, axis=0)
std_err_accA = np.std(pred_accA, axis=0) / np.sqrt(pred_accA.shape[0])
mean_accB = np.mean(pred_accB, axis=0)
std_err_accB = np.std(pred_accB, axis=0) / np.sqrt(pred_accB.shape[0])
x_acc = np.arange(np.shape(pred_accA)[1])

# Compute statistics for Plot 2: Delta Accuracy
mean_offA = np.mean(pred_offA, axis=0)
std_err_offA = np.std(pred_offA, axis=0) / np.sqrt(pred_offA.shape[0])
mean_offB = np.mean(pred_offB, axis=0)
std_err_offB = np.std(pred_offB, axis=0) / np.sqrt(pred_offB.shape[0])
# Define a common x–axis for plots 2, 3, and 4:
x_common = np.arange(max_drop_num)+1

# Compute statistics for Plot 3: Noise Correlation using noise_corr_A (the mean)
# (Assuming noise_corr_A_all has shape (rank, N) so that taking mean over iterations gives a (rank, N) array)
noise_corr_A = np.mean(noise_corr_A_all, axis=0)
noise_corr_B = np.mean(noise_corr_B_all, axis=0)
se_A = np.std(noise_corr_A_all, axis=0) / np.sqrt(noise_corr_A_all.shape[0])
se_B = np.std(noise_corr_B_all, axis=0) / np.sqrt(noise_corr_B_all.shape[0])

# Compute statistics for Plot 4: Intra/Inter Variance
score_A_all = np.array([a[:max_drop_num] for a in intra_inter_var_A_all])
score_B_all = np.array([a[:max_drop_num] for a in intra_inter_var_B_all])
score_A = np.mean(score_A_all, axis=0)
score_B = np.mean(score_B_all, axis=0)
se_score_A = np.std(score_A_all, axis=0) / np.sqrt(score_A_all.shape[0])
se_score_B = np.std(score_B_all, axis=0) / np.sqrt(score_B_all.shape[0])

# Create a figure with 4 subplots.
# Subplot 1 uses its own x-axis, subplots 2-4 share x_common.
fig = plt.figure(figsize=(12, 16))
gs = fig.add_gridspec(4, 1, hspace=0.4)
ax1 = fig.add_subplot(gs[0, 0])              # Plot 1: Accuracy (independent x)
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)              # Plot 2: Delta Accuracy (shared x)
ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)  # Plot 3: Noise Correlation (shared x)
ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)  # Plot 4: Intra/Inter Variance (shared x)

# --- Plot 1: Accuracy ---
ax1.plot(x_acc, mean_accA, label="RNN A", color="blue")
ax1.fill_between(x_acc, mean_accA - std_err_accA, mean_accA + std_err_accA, 
                 color="blue", alpha=0.2)
ax1.plot(x_acc, mean_accB, label="RNN B", color="orange")
ax1.fill_between(x_acc, mean_accB - std_err_accB, mean_accB + std_err_accB, 
                 color="orange", alpha=0.2)
ax1.set_ylabel("Accuracy")
ax1.set_title("Average Accuracy over 10 iterations")
ax1.legend()
ax1.grid()

# --- Plot 2: Delta Accuracy ---
ax2.plot(x_common, mean_offA[:max_drop_num], label="RNN A", color="blue")
ax2.fill_between(x_common, mean_offA[:max_drop_num] - std_err_offA[:max_drop_num],
                 mean_offA[:max_drop_num] + std_err_offA[:max_drop_num],
                 color="blue", alpha=0.2)
ax2.plot(x_common, mean_offB[:max_drop_num], label="RNN B", color="orange")
ax2.fill_between(x_common, mean_offB[:max_drop_num] - std_err_offB[:max_drop_num],
                 mean_offB[:max_drop_num] + std_err_offB[:max_drop_num],
                 color="orange", alpha=0.2)
ax2.set_ylabel("Delta Accuracy")
ax2.set_title(f"Method: {dim_method}")
ax2.legend()
ax2.grid()

# --- Plot 3: Noise Correlation ---
# Here we use noise_corr_A (and noise_corr_B) which are assumed to be 2D arrays of shape (rank, N)
for i in range(rank):
    ax3.plot(x_common, np.abs(noise_corr_A[i, :max_drop_num]), linestyle='-', 
             color=f'C{i}', label=f'A {i}' if i < 10 else None)
    ax3.plot(x_common, np.abs(noise_corr_B[i, :max_drop_num]), linestyle='-.', 
             color=f'C{i}', label=f'B {i}' if i < 10 else None)
    ax3.fill_between(x_common, 
                     np.abs(noise_corr_A[i, :max_drop_num] - se_A[i, :max_drop_num]),
                     np.abs(noise_corr_A[i, :max_drop_num] + se_A[i, :max_drop_num]),
                     color=f'C{i}', alpha=0.2)
    ax3.fill_between(x_common, 
                     np.abs(noise_corr_B[i, :max_drop_num] - se_B[i, :max_drop_num]),
                     np.abs(noise_corr_B[i, :max_drop_num] + se_B[i, :max_drop_num]),
                     color=f'C{i}', alpha=0.2)
ax3.set_ylabel("Absolute Correlation Coefficient")
ax3.set_title(f"Noise Correlation ({dim_method})")
ax3.legend(title="Rank Index", ncol=1, fontsize='small', 
           loc='upper left', bbox_to_anchor=(1.05, 1))
ax3.grid()
ax3.set_xlim([0, max_drop_num])

# --- Plot 4: Intra/Inter Variance ---
ax4.plot(x_common, score_A, label="A: intra/inter var", color='C0')
ax4.plot(x_common, score_B, label="B: intra/inter var", color='C1')
ax4.fill_between(x_common, score_A - se_score_A, score_A + se_score_A, 
                 color='C0', alpha=0.2)
ax4.fill_between(x_common, score_B - se_score_B, score_B + se_score_B, 
                 color='C1', alpha=0.2)
ax4.set_ylabel("Intra/Inter Variance")
ax4.set_title(f"Intra/Inter Variance ({dim_method})")
ax4.legend()
ax4.grid()
ax4.set_xlim([0, max_drop_num])
ax4.set_xlabel("Components")

plt.show()






# plot confmat
confmat_A=np.mean(confmat_A_all,axis=0)
confmat_B=np.mean(confmat_B_all,axis=0)

show_ind=0
plt.figure()
fig, axs = plt.subplots(1,2,figsize=(10, 6))
im0=axs[0].imshow(confmat_A[show_ind]/np.sum(confmat_A[show_ind]),aspect='auto', cmap=parula, interpolation='none',vmin=0,vmax=0.01)
im1=axs[1].imshow(confmat_B[show_ind]/np.sum(confmat_B[show_ind]),aspect='auto', cmap=parula, interpolation='none',vmin=0,vmax=0.01)
axs[0].set_box_aspect(1)
axs[1].set_box_aspect(1)
#axs[1].colorbar()
#cbar = fig.colorbar(im0, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
#plt.suptitle(f'Connection Probability {conProbability[t]}')
plt.show()


# plot offset
offset_mat=np.mean(offset_mat_all,axis=0)
allpop = np.sum(offset_mat[show_ind])
# Use imshow to plot the data
im0 = plt.imshow((offset_mat[show_ind]/allpop).T, aspect='auto', 
                 cmap=parula, 
                 interpolation='none',
                 vmin=0, 
                 vmax=0.002,  
                 origin='lower')
# Set the aspect of the axis to be equal
plt.gca().set_aspect('equal')
# Add color bar
cbar = plt.colorbar(im0, orientation='vertical', fraction=0.02, pad=0.04)
# Set axis labels
plt.xlabel('A offset')
plt.ylabel('B offset')
# Set the title
#plt.title(f'{k}:Connection Probability {conProbability[t]}')
# Show the plot
plt.title(f"method: {dim_method} decoding offset")
plt.show()




# plot categorical and temporal error
catlabel = np.array(["M2 mis, PPC mis", "M2 mis, PPC cor", "M2 cor, PPC mis", "M2 cor, PPC cor"])
cat_error_rate=np.mean(cat_error_rate_all,axis=0)
plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(10, 6))
axs[0].bar(catlabel,cat_error_rate[show_ind,:])

# show temporal error
temp_error_rate=np.mean(temp_error_rate_all,axis=0)
offset_temp=np.arange(-(np.ceil(class_per_trial/2)-1),np.floor(class_per_trial/2)+1)
axs[1].plot(offset_temp,temp_error_rate[show_ind,:])
#plt.suptitle(f'{k}:Connection Probability {conProbability[t]}')
plt.show()


#plot delta accuracy and correlation with noise
# get average correlation with each rank of noise
noise_corr_A_avg=np.mean(noise_corr_A,axis=0)
noise_corr_B_avg=np.mean(noise_corr_B,axis=0)

fig,axs=plt.subplots(2,2,figsize=(8,8))
axs[0, 0].scatter(mean_offA, noise_corr_A_avg)
axs[0, 0].set_xlabel("delta accuracy")
axs[0, 0].set_ylabel("correlation with noise")
axs[0, 0].set_title("RNN A")

axs[1, 0].scatter(mean_offB, noise_corr_B_avg)
axs[1, 0].set_xlabel("delta accuracy")
axs[1, 0].set_ylabel("correlation with noise")
axs[1, 0].set_title("RNN B")

axs[0, 1].scatter(mean_offA, score_A)
axs[0, 1].set_xlabel("delta accuracy")
axs[0, 1].set_ylabel("intra/inter var")
axs[0, 1].set_title("RNN A")

axs[1, 1].scatter(mean_offB, score_B)
axs[1, 1].set_xlabel("delta accuracy")
axs[1, 1].set_ylabel("intra/inter var")
axs[1, 1].set_title("RNN B")
plt.tight_layout()