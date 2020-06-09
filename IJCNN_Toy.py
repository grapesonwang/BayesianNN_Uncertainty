#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:45:30 2020

@author: peng
"""

#%% Import needed packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import edward as ed
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from edward.models import Normal
#%% Air Quality Data Processing
#from os.path import dirname, join as pjoin
#import scipy.io as sio
## Data importation and pre-preprocessing
#data_dir = pjoin('/media/peng/Grapeson/Said_Munir/AirQuality', 'Said_Munir')
##mat_fname = pjoin(data_dir, 'origin_ThreeWeekData.mat')
#mat_fname = pjoin(data_dir, 'norm_ThreeWeekData.mat')
#
#mat_data = sio.loadmat(mat_fname)
#
#FirstWeekData = mat_data['norm_FirstWeekData']
#SecondWeekData = mat_data['norm_SecondWeekData']
#ThirdWeekData = mat_data['norm_ThirdWeekData']
#%% Neural Network Approximation
plt.style.use('ggplot')

#%%

def build_toy_dataset(N=50, noise_std=0.001):
  x = np.linspace(-3, 3, num=N)
  y = np.cos(x) + np.random.normal(0, noise_std, size=N)
  x = x.astype(np.float32).reshape((N, 1))
  y = y.astype(np.float32)
  return x, y
  
def neural_network(x, W_0, W_1, b_0, b_1):
  h = tf.tanh(tf.matmul(x, W_0) + b_0)
  h = tf.tanh(tf.matmul(h, W_1) + b_1)
  return tf.reshape(h, [-1])

#% This is for the original network outputs generation
def neural_network_layerwise(x, W_0, W_1, b_0, b_1):    
  #% Mean of the first layer
  h_0 = tf.tanh(tf.matmul(x, W_0) + b_0)
  h_0_node_0 = tf.reshape(h_0[:, 0], [tf.size(h_0[:, 0]), 1])
  h_0_node_1 = tf.reshape(h_0[:, 1], [tf.size(h_0[:, 1]), 1])
  
  #% Mean of the second layer
  h_1_node_1 = tf.tanh(tf.matmul(h_0, W_1) + b_1)
  
  return tf.reshape(h_0_node_0, [-1]), tf.reshape(h_0_node_1, [-1]), tf.reshape(h_1_node_1, [-1])


def tensor_neural_network_layerwise(x, W_0, W_1, b_0, b_1, N, sample_num):
  
#  stack_h_00 = tf.Variable(shape=(N, sample_num))
#  stack_h_01 = tf.Variable(shape=(N, sample_num))
#  stack_h_11 = tf.Variable(shape=(N, sample_num))
  stack_h_00_list = []
  stack_h_01_list = []
  stack_h_11_list = []
 
#  stack_h_00 = tf.Variable([N, sample_num])
#  stack_h_01 = tf.Variable([N, sample_num])
#  stack_h_11 = tf.Variable([N, sample_num])
  #mus_h_0_node_0, mus_h_0_node_1, mus_h_1_node_1 = old_neural_network_layerwise(x, W_0, W_1, b_0, b_1)
  
  
  for vi_step in range(sample_num):
    mus_h_0_node_0, mus_h_0_node_1, mus_h_1_node_1 = neural_network_layerwise(x, W_0.sample(), W_1.sample(), b_0.sample(), b_1.sample())
    
#    h_0 = tf.sigmoid(tf.matmul(x, W_0) + b_0)
#    mus_h_0_node_0 = tf.reshape(h_0[:, 0], [tf.size(h_0[:, 0]), 1])
#    mus_h_0_node_1 = tf.reshape(h_0[:, 1], [tf.size(h_0[:, 1]), 1])
    stack_h_00_list.append(mus_h_0_node_0)
    stack_h_01_list.append(mus_h_0_node_1)
    stack_h_11_list.append(mus_h_1_node_1)
    
#    stack_h_00[:, vi_step] = mus_h_0_node_0
#    stack_h_01[:, vi_step] = mus_h_0_node_1
#    stack_h_11[:, vi_step] = mus_h_1_node_1
  stack_h_00 = tf.stack(stack_h_00_list)
  stack_h_01 = tf.stack(stack_h_01_list)
  stack_h_11 = tf.stack(stack_h_11_list)
  
  print('/////////////////////////////////')
  print(stack_h_00)
  
  mean_stack_h_00, sigma_stack_h_00 = tf.nn.moments(stack_h_00, [0])
  mean_stack_h_01, sigma_stack_h_01 = tf.nn.moments(stack_h_01, [0])
  mean_stack_h_11, sigma_stack_h_11 = tf.nn.moments(stack_h_11, [0])
 
  print('/////////////////////////////////')
  print(mean_stack_h_00)
#  mean_stack_h_00 = np.mean(stack_h_00, axis = 0)
#  mean_stack_h_01 = np.mean(stack_h_01, axis = 0)
#  mean_stack_h_11 = np.mean(stack_h_11, axis = 0)
#  
#  sigma_stack_h_00 = np.std(stack_h_00, axis = 1)
#  sigma_stack_h_01 = np.std(stack_h_01, axis = 1)
#  sigma_stack_h_11 = np.std(stack_h_11, axis = 1) 
  mean_stack_node_00 = tf.expand_dims(mean_stack_h_00, 1)
  mean_stack_node_01 = tf.expand_dims(mean_stack_h_01, 1)
  mean_stack_node_11 = tf.expand_dims(mean_stack_h_11, 1)
  
  sigma_stack_node_00 = tf.expand_dims(sigma_stack_h_00, 1)
  sigma_stack_node_01 = tf.expand_dims(sigma_stack_h_01, 1)
  sigma_stack_node_11 = tf.expand_dims(sigma_stack_h_11, 1)
  
  
  org_VI_Exp_0 = tf.concat([mean_stack_node_00, mean_stack_node_01], 1)
  org_VI_Cov_0 = tf.concat([sigma_stack_node_00, sigma_stack_node_01], 1)
  
  org_VI_Exp_1 = mean_stack_node_11
  org_VI_Cov_1 = sigma_stack_node_11
  
  return org_VI_Exp_0, org_VI_Cov_0, org_VI_Exp_1, org_VI_Cov_1



#%% Codes for uncertainty quantification


def uncertainty(x, sigmaW_0, sigmaW_1, sigmaB_0, sigmaB_1, meanW_0, meanW_1, meanB_0, meanB_1, Ksi):
    #%% Layer Zero
    #% From the original inputs to generate the nonlinear output of the first layer
    meanLayer_0 = tf.matmul(x, meanW_0) + meanB_0  #meanB_0 is 1*2, same as meanW_0, x is 400*1 so far
    sigmaLayer_0 = tf.square(tf.matmul(x, sigmaW_0)) + tf.square(sigmaB_0) 
    
    #% Cabature points from the non-Standard Gaussian Distribution
    sigLyr_0_Col_1 = tf.reshape(tf.sqrt(sigmaLayer_0[:,0]), [tf.size(sigmaLayer_0[:,0]), 1])
    sigLyr_0_Col_2 = tf.reshape(tf.sqrt(sigmaLayer_0[:,1]), [tf.size(sigmaLayer_0[:,1]), 1])
    
    mnLyr_0_Col_1 = tf.reshape(meanLayer_0[:,0], [tf.size(meanLayer_0[:,0]), 1])
    mnLyr_0_Col_2 = tf.reshape(meanLayer_0[:,1], [tf.size(meanLayer_0[:,1]), 1])
    
    #% Cabature points for non-standard Gaussian distribution --> CKF 2009 
    rawCbrPts_Lyr0_1 = tf.matmul(sigLyr_0_Col_1, Ksi) + tf.concat([mnLyr_0_Col_1, mnLyr_0_Col_1], 1)
    rawCbrPts_Lyr0_2 = tf.matmul(sigLyr_0_Col_2, Ksi) + tf.concat([mnLyr_0_Col_2, mnLyr_0_Col_2], 1)
    
    rawCbrPts_1_Col_1 = tf.reshape(rawCbrPts_Lyr0_1[:,0], [tf.size(rawCbrPts_Lyr0_1[:,0]), 1])
    rawCbrPts_1_Col_2 = tf.reshape(rawCbrPts_Lyr0_1[:,1], [tf.size(rawCbrPts_Lyr0_1[:,1]), 1])
    
    rawCbrPts_2_Col_1 = tf.reshape(rawCbrPts_Lyr0_2[:,0], [tf.size(rawCbrPts_Lyr0_2[:,0]), 1])
    rawCbrPts_2_Col_2 = tf.reshape(rawCbrPts_Lyr0_2[:,1], [tf.size(rawCbrPts_Lyr0_2[:,1]), 1])
    
    #% Cabature Points after nonlinear propagation
    newCbrPts_1_Col_1 = tf.tanh(rawCbrPts_1_Col_1)
    newCbrPts_1_Col_2 = tf.tanh(rawCbrPts_1_Col_2)
    
    newCbrPts_2_Col_1 = tf.tanh(rawCbrPts_2_Col_1)
    newCbrPts_2_Col_2 = tf.tanh(rawCbrPts_2_Col_2) 

    #% Expectation calculation
    inptLyr_1_node_1 = tf.multiply(0.5, (newCbrPts_1_Col_1 + newCbrPts_1_Col_2))
    inptLyr_1_node_2 = tf.multiply(0.5, (newCbrPts_2_Col_1 + newCbrPts_2_Col_2))
   
    #% Covariance calculation
    covLyr_1_node_1 = tf.multiply(0.5, (tf.square(newCbrPts_1_Col_1) + tf.square(newCbrPts_1_Col_2))) - tf.square(inptLyr_1_node_1)
    covLyr_1_node_2 = tf.multiply(0.5, (tf.square(newCbrPts_2_Col_1) + tf.square(newCbrPts_2_Col_2))) - tf.square(inptLyr_1_node_2)
    
    #% Return Expectation and Covariance
    Expt_Lyr_0 = tf.concat([inptLyr_1_node_1, inptLyr_1_node_2], 1)
    Cov_Lyr_0 = tf.concat([covLyr_1_node_1, covLyr_1_node_2], 1)
    
    #% Concatenate mean and covariance
    ExpCov_Lyr_0 = tf.concat([Expt_Lyr_0, Cov_Lyr_0], 1)
    
    #%% Layer One
    #xLyr_1 = tf.tanh(tf.matmul(x, meanW_0) + meanB_0)  # The inputs to Layer 1
    xLyr_1 = Expt_Lyr_0
    
    mnLyr_1 = tf.matmul(xLyr_1, meanW_1) + meanB_1    # Linear combination
    sigLyr_1 = tf.matmul(tf.square(xLyr_1), tf.square(sigmaW_1)) + tf.square(sigmaB_1)
    
    rawCbrPts_Lyr1_1 = tf.matmul(tf.sqrt(sigLyr_1), Ksi) + mnLyr_1

    #% Cabature points for non-standard Gaussian distribution --> CKF 2009    
    rawCbrPts_Lyr1_Col_1 = tf.reshape(rawCbrPts_Lyr1_1[:,0], [tf.size(rawCbrPts_Lyr1_1[:,0]), 1])
    rawCbrPts_Lyr1_Col_2 = tf.reshape(rawCbrPts_Lyr1_1[:,1], [tf.size(rawCbrPts_Lyr1_1[:,1]), 1])
    
    #% Cabature Points after nonlinear propagation
    newCbrPts_Lyr1_Col_1 = tf.tanh(rawCbrPts_Lyr1_Col_1)
    newCbrPts_Lyr1_Col_2 = tf.tanh(rawCbrPts_Lyr1_Col_2)    
    
    #% Expectation calculation
    inptLyr_2_node_1 = tf.multiply(0.5, (newCbrPts_Lyr1_Col_1 + newCbrPts_Lyr1_Col_2))
   
    #% Covariance calculation
    covLyr_2_node_1 = tf.multiply(0.5, (tf.square(newCbrPts_Lyr1_Col_1) + tf.square(newCbrPts_Lyr1_Col_2))) - tf.square(inptLyr_2_node_1)
    
    #% Outputs for uncertainty quantification
    ExpCbr = tf.square(newCbrPts_Lyr1_Col_1) + tf.square(newCbrPts_Lyr1_Col_2)
    ExpExp = tf.square(inptLyr_2_node_1)
    
    #% Concatenate mean and covariance
    ExpCov_Lyr_1 = tf.concat([inptLyr_2_node_1, covLyr_2_node_1], 1)
        
    #%% Return Values
    return ExpCov_Lyr_0, ExpCov_Lyr_1, ExpCbr, ExpExp    
#%% Define Functions
ed.set_seed(42)
N = 50  # number of data points
D = 1   # number of features

#% Num for loss computation
N_loss = N
x_train, y_train = build_toy_dataset(N)

#%% Weight Initialisation
org_f = 5.0   # the scale of the noise
W_0 = Normal(loc=tf.zeros([D, 2]), scale = org_f * tf.ones([D, 2]))
W_1 = Normal(loc=tf.zeros([2, 1]), scale = org_f * tf.ones([2, 1]))
b_0 = Normal(loc=tf.zeros(2), scale = org_f * tf.ones(2))
b_1 = Normal(loc=tf.zeros(1), scale = org_f * tf.ones(1))

Delta = Normal(loc=tf.zeros(1), scale=tf.ones(1)/10)

x = x_train
y = Normal(loc=neural_network(x, W_0, W_1, b_0, b_1),
           scale=0.1 * tf.ones(N)) # y is not used at all
#loc_11, loc_01, loc_00 = neural_network(x, W_0, W_1, b_0, b_1)
#y = Normal(loc=loc_11,
#           scale=0.1 * tf.ones(N)) # y is not used at all
#%% This is where to set the variables
# One dimensional Cabature Points
Ksi = tf.constant([[1.0,-1.0]])
sample_num = 10

# Standard deviation of the network
sigmaW_0 = tf.ones([D, 2])
sigmaW_1 = tf.ones([2, 1])
sigmaB_0 = tf.ones(2)
sigmaB_1 = tf.ones(1)
sigmaDelta = tf.ones(1)

# Mean of the network
meanW_0 = tf.zeros([D, 2])
meanW_1 = tf.zeros([2, 1])
meanB_0 = tf.zeros(2)
meanB_1 = tf.zeros(1)
meanDelta = tf.zeros(1)



#%% For Variational Inference           
qW_0 = Normal(loc=tf.get_variable("qW_0/loc", [D, 2]),
              scale=tf.nn.softplus(tf.get_variable("qW_0/scale", [D, 2])))
qW_1 = Normal(loc=tf.get_variable("qW_1/loc", [2, 1]),
              scale=tf.nn.softplus(tf.get_variable("qW_1/scale", [2, 1])))
qb_0 = Normal(loc=tf.get_variable("qb_0/loc", [2]),
              scale=tf.nn.softplus(tf.get_variable("qb_0/scale", [2])))
qb_1 = Normal(loc=tf.get_variable("qb_1/loc", [1]),
              scale=tf.nn.softplus(tf.get_variable("qb_1/scale", [1])))

qDelta = Normal(loc=tf.get_variable("qDelta/loc", [1]),
              scale=tf.nn.softplus(tf.get_variable("qDelta/scale", [1])))

#%% Parameters setting after Variational Inference
#% Standard deviation of the network
sigmaqW_0 = qW_0.scale
sigmaqW_1 = qW_1.scale
sigmaqB_0 = qb_0.scale
sigmaqB_1 = qb_1.scale
sigmaqDelta = qDelta.scale

#% Mean of the network
meanqW_0 = qW_0.loc
meanqW_1 = qW_1.loc
meanqB_0 = qb_0.loc
meanqB_1 = qb_1.loc
meanqDelta = qDelta.loc

pW_0 = qW_0
pW_1 = qW_1
pb_0 = qb_0
pb_1 = qb_1
                            
#%% Sample functions from variational model to visualize fits.
rs = np.random.RandomState(0)
inputs = np.linspace(-3, 3, num=N, dtype=np.float32)  #The dimension of inputs here is 400 alone
x = tf.expand_dims(inputs, 1) #Expand the dimension of 'inputs' to 400*1 https://stackoverflow.com/questions/39008821/tensorflow-when-use-tf-expand-dims
mus = tf.stack(
    [neural_network(x, qW_0.sample(), qW_1.sample(),
                    qb_0.sample(), qb_1.sample())
     for _ in range(10)]) # 10 is the times of sampling process
 
#% Original data collection
#sample_num = 5
mus_h_0_node_0, mus_h_0_node_1, mus_h_1_node_1 = neural_network_layerwise(x, qW_0.sample(), qW_1.sample(), qb_0.sample(), qb_1.sample())

    
# For Uncertainty Quantification
ExpCov_0, ExpCov_1, ExpCbr, ExpExp  = uncertainty(x, sigmaW_0, sigmaW_1, sigmaB_0, sigmaB_1, meanW_0, meanW_1, meanB_0, meanB_1, Ksi)

#% Call the Uncertainty function after Variational Inference
VI_ExpCov_0, VI_ExpCov_1, VI_ExpCbr, VI_ExpExp = uncertainty(x, sigmaqW_0, sigmaqW_1, sigmaqB_0, sigmaqB_1, meanqW_0, meanqW_1, meanqB_0, meanqB_1, Ksi)

#% VI

#org_VI_Exp_0, org_VI_Cov_0, org_VI_Exp_1, org_VI_Cov_1  = neural_network_layerwise(x, qW_0, qW_1, qb_0, qb_1, N, sample_num)


#% Cost calculation

loss_y_value = tf.reshape(y_train, [tf.size(y_train), 1])

loss_x = tf.placeholder(tf.float32, [None, 1])
loss_y_ = tf.placeholder(tf.float32, [None, 1])


loss_w = tf.Variable(tf.ones([1, 1]))

loss_y_est = loss_w * loss_x

#%
loss = tf.reduce_sum((loss_y_ - loss_y_est) * (loss_y_ - loss_y_est))
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss)


         
#%% FIRST VISUALIZATION (prior)
sess = ed.get_session()
tf.global_variables_initializer().run()

#%%
outputs = mus.eval()
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Iteration: 0")
ax.plot(x_train, y_train, 'ks', alpha=0.5, label='(x, y)')
ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='prior draws')
ax.plot(inputs, outputs[1:].T, 'r', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()

#%% Expectation and Covariance Computation and Visualisation
output_ExpCov_0 = sess.run(ExpCov_0)
output_ExpCov_1 = sess.run(ExpCov_1)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Uncertainty: 0")
ax.plot(x_train, y_train, 'ks', alpha=0.5, label='(x, y)')
#ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='prior draws')
ax.plot(inputs, output_ExpCov_1[:,0], 'r', lw=2, alpha=0.5)
ax.plot(inputs, output_ExpCov_1[:,1], 'g', lw=2, alpha=0.5)

ax.plot(inputs, output_ExpCov_0[:,0], 'b', lw=2, alpha=0.5)
ax.plot(inputs, output_ExpCov_0[:,2], 'm', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()

#%%
try_node_00 = sess.run(mus_h_0_node_0)
try_node_01 = sess.run(mus_h_0_node_1)
try_node_11 = sess.run(mus_h_1_node_1)

#%% Variational Inference 
inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1}, data={y: y_train})
    
#outputs = mus.eval()

#% Step wise running
inf_iter = 500   # Iterative number 
inf_smpl = 50      # Iterative sample number


mean_qW_0_ary = np.zeros(shape=(inf_iter, 2))
mean_qW_1_ary = np.zeros(shape=(inf_iter, 2))
mean_qB_0_ary = np.zeros(shape=(inf_iter, 2))
mean_qB_1_ary = np.zeros(shape=(inf_iter, 1))

sigma_qW_0_ary = np.zeros(shape=(inf_iter, 2))
sigma_qW_1_ary = np.zeros(shape=(inf_iter, 2))
sigma_qB_0_ary = np.zeros(shape=(inf_iter, 2))
sigma_qB_1_ary = np.zeros(shape=(inf_iter, 1))

#% Final results, including layerwise expectation and covariance
mnLyr_0_node_1 = np.zeros(shape=(N, inf_iter))
mnLyr_0_node_2 = np.zeros(shape=(N, inf_iter))

sigmaLyr_0_node_1 = np.zeros(shape=(N, inf_iter))
sigmaLyr_0_node_2 = np.zeros(shape=(N, inf_iter))

mnLyr_1_node_1 = np.zeros(shape=(N, inf_iter))
sigmaLyr_1_node_1 = np.zeros(shape=(N, inf_iter))

new_mnLyr_1_node_1 = np.zeros(shape=(N, inf_iter))
new_sigmaLyr_1_node_1 = np.zeros(shape=(N, inf_iter))

#% Edward Inference Configuration
inference.initialize(n_iter=inf_iter, n_samples = inf_smpl)
tf.global_variables_initializer().run()

#% Loss calculation 

loss_y_value = sess.run(loss_y_value)


#% For the original VI
mnstack_h_0_node_0 = np.zeros(shape=(N, inf_iter))
mnstack_h_0_node_1 = np.zeros(shape=(N, inf_iter))
mnstack_h_1_node_1 = np.zeros(shape=(N, inf_iter))

sigmastack_h_0_node_0 = np.zeros(shape=(N, inf_iter))
sigmastack_h_0_node_1 = np.zeros(shape=(N, inf_iter))
sigmastack_h_1_node_1 = np.zeros(shape=(N, inf_iter))

for _, i in zip(range(inference.n_iter), range(inf_iter)):
  info_dict = inference.update()
  inference.print_progress(info_dict)
  
  #% Parameter storation
  #% Standar derivation
  qW_0_scl = sess.run(qW_0.scale)
  sigma_qW_0_ary[i,:] = qW_0_scl
  
  qW_1_scl = sess.run(qW_1.scale)
  sigma_qW_1_ary[i,:] = qW_1_scl.T

  qB_0_scl = sess.run(qb_0.scale)
  sigma_qB_0_ary[i,:] = qB_0_scl.reshape([1,2])

  qB_1_scl = sess.run(qb_1.scale)
  sigma_qB_1_ary[i,:] = qB_1_scl.reshape([1,1])
  
  #% Mean 
  qW_0_loc = sess.run(qW_0.loc)
  sigma_qW_0_ary[i,:] = qW_0_loc
  
  qW_1_loc = sess.run(qW_1.loc)
  sigma_qW_1_ary[i,:] = qW_1_loc.T

  qB_0_loc = sess.run(qb_0.loc)
  sigma_qB_0_ary[i,:] = qB_0_loc

  qB_1_loc = sess.run(qb_1.loc)
  sigma_qB_1_ary[i,:] = qB_1_loc.reshape([1,1]) 
  
  #% Uncertainty quantification step by step
  otpt_VI_ExpCov_0 = sess.run(VI_ExpCov_0)
  mnLyr_0_node_1[:, i] = otpt_VI_ExpCov_0[:, 0]
  mnLyr_0_node_2[:, i] = otpt_VI_ExpCov_0[:, 1]
  
  sigmaLyr_0_node_1[:, i] = np.sqrt(np.abs(otpt_VI_ExpCov_0[:, 2]))
  sigmaLyr_0_node_2[:, i] = np.sqrt(np.abs(otpt_VI_ExpCov_0[:, 3]))
  
  otpt_VI_ExpCov_1 = sess.run(VI_ExpCov_1)
  mnLyr_1_node_1[:, i] = otpt_VI_ExpCov_1[:, 0]
  sigmaLyr_1_node_1[:, i] = np.sqrt(np.abs(otpt_VI_ExpCov_1[:, 1]))
  
  loss_x_value = tf.reshape(mnLyr_1_node_1[:, i], [tf.size(mnLyr_1_node_1[:, i]), 1])
  loss_x_value = sess.run(loss_x_value)
  
  for step in range(10):
    sess.run(train_step, feed_dict = {loss_x:loss_x_value,
                                    loss_y_:loss_y_value})
  #% Calibrated data
  loss_w_last = sess.run(loss_w)
  loss_opt = 1.0/loss_w_last - 1.0
  new_CbrWeight = 1.0/(2.0*(1.0 + loss_opt))
  print(new_CbrWeight)
  otpt_VI_ExpCbr = sess.run(VI_ExpCbr)
  otpt_VI_ExpExp = sess.run(VI_ExpExp)
  new_mnLyr_1_node_1[:, i] = 2.0 * otpt_VI_ExpCov_1[:, 0] * new_CbrWeight
  sigma_1_inter = new_CbrWeight * otpt_VI_ExpCbr - otpt_VI_ExpExp
  new_sigmaLyr_1_node_1[:, i] = np.sqrt(np.abs(sigma_1_inter[:, 0]))
  
  #% Original data collection
#  org_VI_Exp_0, org_VI_Cov_0, org_VI_Exp_1, org_VI_Cov_1  = neural_network_layerwise(x, qW_0.sample(), qW_1.sample(), qb_0.sample(), qb_1.sample(), sample_num)

  stack_h_00 = np.zeros(shape=(N, sample_num))
  stack_h_01 = np.zeros(shape=(N, sample_num))
  stack_h_11 = np.zeros(shape=(N, sample_num))
  #% Before/After approximation
#  for vi_step in range(sample_num):
#    stack_h_00[:, vi_step] = sess.run(mus_h_0_node_0)
#    stack_h_01[:, vi_step] = sess.run(mus_h_0_node_1)
#    stack_h_11[:, vi_step] = sess.run(mus_h_1_node_1)
    
  #% For uncertainty approximation
  for vi_step in range(sample_num):
    stack_h_00[:, vi_step] = sess.run(mus_h_0_node_0)
    stack_h_01[:, vi_step] = sess.run(mus_h_0_node_1)
    
    mus_h_1_node_1_before_appr = sess.run(mus_h_1_node_1)
    mus_h_1_node_1_feed = tf.reshape(mus_h_1_node_1_before_appr, [tf.size(mus_h_1_node_1_before_appr), 1])
    mus_h_1_node_1_feed = sess.run(mus_h_1_node_1_feed)
    for vi_step_appr in range(10):
      sess.run(train_step, feed_dict = {loss_x:mus_h_1_node_1_feed,
                                        loss_y_:loss_y_value}) 
    loss_w_appr = sess.run(loss_w)
    stack_h_11[:, vi_step] = sess.run(mus_h_1_node_1) * loss_w_appr
  #% For uncertainty approximation
    
     
    
  mean_stack_h_00 = np.mean(stack_h_00, axis = 1)
  mean_stack_h_01 = np.mean(stack_h_01, axis = 1)
  mean_stack_h_11 = np.mean(stack_h_11, axis = 1)
  
  sigma_stack_h_00 = np.std(stack_h_00, axis = 1)
  sigma_stack_h_01 = np.std(stack_h_01, axis = 1)
  sigma_stack_h_11 = np.std(stack_h_11, axis = 1) 
  
#  otpt_org_VI_Exp_0 = sess.run(org_VI_Exp_0)
#  otpt_org_VI_Cov_0 = sess.run(org_VI_Cov_0)
#  
#  otpt_org_VI_Exp_1 = sess.run(org_VI_Exp_1)
#  otpt_org_VI_Cov_1 = sess.run(org_VI_Cov_1)
  
#  mnstack_h_0_node_0[:, i] = otpt_org_VI_Exp_0[:, 0]
#  mnstack_h_0_node_1[:, i] = otpt_org_VI_Exp_0[:, 1]
#  mnstack_h_1_node_1[:, i] = otpt_org_VI_Exp_1[:, 0]
#
#  sigmastack_h_0_node_0[:, i] = otpt_org_VI_Cov_0[:, 0]
#  sigmastack_h_0_node_1[:, i] = otpt_org_VI_Cov_0[:, 1]
#  sigmastack_h_1_node_1[:, i] = otpt_org_VI_Cov_1[:, 0]  
        
  
  mnstack_h_0_node_0[:, i] = mean_stack_h_00
  mnstack_h_0_node_1[:, i] = mean_stack_h_01
  mnstack_h_1_node_1[:, i] = mean_stack_h_11

  sigmastack_h_0_node_0[:, i] = sigma_stack_h_00
  sigmastack_h_0_node_1[:, i] = sigma_stack_h_01
  sigmastack_h_1_node_1[:, i] = sigma_stack_h_11 

   
inference.finalize()

#%% SECOND VISUALIZATION (posterior) After Variational Inference


sigmastack_h_0_node_0 = np.sqrt(sigmastack_h_0_node_0)
sigmastack_h_0_node_1 = np.sqrt(sigmastack_h_0_node_1)
sigmastack_h_1_node_1 = np.sqrt(sigmastack_h_1_node_1) 
#%%
outputs = mus.eval()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Iteration: 1000")
ax.plot(x_train, y_train, 'ks', alpha=0.5, label='(x, y)')
ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='posterior draws')
ax.plot(inputs, outputs[1:].T, 'r', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()

#%% Expectation and Covariance Computation and Visualisation after Variational Inference
output_VI_ExpCov_0 = sess.run(VI_ExpCov_0)
output_VI_ExpCov_1 = sess.run(VI_ExpCov_1)
output_VI_ExpCbr = sess.run(VI_ExpCbr)
output_VI_ExpExp = sess.run(VI_ExpExp)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Uncertainty After VI: 0")
ax.plot(x_train, y_train, 'ks', alpha=0.5, label='(x, y)')
#ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='prior draws')
#ax.plot(inputs, output_VI_ExpCov_1[:,0], 'r', lw=2, alpha=0.5)
#ax.plot(inputs, output_VI_ExpCov_1[:,1], 'g', lw=2, alpha=0.5)

ax.plot(inputs, new_mnLyr_1_node_1[:,inf_iter-1], 'r', lw=2, alpha=0.5)
ax.plot(inputs, new_sigmaLyr_1_node_1[:,inf_iter-1], 'g', lw=2, alpha=0.5)

ax.plot(inputs, output_VI_ExpCov_0[:,0], 'b', lw=2, alpha=0.5)
ax.plot(inputs, output_VI_ExpCov_0[:,2], 'm', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()