#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 23:02:13 2020

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

def build_toy_dataset(N=50, noise_std=0.1):
  x = np.linspace(-3, 3, num=N)
  y = np.cos(x) + np.random.normal(0, noise_std, size=N)
  x = x.astype(np.float32).reshape((N, 1))
  y = y.astype(np.float32)
  return x, y
  
def neural_network(x, W_0, W_1, b_0, b_1):
  h = tf.tanh(tf.matmul(x, W_0) + b_0)
  h = tf.matmul(h, W_1) + b_1
  return tf.reshape(h, [-1])

#%% Codes for uncertainty quantification
def old_uncertainty(x, sigmaW_0, sigmaW_1, sigmaB_0, sigmaB_1, meanW_0, meanW_1, meanB_0, meanB_1, Ksi):
    #%% Layer Zero
    #% From the original inputs to generate the nonlinear output of the first layer
    meanLayer_0 = tf.matmul(x, meanW_0) + meanB_0  #meanB_0 is 1*2, same as meanW_0, x is 400*1 so far
    sigmaLayer_0 = tf.square(tf.matmul(x, sigmaW_0)) + tf.square(sigmaB_0) 
    
    #% Cabature points from the non-Standard Gaussian Distribution
    sigLyr_0_Col_1 = tf.reshape(tf.sqrt(sigmaLayer_0[:,0]), [tf.size(sigmaLayer_0[:,0]), 1])
    sigLyr_0_Col_2 = tf.reshape(tf.sqrt(sigmaLayer_0[:,1]), [tf.size(sigmaLayer_0[:,1]), 1])
    
    mnLyr_0_Col_1 = tf.reshape(meanLayer_0[:,0], [tf.size(meanLayer_0[:,0]), 1])
    mnLyr_0_Col_2 = tf.reshape(meanLayer_0[:,1], [tf.size(meanLayer_0[:,1]), 1])
    
    rawCbrPts_Lyr0_1 = tf.matmul(sigLyr_0_Col_1, Ksi) + tf.concat([mnLyr_0_Col_1, mnLyr_0_Col_1], 1)
    rawCbrPts_Lyr0_2 = tf.matmul(sigLyr_0_Col_2, Ksi) + tf.concat([mnLyr_0_Col_2, mnLyr_0_Col_2], 1)
    
    rawCbrPts_1_Col_1 = tf.reshape(rawCbrPts_Lyr0_1[:,0], [tf.size(rawCbrPts_Lyr0_1[:,0]), 1])
    rawCbrPts_1_Col_2 = tf.reshape(rawCbrPts_Lyr0_1[:,1], [tf.size(rawCbrPts_Lyr0_1[:,1]), 1])
    
    rawCbrPts_2_Col_1 = tf.reshape(rawCbrPts_Lyr0_2[:,0], [tf.size(rawCbrPts_Lyr0_2[:,0]), 1])
    rawCbrPts_2_Col_2 = tf.reshape(rawCbrPts_Lyr0_2[:,1], [tf.size(rawCbrPts_Lyr0_2[:,1]), 1])
    
    #% Cabature points for the Standard Gaussian Distribution
    newCbrPts_1_Col_1 = tf.multiply(tf.sqrt(1.0/np.pi), tf.tanh(tf.multiply(tf.sqrt(tf.multiply(2.0, sigLyr_0_Col_1)), rawCbrPts_1_Col_1) + mnLyr_0_Col_1))
    newCbrPts_1_Col_2 = tf.multiply(tf.sqrt(1.0/np.pi), tf.tanh(tf.multiply(tf.sqrt(tf.multiply(2.0, sigLyr_0_Col_1)), rawCbrPts_1_Col_2) + mnLyr_0_Col_1))
    
    newCbrPts_2_Col_1 = tf.multiply(tf.sqrt(1.0/np.pi), tf.tanh(tf.multiply(tf.sqrt(tf.multiply(2.0, sigLyr_0_Col_2)), rawCbrPts_2_Col_1) + mnLyr_0_Col_2))
    newCbrPts_2_Col_2 = tf.multiply(tf.sqrt(1.0/np.pi), tf.tanh(tf.multiply(tf.sqrt(tf.multiply(2.0, sigLyr_0_Col_2)), rawCbrPts_2_Col_2) + mnLyr_0_Col_2))

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
    xLyr_1 = tf.tanh(tf.matmul(x, meanW_0) + meanB_0)  # The inputs to Layer 1
#    xLyr_1_Col_1 = tf.reshape(xLyr_1[:,0], [tf.size(xLyr_1[:,0]), 1])
#    xLyr_1_Col_2 = tf.reshape(xLyr_1[:,1], [tf.size(xLyr_1[:,1]), 1])
    
    mnLyr_1 = tf.matmul(xLyr_1, meanW_1) + meanB_1    # Linear combination
    sigLyr_1 = tf.matmul(tf.square(xLyr_1), tf.square(sigmaW_1)) + tf.square(sigmaB_1)
    
    rawCbrPts_Lyr1_1 = tf.matmul(tf.sqrt(sigLyr_1), Ksi) + mnLyr_1
    
    rawCbrPts_Lyr1_Col_1 = tf.reshape(rawCbrPts_Lyr1_1[:,0], [tf.size(rawCbrPts_Lyr1_1[:,0]), 1])
    rawCbrPts_Lyr1_Col_2 = tf.reshape(rawCbrPts_Lyr1_1[:,1], [tf.size(rawCbrPts_Lyr1_1[:,1]), 1])

    newCbrPts_Lyr1_Col_1 = tf.multiply(tf.sqrt(1.0/np.pi), tf.tanh(tf.multiply(tf.sqrt(tf.multiply(2.0, sigLyr_1)), rawCbrPts_Lyr1_Col_1) + mnLyr_1))
    newCbrPts_Lyr1_Col_2 = tf.multiply(tf.sqrt(1.0/np.pi), tf.tanh(tf.multiply(tf.sqrt(tf.multiply(2.0, sigLyr_1)), rawCbrPts_Lyr1_Col_2) + mnLyr_1))
    
    #% Expectation calculation
    inptLyr_2_node_1 = tf.multiply(0.5, (newCbrPts_Lyr1_Col_1 + newCbrPts_Lyr1_Col_2))
   
    #% Covariance calculation
    covLyr_2_node_1 = tf.multiply(0.5, (tf.square(newCbrPts_Lyr1_Col_1) + tf.square(newCbrPts_Lyr1_Col_2))) - tf.square(inptLyr_2_node_1)
    
    #% Concatenate mean and covariance
    ExpCov_Lyr_1 = tf.concat([inptLyr_2_node_1, covLyr_2_node_1], 1)
    
    nonLinLyr_1 = tf.tanh(xLyr_1)                    # NonLinear Activation
        
    #%% Return Values
    return ExpCov_Lyr_0,ExpCov_Lyr_1
    
def new_uncertainty(x, sigmaW_0, sigmaW_1, sigmaB_0, sigmaB_1, meanW_0, meanW_1, meanB_0, meanB_1, Ksi):
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
    
#    #% Cabature points for the Standard Gaussian Distribution
#    newCbrPts_1_Col_1 = tf.multiply(tf.sqrt(1.0/np.pi), tf.tanh(tf.multiply(tf.sqrt(tf.multiply(2.0, sigLyr_0_Col_1)), rawCbrPts_1_Col_1) + mnLyr_0_Col_1))
#    newCbrPts_1_Col_2 = tf.multiply(tf.sqrt(1.0/np.pi), tf.tanh(tf.multiply(tf.sqrt(tf.multiply(2.0, sigLyr_0_Col_1)), rawCbrPts_1_Col_2) + mnLyr_0_Col_1))
#    
#    newCbrPts_2_Col_1 = tf.multiply(tf.sqrt(1.0/np.pi), tf.tanh(tf.multiply(tf.sqrt(tf.multiply(2.0, sigLyr_0_Col_2)), rawCbrPts_2_Col_1) + mnLyr_0_Col_2))
#    newCbrPts_2_Col_2 = tf.multiply(tf.sqrt(1.0/np.pi), tf.tanh(tf.multiply(tf.sqrt(tf.multiply(2.0, sigLyr_0_Col_2)), rawCbrPts_2_Col_2) + mnLyr_0_Col_2))

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
    xLyr_1 = tf.tanh(tf.matmul(x, meanW_0) + meanB_0)  # The inputs to Layer 1
#    xLyr_1_Col_1 = tf.reshape(xLyr_1[:,0], [tf.size(xLyr_1[:,0]), 1])
#    xLyr_1_Col_2 = tf.reshape(xLyr_1[:,1], [tf.size(xLyr_1[:,1]), 1])
    
    mnLyr_1 = tf.matmul(xLyr_1, meanW_1) + meanB_1    # Linear combination
    sigLyr_1 = tf.matmul(tf.square(xLyr_1), tf.square(sigmaW_1)) + tf.square(sigmaB_1)
    
    rawCbrPts_Lyr1_1 = tf.matmul(tf.sqrt(sigLyr_1), Ksi) + mnLyr_1

    #% Cabature points for non-standard Gaussian distribution --> CKF 2009    
    rawCbrPts_Lyr1_Col_1 = tf.reshape(rawCbrPts_Lyr1_1[:,0], [tf.size(rawCbrPts_Lyr1_1[:,0]), 1])
    rawCbrPts_Lyr1_Col_2 = tf.reshape(rawCbrPts_Lyr1_1[:,1], [tf.size(rawCbrPts_Lyr1_1[:,1]), 1])
    
    #% Cabature Points after nonlinear propagation
    newCbrPts_Lyr1_Col_1 = tf.tanh(rawCbrPts_Lyr1_Col_1)
    newCbrPts_Lyr1_Col_2 = tf.tanh(rawCbrPts_Lyr1_Col_2)    

#    newCbrPts_Lyr1_Col_1 = tf.multiply(tf.sqrt(1.0/np.pi), tf.tanh(tf.multiply(tf.sqrt(tf.multiply(2.0, sigLyr_1)), rawCbrPts_Lyr1_Col_1) + mnLyr_1))
#    newCbrPts_Lyr1_Col_2 = tf.multiply(tf.sqrt(1.0/np.pi), tf.tanh(tf.multiply(tf.sqrt(tf.multiply(2.0, sigLyr_1)), rawCbrPts_Lyr1_Col_2) + mnLyr_1))
    
    #% Expectation calculation
    inptLyr_2_node_1 = tf.multiply(0.5, (newCbrPts_Lyr1_Col_1 + newCbrPts_Lyr1_Col_2))
   
    #% Covariance calculation
    covLyr_2_node_1 = tf.multiply(0.5, (tf.square(newCbrPts_Lyr1_Col_1) + tf.square(newCbrPts_Lyr1_Col_2))) - tf.square(inptLyr_2_node_1)
    
    #% Concatenate mean and covariance
    ExpCov_Lyr_1 = tf.concat([inptLyr_2_node_1, covLyr_2_node_1], 1)
    
    nonLinLyr_1 = tf.tanh(xLyr_1)                    # NonLinear Activation
        
    #%% Return Values
    return ExpCov_Lyr_0,ExpCov_Lyr_1  

def uncertainty(x, sigmaW_0, sigmaW_1, sigmaB_0, sigmaB_1, meanW_0, meanW_1, meanB_0, meanB_1, Ksi, delta):
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
    inptLyr_1_node_1 = tf.multiply(0.5+delta, (newCbrPts_1_Col_1 + newCbrPts_1_Col_2))
    inptLyr_1_node_2 = tf.multiply(0.5+delta, (newCbrPts_2_Col_1 + newCbrPts_2_Col_2))
   
    #% Covariance calculation
    covLyr_1_node_1 = tf.multiply(0.5+delta, (tf.square(newCbrPts_1_Col_1) + tf.square(newCbrPts_1_Col_2))) - tf.square(inptLyr_1_node_1)
    covLyr_1_node_2 = tf.multiply(0.5+delta, (tf.square(newCbrPts_2_Col_1) + tf.square(newCbrPts_2_Col_2))) - tf.square(inptLyr_1_node_2)
    
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
    inptLyr_2_node_1 = tf.multiply(0.5+delta, (newCbrPts_Lyr1_Col_1 + newCbrPts_Lyr1_Col_2))
   
    #% Covariance calculation
    covLyr_2_node_1 = tf.multiply(0.5+delta, (tf.square(newCbrPts_Lyr1_Col_1) + tf.square(newCbrPts_Lyr1_Col_2))) - tf.square(inptLyr_2_node_1)
    
    #% Concatenate mean and covariance
    ExpCov_Lyr_1 = tf.concat([inptLyr_2_node_1, covLyr_2_node_1], 1)
    
    nonLinLyr_1 = tf.tanh(xLyr_1)                    # NonLinear Activation
        
    #%% Return Values
    return ExpCov_Lyr_0,ExpCov_Lyr_1    
#%% Define Functions
ed.set_seed(42)
N = 50  # number of data points
D = 1   # number of features
x_train, y_train = build_toy_dataset(N)

#%% Weight Initialisation
W_0 = Normal(loc=tf.zeros([D, 2]), scale=tf.ones([D, 2]))
W_1 = Normal(loc=tf.zeros([2, 1]), scale=tf.ones([2, 1]))
b_0 = Normal(loc=tf.zeros(2), scale=tf.ones(2))
b_1 = Normal(loc=tf.zeros(1), scale=tf.ones(1))

Delta = Normal(loc=tf.zeros(1), scale=tf.ones(1))



x = x_train
y = Normal(loc=neural_network(x, W_0, W_1, b_0, b_1),
           scale=0.1 * tf.ones(N)) # y is not used at all

#%% This is where to set the variables
# One dimensional Cabature Points
#Ksi = tf.constant([[1.0,-1.0]])
Ksi = Normal(loc=tf.Variable([[1.0,-1.0]]), scale=tf.Variable([[1.0,1.0]]))

# Standard deviation of the network
sigmaW_0 = tf.ones([D, 2])
sigmaW_1 = tf.ones([2, 1])
sigmaB_0 = tf.ones(2)
sigmaB_1 = tf.ones(1)
sigmaDelta = tf.ones(1)
sigmaKsi = tf.constant([[1.0,1.0]])

# Mean of the network
meanW_0 = tf.zeros([D, 2])
meanW_1 = tf.zeros([2, 1])
meanB_0 = tf.zeros(2)
meanB_1 = tf.zeros(1)
meanDelta = tf.zeros(1)
meanKsi = tf.constant([[1.0,-1.0]])

#meanW_0 = tf.ones([D, 2])
#meanW_1 = tf.ones([2, 1])
#meanB_0 = tf.ones(2)
#meanB_1 = tf.ones(1)
#meanDelta = tf.ones(1)


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

qKsi = Normal(loc=tf.get_variable("qKsi/loc", [D, 2]),
              scale=tf.nn.softplus(tf.get_variable("qKsi/scale", [D, 2])))

#%% Parameters setting after Variational Inference
#% Standard deviation of the network
sigmaqW_0 = qW_0.scale
sigmaqW_1 = qW_1.scale
sigmaqB_0 = qb_0.scale
sigmaqB_1 = qb_1.scale
sigmaqDelta = qDelta.scale
sigmaqKsi = qKsi.scale

#% Mean of the network
meanqW_0 = qW_0.loc
meanqW_1 = qW_1.loc
meanqB_0 = qb_0.loc
meanqB_1 = qb_1.loc
meanqDelta = qDelta.loc
meanqKsi = qKsi.loc
                            
#%% Sample functions from variational model to visualize fits.
rs = np.random.RandomState(0)
inputs = np.linspace(-5, 5, num=400, dtype=np.float32)  #The dimension of inputs here is 400 alone
x = tf.expand_dims(inputs, 1) #Expand the dimension of 'inputs' to 400*1 https://stackoverflow.com/questions/39008821/tensorflow-when-use-tf-expand-dims
mus = tf.stack(
    [neural_network(x, qW_0.sample(), qW_1.sample(),
                    qb_0.sample(), qb_1.sample())
     for _ in range(10)]) # 10 is the times of sampling process
    
# For Uncertainty Quantification
ExpCov_0, ExpCov_1 = uncertainty(x, sigmaW_0, sigmaW_1, sigmaB_0, sigmaB_1, meanW_0, meanW_1, meanB_0, meanB_1, meanKsi, meanDelta)

#% Call the Uncertainty function after Variational Inference
VI_ExpCov_0, VI_ExpCov_1 = uncertainty(x, sigmaqW_0, sigmaqW_1, sigmaqB_0, sigmaqB_1, meanqW_0, meanqW_1, meanqB_0, meanqB_1, meanqKsi, meanqDelta)
         
#%% FIRST VISUALIZATION (prior)
sess = ed.get_session()
tf.global_variables_initializer().run()
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

#%% Variational Inference 
inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1, 
                     Delta: qDelta, Ksi: qKsi}, data={y: y_train})
    
#% Integrated running,
#inference.run(n_iter=100, n_samples=15)


#% Step wise running
inf_iter = 1000     # Iterative number 
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
mnLyr_0_node_1 = np.zeros(shape=(400, inf_iter))
mnLyr_0_node_2 = np.zeros(shape=(400, inf_iter))

sigmaLyr_0_node_1 = np.zeros(shape=(400, inf_iter))
sigmaLyr_0_node_2 = np.zeros(shape=(400, inf_iter))

mnLyr_1_node_1 = np.zeros(shape=(400, inf_iter))
sigmaLyr_1_node_1 = np.zeros(shape=(400, inf_iter))

#% Edward Inference Configuration
inference.initialize(n_iter=inf_iter, n_samples = inf_smpl)
tf.global_variables_initializer().run()

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
  
  sigmaLyr_0_node_1[:, i] = otpt_VI_ExpCov_0[:, 2]
  sigmaLyr_0_node_2[:, i] = otpt_VI_ExpCov_0[:, 3]
  
  otpt_VI_ExpCov_1 = sess.run(VI_ExpCov_1)
  mnLyr_1_node_1[:, i] = otpt_VI_ExpCov_1[:, 0]
  sigmaLyr_1_node_1[:, i] = otpt_VI_ExpCov_1[:, 1]
  
inference.finalize()





#%% SECOND VISUALIZATION (posterior) After Variational Inference
outputs = mus.eval()


##%% Parameters of Interests output
#otptW_0 = sess.run(W_0)
#otptqW_0 = sess.run(qW_0)

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

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Uncertainty After VI: 0")
ax.plot(x_train, y_train, 'ks', alpha=0.5, label='(x, y)')
#ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='prior draws')
ax.plot(inputs, output_VI_ExpCov_1[:,0], 'r', lw=2, alpha=0.5)
ax.plot(inputs, output_VI_ExpCov_1[:,1], 'g', lw=2, alpha=0.5)

ax.plot(inputs, output_VI_ExpCov_0[:,0], 'b', lw=2, alpha=0.5)
ax.plot(inputs, output_VI_ExpCov_0[:,2], 'm', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()