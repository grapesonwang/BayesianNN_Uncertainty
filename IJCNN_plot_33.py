#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 06:59:46 2020

@author: peng
"""


import cycler
import matplotlib as mpl

numColor = 500;

color = plt.cm.hsv(np.linspace(0.4,0.9,numColor))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

#% Bounds of the coordinates
x_lower = -3.0
x_upper = 3.0

y_lower = -1.5
y_upper = 1.5
#%% The variation of qW_0
#fig = plt.figure(figsize=(10, 6))
#ax = fig.add_subplot(111)
#ax.set_title("qW Visualisation: 0")
#ax.plot(range(1000), qW_ary[:,0], 'r', lw=2, alpha=0.5)
#ax.plot(range(1000), qW_ary[:,1], 'g', lw=2, alpha=0.5)
#plt.show()


#%% The Cov of all the Weights and Biases
#plot_sigmaLyr_1_node_1 = sigmaLyr_1_node_1.T
#fig = plt.figure(figsize=(10, 6))
#ax = fig.add_subplot(111)
#ax.set_title("Weights and Biases Visualisation: 0")
#ax.plot(inputs, plot_sigmaLyr_1_node_1[1:].T, 'r', lw=2, alpha=0.5)
#ax.set_xlim([-5, 5])
#ax.set_ylim([-2, 2])
#ax.legend()
#plt.show()
#
#plot_sigmaLyr_0_node_1 = sigmaLyr_0_node_1.T
#fig = plt.figure(figsize=(10, 6))
#ax = fig.add_subplot(111)
#ax.set_title("Weights and Biases Visualisation: 0")
#ax.plot(inputs, plot_sigmaLyr_0_node_1[1:].T, 'g', lw=2, alpha=0.5)
#ax.set_xlim([-5, 5])
#ax.set_ylim([-2, 2])
#ax.legend()
#plt.show()
#
#plot_sigmaLyr_0_node_2 = sigmaLyr_0_node_2.T
#fig = plt.figure(figsize=(10, 6))
#ax = fig.add_subplot(111)
#ax.set_title("Weights and Biases Visualisation: 0")
#ax.plot(inputs, plot_sigmaLyr_0_node_2[1:].T, 'b', lw=2, alpha=0.5)
#ax.set_xlim([-5, 5])
#ax.set_ylim([-2, 2])
#ax.legend()
#plt.show()


#%% The mean of all the Weights and Biases
#plot_mnLyr_1_node_1 = mnLyr_1_node_1.T
#fig = plt.figure(figsize=(10, 6))
#ax = fig.add_subplot(111)
#ax.set_title("Weights and Biases Visualisation: 0")
#ax.plot(inputs, plot_mnLyr_1_node_1[1:].T, 'r', lw=2, alpha=0.5)
#ax.set_xlim([-5, 5])
#ax.set_ylim([-2, 2])
#ax.legend()
#plt.show()
#
#plot_mnLyr_0_node_1 = mnLyr_0_node_1.T
#fig = plt.figure(figsize=(10, 6))
#ax = fig.add_subplot(111)
#ax.set_title("Weights and Biases Visualisation: 0")
#ax.plot(inputs, plot_mnLyr_0_node_1[1:].T, 'g', lw=2, alpha=0.5)
#ax.set_xlim([-5, 5])
#ax.set_ylim([-2, 2])
#ax.legend()
#plt.show()
#
#plot_mnLyr_0_node_2 = mnLyr_0_node_2.T
#fig = plt.figure(figsize=(10, 6))
#ax = fig.add_subplot(111)
#ax.set_title("Weights and Biases Visualisation: 0")
#ax.plot(inputs, plot_mnLyr_0_node_2[1:].T, 'b', lw=2, alpha=0.5)
#ax.set_xlim([-5, 5])
#ax.set_ylim([-2, 2])
#ax.legend()
#plt.show()

#%% Font configuration
#% 
font_legend = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 30,
}

font_axes = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 25,
}

mpl.rc('legend', fontsize = 18)
mpl.rc('xtick', labelsize = 23)
mpl.rc('ytick', labelsize = 23)
mpl.rc('axes', titlesize = 28)
mpl.rc('axes', labelsize = 28)

#%% Cycler color
plot_sigmaLyr_1_node_1 = new_sigmaLyr_1_node_1.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
#ax.set_title("Weights and Biases Visualisation: 0")
for i in range(numColor):
    ax.plot(inputs, plot_sigmaLyr_1_node_1[i,:].T,  lw=2, alpha=0.5)
ax.set_xlim([x_lower, x_upper])
ax.set_ylim([-0.1, 1.0])
plt.xlabel('x',font_axes)
plt.ylabel('node 21 $\sigma$',font_axes)
#mpl.rc('legend', fontsize = 18)
#mpl.rc('xtick', labelsize = 25)
#mpl.rc('ytick', labelsize = 25)
#mpl.rc('axes', titlesize = 25)
#mpl.rc('axes', labelsize = 25)
plt.savefig('./figures/sigmaLyr_1_node_1.eps')
plt.show()

#plot_sigmaLyr_1_node_1 = sigmaLyr_1_node_1.T
#fig = plt.figure(figsize=(10, 6))
#ax = fig.add_subplot(111)
#ax.set_title("Weights and Biases Visualisation: 0")
#for i in range(numColor):
#    ax.plot(inputs, plot_sigmaLyr_1_node_1[i,:].T,  lw=2, alpha=0.5)
#ax.set_xlim([x_lower, x_upper])
#ax.set_ylim([y_lower, y_upper])
#ax.legend()
#plt.show()

plot_sigmaLyr_0_node_1 = sigmaLyr_0_node_1.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
#ax.set_title("Weights and Biases Visualisation: 0")
for i in range(numColor):
    ax.plot(inputs, plot_sigmaLyr_0_node_1[i,:].T, lw=2, alpha=0.5)
ax.set_xlim([x_lower, x_upper])
ax.set_ylim([-0.2, 1.2])
plt.xlabel('x',font_axes)
plt.ylabel('node 11 $\sigma$',font_axes)
#mpl.rc('legend', fontsize = 18)
#mpl.rc('xtick', labelsize = 15)
#mpl.rc('ytick', labelsize = 15)
#mpl.rc('axes', titlesize = 20)
#mpl.rc('axes', labelsize = 20)
plt.savefig('./figures/sigmaLyr_0_node_1.eps')
plt.show()

plot_sigmaLyr_0_node_2 = sigmaLyr_0_node_2.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
#ax.set_title("Weights and Biases Visualisation: 0")
for i in range(numColor):
    ax.plot(inputs, plot_sigmaLyr_0_node_2[i,:].T, lw=2, alpha=0.5)
ax.set_xlim([x_lower, x_upper])
ax.set_ylim([-0.2, 1.2])
plt.xlabel('x',font_axes)
plt.ylabel('node 12 $\sigma$',font_axes)
#mpl.rc('legend', fontsize = 18)
#mpl.rc('xtick', labelsize = 15)
#mpl.rc('ytick', labelsize = 15)
#mpl.rc('axes', titlesize = 20)
#mpl.rc('axes', labelsize = 20)
#ax.legend(loc = 'best')
plt.savefig('./figures/sigmaLyr_0_node_2.eps')
plt.show()

#%% The mean of all the Weights and Biases
plot_mnLyr_1_node_1 = new_mnLyr_1_node_1.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
#ax.set_title("Weights and Biases Visualisation: 0")
for i in range(numColor):
    ax.plot(inputs, plot_mnLyr_1_node_1[i,:].T, lw=2, alpha=0.5)
ax.set_xlim([x_lower, x_upper])
ax.set_ylim([y_lower, y_upper])
plt.xlabel('x',font_axes)
plt.ylabel('node 21 mean',font_axes)
#mpl.rc('legend', fontsize = 18)
#mpl.rc('xtick', labelsize = 15)
#mpl.rc('ytick', labelsize = 15)
#mpl.rc('axes', titlesize = 20)
#mpl.rc('axes', labelsize = 20)
#ax.legend(loc = 'best')
plt.savefig('./figures/mnLyr_1_node_1.eps')
plt.show()

#plot_mnLyr_1_node_1 = mnLyr_1_node_1.T
#fig = plt.figure(figsize=(10, 6))
#ax = fig.add_subplot(111)
#ax.set_title("Weights and Biases Visualisation: 0")
#for i in range(numColor):
#    ax.plot(inputs, plot_mnLyr_1_node_1[i,:].T, lw=2, alpha=0.5)
#ax.set_xlim([x_lower, x_upper])
#ax.set_ylim([y_lower, y_upper])
#ax.legend()
#plt.show()

plot_mnLyr_0_node_1 = mnLyr_0_node_1.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
#ax.set_title("Weights and Biases Visualisation: 0")
for i in range(numColor):
    ax.plot(inputs, plot_mnLyr_0_node_1[i,:].T, lw=2, alpha=0.5)
ax.set_xlim([x_lower, x_upper])
ax.set_ylim([y_lower, y_upper])
plt.xlabel('x',font_axes)
plt.ylabel('node 11 mean',font_axes)
#mpl.rc('legend', fontsize = 18)
#mpl.rc('xtick', labelsize = 15)
#mpl.rc('ytick', labelsize = 15)
#mpl.rc('axes', titlesize = 20)
#mpl.rc('axes', labelsize = 20)
#ax.legend(loc = 'best')
plt.savefig('./figures/mnLyr_0_node_1.eps')
plt.show()

#%%
plot_mnLyr_0_node_2 = mnLyr_0_node_2.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
#ax.set_title("Weights and Biases Visualisation: 0")
for i in range(numColor):
    ax.plot(inputs, plot_mnLyr_0_node_2[i,:].T, lw=2, alpha=0.5)
ax.set_xlim([x_lower, x_upper])
ax.set_ylim([y_lower, y_upper])
plt.xlabel('x',font_axes)
plt.ylabel('node 12 mean',font_axes)
#mpl.rc('legend', fontsize = 18)
#mpl.rc('xtick', labelsize = 15)
#mpl.rc('ytick', labelsize = 15)
#mpl.rc('axes', titlesize = 20)
#mpl.rc('axes', labelsize = 20)
#ax.legend(loc = 'best')
plt.savefig('./figures/mnLyr_0_node_2.eps')
plt.show()

#%% Confidence interval drawing

plot_conf_sigma = new_sigmaLyr_1_node_1.T
plot_conf_mean = new_mnLyr_1_node_1.T

plt_mean = plot_conf_mean[numColor - 1, :].T
plt_sigma = plot_conf_sigma[numColor - 1, :].T

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
#ax.set_title("Confidence Interval Plotting: 0")
#mpl.rc('font', size=23)
ax.plot(inputs, y_train, 'or', label = 'Sample')
ax.plot(inputs, plt_mean, color = 'blue', lw=3, label = 'Prediction')
plt.fill_between(inputs, plt_mean - 2.0 * plt_sigma, plt_mean + 2.0 * plt_sigma,  color = 'blue', alpha=0.2, label = 'Uncertainty')
#for i in range(numColor):
#    ax.plot(inputs, plot_sigmaLyr_1_node_1[i,:].T,  lw=2, alpha=0.5)
ax.set_xlim([x_lower, x_upper])
ax.set_ylim([y_lower, y_upper])
plt.xlabel('x',font_axes)
plt.ylabel('y',font_axes)
#mpl.rc('legend', fontsize = 18)
#mpl.rc('xtick', labelsize = 15)
#mpl.rc('ytick', labelsize = 15)
#mpl.rc('axes', titlesize = 20)
#mpl.rc('axes', labelsize = 20)
ax.legend(loc = 'best')
plt.savefig('./figures/Lyr_1_Uncertainty.png')
plt.show()

#%% This is used to learn how to set up different colors
#import cycler
#import matplotlib as mpl
#
#numColor = 400;
#
#color = plt.cm.viridis(np.linspace(0,1,numColor))
#mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
#
#fig, ax = plt.subplots()
#for i in range(numColor):
#    ax.plot([0,1], [i, 2*i])
#plt.show()
#
#
#fig, ax = plt.subplots()
#ax.plot([0,1], [2,4])
#plt.show()
