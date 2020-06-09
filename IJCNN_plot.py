#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:21:39 2020

@author: peng
"""

import cycler
import matplotlib as mpl

numColor = 110;

color = plt.cm.hsv(np.linspace(0.4,0.9,numColor))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
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

#%% Cycler color
plot_sigmaLyr_1_node_1 = new_sigmaLyr_1_node_1.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Weights and Biases Visualisation: 0")
for i in range(numColor):
    ax.plot(inputs, plot_sigmaLyr_1_node_1[i,:].T,  lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()

plot_sigmaLyr_0_node_1 = sigmaLyr_0_node_1.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Weights and Biases Visualisation: 0")
for i in range(numColor):
    ax.plot(inputs, plot_sigmaLyr_0_node_1[i,:].T, lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()

plot_sigmaLyr_0_node_2 = sigmaLyr_0_node_2.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Weights and Biases Visualisation: 0")
for i in range(numColor):
    ax.plot(inputs, plot_sigmaLyr_0_node_2[i,:].T, lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()

#%% The mean of all the Weights and Biases
plot_mnLyr_1_node_1 = new_mnLyr_1_node_1.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Weights and Biases Visualisation: 0")
for i in range(numColor):
    ax.plot(inputs, plot_mnLyr_1_node_1[i,:].T, lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()

plot_mnLyr_0_node_1 = mnLyr_0_node_1.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Weights and Biases Visualisation: 0")
for i in range(numColor):
    ax.plot(inputs, plot_mnLyr_0_node_1[i,:].T, lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()

plot_mnLyr_0_node_2 = mnLyr_0_node_2.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Weights and Biases Visualisation: 0")
for i in range(numColor):
    ax.plot(inputs, plot_mnLyr_0_node_2[i,:].T, lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()

#%%
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
