#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:24:23 2020

@author: peng
"""

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
plot_sigmaLyr_1_node_1 = sigmastack_h_1_node_1.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
for i in range(numColor):
    ax.plot(inputs, plot_sigmaLyr_1_node_1[i,:].T,  lw=2, alpha=0.5)
ax.set_xlim([x_lower, x_upper])
ax.set_ylim([-0.1, 1.0])
plt.xlabel('x',font_axes)
plt.ylabel('node 21 $\sigma$',font_axes)

plt.savefig('./figures/VI_sigmaLyr_1_node_1.eps')
plt.show()



plot_sigmaLyr_0_node_1 = sigmastack_h_0_node_0.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
for i in range(numColor):
    ax.plot(inputs, plot_sigmaLyr_0_node_1[i,:].T, lw=2, alpha=0.5)
ax.set_xlim([x_lower, x_upper])
ax.set_ylim([-0.2, 1.2])
plt.xlabel('x',font_axes)
plt.ylabel('node 11 $\sigma$',font_axes)

plt.savefig('./figures/VI_sigmaLyr_0_node_1.eps')
plt.show()

plot_sigmaLyr_0_node_2 = sigmastack_h_0_node_1.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
for i in range(numColor):
    ax.plot(inputs, plot_sigmaLyr_0_node_2[i,:].T, lw=2, alpha=0.5)
ax.set_xlim([x_lower, x_upper])
ax.set_ylim([-0.2, 1.2])
plt.xlabel('x',font_axes)
plt.ylabel('node 12 $\sigma$',font_axes)

plt.savefig('./figures/VI_sigmaLyr_0_node_2.eps')
plt.show()

#%% The mean of all the Weights and Biases
plot_mnLyr_1_node_1 = mnstack_h_1_node_1.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
for i in range(numColor):
    ax.plot(inputs, plot_mnLyr_1_node_1[i,:].T, lw=2, alpha=0.5)
ax.set_xlim([x_lower, x_upper])
ax.set_ylim([y_lower, y_upper])
plt.xlabel('x',font_axes)
plt.ylabel('node 21 mean',font_axes)

plt.savefig('./figures/VI_mnLyr_1_node_1.eps')
plt.show()



plot_mnLyr_0_node_1 = mnstack_h_0_node_0.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
for i in range(numColor):
    ax.plot(inputs, plot_mnLyr_0_node_1[i,:].T, lw=2, alpha=0.5)
ax.set_xlim([x_lower, x_upper])
ax.set_ylim([y_lower, y_upper])
plt.xlabel('x',font_axes)
plt.ylabel('node 11 mean',font_axes)

plt.savefig('./figures/VI_mnLyr_0_node_1.eps')
plt.show()


plot_mnLyr_0_node_2 = mnstack_h_0_node_1.T
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
#ax.set_title("Weights and Biases Visualisation: 0")
for i in range(numColor):
    ax.plot(inputs, plot_mnLyr_0_node_2[i,:].T, lw=2, alpha=0.5)
ax.set_xlim([x_lower, x_upper])
ax.set_ylim([y_lower, y_upper])
plt.xlabel('x',font_axes)
plt.ylabel('node 12 mean',font_axes)

plt.savefig('./figures/VI_mnLyr_0_node_2.eps')
plt.show()

#%% Confidence interval drawing

plot_conf_sigma = sigmastack_h_1_node_1.T
plot_conf_mean = mnstack_h_1_node_1.T

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
plt.savefig('./figures/VI_Lyr_1_Uncertainty.png')
plt.show()


