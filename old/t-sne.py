#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 22:08:56 2017

@author: ball
"""

import numpy as np
from matplotlib import pyplot as plt
from tsne import bh_sne
import sys

x_data = []
y_data = []
color = ['black','lightcyan','b']
counter = [0,0,0]

lim = 10000

t = sys.argv[1]

with open (t+'.vec.bk','r') as fp:
    for line in fp:
        data = line[:-1].split('\t')
        if (data[0] == '1' or data[0] == '2') and counter[int(data[0])-1] < lim:
            y_data.append(color[int(data[0])-1])
            counter[int(data[0])-1] = counter[int(data[0])-1] + 1
            x_data.append(np.asarray(data[1].split()).astype('float64'))
        if sum(counter) == 2*(lim):
            break


print(counter)
# convert image data to float64 matrix. float64 is need for bh_sne
x_data = np.asarray(x_data).astype('float64')
x_data = x_data.reshape((x_data.shape[0], -1))

# For speed of computation, only run on a subset

x_data = x_data[:]
y_data = y_data[:]

# perform t-SNE embedding
vis_data = bh_sne(x_data)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

plt.scatter(vis_x, vis_y, s = 8, c=y_data)
#plt.colorbar(ticks=range(3))
#plt.clim(-0.5, 9.5)
#plt.show()

plt.savefig(t+'.png')
