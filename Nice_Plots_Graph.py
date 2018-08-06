#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:05:31 2018

@author: manuel
"""
from __future__ import division

from keras import backend as K
from keras.utils import np_utils, plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Reshape,
    Merge,
    Concatenate,
    Dropout,
    Lambda,
    concatenate
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.datasets import cifar10

from keras.layers.wrappers import Wrapper
from keras.models import Sequential

import pydot, os, copy
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

def hex2RGB(h):
    if h[0] == "#":
        h = h[1:]
    R = int("0x"+h[0:2],16)
    G = int("0x"+h[2:4],16)
    B = int("0x"+h[4:6],16)
    return R,G,B
        


input = Input(shape = (100,100,1))
layer = Conv2D(64,kernel_size=(3,3),activation='relu')(input)
layer = Flatten()(layer)
layer = Dense(1)(layer)

input2 = Conv2D(64,kernel_size=(3,3),activation='relu')(input)
layer2 = Conv2D(64,kernel_size=(3,3),activation='relu')(input2)
layer2 = Conv2D(64,kernel_size=(3,3),activation='relu')(layer2)
layer2 = Conv2D(64,kernel_size=(3,3),activation='relu')(layer2)
layer2 = Flatten()(layer2)
layer2 = Dense(1)(layer2)

layer = concatenate([layer,layer2],axis=1)
model = Model(input,layer) 

SVG(model_to_dot(model).create(prog='dot', format='svg'))


layers = model.layers

nlayers = len(layers)
layer_types = []
labels = []
layers_tag = range(len(layers))
layers_ids = []
conmap = np.zeros((nlayers,nlayers))

# Create graph nodes.
for layer in layers:
    layer_id = str(id(layer))

    # Append a wrapped layer's label to node's label, if it exists.
    layer_name = layer.name
    class_name = layer.__class__.__name__
    if isinstance(layer, Wrapper):
        layer_name = '{}({})'.format(layer_name, layer.layer.name)
        child_class_name = layer.layer.__class__.__name__
        class_name = '{}({})'.format(class_name, child_class_name)

    layer_types.append(class_name)
    labels.append(layer_name)
    layers_ids.append(str(id(layer)))

# Connect nodes with edges.
for i,layer in enumerate(layers):
    layer_id = str(id(layer))
    for j, node in enumerate(layer._inbound_nodes):
        node_key = layer.name + '_ib-' + str(j)
        if node_key in model._container_nodes:
            for inbound_layer in node.inbound_layers:
                inbound_layer_id = str(id(inbound_layer))
                layer_id = str(id(layer))
                print(inbound_layer_id, layer_id)
                print(layer.name,inbound_layer.name)
                _from = layers_ids.index(inbound_layer_id)
                _to = layers_ids.index(layer_id)
                conmap[_from,_to] = 1



font = ImageFont.truetype("UbuntuMono-R.ttf", 220)
radius = 60
top_margin = 10
pad = 2
border = 30

tag_Sizes = []
tags = []

# now fill the conmap
for i,(layer,ltype) in enumerate(zip(layers,layer_types)):
    if (ltype == "InputLayer"):
        tag_color = "#916f6f4f"
        border_color = "#483737ff"
        font_color = "#ffffffff"
    elif (ltype == "Conv2D"):
        tag_color = "#2a7fffff"
        border_color = "#5151c0ff"
        font_color = "#ffffffff"
    elif (ltype == "MaxPooling2D" or class_name == "AveragePooling2D"):
        tag_color = "#ff00004f"    
        border_color = "#800000a9"
        font_color = "#ffffffff"
    elif (ltype == "BatchNormalization"):
        tag_color = "#ffff004f"
        border_color = "#abab4eff"
        font_color = "#000000ff"
    elif (ltype == "Activation"):
        tag_color = "#00ff004f"
        border_color = "#008000a9"
        font_color = "#000000ff"
    elif (ltype == "Flatten"):
        tag_color = "#cd8a63e2"
        border_color = "#a47a4aff"
        font_color = "#ffffffff"
    elif (ltype == "Dense"):
        tag_color = "#d42068e2"
        border_color = "#540023a9"
        font_color = "#000000ff"
    else:
        tag_color = "#e9c6afff"
        border_color = "#005444a9"
        font_color = "#000000ff"
    
    tagSize = font.getsize(ltype)
    
    
    H = 2*top_margin + 2*border + tagSize[1]
    W = 2*radius + 2*border + tagSize[0]
    
    RGB = 255*np.ones((H,W,3))
    img = Image.fromarray(RGB.astype('uint8'))
    draw = ImageDraw.Draw(img)
    
    # draw border rectangle
    draw.rectangle([(radius+border,0),(radius+tagSize[0]+border,H-1)],fill=hex2RGB(border_color))
    draw.ellipse([(pad,pad-1),(2*radius-pad+border,2*radius-pad+border)],fill=hex2RGB(border_color))
    draw.ellipse([(pad,H-2*radius-pad-border),(2*radius-pad+border,H-pad)],fill=hex2RGB(border_color))
    draw.rectangle([(pad,pad+radius+border),(pad+radius+border,H-pad-radius-border)],fill=hex2RGB(border_color))
    draw.ellipse([(W-pad-2*radius-border,pad-1),(W-pad,2*radius+pad)],fill=hex2RGB(border_color))
    draw.ellipse([(W-pad-2*radius-border,H-2*radius-pad),(W-pad,H-pad)],fill=hex2RGB(border_color))
    draw.rectangle([(W-2*radius-pad-border,pad+radius),(W-pad,H-pad-radius)],fill=hex2RGB(border_color))
    
    
    # draw main rectangle
    draw.rectangle([(radius + border,border),(radius+tagSize[0]+border,H-border)],fill=hex2RGB(tag_color))
    draw.ellipse([(pad+border,pad+border-1),(2*radius-pad+border,2*radius-pad+border)],fill=hex2RGB(tag_color))
    draw.ellipse([(pad+border,H-2*radius-pad-border),(2*radius-pad+border,H-pad-border)],fill=hex2RGB(tag_color))
    draw.rectangle([(pad+border,pad+radius+border),(pad+radius+border,H-pad-radius-border)],fill=hex2RGB(tag_color))
    draw.ellipse([(W-pad-2*radius-border,pad+border-1),(W-pad-border,2*radius+pad+border)],fill=hex2RGB(tag_color))
    draw.ellipse([(W-pad-2*radius-border,H-2*radius-pad-border),(W-pad-border,H-pad-border)],fill=hex2RGB(tag_color))
    draw.rectangle([(W-2*radius-pad-border,pad+radius+border),(W-pad-border,H-pad-radius-border)],fill=hex2RGB(tag_color))
    
    draw.text((radius+border,border+pad),ltype,hex2RGB(font_color),font=font)
    
    RGB = np.asarray(img)
    tags.append(RGB)
    plt.imshow(RGB)
    plt.show()
    tag_Sizes.append(RGB.shape[0:2])
        
        
# now let's find the net inputs using the conmap matrix
# the inputs are defined as those columns with sum = 0
graph_starts = [[i] for i,a in enumerate(np.sum(conmap,axis=0)) if (a == 0)]

# the outputs are defined as those rows with sum = 0
graph_ends = [[i] for i,a in enumerate(np.sum(conmap,axis=1)) if (a == 0)]
          
# Now we have to trace each path from start to each of their rights ends
visited_layers = [gs[-1] for gs in graph_starts]

paths = copy.copy(graph_starts)
while (len(visited_layers) < nlayers):
    new_paths = []
    for k,p in enumerate(paths):
        ido = [int(i) for i,sp in enumerate(conmap[p[-1]]) if sp == 1]
        odo = copy.copy([p])
        if (ido != []):
            odo = [p+[ii] for ii in ido]
        visited_layers.append(p[-1])
        #[visited_layers.append(ii) for ii in ido]
        print(ido,odo)
        [new_paths.append(oo) for oo in odo]
    paths = copy.copy(new_paths)
    visited_layers = list(set(visited_layers))
    print(paths)
        
# now find longest path and make it the spinal chord   
# being the spinal chord means that the horizontal position of each of its
# tags will be centered at 0
paths_len = [len(p) for p in paths]
tags_pos = [[] for ii in layers]

paths_idxs = sorted(range(len(paths_len)), key=lambda k: paths_len[k], reverse=True)
paths_len = [paths_len[idd] for idd in paths_idxs]
paths = [paths[idd] for idd in paths_idxs]


visited_layers = np.zeros((nlayers))
paths_left = copy.copy(paths)

h_margin = np.max([ts[1]//2 for ts in tag_Sizes])
v_margin = np.max([ts[0]//2 for ts in tag_Sizes])

arrow_pos = []


x0 = 0
y0 = 0
while (np.sum(visited_layers) < nlayers):
    p = paths_left[0]
    for i,ip in enumerate(p):
        if (i == 0):
            if visited_layers[ip] == 0:
                Wtmp = tag_Sizes[ip][1]//2
                Htmp = tag_Sizes[ip][0]
                tags_pos[ip] = [x0-Wtmp,
                                0,
                                x0+Wtmp,
                                0+Htmp]
                visited_layers[ip] = 1
            else:
                x0 = tags_pos[ip][2] + h_margin
                y0 = tags_pos[ip][3] + v_margin
                
        else:
            if visited_layers[ip] == 0:
                # see the position of the previous layer in the path
                y0 = tags_pos[p[i-1]][-1] + v_margin
                Wtmp = tag_Sizes[ip][1]//2
                Htmp = tag_Sizes[ip][0]
                tags_pos[ip] = [x0-Wtmp,
                                y0,
                                x0+Wtmp,
                                y0+Htmp]
                visited_layers[ip] = 1
            else:
                Wtmp = tag_Sizes[ip][1]
                Htmp = tag_Sizes[ip][0]
                tags_pos[ip][0] = (tags_pos[ip][0]+tags_pos[p[i-1]][0])//2
                tags_pos[ip][2] = Wtmp + tags_pos[ip][0]
                
    paths_left = paths_left[1:]
    
offset = np.min([ts[0] for ts in tags_pos])
for i,sz in enumerate(tags_pos):
    tags_pos[i][0] = sz[0]-offset
    tags_pos[i][2] = sz[2]-offset
    
    
arrow_pos = []
for i,p in enumerate(paths):
    for j,pp in enumerate(p):
        if (j!=0):
            x0 = (tags_pos[p[j-1]][0] + tags_pos[p[j-1]][2])//2
            y0 = tags_pos[p[j-1]][3]
            x1 = (tags_pos[pp][0] + tags_pos[pp][2])//2
            y1 = tags_pos[pp][1]
            arrow_pos.append([x0,y0,x1,y1])
    


_W = np.max([ts[2] for ts in tags_pos])
_H = np.max([ts[3] for ts in tags_pos])

# Now let's create the final RGB image
RGB = 255*np.ones((_H,_W,3)).astype('uint8')
for i,sz in enumerate(tags_pos):
    RGB[sz[1]:sz[3],sz[0]:sz[2],:] = tags[i]

# Now let's add the arrows
for r in arrow_pos:
    cv2.arrowedLine(RGB, tuple(r[0:2]),tuple(r[2:]),(0,0,0),border//2,8)

fig = plt.figure(figsize=(10,10))
plt.imshow(RGB)

import scipy
scipy.misc.imsave('model.png',RGB)
        
        


