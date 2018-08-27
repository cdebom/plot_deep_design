#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 14:41:25 2018

@author: manuel
"""

from keras.utils.vis_utils import model_to_dot
from keras.layers import Wrapper

import numpy as np
import re
import matplotlib.pyplot as plt

import PIL
from PIL import Image, ImageFont, ImageDraw


def _hex2RGB(h):
    if h[0] == "#":
        h = h[1:]
    R = int("0x"+h[0:2],16)
    G = int("0x"+h[2:4],16)
    B = int("0x"+h[4:6],16)
    A = 255
    if len(h) == 8:
        A = int("0x"+h[6:8],16)
    return R,G,B,A

def _get_model_Svg(model,filename=None,display_shapes=True):
    
    # Get model dot (optimal tags locations)
    ddot = model_to_dot(model).create_plain().splitlines() # split linebreaks

    layersInfo = dict()

    zoom = 100

    # Before anything else, let's parse the information contained inside ddot
    i = 1
    ddot_tmp = ddot[i]
    while (ddot_tmp != "stop"):
        ddot_type = ddot_tmp[0:4]
        
        # Regular expressions were built using the online tool: https://regex101.com/r/9y9n85/1
        if ddot_type == "node":
            pattern = re.compile("node (\d+) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) (?:\")(\w+)(?:\:) (\w+)(?:\") (\w+) (\w+) (\w+) (\w+)")
            matches = pattern.findall(ddot_tmp)[0]
            dotId = str(matches[0])
            dotXc = zoom*float(matches[1])
            dotYc = zoom*float(matches[2])
            dotW = zoom*float(matches[3])
            dotH = zoom*float(matches[4])
            
            if dotId in layersInfo:
                layersInfo[dotId]['dotPosition'] = [dotXc,dotYc,dotW,dotH]
            else:
                layersInfo[dotId] = {'dotPosition':[dotXc,dotYc,dotW,dotH]}
            
        i += 1
        ddot_tmp = ddot[i]  



    # get model layers
    layers = model.layers
    SvgTag = ""
    bbox = [np.Inf, np.inf, -1, -1]
    

    for layer in layers:
        layer_id = str(id(layer))
    
        # initialize dictionary with layer information
        if not layer_id in layersInfo:
            layersInfo[layer_id] = dict()
    
        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        layer_type = layer.__class__.__name__
        layer_typeName = layer_type
        if isinstance(layer, Wrapper):
            layer_name = '{}({})'.format(layer_name, layer.layer.name)
            child_class_name = layer.layer.__class__.__name__
            layer_typeName = '{}({})'.format(layer_type, child_class_name)
        
        layersInfo[layer_id]['name'] = layer_name
        layersInfo[layer_id]['type'] = layer_type
        
        oshape = [] # if empty means that oshape did not changed
        
        # Now let's switch the class
        if (layer_type == "InputLayer"):
            tag_color = "#acacace2"
            border_color = "#4d4d4da9"
            font_color = "#ffffffff"
            txt = "Input"
            params = {'shape':layer.input.shape.as_list()}
            oshape = layer.output_shape
        elif (layer_type == "Conv2D"):
            tag_color = "#2a7fffff"
            border_color = "#5151c0ff"
            font_color = "#ffffffff"
            txt = layer_type
            params = {'activation':layer.activation.func_name,
                      'kernel':layer.kernel.shape.as_list(),
                      'padding':layer.padding,
                      'strides':layer.strides}
            oshape = layer.output_shape
        elif (layer_type == "MaxPooling2D"):
            tag_color = "#ff0000ff"    
            border_color = "#800000ff"
            font_color = "#ffffffff"
            txt = "MaxPool2D"
            params = {'pool_size':layer.pool_size,
                      'strides':layer.strides}
            oshape = layer.output_shape
        elif (layer_type == "AveragePooling2D"):
            tag_color = "#ff0000ff"    
            border_color = "#800000ff"
            font_color = "#ffffffff"
            txt = "AvgPool2D"
            params = {'pool_size':layer.pool_size,
                      'strides':layer.strides}
            oshape = layer.output_shape
        elif (layer_type == "BatchNormalization"):
            tag_color = "#ffff00ff"
            border_color = "#808000ac"
            font_color = "#000000ff"
            txt = "Bnorm"
            params = dict()
        elif (layer_type == "Activation"):
            tag_color = "#00ff00ff"
            border_color = "#008000a9"
            font_color = "#000000ff"
            if (layer.activation.func_name == 'relu'):
                layer_typeName = 'ReLU'
            txt = layer_typeName
            params = dict()
        elif (layer_type == "Flatten"):
            tag_color = "#cd8a63e2"
            border_color = "#a47a4aff"
            font_color = "#ffffffff"
            txt = layer_type
            params = dict()
            oshape = layer.output_shape
        elif (layer_type == "Dense"):
            tag_color = "#d42068e2"
            border_color = "#540023a9"
            font_color = "#ffffffff"
            txt = layer_type + str(layer.units)
            params = {'units':layer.units}
            oshape = layer.output_shape
        elif (layer_type == "Concatenate"):
            tag_color = "#ffffffff"
            border_color = "#0000008f"
            font_color = "#000000ff"
            txt = "cat"
            params = dict()
            oshape = layer.output_shape
        elif (layer_type == "Dropout"):
            tag_color = "#00ffffff"
            border_color = "#006680ff"
            font_color = "#000000ff"
            txt = "Dropout"
            params = {'rate':layer.rate}
        else:
            tag_color = "#e9c6afff"
            border_color = "#005444a9"
            font_color = "#0000004f"
            txt = layer_type
            params = dict()
        
        layersInfo[layer_id]['typeName'] = layer_typeName
        layersInfo[layer_id]['tag'] = txt
        layersInfo[layer_id]['tagColor'] = tag_color
        layersInfo[layer_id]['borderColor'] = border_color
        layersInfo[layer_id]['fontColor'] = font_color
        layersInfo[layer_id]['params'] = params
        layersInfo[layer_id]['output_shape'] = oshape
        
        # Now let's calculate the size of this tag
        font = ImageFont.truetype("UbuntuMono-R.ttf", 40)
        tagSize = font.getsize(txt)
        h_border = 10
        bradius = 20
        border = 5
        
        layersInfo[layer_id]['tagSize'] = tagSize
        
        X0 = layersInfo[layer_id]['dotPosition'][0]
        Y0 = layersInfo[layer_id]['dotPosition'][1]
        W0 = layersInfo[layer_id]['dotPosition'][2]
        H0 = layersInfo[layer_id]['dotPosition'][3]
        
        bbox[0] = np.min((bbox[0],X0-tagSize[0]//2 - h_border-border))
        bbox[1] = np.min((bbox[1],Y0-border))
        bbox[2] = np.max((bbox[2],X0-tagSize[0]//2 + tagSize[0] + h_border+border))
        bbox[3] = np.max((bbox[3],Y0+H0+border))
        
        #_H = np.max((_H,Y0+H0+border))
        #_W = np.max((_W,X0+tagSize[0] + 2*h_border+border))
        
    #_H -= 4*border
    SvgTag = ""
    
    # define the arrow marker
    SvgTag += '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="0" refY="3" orient="auto" markerUnits="strokeWidth"> <path d="M0,1.5 L0,4.5 L3.5,3 z" fill="#000000" /> </marker> </defs>'
    
    _H = bbox[-1] + 100
    for layer_id in layersInfo:
        
        X0 = layersInfo[layer_id]['dotPosition'][0]
        Y0 = _H - layersInfo[layer_id]['dotPosition'][1]
        W0 = layersInfo[layer_id]['dotPosition'][2]
        H0 = layersInfo[layer_id]['dotPosition'][3]
        
        tagSize = layersInfo[layer_id]['tagSize']
        txt = layersInfo[layer_id]['tag']
        tag_color = layersInfo[layer_id]['tagColor']
        border_color = layersInfo[layer_id]['borderColor']
        font_color = layersInfo[layer_id]['fontColor']
        
        # it is important than H0 > tagSize[1]
        outter_rectangle_svg = '<rect x="%f" y="%f" width="%f" height="%f" rx="%f" ry="%f" fill="%s" fill-opacity="%1.2f" />'%(X0-tagSize[0]//2 - h_border-border,Y0-border,tagSize[0] + 2*h_border+2*border,H0+2*border,1.2*bradius,1.2*bradius,border_color[0:-2],_hex2RGB(border_color)[3]/255.)
        inner_rectangle_svg = '<rect x="%f" y="%f" width="%f" height="%f" rx="%f" ry="%f" fill="%s" fill-opacity="%1.2f" />'%(X0-tagSize[0]//2 - h_border,Y0,tagSize[0] + 2*h_border,H0,bradius,bradius,tag_color[0:-2],_hex2RGB(tag_color)[3]/255.)
        text_svg = '<text x="%f" y="%f" text-anchor="middle" fill="%s" fill-opacity="%1.2f" font-size="30px" font-family="Ubuntu Light" dy=".3em">%s</text>'%(X0,Y0+H0//2,font_color[0:-2],_hex2RGB(font_color)[3]/255.,txt)
        
        
        x0 = X0 - tagSize[0]//2 - h_border - border
        x1 = X0 - tagSize[0]//2 + tagSize[0] + h_border + border
        y0 = Y0 - border
        y1 = Y0 + H0 + border
        layersInfo[layer_id]['outter_bbox'] = [x0,y0,x1,y1] 
        
        tagSvg = "<g>" + outter_rectangle_svg + inner_rectangle_svg + text_svg + "</g>"    
        
        SvgTag += tagSvg
        
    # Now we need to add the edges (lines)
    stroke_width = 4
    i = 1
    ddot_tmp = ddot[i]
    while (ddot_tmp != "stop"):
        ddot_type = ddot_tmp[0:4]
        if ddot_type == "edge":
            pattern = re.compile("edge (\d+) (\d+) (\d+) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) (\w+) (\w+)")
            matches = pattern.findall(ddot_tmp)[0]
            startId = matches[0]
            endId = matches[1]
            
            xy0 = layersInfo[startId]['outter_bbox']
            xy1 = layersInfo[endId]['outter_bbox']
            
            x0 = (xy0[2]-xy0[0])/2 + xy0[0]
            y0 = xy0[3]
            
            x1 = (xy1[2]-xy1[0])/2 + xy1[0]
            y1 = xy1[1]
            
            bezierSvg = '<path stroke-width="%i" d="M%f,%f C %f %f, %f %f, %f %f" stroke="black" fill="none" marker-end="url(#arrow)" />'%(stroke_width,x0,y0,x0,y1-13,x1,y0,x1,y1-13)
            
            # Now add the output_shape tag
            if (layersInfo[startId]['output_shape'] != [] and display_shapes):
                shapeTagSvg = '<text x="%f" y="%f" text-anchor="start" alignment-baseline="hanging" fill="#000000" font-size="20px" font-family="Ubuntu Light" dy=".3em">%s</text>'%(np.abs((x0+x1)/2)+5,y0+10,str(layersInfo[startId]['output_shape'][1:]))

            SvgTag += bezierSvg + shapeTagSvg
        
        i += 1
        ddot_tmp = ddot[i]  
        #print(ddot_tmp)
        
    # add final shape in case we need it
    if (display_shapes):
        # check if layer is output
        for l in layers:
            for o in model.outputs:
                if (l.output == o):
                    xy0 = layersInfo[str(id(l))]['outter_bbox']
                    x0 = (xy0[2]-xy0[0])/2 + xy0[0]
                    y0 = xy0[3]
                    shapeTagSvg = '<text x="%f" y="%f" text-anchor="middle" alignment-baseline="hanging" fill="#000000" font-size="20px" font-family="Ubuntu Light" dy=".3em">%s</text>'%(x0,y0+10,str(l.output_shape[1:]))
                    SvgTag += shapeTagSvg
    
    SvgTag = '<svg viewBox="%i %i %i %i">'%(bbox[0],_H-bbox[3],bbox[2]-bbox[0]+30,_H) + SvgTag + '</svg>'

    # write to file
    if filename is None:
        filename = "model.svg"
    else:
        if not ".svg" in filename:
            filename += ".svg"
    svgFile = open(filename,"w")
    svgFile.write(SvgTag)
    svgFile.close()

