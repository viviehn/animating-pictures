from layer import BoatLayer, WaterLayer, PlantLayer, CloudLayer
import sys
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import imageio

def generate_boat_layer(im, alpha, misc):
    # open file in matplotlib, do some user input
    plt.figure()
    im_plot = plt.imshow(im)
    pts = plt.ginput(2, timeout=-1)
    return BoatLayer(im, alpha, pts[0][0], pts[0][1], misc)

def generate_plant_layer(im, alpha, misc):
    plt.figure()
    im_plt = plt.imshow(im)
    pts = plt.ginput(2, timeout=-1)
    return PlantLayer(im, alpha, pts[0][0], pts[0][1], pts[1][0], pts[1][1], misc)

def generate_water_layer(im, alpha, misc):
    plt.figure()
    im_plot = plt.imshow(im)
    pts = plt.ginput(2, timeout=-1)
    return WaterLayer(im, alpha, misc)

def generate_cloud_layer(im, alpha, misc):
    return CloudLayer(im, alpha, misc)

def generate_rock_layer(im, alpha, misc):
    pass

layers = []
bg = None
outdir = None
if len(sys.argv)> 2:
    outdir = sys.argv[2]
with open(sys.argv[1]) as f:
    line = f.readline()
    while line:
        split_line = line.split()
        if split_line[0] == 'bg':
            bg = imageio.imread(split_line[1])[:,:,0:3]
        else:
            l = line.split()
            im_name, alpha_name, motion_type = l[0], l[1], l[2]
            misc = 0
            if len(l) > 3:
                misc = l[3]
            im = imageio.imread(im_name)[:,:,0:3]
            alpha = color.rgb2gray(imageio.imread(alpha_name))
            if motion_type == 'boat':
                layer = generate_boat_layer(im, alpha, misc)
            elif motion_type == 'plant':
                layer = generate_plant_layer(im, alpha, misc)
            elif motion_type == 'water':
                layer = generate_water_layer(im, alpha, misc)
            elif motion_type == 'cloud':
                layer = generate_cloud_layer(im, alpha, misc)
            elif motion_type == 'rock':
                layer = generate_rock_layer(im, alpha, misc)
            layers.append(layer)
        line = f.readline()

for t in range(1,300):
    composite = bg
    for layer in layers:
        C_ = np.zeros(layer.im.shape)
        a_ = np.zeros(layer.alpha.shape)
        if layer.motion == 'boat' or layer.motion == 'cloud':
            d = layer.generate_disp(t)
        for i in range(C_.shape[0]):
            if layer.motion == 'water':
                d = layer.generate_disp(t, i=i)
            for j in range(C_.shape[1]):
                if layer.motion == 'plant':
                    d = layer.generate_disp(t, i, j)
                p = [i, j]
                inv_d = -1 * d
                lookup_p = p + inv_d
                if layer.motion == 'cloud':
                    lookup_p[0] = lookup_p[0] % layer.im.shape[0]
                    lookup_p[1] = lookup_p[1] % layer.im.shape[1]
                else:
                    lookup_p[0] = max(0, min(lookup_p[0], layer.im.shape[0] - 1))
                    lookup_p[1] = max(0, min(lookup_p[1], layer.im.shape[1] - 1))
                C_[i, j] = layer.im[lookup_p[0], lookup_p[1]]
                a_[i, j] = layer.alpha[lookup_p[0], lookup_p[1]]

        a_ = a_/255
        a_ = np.dstack([a_, a_, a_])
        composite = a_*C_ + (1-a_)*composite
    if outdir is not None:
        composite = composite/255
        plt.imsave(outdir+"/composite_"+str(t), np.clip(composite, 0, 1))
    else:
        plt.imsave("composite_"+str(t), composite/255)
