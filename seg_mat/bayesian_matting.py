import sys
import math
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from cv2 import cv2
from time import time

im_name = sys.argv[1]
mask_name = sys.argv[2]
im = plt.imread(im_name)/255.
mask = plt.imread(mask_name)

def get_background_pix(im, mask):
    masked_pixels= np.where(mask <= .1, im, 0)
    return masked_pixels

def get_foreground_pix(im, mask):
    masked_pixels= np.where(mask >= .9, im, 0)
    return masked_pixels

def get_unknown_pix(im, mask):
    im_r = im[:,:,0]
    im_g = im[:,:,1]
    im_b = im[:,:,2]
    mask = mask[:,:,0]/255.

    masked_r = np.where(mask < .9, im_r, 0)
    masked_g = np.where(mask < .9, im_g, 0)
    masked_b = np.where(mask < .9, im_b, 0)
    masked_r = np.where(mask > .1, masked_r, 0)
    masked_g = np.where(mask > .1, masked_g, 0)
    masked_b = np.where(mask > .1, masked_b, 0)

    masked_pixels = np.dstack((masked_r, masked_g, masked_b))
    return masked_pixels

def split_masks(mask):
    mask = mask[:,:,0]/255.
    fg_mask = np.where(mask > .9, 1, 0)
    bg_mask = np.where(mask < .1, 1, 0)
    uk_mask = np.where(mask < .9, 1, 0)
    uk_mask = np.where(mask > .1, uk_mask, 0)
    return fg_mask, bg_mask, uk_mask

# https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
def cluster_colors(n_colors, im):
    # First, flatten each WxH color channel
    # w, h, d = im.shape
    # im_flat = np.reshape(im, (w*h, d))
    w = im.shape[1]**(1/2)
    im_flat = im.reshape(im.shape[1], 3)

    # Next, fit a model to a small sub-sample of the image
    im_sample = shuffle(im_flat, random_state=0)[:int(w/4.)]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(im_sample)

    # Using this model, label each point
    labels = kmeans.predict(im_flat)

    # Create a codebook to cluster colors
    cb_random = shuffle(im_flat, random_state=0)[:n_colors]
    labels_random = pairwise_distances_argmin(cb_random, im_flat, axis=0)

    return cb_random, labels_random

def quantized_color_image(cb, labels, im):
    w, h, d = im.shape
    d_ = cb.shape[1]
    q_im = np.zeros((w, h, d_))
    label_ = 0
    for i in range(w):
        for j in range(h):
            q_im[i][j] = cb[labels[label_]]
            label_ += 1
    return q_im

def neighborhood(pixel, N_radius):
    r = 2*(N_radius**2) + 2*N_radius
    r = int(r)
    pixel = int(pixel)
    return range(pixel - r, pixel + r + 1)

    

def est_foreground(im_stack, num_clusters, nb, g):
    alpha = im_stack[12]
    fg_mask = im_stack[3]

    fg = np.where(fg_mask, im_stack[0:3], im_stack[6:9])

    w = np.multiply(alpha[nb], alpha[nb], g)
    F = fg[:,nb]

    estimates = []
    # Cluster by color the neighborhood around p
    cb, labels = cluster_colors(num_clusters, F)
    for c in range(len(cb)):
        W = np.sum(w[np.where(labels == c)])
        w_c = w[np.where(labels == c)]
        F_c = F[:,np.where(labels == c)][:,0,:]
        F_ = (1./W) * np.sum(np.multiply(w_c, F_c), axis = 1)
        F_dif = np.transpose(F_c) - F_
        cov_F = [np.outer(f, f.T) for f in F_dif]
        cov_F = np.array(cov_F)
        cov_F = np.multiply(w_c[:,np.newaxis, np.newaxis], cov_F)
        cov_F = (1./W) * np.sum(cov_F, axis=0)
        print(cov_F)
        print(cov_F.shape)
        estimates.append((F_, cov_F))
    return estimates

def est_background(im_stack, num_clusters, nb, g):
    alpha = im_stack[12]
    bg_mask = im_stack[4]

    bg = np.where(bg_mask, im_stack[0:3], im_stack[6:9])

    w = np.multiply((1. - alpha[nb]), (1. - alpha[nb]), g)
    B = bg[:,nb]

    # Cluster by color the neighborhood around p
    cb, labels = cluster_colors(num_clusters, B)
    for c in range(len(cb)):
        W = np.sum(w[np.where(labels == c)])
        w_c = w[np.where(labels == c)]
        B_c = B[:,np.where(labels == c)][:,0,:]
        B_ = (1/W) * np.sum(np.multiply(w_c, B_c), axis = 1)
        B_dif = np.transpose(B_c) - B_
        prod = np.dot(np.transpose(B_dif[0]), B_dif[0])
        
        cov_b = []
        estimates.append((B_, cov_B))
    return estimates


def estimate_colors(a, imstack, p, N_radius, num_clusters):
    r = 2*(N_radius**2) + 2*N_radius
    kernel = cv2.getGaussianKernel(8, -1)
    g = kernel #* kernel.T
    nb = neighborhood(p, N_radius) # Get the neighborhood of pixels around p

    fg_est = est_foreground(imstack, num_clusters, nb, g)
    bg_est = est_background(imstack, num_clusters, nb, g)
    C = imstack[0:3, p] # the observed color at this pixel
    print(C)
    # solve linear equation for each pair of pair of fg and bg clusters
    # f_ will have format (FBar, F_cov) for a particular cluster
    for f_ in fg_est:
        for b_ in bg_est:
            f, b = solve(f_,b_)
            if L(f, b) > cur_L:
                best_f = f
                best_b = b

def estimate_alpha(F, B):
    alpha = np.dot((C-B), (F-B))
    alpha /= np.linalg.norm(F - B)

def solve_unknown(im, mask):
    N_radius = 32 # in 2D space
    num_iter = 5
    num_clusters = 4

    w, h, d = im.shape
    # Create uk_stack, with layers uk_fg_r, uk_fg_b, uk_fg_g, uk_bg_r, uk_bg_b, uk_bg_g, a
    uk_stack = np.zeros((7, w*h))
    # Create im_stack, with layers im_r, im_b, im_g, fg_mask, bg_mask, uk_mask, uk_stack
    im_flat = np.reshape(im, (d, w*h))
    fg_mask, bg_mask, uk_mask = split_masks(mask)
    fg_mask = fg_mask.reshape(1, w*h)
    bg_mask = bg_mask.reshape(1, w*h)
    uk_mask = uk_mask.reshape(1, w*h)
    uk_stack[6] = np.where(fg_mask[0] == 1, 1, 0)
    uk_stack[6] = np.where(bg_mask[0] == 1, 1, 0)
    im_stack = np.concatenate((im_flat, fg_mask, bg_mask, uk_mask, uk_stack), axis=0)

    unknown_area = np.where(uk_mask)[0]
    a = .5
    for p in unknown_area[::-1]:
        for _ in range(num_iter):
            # TODO: Init a as average alpha in neighborhood
            F, B = estimate_colors(a, im_stack, p, N_radius, 2)
            # TODO: Store F,B into im_stack
            a = estimate_alpha(F,B)
            # TODO: Store alpha into im_stack





''' Show masks
unknown = get_unknown_area(im, mask)
foreground = get_foreground(im, mask)
background = get_background(im, mask)
plt.imshow(foreground.reshape(im.shape))
plt.show()
plt.imshow(background.reshape(im.shape))
plt.show()
plt.imshow(unknown.reshape(im.shape))
plt.show()
'''

''' Quantize image colors
cb, labels = cluster_colors(16, im)
q_im = quantized_color_image(cb, labels, im)
plt.imshow(q_im)
plt.show()
plt.imsave("quantized_corgi.jpg", q_im)
'''

solve_unknown(im, mask)
