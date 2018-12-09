import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle

        
####################################
# Initialization functions
####################################

def split_mask(mask):
    fg_mask = mask == 255
    bg_mask = mask == 0
    uk_mask = np.logical_not(np.logical_or(fg_mask, bg_mask))
    return fg_mask, bg_mask, uk_mask

def get_masked_pix(im, mask):
    return im * np.repeat(mask[:,:,np.newaxis], 3, axis=2)

####################################
# Calculation Helpers
####################################

# Get an N_radius by N_radius region around (p_x, p_y) in im, padded if needed
def neighborhood(im, p_x, p_y, N_radius):
    if (len(im.shape) < 3):
        im = im[:,:,np.newaxis]
    h,w,d = im.shape
    neighborhood = np.zeros((N_radius*2 + 1, N_radius*2 + 1, d))
    min_y = p_y - N_radius if (p_y - N_radius >= 0) else 0
    max_y = p_y + N_radius + 1 if (p_y + N_radius + 1 < im.shape[0]) else im.shape[0]

    min_x = p_x - N_radius if (p_x - N_radius >= 0) else 0
    max_x = p_x + N_radius + 1 if (p_x + N_radius + 1 < im.shape[1]) else im.shape[1]
    n_min_x = N_radius + min_x - p_x
    n_max_x = N_radius + max_x - p_x
    n_min_y = N_radius + min_y - p_y
    n_max_y = N_radius + max_y - p_y
    neighborhood[n_min_y:n_max_y, n_min_x:n_max_x] = im[min_y:max_y, min_x:max_x]
    return neighborhood

####################################
# Color Clustering 
####################################

# https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
def cluster_colors(n_colors, im):
    # First, flatten each WxH color channel
    # w, h, d = im.shape
    # im_flat = np.reshape(im, (w*h, d))
    #w = im.shape[1]**(1/2)
    #im_flat = im.reshape(im.shape[1], 3)
    im_flat = im

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


####################################
# Main Functions
####################################

#w and pixels are flattened by channel
def calculate(im, num_clusters, w, pixels, N_radius):
    estimates = []
    # flatten weights and each color channel of pixels

    cb, labels = cluster_colors(num_clusters, pixels)
    # labels is a linearized list!!
    for c in range(len(cb)):
        W = np.sum(w[np.where(labels == c)])
        w_c = w[np.where(labels == c)]
        F_c = pixels[np.where(labels == c),:][0]

        product = (w_c * F_c.T).T
        summation = np.sum(product, axis=0)
        F_bar = (1./W) * summation

        diff = F_c - F_bar
        outer = np.array([np.outer(diff_, diff_).flatten() for diff_ in diff])
        product = (w_c*outer.T).T
        summation = np.sum(product, axis=0)
        sig_F = np.reshape((1./W) * summation, (3,3)) + 1e-5*np.eye(3)
        estimates.append((F_bar, sig_F))

    return estimates

def solve_pixel(F_bar, sig_F, B_bar, sig_B, a, C, num_iter):
    inv_sigma_c_sq = 1./(.01**2)
    I = np.identity(3);
    sig_B_inv = np.linalg.inv(sig_B)
    sig_F_inv = np.linalg.inv(sig_F)
    for _ in range(num_iter):
        A_ul = sig_F_inv + I * (a**2)*inv_sigma_c_sq
        A_ur = I*a*(1-a)*inv_sigma_c_sq
        A_ll = A_ur
        A_lr = sig_B_inv + I*((1-a)**2)*inv_sigma_c_sq

        b_up = np.dot(sig_F_inv, F_bar) + C * a * inv_sigma_c_sq
        b_lw = np.dot(sig_B_inv, B_bar) + C * (1-a) * inv_sigma_c_sq

        A_up = np.hstack((A_ul, A_ur))
        A_lw = np.hstack((A_ll, A_lr))
        A = np.vstack((A_up, A_lw))
        b = np.hstack((b_up, b_lw))
        b = b.T

        FB = np.linalg.solve(A,b)
        FB = FB.T
        F = np.minimum(np.maximum(0, FB[:3]),255)
        B = np.minimum(np.maximum(0, FB[3:]),255)

        a = np.dot(C - B, F-B) / (np.sum((F - B)**2))
        a = min(max(0, a), 1)
    return F,B,a

def likelihood(C, F, F_bar, sig_F, B, B_bar, sig_B, A):
    inv_sigma_c_sq = 1./(.01**2)
    L_C = -1 * np.sum((C - A*F - (1-A)*B)**2) * inv_sigma_c_sq
    L_F = -1 * np.dot((F - F_bar).T, np.dot(np.linalg.inv(sig_F), F-F_bar))/2
    L_B = -1 * np.dot((B - B_bar).T, np.dot(np.linalg.inv(sig_B), B-B_bar))/2
    return L_C + L_F + L_B


def solve_unknown_region(im, fg_pixels, bg_pixels, uk_mask, alpha, g):
    N_radius = 25
    num_iter = 50 # Per pixel iterations
    num_clusters = 1

    uk_Y, uk_X = np.where(uk_mask)

    for i in range(len(uk_Y)):
        if (i % 100 == 0):
            print(i, " ", len(uk_Y))
        x, y = uk_X[i], uk_Y[i]
        a = neighborhood(alpha, x, y, N_radius)[:,:,0]

        f_w = np.multiply(a*a, g).flatten()
        valid = np.nan_to_num(f_w) > 0
        f_w = f_w[valid]
        f = neighborhood(fg_pixels, x, y, N_radius)
        f = np.reshape(f, ((N_radius*2 + 1)**2,3))
        f = f[valid,:]

        b_w = np.multiply((1 - a)*(1-a), g).flatten()
        valid = np.nan_to_num(b_w) > 0
        b_w = b_w[valid]
        b = neighborhood(bg_pixels, x, y, N_radius)
        b = np.reshape(b, ((N_radius*2 + 1)**2,3))
        b = b[valid,:]


        if len(f_w) < 30 or len(b_w) < 30:
            print("Passing")
            continue

        mean_a = np.nanmean(a.flatten())

        fg_calculations = calculate(im, num_clusters, f_w, f, N_radius)
        bg_calculations = calculate(im, num_clusters, b_w, b, N_radius)

        maxL = float('-inf')
        C = im[y,x]

        for fg_pair in fg_calculations:
            for bg_pair in bg_calculations:
                f_bar = fg_pair[0]
                sig_f = fg_pair[1]
                b_bar = bg_pair[0]
                sig_b = bg_pair[1]
                F, B, A = solve_pixel(f_bar, sig_f, b_bar, sig_b, mean_a, C, num_iter)
                L = likelihood(C, F, f_bar, sig_f, B, b_bar, sig_b, A)
                if L > maxL:
                    maxL = L
                    bestF, bestB, bestA = F, B, A

        fg_pixels[y,x] = F
        bg_pixels[y,x] = B
        alpha[y,x] = A
    return fg, bg, alpha

def computeMatte(im, mask):
    w, h, d = im.shape

    fg_mask, bg_mask, uk_mask = split_mask(mask)
    fg = get_masked_pix(im, fg_mask)
    bg = get_masked_pix(im, bg_mask)
    alpha = np.zeros((w,h))
    alpha[fg_mask] = 1
    alpha[uk_mask] = np.nan

    width = 51 # hand calculated for sigma=8
    sigma = 8
    kernel = cv2.getGaussianKernel(width, sigma)
    kernel = kernel * kernel.T
    
    fg_new, bg_new, alpha_new = solve_unknown_region(im, fg, bg, uk_mask, alpha, kernel)
    # return fg + uk, alpha mask, bg with hole
    return fg_new, alpha_new, bg

if __name__ == '__main__':
    im_name = sys.argv[1]
    mask_name = sys.argv[2]
    im = plt.imread(im_name)
    w, h, d = im.shape
    mask = plt.imread(mask_name)[:,:,0]

    fg_mask, bg_mask, uk_mask = split_mask(mask)
    fg = get_masked_pix(im, fg_mask)
    bg = get_masked_pix(im, bg_mask)
    alpha = np.zeros((w,h))
    alpha[fg_mask] = 1
    alpha[uk_mask] = np.nan

    width = 51 # hand calculated for sigma=8
    sigma = 8
    kernel = cv2.getGaussianKernel(width, sigma)
    kernel = kernel * kernel.T
    
    plt.imshow(fg)
    plt.show()
    fg_new, bg_new, alpha_new = solve_unknown_region(im, fg, bg, uk_mask, alpha, kernel)
    plt.imshow(fg_new)
    plt.show()
    print("done")
