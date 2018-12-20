import sys
import matplotlib.pyplot as plt
from matting import computeMatte
import numpy as np
import scipy.ndimage.filters as filters
from scipy.signal import convolve2d
from skimage import color

def get_patch(p_x, p_y, radius, data):
    # return the patch with half-width radius, centered at p_x p_y, from data array
    # correctly padded if needed
    if (len(data.shape) < 3):
        data = data[:,:,np.newaxis]
    h,w,d = data.shape
    neighborhood = np.zeros((radius*2 + 1, radius*2 + 1, d))
    min_y = p_y - radius if (p_y - radius >= 0) else 0
    max_y = p_y + radius + 1 if (p_y + radius + 1 < h) else h

    min_x = p_x - radius if (p_x - radius >= 0) else 0
    max_x = p_x + radius + 1 if (p_x + radius + 1 < w) else w
    n_min_x = radius + min_x - p_x
    n_max_x = radius + max_x - p_x
    n_min_y = radius + min_y - p_y
    n_max_y = radius + max_y - p_y
    neighborhood[n_min_y:n_max_y, n_min_x:n_max_x] = data[min_y:max_y, min_x:max_x]
    size = (n_max_y - n_min_y) * (n_max_x - n_min_x)
    return neighborhood, size


def get_fill_front(mask):
    fill_front = np.where(filters.laplace(mask) < 0)
    return fill_front

def calc_priority(cur_conf, cur_image, cur_mask, fill_front, radius):
    conf = update_conf(cur_conf, fill_front, radius)
    data = update_data(cur_image, cur_mask, fill_front, radius)
    priority = conf * data
    return priority

def update_conf(cur_conf, fill_front, radius):
    h,w = cur_conf.shape
    updated_conf = np.zeros((h,w))
    for i in range(len(fill_front[0])):
        p_y, p_x = fill_front[0][i], fill_front[1][i]
        patch, size = get_patch(p_x, p_y, radius, cur_conf)
        updated_conf[p_y, p_x] = np.sum(patch)/size
    return updated_conf 

def update_data(cur_image, cur_mask, fill_front, radius):
    normals = calc_normals(cur_mask)
    gradients = calc_gradient_magnitudes(cur_image, fill_front, radius)
    data = (normals[:,:,0] * gradients)**2 + (normals[:,:,1] * gradients)**2
    data /= 255.
    return data

def calc_normals(cur_mask):
    # calculating the gradient of the curve (mask) is equivalent
    # to calculating the normal
    updated_normals = np.zeros(cur_mask.shape)

    y_filt = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    x_filt = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    filtered_y = convolve2d(cur_mask, y_filt, mode='same')
    filtered_x = convolve2d(cur_mask, x_filt, mode='same')

    updated_normals = np.dstack((filtered_y, filtered_x))
    normalization = filtered_y**2 + filtered_x**2
    normalization = np.sqrt(normalization)
    normalization = np.dstack((normalization, normalization))
    normalization[normalization == 0] = 1
    updated_normals /= normalization

    return updated_normals

def calc_gradient_magnitudes(cur_image, fill_front, radius):
    # store maximum gradient value in the patch surrounding each p
    h,w,d = cur_image.shape
    gradients = np.gradient(color.rgb2gray(cur_image))
    gradient_magnitudes = gradients[0]**2 + gradients[1]**2
    gradient_magnitudes = np.sqrt(gradient_magnitudes)
    max_gradient_magnitudes = np.zeros((cur_image.shape[0], cur_image.shape[1]))
    for i in range(len(fill_front[0])):
        p_y, p_x = fill_front[0][i], fill_front[1][i]
        patch, size = get_patch(p_x, p_y, radius, gradient_magnitudes)
        patch = patch[:,:,0]
        max_gradient_magnitudes[p_y, p_x] = np.max(patch)
    
    return max_gradient_magnitudes

def find_exemplar(p_x, p_y, cur_image, cur_mask, radius):
    cielab_im = color.rgb2lab(cur_image)
    h,w,d = cur_image.shape
    target_patch,size = get_patch(p_x, p_y, radius, cielab_im)
    masked_target,size = get_patch(p_x, p_y, radius, cur_mask)
    masked_target = 1 - masked_target
    masked_target = masked_target
    target_patch = masked_target*target_patch
    min_diff = float('Inf')
    best_x, best_y = 0,0
    for y in range(radius+1, h - radius):
        for x in range(radius+1, w - radius):
            patch, size = get_patch(x,y,radius,cur_mask)
            if not np.isin(1, patch[:,:,0]):
                src_patch, size = get_patch(x, y, radius, cielab_im)
                src_patch = masked_target*src_patch
                ssd = np.sum((target_patch - src_patch)**2)
                if ssd < min_diff:
                    min_diff = ssd
                    best_x = x
                    best_y = y
    print(best_x, best_y)
    return best_x, best_y

def copy_patch(t_x, t_y, s_x, s_y, cur_image, cur_mask, cur_conf, radius):
    update_image = np.copy(cur_image)
    update_mask = np.copy(cur_mask)
    update_conf = np.copy(cur_conf)
    h,w = cur_image.shape[:2]
    for y in range(-radius, radius+1):
        for x in range(-radius, radius+1):

            if t_y-y < cur_mask.shape[0] and t_x-x < cur_mask.shape[1] and cur_mask[t_y-y, t_x-x] == 1:
                update_image[t_y-y, t_x-x] = cur_image[s_y-y, s_x-x]
                update_mask[t_y-y, t_x-x] = 0
                update_conf[t_y-y, t_x-x] = cur_conf[t_y, t_x]
    return update_image, update_mask, update_conf


def inpaint(mask, bg, out_dir):
    target_region = np.where(mask > 0)
    h,w,d = bg.shape

    cur_mask = np.zeros((h,w))
    cur_mask[target_region[0], target_region[1]] = 1
    # cur_mask is 1s inside the region TO BE inpainted

    conf_map = np.ones((h,w))
    conf_map[target_region[0], target_region[1]] = 0

    cur_image = bg
    done = False
    radius = 4
    i = 0
    
    while not done:
        i = i + 1
        if (i%20 == 0):
            if out_dir is not None:
                plt.imsave(out_dir+"/inpainted_" + str(i) + ".jpg", cur_image)
            else:
                plt.imsave("inpainted_" + str(i) + ".jpg", cur_image)
        fill_front = get_fill_front(cur_mask)
        priority_map = calc_priority(conf_map, cur_image, cur_mask, fill_front, radius)
        priority_map = priority_map[fill_front[0], fill_front[1]]
        target_y, target_x = fill_front[0][priority_map.argmax()], fill_front[1][priority_map.argmax()]
        print(target_x, target_y)
        source_x, source_y = find_exemplar(target_x, target_y, cur_image, cur_mask, radius)
        cur_image, cur_mask, conf_map = copy_patch(target_x, target_y,
                                                   source_x, source_y,
                                                   cur_image, cur_mask,
                                                   conf_map, radius)
        print(np.sum(cur_mask))
        if np.sum(cur_mask) == 0:
            done = True
    return cur_image

if __name__ == '__main__':
    '''
    im_name = sys.argv[1]
    mask_name = sys.argv[2]
    im = plt.imread(im_name)
    mask = plt.imread(mask_name)[:,:,0]
    fg, alpha, bg = computeMatte(im, mask)
    '''
    bg_name = sys.argv[1]
    mask_name = sys.argv[2]
    out_dir = None
    if (len(sys.argv)> 3):
        out_dir = sys.argv[3]
    bg = plt.imread(bg_name)[:,:,0:3]/255.
    mask = plt.imread(mask_name)[:,:,0]
    mask_ = (1-np.repeat((mask)[:,:,np.newaxis], 3, axis=2)/255.)
    bg = bg * mask_
    new_bg = inpaint(mask, bg,out_dir)
    if out_dir is not None:
        plt.imsave(out_dir+"/inpainted.jpg", new_bg)
    else:
        plt.imsave("inpainted.jpg", new_bg)
