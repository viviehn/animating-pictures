import sys
import matplotlib.pyplot as plt
from matting import computeMatte
import numpy as np
import scipy.ndimage.filters as filters
from scipy.signal import convolve2d
from skimage import color

def get_fill_front(im):
    # find the edge of the unknown region by applying laplacian filter
    # returns np.array of 2d indices to the fill_front
    fill_front = np.where(filters.laplace(im) < 0)
    return fill_front

def find_exemplar(p_x, p_y, cur_image, cur_mask, radius):
    cielab_im = color.rgb2lab(cur_image)
    h,w,d = cur_image.shape
    rows, cols, size = get_patch(p_x, p_y, radius, h, w)
    target = cielab_im[rows, cols, :]
    min_diff = float('Inf')
    best_x, best_y = 0, 0
    for y in range(radius, h - radius + 1):
        for x in range(radius, w - radius + 1):
            if cur_mask[y,x] != 1:
                rows, cols, size = get_patch(x, y, radius, h, w)
                source = (cielab_im*cur_mask[:,:,np.newaxis])[rows, cols, :]
                ssd = np.sum((target - source)**2)
                if ssd < min_diff:
                    min_diff = ssd
                    best_x = x
                    best_y = y
        print(y)
    print('found best')
    print(best_x, best_y)
    return best_x, best_y

def copy_patch(t_x, t_y, s_x, s_y, cur_image, cur_mask, radius):
    update_image = np.copy(cur_image)
    update_mask = np.copy(cur_mask)
    h,w,d = cur_image.shape
    flip_mask = np.logical_not(cur_mask)
    rows, cols, size = get_patch(s_x, s_y, radius, h, w)
    source = (cur_image*flip_mask[:,:,np.newaxis])[rows, cols, :]
    source_canvas = np.zeros(cur_image.shape, dtype=np.uint8)

    rows, cols, size = get_patch(t_x, t_y, radius, h, w)
    target = cur_image[rows, cols, :]
    source_canvas[rows,cols,:] = source
    source_canvas = source_canvas*cur_mask[:,:,np.newaxis]
    source_canvas = source_canvas.astype(np.uint8)
    plt.imshow(source_canvas)
    plt.show()

#    source_canvas = source_canvas.astype(uint8)
    update_image += source_canvas

    update_mask[rows, cols] = 1
    return update_image, update_mask 


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
        rows, cols, size = get_patch(p_x, p_y, radius, h, w)
        max_gradient_magnitudes[p_y, p_x] = max(gradient_magnitudes[rows, cols])
    
    return max_gradient_magnitudes

def update_conf(cur_conf, fill_front, radius):
    updated_conf = np.copy(cur_conf)
    h,w = cur_conf.shape
    for i in range(len(fill_front[0])):
        p_y, p_x = fill_front[0][i], fill_front[1][i]
        rows, cols, size = get_patch(p_x, p_y, radius, h, w)
        updated_conf[p_y, p_x] = np.sum(cur_conf[rows, cols])/size

    return updated_conf 

def update_data(cur_image, cur_mask, fill_front, radius):
    normals = calc_normals(cur_mask)
    gradients = calc_gradient_magnitudes(cur_image, fill_front, radius)
    data = (normals[:,:,0] * gradients)**2 + (normals[:,:,1] * gradients)**2
    data /= 255.
    return data

def calc_priority(cur_conf, cur_image, cur_mask, fill_front, radius):
    conf = update_conf(cur_conf, fill_front, radius)
    data = update_data(cur_image, cur_mask, fill_front, radius)
    priority = conf * data
    return priority


def get_patch(p_x, p_y, radius, h, w):
    min_y = p_y - radius if p_y - radius >= 0 else 0
    max_y = p_y + radius if p_y + radius <= h else h
    min_x = p_x - radius if p_x - radius >= 0 else 0
    max_x = p_x + radius if p_x + radius <= w else w
    rows = np.repeat(np.arange(min_y, max_y), max_x - min_x)
    cols = np.tile(np.arange(min_x, max_x), max_y - min_y)
    return rows, cols, len(rows)

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


def inpaint(mask, bg):
    target_region = np.where(mask > 0)
    h,w,d = bg.shape
    conf_map = np.ones((h,w))
    conf_map[target_region[0], target_region[1]] = 0
    cur_mask = np.zeros((h, w))
    cur_mask[target_region[0], target_region[1]] = 1
    plt.imshow(cur_mask.astype(float))
    plt.show()
    cur_image = bg
    done = False
    radius = 128
    print(np.sum(cur_mask))
    while not done:
        print(np.sum(cur_mask))
        fill_front = get_fill_front(cur_mask)
        priority_map = calc_priority(conf_map, cur_image, cur_mask, fill_front, radius)
        target_x, target_y = np.unravel_index(priority_map.argmax(), priority_map.shape)
        source_x, source_y = find_exemplar(target_x, target_y, cur_image, cur_mask, radius)
        cur_image, cur_mask = copy_patch(target_x, target_y, source_x, source_y, cur_image, cur_mask, radius)
        if (np.sum(cur_mask) == h*w):
            done = True
        plt.imshow(cur_image)
        plt.show()
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
    bg = plt.imread(bg_name)
    mask = plt.imread(mask_name)[:,:,0]
    new_bg = inpaint(mask, bg)
    plt.imshow(new_bg/255)
    plt.show()
    plt.imsave("inpainted.jpg", new_bg.astype(int))

