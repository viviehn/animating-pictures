from layer import BoatLayer, WaterLayer
import sys
import matplotlib.pyplot as plt

def generate_boat_layer(im, alpha):
    # open file in matplotlib, do some user input
    plt.figure()
    im_plot = plt.imshow(im)
    pts = plt.ginput(2, timeout=-1)
    return BoatLayer(im, alpha, pts[0][0], pts[0][1], pts[1][0], pts[1][1])

def generate_plant_layer(im, alpha):
    pass

def generate_water_layer(im, alpha):
    plt.figure()
    im_plot = plt.imshow(im)
    pts = plt.ginput(2, timeout=-1)
    return WaterLayer(im, alpha, pts[0][0], pts[0][1], pts[1][0], pts[1][1])

def generate_cloud_layer(im, alpha):
    pass

def generate_rock_layer(im, alpha):
    pass

layers = []
bg = None
with open(sys.argv[1]) as f:
    line = f.readline()
    while line:
        split_line = line.split()
        if split_line[0] == 'bg':
            bg = plt.imread(split_line[1])
        else:
            im_name, alpha_name, motion_type = line.split()
            im = plt.imread(im_name)
            alpha = plt.imread(alpha_name)
            if motion_type == 'boat':
                layer = generate_boat_layer(im, alpha)
            elif motion_type == 'plant':
                layer = generate_plant_layer(im, alpha)
            elif motion_type == 'water':
                layer = generate_water_layer(im, alpha)
            elif motion_type == 'cloud':
                layer = generate_cloud_layer(im, alpha)
            elif motion_type == 'rock':
                layer = generate_rock_layer(im, alpha)
            layers.append(layer)

composite = bg
for t in range(1,4):
    for layer in layers:
        C_ = np.zeros(layer.im.shape)
        a_ = np.zeros(layer.alpha.shape)
        for i in range(C_.shape[0]):
            for j in range(C_.shape[1]):
                p = [i, j, 1]
                d = layer.generate_disp(p,t)
                inv_d = -d
                lookup_p = p + inv_d
                C_[i, j, :] = layer.im[lookup_p[0], lookup_p[1], :]
                a_[i, j] = layer.alpha[lookup_p[0], lookup_p[1]]

        a_ = a_[:,:,np.newaxis]
        composite = a_*C_ + (1-a_)*composite
    plt.imsave("composite_"+str(t), composite/255)
