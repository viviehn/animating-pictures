from math import pi
import numpy as np

class BoatLayer():
    def __init__(self, im, alpha, x1, y1, misc):
        self.im = im
        self.alpha = alpha
        self.motion = 'boat'
        self.x = x1
        self.y = y1
        self.misc = int(misc)

    def generate_disp(self, t, i=0, j=0):
        # ducks
        #t = t/10.
        #x = np.sin(2*t + 4 + self.y%50) + np.sin(5*t + 3 + self.y%50) + np.sin(4*t + 5 + self.y%50) + np.sin(3*t + .25 + self.y%50)
        #x = np.sin(t/14 + 2*(i%50))*2 + np.sin(t/5 + 2*(i%50))*4 + np.sin(t/3 + 2*(i%50))*2
        # circles
        x = np.exp(-t/(80*self.misc)) * np.sin(t/(3*self.misc))*9
        return np.array([int(x), 0])
            
class WaterLayer():
    def __init__(self, im, alpha, misc):
        self.im = im
        self.alpha = alpha
        self.motion = 'water'
        self.misc = misc
        

    def generate_disp(self, t, i=0, j=0):
        # ducks
        t = t/10
        x = np.sin(2*t + 4 + i%50) + np.sin(5*t + 3 + i%50) + np.sin(4*t + 5 + i%50) + np.sin(3*t + .25 + i%50)
        #x = np.sin(t/14 + 2*(i%50))*2 + np.sin(t/5 + 2*(i%50))*4 + np.sin(t/3 + 2*(i%50))*2
        return np.array([int(x), 0])

    def gen_gauss_noise(self):
        np.random.normal(size=(self.N, self.N))

class PlantLayer():
    def __init__(self, im, alpha, x1, y1, x2, y2, misc = 0):
        self.im = im
        self.alpha = alpha
        self.motion = 'plant'
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.misc = int(misc)

    def generate_disp(self, t, i=0, j=0):
        t = (t + self.misc*2)/10
        diff = self.y1 - min(i, self.y2)
        dist = self.y1 - self.y2
        a = 1. - diff/dist
        s = 0
        c = a*(np.cos(3*t + 1) + np.cos(2*t + 4) + np.cos(5*t + 3) + np.cos(4*t + 1))*2
        return np.array([int(s), int(8 - c)])

class CloudLayer():

    def __init__(self, im, alpha, misc = 0):
        self.im = im
        self.alpha = alpha
        self.motion = 'cloud'
        self.misc = int(misc)

    def generate_disp(self, t):
        return np.array([0, int(t*2)])
