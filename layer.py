
class BoatLayer():
    def __init__(self, img, alpha, x1, y1, x2, y2):
        self.img = img
        self.alpha = alpha
        self.pt_a = [x1, y1]
        self.pt_b = [x2, y2]

    def generate_disp(self, p, t):
        return [3,1]
            
class WaterLayer():
    def __init__(self, img, alpha, x1, y1, x2, y2, wind_speed, wind_dir):
        self.img = img
        self.alpha = alpha
        self.pt_a = [x1, y1]
        self.pt_b = [x2, y2]
        self.a = .2 # scales height of ripples
        self.B = 5 # roughness
        self.N = 256
        self.wind_speed = wind_speed
        self.wind_dir = wind_dir
        

    def generate_disp(self, p, t):
        return [3,1]
