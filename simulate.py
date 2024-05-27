import numpy as np

class observer():
    def __init__(self,x0,v0):
        # Input X0,V0 in RTN, V0 must be normal to X0
        # right now velocity really means nothing relative to the angular velocity
        self.orbital_period = 60
        self.radius = np.linalg.norm(x0)
        self.velocity = np.linalg.norm(v0)
        self.x_unit = x0/self.radius
        self.vel_unit = v0/self.velocity
    def pos_at_t(self,t):
        #returns in the RTN frame, so all are orbiting about [r,t,n] = [0,0,0]
        return self.radius * np.sin(t*np.pi*2/self.orbital_period) * self.vel_unit + self.radius * np.cos(t*np.pi*2/self.orbital_period) * self.x_unit
    


