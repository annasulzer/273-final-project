import numpy as np
from scipy.spatial.transform import Rotation as R


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
    
class debris():
    def __init__(self,features,omega):
        # features is a np array of starting feature points
        # omega is an arbitrary angular velocity
        self.omega = omega #quaternion #rad/s
        self.omega_unit = np.linalg.norm(omega)
        self.features = features
        self.radii = np.linalg.norm(features,axis=0)

    
    def pos_at_t(self,t):
        #returns all of the features at a timestep t
        full_rot = 2*np.arccos(self.omega[3]) * t # will get the absolute rotation in rad
        quat_vec = self.omega[:3]/np.linalg.norm(self.omega[:3])
        quat_vec = quat_vec * np.sin(full_rot/2)
        new_quat = np.array([quat_vec[0],quat_vec[1],quat_vec[2],np.cos(full_rot/2)])
        rotation = R.from_quat(new_quat)
        rotated_features = rotation.apply(self.features.T)
        return rotated_features


