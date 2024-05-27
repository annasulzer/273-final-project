#I guess this is where we will try to be good coders and write unit tests
import numpy as np
from scipy.spatial.transform import Rotation as R
import simulate as sim


def test_sim():
    print("Testing Simulation")
    try:
        s1x = np.array([[1,0,0]]).T
        s1v = np.array([[0,1,1]])
        o = sim.observer(s1x,s1v)
    except Exception as e:
        print("Failure to Initialize Observer", e)

    try:
        o.pos_at_t(10)
    except Exception as e:
        print("Failure to Propogate Orbit", e)

    try:

        d = sim.debris(np.array([1,0,0]),np.array([0,1*np.sin(np.pi/2),0,np.cos(np.pi/2)]))
    except Exception as e:
        print("Failue to Initialize Debris", e)

    try:
        d.pos_at_t(1)
    
    except Exception as e:
        print("Failure to Properly Propogate Debris",e)

    try:

        dm = sim.debris(np.array([[1,0,0],[0,1,0],[0,0,1],[200,34,269]]).T,np.array([0,1*np.sin(np.pi/4),0,np.cos(np.pi/4)]))
    except Exception as e:
        print("Failue to Initialize Many Features", e)

    try:
        dm.pos_at_t(1)
    
    except Exception as e:
        print("Failure to Properly Propogate Many Features",e)
    



if __name__ == "__main__":
    test_sim()