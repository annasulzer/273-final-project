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
        (o[10])

    except Exception as e:
        print("Failure to Propogate Orbit", e)

    try:

        d = sim.debris(np.array([1,0,0]),np.array([0,1*np.sin(np.pi/2),0,np.cos(np.pi/2)]))
    except Exception as e:
        print("Failue to Initialize Debris", e)

    try:
        d[1]
    
    except Exception as e:
        print("Failure to Properly Propogate Debris",e)

    try:

        dm = sim.debris(np.array([[1,0,0],[0,1,0],[0,0,1],[200,34,269]]).T,np.array([0,1*np.sin(np.pi/4),0,np.cos(np.pi/4)]))
    except Exception as e:
        print("Failue to Initialize Many Features", e)

    try:
        
        (dm[1])
            
    except Exception as e:
        print("Failure to Properly Propogate Many Features",e)
    
    try:
        m = sim.MeasurementModel(dm,np.array([o]))
        (o[1])
        # (m[1])

    except Exception as e:
        print("Problem Making Measurement Model",e)

    try:
        print("ctest")
        orbit_radius = 10
        sat1_x0 = np.array([[orbit_radius,0,0]]).T #units u
        sat1_nv0 = np.array([[0,1,1]]).T/np.linalg.norm(np.array([0,1,1])).T

        sat2_x0 = np.array([[0,orbit_radius,0]]).T #units u
        sat2_nv0 = np.array([[1,0,1]]).T/np.linalg.norm(np.array([1,0,1])).T

        sat3_x0 = np.array([[0,0,orbit_radius]]).T #units u
        sat3_nv0 = np.array([[1,1,0]]).T/np.linalg.norm(np.array([1,1,0])).T

        o1 = sim.observer(sat1_x0,sat1_nv0)
        o2 = sim.observer(sat2_x0,sat2_nv0)
        o3 = sim.observer(sat3_x0,sat3_nv0)
        debris_points = np.array([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1],[3,0,1],[3,0,-1],[-2,0,-1]]).T
        omega = .4 # rad/s
        quat = np.array([0,1*np.sin(omega/2),0,np.cos(omega/2)])
        deb = sim.debris(debris_points,quat)
        mtest = sim.MeasurementModel(deb,np.array([o1,o2,o3]))
        print(mtest[4][1,1])
    except Exception as e:
        print("Problem with multiple satellite formulation",e)
    
    print("Tests Complete")


if __name__ == "__main__":
    test_sim()