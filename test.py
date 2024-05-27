#I guess this is where we will try to be good coders and write unit tests
import numpy as np

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
    



if __name__ == "__main__":
    test_sim()