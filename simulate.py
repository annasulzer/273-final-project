import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

class observer():
    def __init__(self, x0, v0, del_t,orb_prd=60):
        # Input X0,V0 in RTN, V0 must be normal to X0
        # right now velocity really means nothing relative to the angular velocity
        self.orbital_period = orb_prd
        self.radius = np.linalg.norm(x0)
        self.velocity = np.linalg.norm(v0)
        self.x_unit = x0 / self.radius
        self.vel_unit = v0 / self.velocity
        self.del_t = del_t

    def __getitem__(self, i):
        t = i * self.del_t
        # returns in the RTN frame, so all are orbiting about [r,t,n] = [0,0,0]
        # output shape = [3,1]

        chunk1 = self.radius * np.sin(t * np.pi * 2 / self.orbital_period) * self.vel_unit
        chunk2 = self.radius * np.cos(t * np.pi * 2 / self.orbital_period) * self.x_unit

        return chunk1 + chunk2

class debris():
    def __init__(self, features, omega, del_t):
        # features is a np array of starting feature points
        # omega is an arbitrary angular velocity
        self.omega = omega  # quaternion #rad/s
        self.features = features
        self.radii = np.linalg.norm(features, axis=0)
        self.del_t = del_t

    def __getitem__(self, i):
        t = i*self.del_t
        # returns all of the features at a timestep t

        theta = np.linalg.norm(self.omega[:3]) * t  # Changed line
        # Normalize the rotation axis
        axis = self.omega[:3] / np.linalg.norm(self.omega[:3])  # Changed line
        # Create a new quaternion for the rotation at time t
        new_quat = np.array([  # Changed lines
            axis[0] * np.sin(theta / 2),
            axis[1] * np.sin(theta / 2),
            axis[2] * np.sin(theta / 2),
            np.cos(theta / 2)
        ])

        full_rot = 2 * np.arccos(self.omega[3]) * t  # will get the absolute rotation in rad
        quat_vec = self.omega[:3] / np.linalg.norm(self.omega[:3])
        quat_vec = quat_vec * np.sin(full_rot / 2)
        new_quat = np.array([quat_vec[0], quat_vec[1], quat_vec[2], np.cos(full_rot / 2)])
        rotation = R.from_quat(new_quat)
        rotated_features = rotation.apply(self.features.T).T

        return rotated_features

class MeasurementModel:
    def __init__(self, debris_init, observers_init, del_t, n_blind=0):
        # Takes in a debris state (only one) and an array of observers

        self.debris = debris_init
        self.observers = observers_init
        self.n_blind = n_blind
        self.del_t = del_t

    def __getitem__(self, i):
        t = i#*self.del_t
        # Returns the measurement state ... This is of the shape y = [number of observers, number of targets]
        # this is meant to make it easier to read/understand, as the index is just lst[observer,thing it is observing],
        # but for use in the EKF filter you may have to flatten/reshape

        # Seed the RNG so it has the same output every time with respect to the timestep
        np.random.seed(24601 + i)
        lst = []
        for o in self.observers:
            dist = np.linalg.norm(self.debris[t] - o[t], axis=0)


        total_points = self.debris.features.shape[1]
        blind_indices = np.random.choice(range(total_points), self.n_blind, replace=False) if self.n_blind > 0 else []

        for o in self.observers:
            dist = np.linalg.norm(self.debris[t] - o[t], axis=0)
            dist[blind_indices] = np.nan
            lst.append(dist)


        return np.array(lst)

if __name__ == "__main__":
    # Here is an example of the satellite formulation and observers running
    del_t = 0.1
    # Initialize the Observer Starting Positions and directions of motion
    # Observer speed is determined by the orbital period, defined in Observer Class
    orbit_radius = 10
    sat1_x0 = np.array([[orbit_radius, 0, 0]]).T  # units u
    sat1_nv0 = np.array([[0, 1, 1]]).T / np.linalg.norm(np.array([0, 1, 1])).T

    sat2_x0 = np.array([[0, orbit_radius, 0]]).T  # units u
    sat2_nv0 = np.array([[1, 0, 1]]).T / np.linalg.norm(np.array([1, 0, 1])).T

    sat3_x0 = np.array([[0, 0, orbit_radius]]).T  # units u
    sat3_nv0 = np.array([[1, 1, 0]]).T / np.linalg.norm(np.array([1, 1, 0])).T

    # Initialize 3 Observer Satellites
    o1 = observer(sat1_x0, sat1_nv0, del_t)
    o2 = observer(sat2_x0, sat2_nv0, del_t)
    o3 = observer(sat3_x0, sat3_nv0, del_t)

    print(o1[1])

    # Initialize the Debris Points
    debris_points = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1], [3, 0, 1], [3, 0, -1], [-2, 0, -1]]).T

    # Initialize the Quaternion for Debris Rotation.
    # Given any normalized direction vector [nx, ny, nz] and angular rotation [omega], the quaternion is of the
    # form: [nx*sin(omega/2),ny*sin(omega/2),nz*sin(omega/2),cos(omega/2)]
    omega = 0.1  # rad/s

    quat = np.array([0, 1 * np.sin(omega / 2), 0, np.cos(omega / 2)])
    print(quat)


    # Initialize the Debris Object
    deb = debris(debris_points, quat, del_t)

    # Initialize the Measurement model
    mtest = MeasurementModel(deb, np.array([o1, o2, o3]),del_t, n_blind=0)
    print(mtest[4])  # Transposed for Viewing Pleasure

    # This should be the correct format for output y(t). will be [o1p1,o1p2,o1p3 ... o3p9,o3p10,o3p11]
    # o = observer number, p = point number

    print(mtest[4].flatten())
    posehist = np.zeros((int(60/del_t), 3))
    for i in range(int(60/del_t)):
        
        debris = deb[i]
        posehist[i, :] = debris[:, 2]

    time_step = del_t
    time_array = np.arange(0, 60, time_step)

    plt.figure()

    plt.plot(time_array, np.linalg.norm(posehist, axis=1))

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    axs[0].plot(time_array, posehist[:, 0], label="True X")
    axs[0].set_xlabel('time [s]')
    axs[0].set_ylabel('X [m]')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title('Estimated vs. True X Position EKF')

    # Plot Y Position
    axs[1].plot(time_array, posehist[:, 1], label="True Y")

    axs[1].set_xlabel('time [s]')
    axs[1].set_ylabel('Y [m]')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title('Estimated vs. True Y Position EKF')

    axs[2].plot(time_array, posehist[:, 2], label="True Heading")
    axs[2].set_xlabel('time [s]')
    axs[2].set_ylabel('Heading Angle [rad]')
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_title('Estimated vs. True Heading EKF')

    plt.tight_layout()
    plt.show()



