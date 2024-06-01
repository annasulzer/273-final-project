import numpy as np
from scipy.sparse import block_diag

class KalmanFilter:
    def __init__(self, mu0, sigma0, del_t):
        self.sigma = sigma0
        self.mu = mu0
        self.del_t = del_t
        self.Q = 0.01 * np.eye(len(mu0))
        self.R = 1 * np.eye(11)

    def predict(self):
        omega_x = self.mu[0]
        omega_y = self.mu[1]
        omega_z = self.mu[2]

        STM = np.eye(len(self.mu))
        for i in range(3, len(self.mu), 3):
            STM[i:i+3, 0:3] = [[0, self.del_t * self.mu[i+2], -self.del_t * self.mu[i+1]],
                            [-self.del_t * self.mu[i+2], 0, self.del_t * self.mu[i]],
                            [self.del_t * self.mu[i+1], -self.del_t * self.mu[i], 0]]
            STM[i:i+3, i:i+3] = [[1, -self.del_t * omega_z, self.del_t * omega_y],
                                [self.del_t * omega_z, 1, -self.del_t * omega_x],
                                [-self.del_t * omega_y, self.del_t * omega_x, 1]]
   
        self.A = STM
    
        mu_predict = self.A @ self.mu
        sigma_predict = self.A @ self.sigma @ self.A.T + self.Q
   
        return mu_predict, sigma_predict

    def update(self, y, mu_est, sigma_est, sat_pos):
        self.C = np.zeros((len(y), len(mu_est)))
        valid_indices = []
        g = []
        for i in range(len(y)):
            if not np.isnan(y[i]):
                j = 3 + 3 * i
                point_pos = self.mu[j:j+3]
    
                dist = np.linalg.norm(point_pos - sat_pos)
                self.C[i, j:j+3] = (point_pos - sat_pos)/dist
                print((point_pos - sat_pos)/dist @ self.mu[j:j+3])
                print('dist', dist)
                valid_indices.append(i)
                g.append(dist)

        print('C',self.C)
        if len(valid_indices) == 0:
            print("No valid measurements available for update.")
            return mu_est, sigma_est

        K = sigma_est @ self.C.T @ np.linalg.inv(self.C @ sigma_est @ self.C.T + self.R)

        y_valid = y.copy()
        y_valid[np.isnan(y_valid)] = 0
     
        self.mu = mu_est + K @ (y_valid - g)
        self.sigma = sigma_est - K @ self.C @ sigma_est

        return mu_est,sigma_est

if __name__ == "__main__":
    # Example initialization
    mu0 = np.random.rand(36)
    sigma0 = np.eye(36)
    kf = KalmanFilter(mu0, sigma0)

    # Example predict and update
    mu_pred, sigma_pred = kf.predict()
    y = np.random.rand(11)
    y[3] = np.nan  # Example with one NaN
    sat_pos = np.random.rand(3)

    mu_upd, sigma_upd = kf.update(y, mu_pred, sigma_pred, sat_pos)
 
