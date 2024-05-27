import numpy as np
from scipy.sparse import block_diag

class KalmanFilter:
    def __init__(self, mu0,
                sigma0, del_t = 1):
        self.sigma = sigma0
        self.mu = mu0
        self.del_t = del_t
        self.Q = np.eye(39)
        self.R = 9 * np.eye(11)

        

    def predict(self):
        #W = np.random.multivariate_normal(np.zeros(2), self.Q[2:, 2:])

        omega_R = self.mu[3]
        omega_T = self.mu[4]
        omega_N = self.mu[5]
        
        STM_COM = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
        STM_omega = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
        STM_points = np.array([[1, -self.del_t*omega_N, self.del_t*omega_T],[self.del_t * omega_N, 1, self.del_t*omega_R], [-self.del_t * omega_T, -self.del_t*omega_R, 1]])
        matrices = [STM_COM, STM_omega] + [STM_points] * 11
        matrices = [np.array(matrix, dtype=np.float64) for matrix in matrices]

        self.A = block_diag(matrices).toarray()
  

        mu_predict = self.A @ self.mu
        sigma_predict = self.A @ self.sigma @ self.A.T + self.Q
        return mu_predict, sigma_predict
    
    def update(self, y, mu_est, sigma_est, sat_pos):
        #V = np.random.multivariate_normal(np.zeros(2), self.Q[2:, 2:])

        

        self.C = np.zeros((11, 39))

        for i in range(len(y)):
            j = int(6 + 3*i)
            point_pos = self.mu[j:j+3:]
            dist = np.linalg.norm(sat_pos - point_pos)
            self.C[i, j:j+3] = np.array([-(sat_pos[0] - point_pos[0])/dist, -(sat_pos[1] - point_pos[1])/dist, -(sat_pos[2] - point_pos[2])/dist]).reshape(-1)

        
        K = sigma_est @ self.C.T @np.linalg.inv(self.C @ sigma_est @ self.C.T + self.R)
  
        self.mu = mu_est + K @ (y - self.C @ mu_est)
        self.sigma = sigma_est - K @ self.C @ sigma_est

        return self.mu, self.sigma