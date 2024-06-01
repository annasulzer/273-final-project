import numpy as np
from scipy.sparse import block_diag

class KalmanFilter:
    def __init__(self, mu0, sigma0, del_t=1):
        self.sigma = sigma0
        self.mu = mu0
        self.del_t = del_t
        self.Q = 0.005 * np.eye(len(mu0))
        self.R = 0.00001 * np.eye(11)

    def predict(self):
        omega_R = self.mu[3]
        omega_T = self.mu[4]
        omega_N = self.mu[5]
        
        STM_COM = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        STM_omega = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        STM_points = np.array([
            [1, -self.del_t * omega_N, self.del_t * omega_T],
            [self.del_t * omega_N, 1, self.del_t * omega_R],
            [-self.del_t * omega_T, -self.del_t * omega_R, 1]
        ])
        
        matrices = [STM_COM, STM_omega] + [STM_points] * 11
        matrices = [np.array(matrix, dtype=np.float64) for matrix in matrices]

        self.A = block_diag(matrices).toarray()
    

        mu_predict = self.A @ self.mu
        sigma_predict = self.A @ self.sigma @ self.A.T + self.Q
   
        return mu_predict, sigma_predict

    def update(self, y, mu_est, sigma_est, sat_pos):
        self.C = np.zeros((11, 39))
        valid_indices = []

        for i in range(len(y)):
            if not np.isnan(y[i]):
                j = 6 + 3 * i
                point_pos = mu_est[j:j+3]
                print(np.linalg.norm(sat_pos - point_pos))
                dist = np.linalg.norm(sat_pos - point_pos)
                self.C[i, j:j+3] = np.array([
                    -(sat_pos[0] - point_pos[0]) / dist,
                    -(sat_pos[1] - point_pos[1]) / dist,
                    -(sat_pos[2] - point_pos[2]) / dist
                ])
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            print("No valid measurements available for update.")
            return mu_est, sigma_est

        K = sigma_est @ self.C.T @ np.linalg.inv(self.C @ sigma_est @ self.C.T + self.R)
       

        y_valid = y.copy()
        y_valid[np.isnan(y_valid)] = 0
        self.mu = mu_est + K @ (y_valid - self.C @ mu_est)
        self.sigma = sigma_est - K @ self.C @ sigma_est



        return self.mu, self.sigma

if __name__ == "__main__":
    # Example initialization
    mu0 = np.random.rand(39)
    sigma0 = np.eye(39)
    kf = KalmanFilter(mu0, sigma0)

    # Example predict and update
    mu_pred, sigma_pred = kf.predict()
    y = np.random.rand(11)
    y[3] = np.nan  # Example with one NaN
    sat_pos = np.random.rand(3)

    mu_upd, sigma_upd = kf.update(y, mu_pred, sigma_pred, sat_pos)
    print(f"Final mu:\n{mu_upd}")
    print(f"Final sigma:\n{sigma_upd}")
