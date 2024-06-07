import numpy as np



class KalmanFilter:
    def __init__(self, mu0, sigma0, del_t):
        self.sigma = sigma0
        self.mu = mu0
        self.del_t = del_t
        self.Q = 1 * del_t * np.eye(len(mu0))
        self.Q[:3, :3] = 0.000001 * np.eye(3)#angular velocity
        self.R = 0.01 * np.eye(33)

    def predict(self):

        omega_x = self.mu[0]
        omega_y = self.mu[1]
        omega_z = self.mu[2]

        
        STM = np.eye(len(self.mu))
        self.del_t = 0.5*self.del_t 
        for i in range(3, len(self.mu), 3):
            
            STM[i:i+3, 0:3] = [[0, self.del_t * self.mu[i+2], -self.del_t * self.mu[i+1]],
                            [-self.del_t * self.mu[i+2], 0, self.del_t * self.mu[i]],
                            [self.del_t * self.mu[i+1], -self.del_t * self.mu[i], 0]]
            
            
            STM[i:i+3, i:i+3] = [[1, -self.del_t * omega_z, self.del_t * omega_y],
                                [self.del_t * omega_z, 1, -self.del_t * omega_x],
                                [-self.del_t * omega_y, self.del_t * omega_x, 1]]
                                
        self.del_t = 2*self.del_t 
                          
        
        self.A = STM
        
        mu_predict = self.A @ self.mu 
        sigma_predict = self.A @ self.sigma @ self.A.T + self.Q
      
        return mu_predict, sigma_predict

    def update(self, y, mu_est, sigma_est, sat_pos):

        
        valid_indices = [i for i in range(len(y)) if not np.isnan(y[i])]
   
       
        # Initialize 
        self.C = np.zeros((len(valid_indices), len(mu_est)))
        g = np.zeros(len(valid_indices))

        for idx, i in enumerate(valid_indices): #i = 0-32 except for the blind ones, idx = 0-(32-blind)
         
            if (i > 21):
                n = 2
            elif (i > 10):
                n = 1
            else:
                n = 0

            j = 3 + 3 * np.mod(i, 11)
            point_pos = mu_est[j:j+3]
            
            dist = np.linalg.norm(point_pos - sat_pos[n])
        

            self.C[idx, j:j+3] = (point_pos - sat_pos[n]) / dist
            g[idx] = dist


        if len(valid_indices) == 0:
            print("No valid measurements available for update.")
            return mu_est, sigma_est

        #plt.spy(self.C)

        valid_indices = np.array(valid_indices)

  
        R_valid = self.R[valid_indices[:, None], valid_indices]
        R_valid += np.eye(R_valid.shape[0]) * 1e-9

        K = sigma_est @ self.C.T @ np.linalg.inv(self.C @ sigma_est @ self.C.T + R_valid) #eliminate all dimensions that are nan 
        #plt.spy() visualize sparsity structure of matrix -> check if sigma_est and C if block diagonalor sufficially close then we can do incremental update
        #information filter (not recommended)
        y_valid = y[valid_indices]

        self.mu = mu_est + K @ (y_valid - g)
       
        self.sigma = sigma_est - K @ self.C @ sigma_est
        O = np.vstack((self.C, self.C@self.A, self.C@self.A@self.A.T))

       
        return self.mu, self.sigma, np.linalg.matrix_rank(O)


 
