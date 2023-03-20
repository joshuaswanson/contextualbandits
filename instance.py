class Linear():
    
    def __init__(X, theta, theta_star):
        
        # arms
        self.X = X
        self.n, self.d = X.shape
        
        # 
        self.theta = np.zeros(self.d)
        self.theta_star = theta_star
        
        self.V = np.eye(self.d)
        self.B = np.matmul(X.reshape(-1,self.d,1), X.reshape(-1,1,self.d))
        
        self.T = T  # time steps
        self.sigma = sigma
        self.name = name
        self.arms_chosen = []