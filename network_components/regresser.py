import torch
import numpy as np
    
class BayesRegresser:

    def __init__(self, device, n_feats, n_labels, sig_W, sig_y, lr):

        self.device = device
        self.N = n_feats
        self.M = n_labels
        self.sig_W = sig_W
        self.sig_y = sig_y
        self.lr = lr

        self.W = torch.tensor(np.random.rand(self.M, self.N) * self.sig_W, dtype=torch.float32, device = self.device)

    def weight_posterior(self, batch):

        X, Y = batch
        Y_T_X = Y.T @ X
        X_T_X = X.T @ X

        mu_W = torch.linalg.solve(X_T_X + (self.sig_y/self.sig_W)**2 * torch.eye(self.N, device=self.device), Y_T_X.T).T
        return mu_W
    
    def parameters_step(self, batch):
        weight_diff = self.weight_posterior(batch) - self.W
        self.W = self.W + self.lr * weight_diff