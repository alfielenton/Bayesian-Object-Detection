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

        self.Mu_W = torch.zeros(self.M, self.N, dtype = torch.float32, device = self.device)
        self.inv_Sig_W = self.sig_W ** 2 * torch.eye(self.N, dtype = torch.float32, device = self.device)

    def weight_posterior(self, batch):

        X, Y = batch
        Y_T_X = Y.T @ X
        X_T_X = X.T @ X
        inv_Sig_W = X_T_X + (self.sig_y/self.sig_W)**2 * torch.eye(self.N, device=self.device)

        Mu_W = torch.linalg.solve(inv_Sig_W, Y_T_X.T).T
        inv_Sig_W = inv_Sig_W / self.sig_y ** 2

        return Mu_W, inv_Sig_W
    
    def parameters_step(self, batch):

        Mu_W, inv_Sig_W = self.weight_posterior(batch)

        Mu_diff = Mu_W - self.Mu_W
        inv_Sig_diff = inv_Sig_W - self.inv_Sig_W

        self.Mu_W = self.Mu_W + self.lr * Mu_diff
        self.inv_Sig_W = self.inv_Sig_W + self.lr * inv_Sig_diff

    def predict_boxes(self, X):
        
        mu_y = self.Mu_W @ X.T
        sig_y = self.sig_y ** 2 * (1 + (X * torch.linalg.solve(self.inv_Sig_W, X.T).T).sum(dim=1))

        I = torch.eye(self.M, device=X.device).expand(X.size(0), self.M, self.M)
        sig_y = sig_y[:, None, None] * I

        return mu_y.T, sig_y