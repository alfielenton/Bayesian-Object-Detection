import torch
import numpy as np
    
class BayesRegresser:

    def __init__(self, device, n_feats, n_labels, sig_wc, sig_y):

        self.device = device
        self.N = n_feats
        self.M = n_labels
        self.sig_wc = sig_wc
        self.sig_y = sig_y

        self.Mu_w = torch.zeros(self.M, self.N, dtype=torch.float32, device=self.device)
        self.inv_Sig_wc = (1 / self.sig_wc ** 2) * torch.eye(self.N, dtype=torch.float32, device=self.device)
        self.Sig_y = (self.sig_y ** 2) * torch.eye(self.M, device=self.device, dtype=torch.float32)

    
    def parameters_step(self, batch):

        with torch.no_grad():
            X, Y = batch

            XtX = X.T @ X           
            YtX = Y.T @ X          

            inv_Sig_wc = self.inv_Sig_wc
            self.inv_Sig_wc = self.inv_Sig_wc + XtX
            self.Mu_w = (self.Mu_w @ inv_Sig_wc + YtX) @ torch.linalg.inv(self.inv_Sig_wc)

    def predict_boxes(self, X):

        mu_y = (self.Mu_w @ X.T).T 
        sig_y = 1 + (X * torch.linalg.solve(self.inv_Sig_wc, X.T).T).sum(dim=1)
        sig_y = sig_y[:, None, None] * self.Sig_y

        return mu_y, sig_y