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
        self.inv_Sig_W = ((1/self.sig_W ** 2) * torch.eye(self.N, dtype = torch.float32, device = self.device)).expand(self.M, self.N, self.N)
    
    def parameters_step(self, batch):
        X, Y = batch

        XtX = X.T @ X           
        YtX = Y.T @ X          

        inv_Sig_W_prior = self.inv_Sig_W

        inv_Sig_W_post = inv_Sig_W_prior + (1 / self.sig_y ** 2) * XtX
        inv_Sig_W_post = 0.5 * (inv_Sig_W_post + torch.transpose(inv_Sig_W_post, 1, 2))

        rhs = (inv_Sig_W_prior * self.Mu_W[:, None, :]).sum(dim=1) + (1 / self.sig_y ** 2) * YtX

        self.Mu_W = torch.linalg.solve(inv_Sig_W_post, rhs[..., None]).squeeze(-1)
        self.inv_Sig_W = inv_Sig_W_post


    def predict_boxes(self, X):

        mu_y = (self.Mu_W @ X.T).T 

        X_exp = X.T.unsqueeze(0).expand(self.M, -1, -1)

        tmp = torch.linalg.solve(self.inv_Sig_W, X_exp)
        quad = (X_exp * tmp).sum(dim=1)

        var_y = self.sig_y ** 2 * (1 + quad.T)

        I = torch.eye(self.M, device=self.device).expand(X.size(0), self.M, self.M)
        sig_y = var_y[:, None, :] * I

        return mu_y, sig_y