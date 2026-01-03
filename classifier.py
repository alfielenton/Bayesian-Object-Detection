import torch
import numpy as np

class BayesClassifier:

    def __init__(self, device, n_feats, n_cats, sig_Mu, sig_Sig, lr):

        self.lr = lr

        self.device = device
        self.C = n_cats
        self.N = n_feats

        self.sig_Mu = sig_Mu
        self.sig_Sig = sig_Sig

        self.log_nu_Sig = torch.tensor(np.random.rand(), device = self.device, dtype = torch.float32)
        self.log_nu_Mu = torch.tensor(np.random.rand(), device = self.device, dtype = torch.float32)

        self.lambda_Sig = torch.tensor(np.random.rand(self.N, self.C), device = self.device, dtype = torch.float32) * self.sig_Sig
        self.lambda_Mu = torch.tensor(np.random.rand(self.N, self.C), device = self.device, dtype = torch.float32) * self.sig_Mu

    def predict_Mu_Sig(self):
        nu_Sig = torch.exp(self.log_nu_Sig)
        return self.lambda_Mu, torch.exp(self.lambda_Sig + (nu_Sig**2)/2)
    
    def direction_lambda_Mu(self, batch):

        X, Y = batch
        nu_Sig = torch.exp(self.log_nu_Sig)
        alpha = torch.exp(-self.lambda_Sig + (nu_Sig**2)/2)

        mult = torch.zeros_like(self.lambda_Mu)
        mult.index_add_(1, Y, (X.T - self.lambda_Mu[:, Y]))

        return alpha * mult - self.lambda_Mu / (self.sig_Mu ** 2)
    
    def direction_lambda_Sig(self, batch):

        X, Y = batch
        nu_Sig = torch.exp(self.log_nu_Sig)
        alpha = torch.exp(-self.lambda_Sig + (nu_Sig**2)/2)

        nu_Mu = torch.exp(self.log_nu_Mu)

        mult = torch.zeros_like(self.lambda_Sig)
        mult.index_add_(1, Y, (1 - ((X.T - self.lambda_Mu[:, Y]) ** 2 + nu_Mu ** 2) * alpha[:, Y]))
        return -.5 * mult - self.lambda_Sig / (self.sig_Sig ** 2) - 2
    
    def direction_log_nu_Mu(self, batch):

        _, Y = batch
        nu_Sig = torch.exp(self.log_nu_Sig)
        alpha = torch.exp(-self.lambda_Sig + (nu_Sig**2)/2)

        nu_Mu = torch.exp(self.log_nu_Mu)

        grad = -nu_Mu * alpha[:, Y].sum() - (self.N * self.C * nu_Mu) / (self.sig_Mu ** 2) - (self.N * self.C) / nu_Mu
        return nu_Mu * grad
    
    def direction_log_nu_Sig(self, batch):

        X, Y = batch
        nu_Sig = torch.exp(self.log_nu_Sig)
        alpha = torch.exp(-self.lambda_Sig + (nu_Sig**2)/2)

        nu_Mu = torch.exp(self.log_nu_Mu)

        grad = -.5 * nu_Sig * (((X.T - self.lambda_Mu[:, Y]) ** 2 + nu_Mu ** 2) * alpha[:, Y]).sum() - self.N * self.C * nu_Sig/(self.sig_Sig ** 2) - self.N * self.C / nu_Sig
        return nu_Sig * grad
    
    def parameters_step(self, batch):

        self.lambda_Mu = self.lambda_Mu + self.lr * self.direction_lambda_Mu(batch)
        self.lambda_Sig = self.lambda_Sig + self.lr * self.direction_lambda_Sig(batch)
        self.log_nu_Mu = self.log_nu_Mu + self.lr * self.direction_log_nu_Mu(batch)
        self.log_nu_Sig = self.log_nu_Sig + self.lr * self.direction_log_nu_Sig(batch)

    def predict_cats(self, xs):

        Mu, Sig = self.predict_Mu_Sig()
        log_p = -.5 * self.N * torch.log(torch.tensor(2 * torch.pi)) -.5 * torch.log(Sig).sum(dim=0) \
                -.5 * ((xs[..., None] - Mu) * (1/Sig) * (xs[..., None] - Mu)).sum(dim=1)
    
        log_p = log_p - torch.logsumexp(log_p, dim = 1, keepdim = True)

        return log_p