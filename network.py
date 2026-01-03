import torch
from torch import nn
import numpy as np

class CNNhead(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn_head = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 10, stride = 2),
                                      nn.ReLU(), 
                                      nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 6, stride = 2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size = 4, stride = 2))
        
        self.cnn_classification_tail1 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2),
                                                     nn.ReLU())
        self.cnn_classification_tail2 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 2, stride = 2),
                                                     nn.ReLU())
        self.cnn_classification_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        

        self.cnn_regression_tail1 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2),
                                                 nn.ReLU())
        self.cnn_regression_tail2 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 2, stride = 2),
                                                 nn.ReLU())
        self.cnn_regression_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        

        self.ffn_classification_tail = nn.Sequential(nn.Linear(in_features = 12 * 12 * 64, out_features = 2048),
                                                     nn.ReLU(), 
                                                     nn.Linear(in_features = 2048, out_features = 1024),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features = 1024, out_features = 512), 
                                                     nn.Tanh())
        
        self.ffn_regression_tail = nn.Sequential(nn.Linear(in_features = 12 * 12 * 64, out_features = 2048),
                                                     nn.ReLU(), 
                                                     nn.Linear(in_features = 2048, out_features = 1024),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features = 1024, out_features = 512), 
                                                     nn.Sigmoid())
        
    def forward(self, im):

        im_head = self.cnn_head(im)

        im_class1 = self.cnn_classification_tail1(im_head)
        im_reg1 = self.cnn_regression_tail1(im_head)

        im_class2 = self.cnn_classification_tail2(im_class1)
        im_reg2 = self.cnn_regression_tail2(im_reg1)

        im_class_pool = self.cnn_classification_pool(im_class2)
        im_reg_pool = self.cnn_regression_pool(im_reg2)

        im_class_pool = im_class_pool.view(-1, 12 * 12 * 64)
        im_reg_pool = im_reg_pool.view(-1, 12 * 12 * 64)

        im_class_feature = self.ffn_classification_tail(im_class_pool)
        im_reg_feature = self.ffn_regression_tail(im_reg_pool)

        return im_class_feature, im_reg_feature
    

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
        log_p = -.5 * self.N * torch.log(2 * torch.pi) -.5 * torch.log(Sig).sum(dim=0) \
                -.5 * ((xs[..., None] - Mu) * (1/Sig) * (xs[..., None] - Mu)).sum(dim=1)
        
        log_p = log_p - torch.log(torch.exp(log_p).sum(dim=1))
        return log_p
    
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

        mu_W = torch.linalg.solve(X_T_X + (self.sig_y/self.sig_W)**2 * torch.eye(self.N), Y_T_X.T).T
        return mu_W
    
    def parameters_step(self, batch):
        weight_diff = self.weight_posterior(batch) - self.W
        self.W = self.W + self.lr * weight_diff