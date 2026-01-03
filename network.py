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