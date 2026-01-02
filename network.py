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

    def __init__(self, n_feats, n_cats, Mu_scale, log_Sig_scale):

        self.n_cats = n_cats
        self.n_feats = n_feats
        self.Mu_scale = Mu_scale
        self.log_Sig_scale = log_Sig_scale

        self.log_Sig = torch.tensor(np.random.rand(self.n_feats, self.n_cats))
        self.Sig = torch.exp(self.log_Sig)
        self.Mu = torch.tensor(np.random.rand(self.n_feats, self.n_cats))
        