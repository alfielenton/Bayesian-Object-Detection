from torch import nn

class CNNhead(nn.Module):

    def __init__(self):

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
        

        self.ffn_classification_tail = nn.Sequential(nn.Linear(in_features = 11 * 11 * 64, out_features = 2048),
                                                     nn.ReLU(), 
                                                     nn.Linear(in_features = 2048, out_features = 1024),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features = 1024, out_features = 512), 
                                                     nn.Tanh())
        
        self.ffn_regression_tail = nn.Sequential(nn.Linear(in_features = 11 * 11 * 64, out_features = 2048),
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

        im_class_feature = self.ffn_classification_tail(self.cnn_classification_pool(im_class2))
        im_reg_feature = self.ffn_regression_tail(self.cnn_regression_pool(im_reg2))

        return im_class_feature, im_reg_feature