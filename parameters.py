
class Parameters:

    def __init__(self):

        self.device = 'cuda'
        self.n_epochs = 20
        self.batch_size = 32
        self.feat_dim = 512
        self.box_dim = 4
        self.n_cats = 54

        self.sig_Sig_c = 1.
        self.sig_Mu_c = 1.
        self.lr_c = 1e-4
        self.lr_r = 1e-4
        self.lr_n = 1e-4

        self.sig_W_r = 1.
        self.sig_y_r = 1.