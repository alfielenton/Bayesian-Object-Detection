
class Parameters:

    def __init__(self):

        self.device = 'cuda'
        self.n_epochs = 80
        self.batch_size = 32
        self.feat_dim = 512
        self.box_dim = 4
        self.n_cats = 54
        self.lr_net = 1e-3

        self.log_Uq = 1.
        self.log_Vq = 1.
        self.Up = 1.
        self.Vp = 1.
        self.alpha = 1.
        self.beta = 1.
        self.lr_c = 1e-3

        self.sig_W_r = 1.
        self.sig_y_r = 1.