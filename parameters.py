
class Parameters:

    def __init__(self):

        self.device = 'cuda'
        self.batch_size = 32

        self.sig_Sig_c = 1.
        self.sig_Mu_c = 1.
        self.lr_c = 1e-4

        self.sig_W_r = 1.
        self.sig_y_r = 1.