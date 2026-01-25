import torch
import math

class BayesClassifier:

    def __init__(self, device, n_feats, n_cats, log_Uq, log_Vq, Up, Vp, alpha, beta, lr):

        self.lr = lr

        self.device = device
        self.C = n_cats
        self.N = n_feats

        self.Mq = torch.zeros((self.N, self.C), dtype=torch.float32, device=self.device, requires_grad=True)
        self.log_Uq = torch.tensor(log_Uq, device=self.device, requires_grad=True)
        self.log_Vq = torch.tensor(log_Vq, device=self.device, requires_grad=True)

        self.log_tauq = torch.ones(self.C, device=self.device, dtype=torch.float32, requires_grad=True)
        self.log_epsq = torch.ones(self.C, device=self.device, dtype=torch.float32, requires_grad=True)

        self.optimiser = torch.optim.Adam([self.Mq, self.log_Uq, self.log_Vq, self.log_tauq, self.log_epsq], 
                                          lr = self.lr)
        
        self.Up = Up
        self.Vp = Vp

        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)

    def predict_cats(self, x):

        M = self.Mq
        Sigma = (torch.exp(self.log_epsq) / torch.exp(self.log_tauq)) * (1 + torch.exp(self.log_Uq + self.log_Vq))
        nu = 2 * torch.exp(self.log_tauq)

        log_p = torch.lgamma((nu + self.N) * .5) - torch.lgamma(.5 * nu) \
                -.5 * self.N * torch.log(nu * torch.pi) -.5 * self.N * torch.log(Sigma) \
                -.5 * (nu + self.N) * torch.log(1 + ((x[..., None] - M)**2).sum(dim=1) / (nu * Sigma))
        
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        return log_p
        
    def parameters_step(self, batch):

        x_d, C_d = batch
        D = x_d.size(0)

        exp_log_q_M = -.5 * (self.N * self.C) * (1 + math.log(2 * torch.pi)) \
                      - self.C * self.log_Uq - self.N * self.log_Vq
        
        exp_log_q_sig = -torch.lgamma(torch.exp(self.log_tauq)) - self.log_epsq + \
                    (torch.exp(self.log_tauq) + 1) * torch.digamma(torch.exp(self.log_tauq)) - torch.exp(self.log_tauq)
        exp_log_q_sig = exp_log_q_sig.sum()

        exp_log_q = exp_log_q_M + exp_log_q_sig

        tr_Up_inv_Vp_inv_Mq2 = (self.Mq ** 2).sum() / torch.exp(self.log_Uq + self.log_Vq)
        tr_Up_inv_Uq = self.N * (torch.exp(self.log_Uq * 2) / self.Up ** 2)
        tr_Vp_inv_vq = self.N * (torch.exp(2 * self.log_Vq) / self.Vp ** 2)

        exp_log_p_M = -.5 * self.N * self.C * math.log(2 * torch.pi) - self.C * self.log_Uq - self.N * self.log_Vq \
                      -.5 * (tr_Up_inv_Vp_inv_Mq2 + tr_Up_inv_Uq * tr_Vp_inv_vq)
        
        exp_log_p_sig = self.C * (self.alpha * torch.log(self.beta) - torch.lgamma(self.alpha)) \
                        -(self.alpha + 1) * (self.log_epsq - torch.digamma(torch.exp(self.log_tauq))).sum() \
                        -self.beta * (torch.exp(self.log_tauq - self.log_epsq)).sum()
        
        n_c = torch.zeros(self.C, device=self.device)
        for c in range(self.C):
            n_c[c] = (C_d == c).sum()

        M_cd = self.Mq[:, C_d].T
        tau_cd = torch.exp(self.log_tauq[C_d])
        eps_cd = torch.exp(self.log_epsq[C_d])

        sq_dist = ((x_d - M_cd)**2).sum(dim=1)

        exp_log_like = -D * math.log(self.C) -.5 * D * self.N * math.log(2 * torch.pi) \
                       -self.N * (n_c * (self.log_epsq - torch.digamma(torch.exp(self.log_tauq)))).sum() \
                       -.5 * (((tau_cd * (tau_cd + 1)) / eps_cd ** 2) * (sq_dist + self.N * torch.exp(self.log_Uq) * torch.exp(self.log_Vq))).sum()
        
        exp_log_p = exp_log_like + exp_log_p_M + exp_log_p_sig

        loss = exp_log_q - exp_log_p
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        with torch.no_grad():
            self.log_tauq.clamp(min=torch.tensor(1e-5, device=self.device))