import json
import torch
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from parameters import Parameters
from stats_collector import ProgressTracker
from network_components import classifier, regresser, network
from data_handlings import data_handling_functions

opt = Parameters()

with open('dataset//train//measurements.json','r') as f:
    meas = json.load(f)['category_count']

with open('dataset//ids.json', 'r') as f:
    ids = json.load(f)

with open('dataset//categories.json', 'r') as f:
    categories = json.load(f)['categories']

train_ids = ids['train']
valid_ids = ids['valid']
test_ids = ids['test']

print('Making training sampler...')
labels = [data_handling_functions.get_labels(id)[0] for id in train_ids]
weights = torch.tensor([1 / meas[categories[c]] for c in labels])
weights /= weights.sum()

sampler = WeightedRandomSampler(weights, num_samples=weights.size(0), replacement = True)
train_dl = DataLoader(train_ids, batch_size = opt.batch_size, shuffle = False)
print('Data loader and sampler created.\n')

print('Getting models...')
net = network.CNNhead().to(opt.device)
net_optim = optim.Adam(net.parameters(), opt.lr_net, weight_decay = 1e-8)
bc = classifier.BayesClassifier(opt.device, opt.feat_dim, opt.n_cats, opt.log_Uq, opt.log_Vq, opt.Up, opt.Vp, opt.alpha, opt.beta, opt.lr_c)
br = regresser.BayesRegresser(opt.device, opt.feat_dim, opt.box_dim, opt.sig_W_r, opt.sig_y_r)
print('Models obtained\n')

print('Getting progress tracker...')
pt = ProgressTracker()
pt.start_timer()
print('Progress tracker obtained and timer started')

print('Starting training...\n')
n_epochs = opt.n_epochs

for epoch in range(n_epochs):

    pt.start_epoch()
    for ids in train_dl:

        x_batch, y_c, y_r = data_handling_functions.generate_batch(ids)
        x_batch, y_c, y_r = x_batch.to(opt.device), y_c.to(opt.device), y_r.to(opt.device)

        net.eval()
        data_handling_functions.freeze(net)

        with torch.no_grad():
            c_feat, r_feat = net(x_batch)

        bc.parameters_step((c_feat, y_c))
        br.parameters_step((r_feat, y_r))

        net.train()
        data_handling_functions.unfreeze(net)

        c_feat, r_feat = net(x_batch)

        c_logp = bc.predict_cats(c_feat)
        c_loss = -c_logp[torch.arange(y_c.size(0)), y_c].mean()

        r_mu, r_sig = br.predict_boxes(r_feat)
        var = torch.diagonal(r_sig, dim1=1, dim2=2)

        r_loss = (
            0.5 * torch.log(var).sum(dim=1)
            + 0.5 * ((y_r - r_mu) ** 2 / var).sum(dim=1)
        ).mean()

        loss = c_loss + r_loss
        net_optim.zero_grad()
        loss.backward()
        net_optim.step()

        pt.record_losses(c_loss.detach(), r_loss.detach(), loss.detach())

    pt.end_epoch()