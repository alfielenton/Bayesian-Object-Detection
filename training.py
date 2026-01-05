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
net_optim = optim.Adam(net.parameters(), opt.lr_n, weight_decay = 1e-8)
bc = classifier.BayesClassifier(opt.device, opt.feat_dim, opt.n_cats, opt.sig_Mu_c, opt.sig_Sig_c, opt.lr_c)
br = regresser.BayesRegresser(opt.device, opt.feat_dim, opt.box_dim, opt.sig_W_r, opt.sig_y_r)
print('Models obtained\n')

print('Getting progress tracker...')
pt = ProgressTracker()
pt.start_timer()
print('Progress tracker obtained and timer started')

print('Starting training...\n')
net.train()
n_epochs = opt.n_epochs

for epoch in range(n_epochs):

    pt.start_epoch()
    for ids in train_dl:

        x_batch, y_c, y_r = data_handling_functions.generate_batch(ids)
        x_batch, y_c, y_r = x_batch.to(opt.device), y_c.to(opt.device), y_r.to(opt.device)

        c_feat, r_feat = net(x_batch)

        c_preds = bc.predict_cats(c_feat)
        r_preds_mu, r_preds_sig = br.predict_boxes(r_feat)

        c_loss = -c_preds[torch.arange(y_c.size(0), device = opt.device), y_c].sum()

        var = torch.diagonal(r_preds_sig, dim1=1, dim2=2)
        r_loss = .5 * torch.log(var).sum(dim=1) \
                 +.5 * ((y_r - r_preds_mu) / var).sum(dim=1)
        r_loss = r_loss.sum()

        net_optim.zero_grad()

        loss = r_loss + c_loss
        loss.backward()

        net_optim.step()

        pt.record_losses(c_loss.detach(), r_loss.detach(), loss.detach())
        with torch.no_grad():
            bc.parameters_step((c_feat, y_c))
            br.parameters_step((r_feat, y_r))

    pt.end_epoch()