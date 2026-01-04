import json
import torch
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from parameters import Parameters
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
br = regresser.BayesRegresser(opt.device, opt.feat_dim, opt.box_dim, opt.sig_W_r, opt.sig_y_r, opt.lr_r)
print('Models obtained\n')