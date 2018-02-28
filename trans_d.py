import os
import logging
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as f
from config import config
from torch.optim import Adam, SGD, Adagrad
from torch.autograd import Variable
from data_utils import batch_by_num
from base_model import BaseModel, BaseModule

class TransDModule(BaseModule):
    def __init__(self, n_ent, n_rel, config):
        super(TransDModule, self).__init__()
        self.margin = config.margin
        self.p = config.p
        self.temp = config.get('temp', 1)
        self.rel_embed = nn.Embedding(n_rel, config.dim)
        self.ent_embed = nn.Embedding(n_ent, config.dim)
        self.proj_rel_embed = nn.Embedding(n_rel, config.dim)
        self.proj_ent_embed = nn.Embedding(n_ent, config.dim)
        self.init_weight()

    def init_weight(self):
        for param in self.parameters():
            param.data.normal_(1 / param.size(1) ** 0.5)
            param.data.renorm_(2, 0, 1)

    def forward(self, src, rel, dst):
        src_proj = self.ent_embed(src) +\
                   t.sum(self.proj_ent_embed(src) * self.ent_embed(src), dim=-1, keepdim=True) * self.proj_rel_embed(rel)
        dst_proj = self.ent_embed(dst) +\
                   t.sum(self.proj_ent_embed(dst) * self.ent_embed(dst), dim=-1, keepdim=True) * self.proj_rel_embed(rel)
        return t.norm(dst_proj - self.rel_embed(rel) - src_proj + 1e-30, p=self.p, dim=-1)

    def dist(self, src, rel, dst):
        return self.forward(src, rel, dst)

    def score(self, src, rel, dst):
        return self.forward(src, rel, dst)

    def prob_logit(self, src, rel, dst):
        return -self.forward(src, rel ,dst) / self.temp

    def constraint(self):
        for param in self.parameters():
            param.data.renorm_(2, 0, 1)

class TransD(BaseModel):
    def __init__(self, n_ent, n_rel, config):
        super(TransD, self).__init__()
        self.mdl = TransDModule(n_ent, n_rel, config)
        self.mdl.cuda()
        self.config = config

    def load_vec(self, path):
        ent_mat = np.loadtxt(os.path.join(path, 'entity2vec.vec'))
        self.mdl.ent_embed.weight.data.copy_(t.from_numpy(ent_mat))
        rel_mat = np.loadtxt(os.path.join(path, 'relation2vec.vec'))
        n_rel = rel_mat.shape[0]
        self.mdl.rel_embed.weight.data.copy_(t.from_numpy(rel_mat))
        a_mat = np.loadtxt(os.path.join(path, 'A.vec'))
        self.mdl.proj_rel_embed.weight.data.copy_(t.from_numpy(a_mat[:n_rel, :]))
        self.mdl.proj_ent_embed.weight.data.copy_(t.from_numpy(a_mat[n_rel:, :]))
        self.mdl.cuda()

    def pretrain(self, train_data, corrupter, tester):
        src, rel, dst = train_data
        n_train = len(src)
        optimizer = Adam(self.mdl.parameters())
        #optimizer = SGD(self.mdl.parameters(), lr=1e-4)
        n_epoch = self.config.n_epoch
        n_batch = self.config.n_batch
        best_perf = 0
        for epoch in range(n_epoch):
            epoch_loss = 0
            rand_idx = t.randperm(n_train)
            src = src[rand_idx]
            rel = rel[rand_idx]
            dst = dst[rand_idx]
            src_corrupted, dst_corrupted = corrupter.corrupt(src, rel, dst)
            src_cuda = src.cuda()
            rel_cuda = rel.cuda()
            dst_cuda = dst.cuda()
            src_corrupted = src_corrupted.cuda()
            dst_corrupted = dst_corrupted.cuda()
            for s0, r, t0, s1, t1 in batch_by_num(n_batch, src_cuda, rel_cuda, dst_cuda, src_corrupted, dst_corrupted,
                                                  n_sample=n_train):
                self.mdl.zero_grad()
                loss = t.sum(self.mdl.pair_loss(Variable(s0), Variable(r), Variable(t0), Variable(s1), Variable(t1)))
                loss.backward()
                optimizer.step()
                self.mdl.constraint()
                epoch_loss += loss.data[0]
            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, n_epoch, epoch_loss / n_train)
            if (epoch + 1) % self.config.epoch_per_test == 0:
                test_perf = tester()
                if test_perf > best_perf:
                    self.save(os.path.join(config().task.dir, self.config.model_file))
                    best_perf = test_perf
        return best_perf
