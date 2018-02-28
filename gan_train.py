import os
import logging
import datetime
import torch
from random import sample, random

from config import config, overwrite_config_with_args, dump_config
from read_data import index_ent_rel, graph_size, read_data
from data_utils import heads_tails, inplace_shuffle, batch_by_num
from trans_e import TransE
from trans_d import TransD
from distmult import DistMult
from compl_ex import ComplEx
from logger_init import logger_init
from select_gpu import select_gpu
from corrupter import BernCorrupterMulti


logger_init()
torch.cuda.set_device(select_gpu())
overwrite_config_with_args()
dump_config()

task_dir = config().task.dir
kb_index = index_ent_rel(os.path.join(task_dir, 'train.txt'),
                         os.path.join(task_dir, 'valid.txt'),
                         os.path.join(task_dir, 'test.txt'))
n_ent, n_rel = graph_size(kb_index)

models = {'TransE': TransE, 'TransD': TransD, 'DistMult': DistMult, 'ComplEx': ComplEx}
gen_config = config()[config().g_config]
dis_config = config()[config().d_config]
gen = models[config().g_config](n_ent, n_rel, gen_config)
dis = models[config().d_config](n_ent, n_rel, dis_config)
gen.load(os.path.join(task_dir, gen_config.model_file))
dis.load(os.path.join(task_dir, dis_config.model_file))

train_data = read_data(os.path.join(task_dir, 'train.txt'), kb_index)
inplace_shuffle(*train_data)
valid_data = read_data(os.path.join(task_dir, 'valid.txt'), kb_index)
test_data = read_data(os.path.join(task_dir, 'test.txt'), kb_index)
filt_heads, filt_tails = heads_tails(n_ent, train_data, valid_data, test_data)
valid_data = [torch.LongTensor(vec) for vec in valid_data]
test_data = [torch.LongTensor(vec) for vec in test_data]
tester = lambda: dis.test_link(valid_data, n_ent, filt_heads, filt_tails)
train_data = [torch.LongTensor(vec) for vec in train_data]

dis.test_link(test_data, n_ent, filt_heads, filt_tails)

corrupter = BernCorrupterMulti(train_data, n_ent, n_rel, config().adv.n_sample)
src, rel, dst = train_data
n_train = len(src)
n_epoch = config().adv.n_epoch
n_batch = config().adv.n_batch
mdl_name = 'gan_dis_' + datetime.datetime.now().strftime("%m%d%H%M%S") + '.mdl'
best_perf = 0
avg_reward = 0
for epoch in range(n_epoch):
    epoch_d_loss = 0
    epoch_reward = 0
    src_cand, rel_cand, dst_cand = corrupter.corrupt(src, rel, dst, keep_truth=False)
    for s, r, t, ss, rs, ts in batch_by_num(n_batch, src, rel, dst, src_cand, rel_cand, dst_cand, n_sample=n_train):
        gen_step = gen.gen_step(ss, rs, ts, temperature=config().adv.temperature)
        src_smpl, dst_smpl = next(gen_step)
        losses, rewards = dis.dis_step(s, r, t, src_smpl.squeeze(), dst_smpl.squeeze())
        epoch_reward += torch.sum(rewards)
        rewards = rewards - avg_reward
        gen_step.send(rewards.unsqueeze(1))
        epoch_d_loss += torch.sum(losses)
    avg_loss = epoch_d_loss / n_train
    avg_reward = epoch_reward / n_train
    logging.info('Epoch %d/%d, D_loss=%f, reward=%f', epoch + 1, n_epoch, avg_loss, avg_reward)
    if (epoch + 1) % config().adv.epoch_per_test == 0:
        #gen.test_link(valid_data, n_ent, filt_heads, filt_tails)
        perf = dis.test_link(valid_data, n_ent, filt_heads, filt_tails)
        if perf > best_perf:
            best_perf = perf
            dis.save(os.path.join(config().task.dir, mdl_name))
dis.load(os.path.join(config().task.dir, mdl_name))
dis.test_link(test_data, n_ent, filt_heads, filt_tails)
