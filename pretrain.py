import os
import logging
import torch
from corrupter import BernCorrupter, BernCorrupterMulti
from read_data import index_ent_rel, graph_size, read_data
from config import config, overwrite_config_with_args
from logger_init import logger_init
from data_utils import inplace_shuffle, heads_tails
from select_gpu import select_gpu
from trans_e import TransE
from trans_d import TransD
from distmult import DistMult
from compl_ex import ComplEx

logger_init()
torch.cuda.set_device(select_gpu())
overwrite_config_with_args()

task_dir = config().task.dir
kb_index = index_ent_rel(os.path.join(task_dir, 'train.txt'),
                         os.path.join(task_dir, 'valid.txt'),
                         os.path.join(task_dir, 'test.txt'))
n_ent, n_rel = graph_size(kb_index)

train_data = read_data(os.path.join(task_dir, 'train.txt'), kb_index)
inplace_shuffle(*train_data)
valid_data = read_data(os.path.join(task_dir, 'valid.txt'), kb_index)
test_data = read_data(os.path.join(task_dir, 'test.txt'), kb_index)
heads, tails = heads_tails(n_ent, train_data, valid_data, test_data)
valid_data = [torch.LongTensor(vec) for vec in valid_data]
test_data = [torch.LongTensor(vec) for vec in test_data]
tester = lambda: gen.test_link(valid_data, n_ent, heads, tails)
train_data = [torch.LongTensor(vec) for vec in train_data]

mdl_type = config().pretrain_config
gen_config = config()[mdl_type]
if mdl_type == 'TransE':
    corrupter = BernCorrupter(train_data, n_ent, n_rel)
    gen = TransE(n_ent, n_rel, gen_config)
elif mdl_type == 'TransD':
    corrupter = BernCorrupter(train_data, n_ent, n_rel)
    gen = TransD(n_ent, n_rel, gen_config)
elif mdl_type == 'DistMult':
    corrupter = BernCorrupterMulti(train_data, n_ent, n_rel, gen_config.n_sample)
    gen = DistMult(n_ent, n_rel, gen_config)
elif mdl_type == 'ComplEx':
    corrupter = BernCorrupterMulti(train_data, n_ent, n_rel, gen_config.n_sample)
    gen = ComplEx(n_ent, n_rel, gen_config)
gen.pretrain(train_data, corrupter, tester)
gen.load(os.path.join(task_dir, gen_config.model_file))
gen.test_link(test_data, n_ent, heads, tails)