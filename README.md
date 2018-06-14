# KBGAN
Liwei Cai and William Yang Wang, "KBGAN: Adversarial Learning for Knowledge Graph Embeddings", in *Proceedings of The 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL HLT 2018)*.

Paper: https://arxiv.org/abs/1711.04071

Our lab: http://nlp.cs.ucsb.edu/index.html

## Dependencies
* Python 3
* PyTorch 0.2.0
* PyYAML
* nvidia-smi

## Usage
1. Unzip `data.zip`.
2. Pretrain: `python3 pretrain.py --config=config_<dataset_name>.yaml --pretrain_config=<model_name>` (this will generate a pretrained model file)
2. Adversarial train: `python3 gan_train.py --config=config_<dataset_name>.yaml --g_config=<G_model_name> --d_config=<D_model_name>` (make sure that G model and D model are both pretrained)

Feel free to explore and modify parameters in config files. Default parameters are those used in experiments reported in the paper.

Decrease `adv.test_batch_size` if you experience GPU memory exhaustion. (this would make the program runs slower, but would not affect the test result)
