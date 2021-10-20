#%%
# import things
# model code
import torch
from torch.utils.data import DataLoader
from utils.eval_utils import NLVLEvaluator
from utils.loss import NLVLLoss

# logging code
from torch.utils.tensorboard import SummaryWriter

# etc
import random
import numpy as np
import os
from os.path import join
from tqdm import tqdm
from copy import deepcopy
from yacs.config import CfgNode
from yaml import dump as dump_yaml
import argparse

#%%
# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model",'-m',type=str,default="CrossModalityTwostageAttention")
parser.add_argument("--config",'-c',type=str,default="configs/cha_simple_model/simplemodel_cha_BS256_two-stage_attention.yml")
parser.add_argument("--pre_trained", type=str, default="pretrained_weight.pth")
parser.add_argument("--seed", '-s',type=int,default=0)
parser.add_argument("--reg_w", type=float, default=1.0)
args = parser.parse_args()

dict_args = {
    "model": args.model,
    "confg": args.config
}

random.seed(int(args.seed))
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(int(args.seed))
torch.manual_seed(int(args.seed))
torch.cuda.manual_seed(int(args.seed))
torch.cuda.manual_seed_all(int(args.seed))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

#%%
# load things according to arguments
if args.model == "SimpleModel":
    from models.simple_model import SimpleModel as Model
    from config_parsers.simple_model_config import _C as TestCfg
elif args.model == "CrossModalityTwostageAttention":
    from models.simple_model_cross_modal_two_stage import SimpleModel as Model
    from config_parsers.simple_model_cross_modality_twostage_attention_config import _C as TestCfg
else:
    raise ValueError("No such model: {}".format(args.model))


#%%
# constants
CONFIG_PATH = args.config
print("config file path: ", CONFIG_PATH)
TestCfg.merge_from_file(CONFIG_PATH)
cfg = TestCfg
device = torch.device("cuda")
DATA_PATH = cfg.DATASET.DATA_PATH
ANNO_PATH ={"train": cfg.DATASET.TRAIN_ANNO_PATH,
            "test": cfg.DATASET.TEST_ANNO_PATH}
VID_PATH = cfg.DATASET.VID_PATH

#%%
# function for dumping yaml
def cfg_to_dict(cfg):
    dict_cfg = dict(cfg)
    for k,v in dict_cfg.items():
        if isinstance(v,CfgNode):
            dict_cfg[k] = cfg_to_dict(v)
    return dict_cfg

#%%
# Load dataloader
dataset_name = cfg.DATASET.NAME
if dataset_name == "Charades":
    from dataset.charades_basic import CharadesBasicDatasetBuilder
    dataloaders = CharadesBasicDatasetBuilder(cfg,data_path=DATA_PATH,anno_path=ANNO_PATH,vid_path=VID_PATH).make_dataloaders()
elif dataset_name == "AnetCap":
    from dataset.anetcap_basic import AnetCapBasicDatasetBuilder
    dataloaders = AnetCapBasicDatasetBuilder(cfg,data_path=DATA_PATH,anno_path=ANNO_PATH,vid_path=VID_PATH).make_dataloaders()
else:
    raise ValueError("No such dataset: {}".format(dataset_name))

#%%
# load training stuff
## model and loss
model = Model(cfg).to(device)
model.load_state_dict(torch.load(args.pre_trained))

## evaluator
evaluator = NLVLEvaluator(cfg)
batch_to_device = dataloaders['test'].dataset.batch_to_device

# information print out
print("====="*10)
print(args)
print("====="*10)
print(cfg)
print("====="*10)

#%%
# test loop
model.eval()
pbar = tqdm(range(1))
for epoch in pbar:
    eval_results_list = []
    for batch_idx,item in enumerate(dataloaders['test']):
        # update progress bar
        pbar.set_description("test {}/{}".format(batch_idx,len(dataloaders['test'])))
        # make evaluation
        with torch.no_grad():
            batch_to_device(item,device)
            model_outputs = model(item)
            eval_results = evaluator(model_outputs,item)
        # add to mean eval results
        eval_results_list.append(eval_results)
    # write test information
    ## make mean metric dict
    mean_eval_results_dict = {}
    for k in list(eval_results_list[0].keys()):
        mean_eval_results_dict[k] = torch.mean(torch.Tensor([x[k] for x in eval_results_list]))

    # print test information
    tqdm.write("****** epoch:{} ******".format(epoch))
    for k,v in mean_eval_results_dict.items():
        tqdm.write("\t{}: {}".format(k,v))
