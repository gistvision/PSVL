#%%
"""
charades_basic.py
****
the basic charades dataset class.
"""
#%%
# import things
import os
from os.path import join
import json
import h5py
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from random import random, randint, choice
from collections import Counter

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config_parsers.simple_model_config import _C as TestCfg


#%%
# dataset builder class
class CharadesBasicDatasetBuilder():
    def __init__(self,cfg,data_path=None,anno_path=None,vid_path=None):
        # make variables that will be used in the future
        self.cfg = cfg
        self.splits = ["train","test"]  # list of splits
        # make paths
        if len(data_path) > 0:
            self.vid_path = join(data_path,"i3d_finetuned")
            self.anno_path = {s: join(data_path,"annotations","charades_sta_{}_pos_original.json".format(s)) for s in self.splits}
        if len(vid_path) > 0:
            self.vid_path = vid_path
        if len(anno_path) > 0:
            self.anno_path = anno_path
        # read annotations
        self.annos = self._read_annos(self.anno_path)
        # make dictionary of word-index correspondence
        self.wtoi, self.itow = self._make_word_dictionary(self.annos)
    
    def _read_annos(self,anno_path):
        # read annotations
        
        annos = {s: None for s in self.splits}
        for s in self.splits:
            with open(anno_path[s],'r') as f:
                #annos[s] = json.load(f)[:100]
                annos[s] = json.load(f)
        return annos
        
    def _make_word_dictionary(self,annos):
        """
        makes word tokens - number idx correspondences
        ARGS:
            - annos: annotations read
        RETURNS:
            - wtoi: word -> index dictionary
            - itow: index -> word dictionary
        PARAMS:
            - DATASET.SHOW_TOP_VOCAB: the number of top-n tokens to print
        """
        # get training annos
        train_annos = self.annos["train"]
        # read tokens
        tokens_list = []
        for ann in train_annos:
            tokens_list += [tk for tk in ann["tokens"]]
        # print results: count tokens and show top-n
        print("Top-{} tokens list:".format(self.cfg.DATASET.SHOW_TOP_VOCAB))
        tokens_count = sorted(Counter(tokens_list).items(), key=lambda x:x[1])
        for tk in tokens_count[-self.cfg.DATASET.SHOW_TOP_VOCAB:]:
            print("\t- {}: {}".format(tk[0],tk[1]))
        # make wtoi, itow
        wtoi = {}
        wtoi["<PAD>"], wtoi["<UNK>"] = 0, 1
        wtoi["<S>"], wtoi["<E>"] = 2, 3
        for i,(tk,cnt) in enumerate(tokens_count):
            idx = i+4   # idx start at 4
            wtoi[tk] = idx
        itow = {v:k for k,v in wtoi.items()}
        self.cfg.MODEL.QUERY.EMB_IDIM = len(wtoi)
        return wtoi, itow
        
    def make_dataloaders(self):
        """
        makes actual dataset class
        RETURNS:
            - dataloaders: dataset classes for each splits. dictionary of {split: dataset}
        """
        # read annotations
        annos = self._read_annos(self.anno_path)
        # make dictionary of word-index correspondence
        wtoi, itow = self._make_word_dictionary(annos)
        batch_size = self.cfg.TRAIN.BATCH_SIZE
        num_workers = self.cfg.TRAIN.NUM_WORKERS
        dataloaders = {}
        for s in self.splits:
            if "train" in s:
                dataset = CharadesBasicDataset(self.cfg, self.vid_path, s, wtoi, itow, annos[s])
                dataloaders[s] = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=dataset.collate_fn, drop_last=True, shuffle=True)
            else:
                dataset = CharadesBasicDataset(self.cfg, self.vid_path, s, wtoi, itow, annos[s])
                dataloaders[s] = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=dataset.collate_fn, drop_last=False, shuffle=False)
        return dataloaders


#%%
# Charades Dataset Class
class CharadesBasicDataset(Dataset):
    def __init__(self, cfg, vid_path, split, wtoi, itow, annos):
        self.cfg = cfg
        self.vid_path = vid_path
        self.split = split
        self.wtoi = wtoi
        self.itow = itow
        self.annos = annos
        self.feats = self._load_vid_feats()
        self.num_segments = self.cfg.DATASET.NUM_SEGMENT
        self.sentence_max_length = self.cfg.DATASET.MAX_LENGTH
    
    def _load_vid_feats(self):
        feats = {}
        vid_list = [x['vid'] for x in self.annos]
        for vid in tqdm(vid_list, desc="loading video features"):
            feats[vid] = np.load(join(self.vid_path,"{}.npy".format(vid))).squeeze()
        return feats

    def _tokens_to_index(self,tokens):
        """
        translates list of tokens into trainable index format. also does padding.
        """
        wids = []
        for tk in tokens:
            if tk in self.wtoi.keys():
                wids.append(self.wtoi[tk])
            else:
                wids.append(1)  # <UNK>
        for _ in range(self.sentence_max_length - len(wids)):
            wids.append(0)
        if len(wids) > self.sentence_max_length:
            wids = wids[:self.sentence_max_length]
        return wids
    
    def get_fixed_length_feat(self, feat, num_segment, start_pos, end_pos):
        """
        makes fixed length feature. adopted from LGI code.
        """
        nfeats = feat[:,:].shape[0]
        if nfeats <= self.num_segments:
            stride = 1
        else:
            stride = nfeats * 1.0 / num_segment
        if self.split != "train":
            spos = 0
        else:
            random_end = -0.5 + stride
            if random_end == np.floor(random_end):
                random_end = random_end - 1.0
            spos = np.random.random_integers(0,random_end)
        s = np.round( np.arange(spos, nfeats-0.5, stride) ).astype(int)
        start_pos =  float(nfeats-1.0) * start_pos
        end_pos = float(nfeats-1.0) * end_pos

        if not (nfeats < self.num_segments and len(s) == nfeats) \
                and not (nfeats >= self.num_segments and len(s) == num_segment):
            s = s[:num_segment] # ignore last one
        assert (nfeats < self.num_segments and len(s) == nfeats) \
                or (nfeats >= self.num_segments and len(s) == num_segment), \
                "{} != {} or {} != {}".format(len(s), nfeats, len(s), num_segment)

        start_index, end_index =  None, None
        for i in range(len(s)-1):
            if s[i] <= end_pos < s[i+1]:
                end_index = i
            if s[i] <= start_pos < s[i+1]:
                start_index = i

        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = num_segment-1

        cur_feat = feat[s, :]
        nfeats = min(nfeats, num_segment)
        out = np.zeros((num_segment, cur_feat.shape[1]))
        out [:nfeats,:] = cur_feat
        return out, nfeats, start_index, end_index

    def make_attention_mask(self,start_index,end_index):
        attn_mask = np.zeros([self.num_segments])
        attn_mask[start_index:end_index+1] = 1
        attn_mask = torch.Tensor(attn_mask)
        return attn_mask

    def __getitem__(self,idx):
        anno = self.annos[idx]
        vid = anno["vid"]
        duration = anno['duration']
        timestamp = [x*duration for x in anno['timestamp']]
        start_pos, end_pos = anno['timestamp']
        query_label = self._tokens_to_index(anno['tokens'])
        query_length = len(anno['tokens'])
        vid_feat = self.feats[vid]
        
        fixed_vid_feat, nfeats, start_index, end_index = self.get_fixed_length_feat(vid_feat, self.num_segments, start_pos, end_pos)
        # get video masks
        vid_mask = np.zeros((self.num_segments, 1))
        vid_mask[:nfeats] = 1
        # make attn mask
        instance = {
            "vids": vid,
            "qids": idx,
            "timestamps": timestamp, # GT location [s, e] (second)
            "duration": duration, # video span (second)
            "query_lengths": query_length,
            "query_labels": torch.LongTensor(query_label).unsqueeze(0),     # [1,L_q_max]
            "query_masks": (torch.FloatTensor(query_label)>0).unsqueeze(0), # [1,L_q_max]
            "grounding_start_pos": torch.FloatTensor([start_pos]), # [1]; normalized
            "grounding_end_pos": torch.FloatTensor([end_pos]),     # [1]; normalized
            "nfeats": torch.FloatTensor([nfeats]),
            "video_feats": torch.FloatTensor(fixed_vid_feat), # [L_v,D_v]
            "video_masks": torch.ByteTensor(vid_mask), # [L_v,1]
            "attention_masks": self.make_attention_mask(start_index,end_index),
        }
        return instance

    def collate_fn(self, data):
        seq_items = ["video_feats", "video_masks","attention_masks"]
        tensor_items = [
            "query_labels", "query_masks", "nfeats",
            "grounding_start_pos", "grounding_end_pos"
        ]
        batch = {k: [d[k] for d in data] for k in data[0].keys()}
        if len(data) == 1:
            for k,v in batch.items():
                if k in tensor_items:
                    batch[k] = torch.cat(batch[k], 0)
                elif k in seq_items:
                    batch[k] = torch.nn.utils.rnn.pad_sequence(
                            batch[k], batch_first=True)
                else:
                    batch[k] = batch[k][0]
        else:
            for k in tensor_items:
                batch[k] = torch.cat(batch[k], 0)
            for k in seq_items:
                batch[k] = torch.nn.utils.rnn.pad_sequence(batch[k], batch_first=True)
        return batch

    def batch_to_device(self,batch,device):
        for k,v in batch.items():
            if isinstance(v,torch.Tensor):
                batch[k] = v.to(device)

    def __len__(self):
        return len(self.annos)


#%%
# test
if __name__ == "__main__":
    DATA_PATH = "/home/skaws2003/projects/didemo/localglobal/data/charades"
    dataloaders = CharadesBasicDatasetBuilder(TestCfg,DATA_PATH).make_dataloaders()
    for item in dataloaders['train']:
        _ = item['vids']

