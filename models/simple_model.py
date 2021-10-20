#%%
"""
simple_model.py
****
simple, basic model for NLVL.
 - Query-Video matching with (Multi-Head Attention + ConvBNReLU) with residual connection
 - Video Encoding with simple GRU
"""

#%%
# import things
import torch
import torch.nn as nn

#%%
# model
class SimpleSentenceEmbeddingModule(nn.Module):
    """
    A Simple Query Embedding class
    """
    def __init__(self, cfg):
        super().__init__()
        # config params
        self.cfg = cfg
        self.query_length = self.cfg.DATASET.MAX_LENGTH
        # embedding Layer
        emb_idim = self.cfg.MODEL.QUERY.EMB_IDIM
        emb_odim = self.cfg.MODEL.QUERY.EMB_ODIM
        self.embedding = nn.Embedding(emb_idim, emb_odim)
        # RNN Layer
        gru_hidden = self.cfg.MODEL.QUERY.GRU_HDIM
        self.gru = nn.GRU(input_size=emb_odim,hidden_size=gru_hidden,num_layers=1,batch_first=True,bidirectional=True)
        # feature adjust
        emb_dim = self.cfg.MODEL.FUSION.EMB_DIM
        self.feature_aggregation = nn.Sequential(
            nn.Linear(in_features=gru_hidden*2,out_features=emb_dim),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, query_labels, query_masks):
        """
        encode query sequence using RNN and return logits over proposals.
        code adopted from LGI
        Args:
            query_labels: query_labels vectors of query; [B, vocab_size]
            query_masks: mask for query; [B,L]
            out_type: output type [word-level | sentenve-level | both]
        Returns:
            w_feats: word-level features; [B,L,2*h]
            s_feats: sentence-level feature; [B,2*h]
        """
        # embedding query_labels data
        wemb = self.embedding(query_labels) # [B,L,emb_odim]
        # encoding query_labels data.
        max_len = query_labels.size(1) # == L
        # make word-wise feature
        length = query_masks.sum(1) # [B,]
        pack_wemb = nn.utils.rnn.pack_padded_sequence(wemb, length, batch_first=True, enforce_sorted=False)
        w_feats, _ = self.gru(pack_wemb)
        w_feats, max_ = nn.utils.rnn.pad_packed_sequence(w_feats, batch_first=True, total_length=max_len)
        w_feats = w_feats.contiguous() # [B,L,2*h]
        # get sentence feature
        B, L, H = w_feats.size()
        idx = (length-1).long() # 0-indexed
        idx = idx.view(B, 1, 1).expand(B, 1, H//2)
        fLSTM = w_feats[:,:,:H//2].gather(1, idx).view(B, H//2)
        bLSTM = w_feats[:,0,H//2:].view(B,H//2)
        s_feats = torch.cat([fLSTM, bLSTM], dim=1)
        # aggregae features
        w_feats = self.feature_aggregation(w_feats)
        return w_feats, s_feats


class SimpleVideoEmbeddingModule(nn.Module):
    """
    A simple Video Embedding Class
    """
    def __init__(self, cfg):
        super().__init__() # Must call super __init__()
        # get configuration
        self.cfg = cfg
        # video gru
        vid_idim = self.cfg.MODEL.VIDEO.IDIM
        vid_gru_hdim = self.cfg.MODEL.VIDEO.GRU_HDIM
        self.gru = nn.GRU(input_size=vid_idim,hidden_size=vid_gru_hdim,batch_first=True,dropout=0.5,bidirectional=True)
        # video feature aggregation module
        catted_dim = vid_idim + vid_gru_hdim*2
        emb_dim = self.cfg.MODEL.FUSION.EMB_DIM
        self.feature_aggregation = nn.Sequential(
            nn.Linear(in_features=catted_dim,out_features=emb_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, vid_feats, vid_masks):
        """
        encode video features. Utilizes GRU.
        Args:
            vid_feats: video features
            vid_masks: mask for video
        Return:
            vid_features: hidden state features of the video
        """
        length = vid_masks.sum(1).squeeze(1)
        packed_vid = nn.utils.rnn.pack_padded_sequence(vid_feats, length, batch_first=True, enforce_sorted=False)
        vid_hiddens, _ = self.gru(packed_vid)
        vid_hiddens, max_ = nn.utils.rnn.pad_packed_sequence(vid_hiddens, batch_first=True, total_length=vid_feats.shape[1])
        vid_catted = torch.cat([vid_feats,vid_hiddens],dim=2)
        vid_output = self.feature_aggregation(vid_catted)
        return vid_output


class FusionConvBNReLU(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        # get configuration
        self.cfg = cfg
        # modules
        emb_dim = self.cfg.MODEL.FUSION.EMB_DIM
        kernel_size = self.cfg.MODEL.FUSION.CONVBNRELU.KERNEL_SIZE
        padding = self.cfg.MODEL.FUSION.CONVBNRELU.PADDING
        self.module = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim,out_channels=emb_dim,kernel_size=kernel_size,padding=padding),
            nn.BatchNorm1d(num_features=emb_dim),
            nn.ReLU())

    def forward(self,feature):
        transposed_feature = torch.transpose(feature,1,2)   # to [B,D,L] format (channels first)
        convolved_feature = self.module(transposed_feature)
        return torch.transpose(convolved_feature,1,2)


class AttentionBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        # get configuration
        self.cfg = cfg
        # modules
        emb_dim = emb_dim = self.cfg.MODEL.FUSION.EMB_DIM
        num_head = self.cfg.MODEL.FUSION.NUM_HEAD
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim,num_heads=num_head)
        self.convbnrelu = FusionConvBNReLU(cfg)

    def forward(self,vid_feats,query_feats,query_masks):
        # attnetion
        key_padding_mask = query_masks < 0.1    # if true, not allowed to attend. if false, attend to it.
        attended_feature, weights = self.attention(
            query=torch.transpose(vid_feats,0,1),
            key=torch.transpose(query_feats,0,1),
            value=torch.transpose(query_feats,0,1),
            key_padding_mask=key_padding_mask,)
        attended_feature = torch.transpose(attended_feature,0,1)    # to [B,L,D] format
        # convolution
        convolved_feature = self.convbnrelu(attended_feature) + vid_feats
        return convolved_feature


class SimpleFusionModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # get configuration
        self.cfg = cfg
        # attention module
        num_layers = self.cfg.MODEL.FUSION.NUM_LAYERS
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(AttentionBlock(cfg))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, query_feats, query_masks, vid_feats, vid_masks):
        attended_vid_feats = vid_feats
        for attn_layer in self.layers:
            attended_vid_feats = attn_layer(vid_feats=attended_vid_feats, query_feats=query_feats, query_masks=query_masks)
        return attended_vid_feats


class NonLocalBlock(nn.Module):
    """
    Nonlocal block used for obtaining global feature.
    code borrowed from LGI
    """
    def __init__(self, cfg):
        super(NonLocalBlock, self).__init__()
        self.cfg = cfg
        # dims
        self.idim = self.cfg.MODEL.FUSION.EMB_DIM
        self.odim = self.cfg.MODEL.FUSION.EMB_DIM
        self.nheads = self.cfg.MODEL.NONLOCAL.NUM_HEAD

        # options
        self.use_bias = self.cfg.MODEL.NONLOCAL.USE_BIAS

        # layers
        self.c_lin = nn.Linear(self.idim, self.odim*2, bias=self.use_bias)
        self.v_lin = nn.Linear(self.idim, self.odim, bias=self.use_bias)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(self.cfg.MODEL.NONLOCAL.DROPOUT)

    def forward(self, m_feats, mask):
        """
        Inputs:
            m_feats: segment-level multimodal feature     [B,nseg,*]
            mask: mask                              [B,nseg]
        Outputs:
            updated_m: updated multimodal  feature  [B,nseg,*]
        """

        mask = mask.float()
        B, nseg = mask.size()

        # key, query, value
        m_k = self.v_lin(self.drop(m_feats)) # [B,num_seg,*]
        m_trans = self.c_lin(self.drop(m_feats))  # [B,nseg,2*]
        m_q, m_v = torch.split(m_trans, m_trans.size(2) // 2, dim=2)

        new_mq = m_q
        new_mk = m_k

        # applying multi-head attention
        w_list = []
        mk_set = torch.split(new_mk, new_mk.size(2) // self.nheads, dim=2)
        mq_set = torch.split(new_mq, new_mq.size(2) // self.nheads, dim=2)
        mv_set = torch.split(m_v, m_v.size(2) // self.nheads, dim=2)
        for i in range(self.nheads):
            mk_slice, mq_slice, mv_slice = mk_set[i], mq_set[i], mv_set[i] # [B, nseg, *]

            # compute relation matrix; [B,nseg,nseg]
            m2m = mk_slice @ mq_slice.transpose(1,2) / ((self.odim // self.nheads) ** 0.5)
            m2m = m2m.masked_fill(mask.unsqueeze(1).eq(0), -1e9) # [B,nseg,nseg]
            m2m_w = torch.nn.functional.softmax(m2m, dim=2) # [B,nseg,nseg]
            w_list.append(m2m_w)

            # compute relation vector for each segment
            r = m2m_w @ mv_slice if (i==0) else torch.cat((r, m2m_w @ mv_slice), dim=2)

        updated_m =m_feats + r
        return updated_m


class AttentivePooling(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        super(AttentivePooling, self).__init__()
        self.att_n = 1
        self.feat_dim = self.cfg.MODEL.FUSION.EMB_DIM
        self.att_hid_dim = self.cfg.MODEL.FUSION.EMB_DIM // 2
        self.use_embedding = True

        self.feat2att = nn.Linear(self.feat_dim, self.att_hid_dim, bias=False)
        self.to_alpha = nn.Linear(self.att_hid_dim, self.att_n, bias=False)
        if self.use_embedding:
            edim = self.cfg.MODEL.FUSION.EMB_DIM
            self.fc = nn.Linear(self.feat_dim, edim)

    def forward(self, feats, f_masks=None):
        """ 
        Compute attention weights and attended feature (weighted sum)
        Args:
            feats: features where attention weights are computed; [B, A, D]
            f_masks: mask for effective features; [B, A]
        """
        # check inputs
        assert len(feats.size()) == 3 or len(feats.size()) == 4
        assert f_masks is None or len(f_masks.size()) == 2

        # dealing with dimension 4
        if len(feats.size()) == 4:
            B, W, H, D = feats.size()
            feats = feats.view(B, W*H, D)

        # embedding feature vectors
        attn_f = self.feat2att(feats)   # [B,A,hdim]

        # compute attention weights
        dot = torch.tanh(attn_f)        # [B,A,hdim]
        alpha = self.to_alpha(dot)      # [B,A,att_n]
        if f_masks is not None:
            alpha = alpha.masked_fill(f_masks.float().unsqueeze(2).eq(0), -1e9)
        attw =  torch.nn.functional.softmax(alpha.transpose(1,2), dim=2) # [B,att_n,A]

        att_feats = attw @ feats # [B,att_n,D]
        att_feats = att_feats.squeeze(1)
        attw = attw.squeeze(1)
        if self.use_embedding: att_feats = self.fc(att_feats)

        return att_feats, attw


class AttentionLocRegressor(nn.Module):
    def __init__(self, cfg):
        super(AttentionLocRegressor, self).__init__()
        self.cfg = cfg
        self.tatt = AttentivePooling(self.cfg)
        # Regression layer
        idim = self.cfg.MODEL.FUSION.EMB_DIM
        gdim = self.cfg.MODEL.FUSION.EMB_DIM
        nn_list = [ nn.Linear(idim, gdim), nn.ReLU(), nn.Linear(gdim, 2), nn.ReLU()]
        self.MLP_reg = nn.Sequential(*nn_list)

    def forward(self, semantic_aware_seg_feats, masks):
        # perform Eq. (13) and (14)
        summarized_vfeat, att_w = self.tatt(semantic_aware_seg_feats, masks)
        # perform Eq. (15)
        loc = self.MLP_reg(summarized_vfeat) # loc = [t^s, t^e]
        return loc, att_w


class SimpleModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.query_encoder = SimpleSentenceEmbeddingModule(cfg)
        self.video_encoder = SimpleVideoEmbeddingModule(cfg)
        self.fusor = SimpleFusionModule(cfg)
        self.n_non_local = self.cfg.MODEL.NONLOCAL.NUM_LAYERS
        self.non_locals = nn.ModuleList([NonLocalBlock(cfg) for _ in range(self.n_non_local)])
        self.loc_regressor = AttentionLocRegressor(cfg)

    def forward(self,inputs):
        # encode query
        query_labels = inputs['query_labels']
        query_masks = inputs['query_masks']
        encoded_query, encoded_sentence = self.query_encoder(query_labels, query_masks)
        # encode video
        vid_feats = inputs['video_feats']
        vid_masks = inputs['video_masks']
        encoded_video = self.video_encoder(vid_feats,vid_masks)
        attended_vid = self.fusor(encoded_query, query_masks, encoded_video, vid_masks)
        global_vid = attended_vid
        for non_local_layer in self.non_locals:
            global_vid = non_local_layer(global_vid,vid_masks.squeeze(2))
        loc,attn_weight = self.loc_regressor(global_vid,vid_masks.squeeze(2))
        return {"timestamps": loc,
                "attention_weights": attn_weight}




