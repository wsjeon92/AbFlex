import torch
import torch.nn as nn
from egnn_pytorch import EGNN
import numpy as np

vocab = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', '*']

class AbFlex(nn.Module):
    def __init__(self, dim, res_dim, n_layer, num_nearest_neighbors =0, valid_rad=16.):
        super(AbFlex, self).__init__()
        self.dim = dim
        self.num_nearest_neighbors = num_nearest_neighbors 
        self.valid_radius = valid_rad
        for i in range(n_layer):
            exec(f'self.layer{i+1}=EGNN(dim = self.dim, num_nearest_neighbors = self.num_nearest_neighbors, m_dim=self.dim,\
                                       norm_feats =True, norm_coors=True, valid_radius=self.valid_radius,\
                                       m_pool_method = "sum", coor_weights_clamp_value = 2, update_feats=True)') 
            
        self.num_token = 21
        self.seq_emb = nn.Embedding(self.num_token, res_dim)
        self.chain_emb = nn.Embedding(2, 2)
        
    def forward(self, features, input_mask, coords):
        input_seq, input_chain = features
        input_seq = self.seq_emb(torch.tensor([vocab.index(i) for i in input_seq]).cuda()).unsqueeze(0)

        input_chain = self.chain_emb(torch.from_numpy(input_chain).unsqueeze(0).int().cuda())
        input_feat = torch.cat([input_seq, input_chain], dim=-1)

        input_mask = input_mask.float().cuda().requires_grad_(False)

        try:
            input_coor = torch.from_numpy(np.expand_dims(coords,0)).cuda().float()
        except:
            input_coor = coords

    
        feats1, coors1 = self.layer1(input_feat, input_coor)
        coors1 = input_mask*coors1+(1-input_mask)*input_coor

        feats2, coors2  = self.layer2(feats1, coors1)
        coors2 = input_mask*coors2+(1-input_mask)*input_coor

        feats3, coors3  = self.layer3(feats2, coors2)
        coors3 = input_mask*coors3+(1-input_mask)*input_coor

        feats4, coors4  = self.layer4(feats3, coors3)
        coors4 = input_mask*coors4+(1-input_mask)*input_coor

        feats5, coors5 = self.layer5(feats4, coors4)
        coors5 = input_mask*coors5+(1-input_mask)*input_coor

        feats6, coors6  = self.layer6(feats5, coors5)
        coors6 = input_mask*coors6+(1-input_mask)*input_coor

        feats7, coors7  = self.layer7(feats6, coors6)
        coors7 = input_mask*coors7+(1-input_mask)*input_coor

        feats8, coors8  = self.layer8(feats7, coors7)
        coors8 = input_mask*coors8+(1-input_mask)*input_coor

        feats9, coors9  = self.layer9(feats8, coors8)
        coors9 = input_mask*coors9+(1-input_mask)*input_coor

        feats10, coors10  = self.layer10(feats9, coors9)
        coors10 = input_mask*coors10+(1-input_mask)*input_coor

        feats11, coors11  = self.layer11(feats10, coors10)
        coors11 = input_mask*coors11+(1-input_mask)*input_coor

        feats12, coors12  = self.layer12(feats11, coors11)
        coors12 = input_mask*coors12+(1-input_mask)*input_coor

        feats13, coors13  = self.layer13(feats12, coors12)
        coors13 = input_mask*coors13+(1-input_mask)*input_coor

        feats14, coors14  = self.layer14(feats13, coors13)
        coors14 = input_mask*coors14+(1-input_mask)*input_coor

        feats15, coors15  = self.layer15(feats14, coors14)
        coors15 = input_mask*coors15+(1-input_mask)*input_coor

        feats16, coors16  = self.layer16(feats15, coors15)
        coors16 = input_mask*coors16+(1-input_mask)*input_coor

        return feats16, coors16


class pred_seq(nn.Module):
    def __init__(self, n_in, n_out):
        super(pred_seq, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear1 = nn.Linear(n_in, n_in//2)
        self.linear2 = nn.Linear(n_in//2, n_out)
        self.gelu = nn.GELU()
    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))
        
            
