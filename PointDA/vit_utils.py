import sys, os, pdb
# from tkinter import X
import torch
import math
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models import create_model
from collections import OrderedDict
from timm.models.vision_transformer import VisionTransformer, _create_vision_transformer, _init_vit_weights, checkpoint_filter_fn, default_cfgs
from timm.models.layers import PatchEmbed, DropPath, Mlp, trunc_normal_, lecun_normal_
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
# from scripts.utils_model import vit_stage_layer_mapping
from PointDA.model_utils import Pointnet_Encoder, Relative_Encoder, Dgcnn_Encoder
from model_utils import transform_net, conv_2d, get_graph_feature, fc_layer, classifier, Group, Encoder, TransformerEncoder, PointNetFeaturePropagation
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from Models import DGCNN_Propagation, RegionReconstruction,fps

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class ViT(VisionTransformer):
    def __init__(self, config, #img_size=224, patch_size=16, in_chans=3, #num_classes=1000, embed_dim=768, depth=12,
                 #num_heads=12, 
                 mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0.5, attn_drop_rate=0.5, drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        super().__init__()
        self.config = config
        num_classes = config.cls_dim #num_classes
        self.num_classes = num_classes
        embed_dim = config.trans_dim
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        
        depth = config.depth 
        drop_path_rate = config.drop_path_rate
        num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group  

        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        self.encoder_dims =  config.encoder_dims
        self.encoder_type = config.encoder_type
        if self.encoder_type=='Encoder':
            self.encoder = Encoder(encoder_channel = self.encoder_dims,use_relative=True)
        elif self.encoder_type=='Pointnet_Encoder':
            self.encoder = Pointnet_Encoder(config, self.encoder_dims)
        elif self.encoder_type=='Dgcnn_Encoder':
            self.encoder = Dgcnn_Encoder(config, self.encoder_dims)
        elif self.encoder_type=='Relative_Encoder':
            self.encoder = Relative_Encoder(encoder_channel=self.encoder_dims)
        print('------------------------our encoder type is', self.encoder_type)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims,  embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embedd = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, embed_dim)) 
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()



    
        self.build_loss_func()

        ### for deformation reconstruction
        self.propagation_2 = PointNetFeaturePropagation(in_channel= self.embed_dim + 3, mlp = [self.embed_dim * 4, self.embed_dim])
        self.propagation_1= PointNetFeaturePropagation(in_channel= self.embed_dim + 3, mlp = [self.embed_dim * 4, self.embed_dim])
        self.propagation_0 = PointNetFeaturePropagation(in_channel= self.embed_dim + 3, mlp = [self.embed_dim * 4, self.embed_dim])
        self.dgcnn_pro_1 = DGCNN_Propagation(k = 4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k = 4)

        self.DefRec = RegionReconstruction(self.config,self.embed_dim * 2 + self.embed_dim)
        
        self.init_weights(weight_init)



    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        # trunc_normal_(self.pos_embedd, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    # def forward_features(self, feat=[]):
    #     xy_embed_list = []
    #     xy, xy_embed, nns = feat

    #     _, xy_embed, _, xy_embed_list = self.blocks([xy, xy_embed, nns, xy_embed_list]) # [1, 145, 384] -> [1, 145, 384]
    #     xy_embed = self.norm(xy_embed)
    #     res = sum(xy_embed_list) / len(xy_embed_list)
    #     return res
        # return xy_embed

    def forward(self, pts, activate_DefRec=False):
        batch_size = pts.size(0)
        num_points = pts.size(1)
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # print('neighborhood', neighborhood.shape)
        # encoder the input cloud blocks
        if self.encoder_type=='Relative_Encoder':
            group_input_tokens = self.encoder(neighborhood, center)
        else:
            group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # add pos embedding
        pos = self.pos_embedd(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = self.pos_drop(x + pos)
        x=self.blocks(x)
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        # x = self.blocks(x)
        x = self.norm(x)

        concat_f = torch.cat([x[:,0], x[:, 1:].max(1)[0]], dim = -1)

        if activate_DefRec:
            feature_list = [self.norm(t)[:,1:].transpose(-1, -2).contiguous() for t in feature_list] #32 384 64
            center_level_0 = pts.transpose(-1, -2).contiguous()
            f_level_0 = center_level_0

            center_level_1 = fps(pts, 512).transpose(-1, -2).contiguous()            # 512 # B 3 N0
            f_level_1 = center_level_1
            center_level_2 = fps(pts, 256).transpose(-1, -2).contiguous()            # 256 # B 3 N1
            f_level_2 = center_level_2
            center_level_3 = center.transpose(-1, -2).contiguous()                   # 128 # B 3 G
            # print('center_level_1,center_level_2,center_level_3',center_level_1.shape,center_level_2.shape,center_level_3.shape)
            
            # init the feature by 3nn propagation
            f_level_3 = feature_list[2]
            f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, feature_list[1]) #B 384 N1-256 
            f_level_1 = self.propagation_1(center_level_1, center_level_3, f_level_1, feature_list[0]) #B 384 N2-512
            # print('f_level_1,f_level_2,f_level_3',f_level_1.shape,f_level_2.shape,f_level_3.shape)
            # bottom up
            f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2) #B 384 N1-256 
            f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1) #B 384 N2-512 
            f_level_0 =  self.propagation_0(center_level_0, center_level_1, f_level_0, f_level_1) #get point-wise feature
            # print('f_level_0,f_level_1,f_level_2',f_level_0.shape,f_level_1.shape,f_level_2.shape)
            # print('get point-wise feature', f_level_0.shape)
            
            DefRec_input = torch.cat((f_level_0, concat_f.unsqueeze(2).repeat(1, 1, num_points)), dim=1)
            ret = self.DefRec(DefRec_input)
            return ret


        ret = self.head(concat_f)
        return ret


        # x = self.forward_features(feat=feat)
        # return x
    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
    
    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]


        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print('missing_keys', logger = 'Transformer')
            print(
                get_missing_parameters_message(incompatible.missing_keys),
            )
        if incompatible.unexpected_keys:
            print('unexpected_keys', logger = 'Transformer')
            print(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
            )

        print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')