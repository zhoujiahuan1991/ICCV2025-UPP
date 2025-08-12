import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
import ipdb
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from .modules import square_distance, index_points
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import emd 
import pytorch3d.ops
from .Point_MAE_pretask_dev import Block, RectifyPrompter, TransformerDecoder

def propagate(xyz1, xyz2, points1, points2, de_neighbors=64, dist_e=1e-8):
    """
    Input:
        xyz1: input points position data, [B, N, 3]
        xyz2: sampled input points position data, [B, S, 3]
        points1: input points data, [B, N, D']
        points2: input points data, [B, S, D'']
    Return:
        new_points: upsampled points data, [B, N, D''']
    """

    B, N, C = xyz1.shape
    _, S, _ = xyz2.shape

    dists = square_distance(xyz1, xyz2)
    dists, idx = dists.sort(dim=-1)
    dists, idx = dists[:, :, :de_neighbors], idx[:, :, :de_neighbors]  # [B, N, S]

    dist_recip = 1.0 / (dists + dist_e)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    weight = weight.view(B, N, -1, 1)
    interpolated_points = torch.sum(index_points(points2, idx) * weight, dim=2)#B, N, 6, C->B,N,C

    new_points = points1+0.3*interpolated_points # B,N,C

    return new_points


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, require_index=False, gather_idx=False):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center,center_idx = misc.fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        if not gather_idx:
            idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
            idx = idx + idx_base
            idx = idx.view(-1)

            center_idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1) * num_points
            center_idx = center_idx + center_idx_base
            center_idx = center_idx.view(-1)

            neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
            neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        else:
            neighborhood = torch.gather(xyz, 1, idx.reshape(batch_size, -1, 1).expand(-1,-1,3))
            neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
            center_idx = center_idx.long()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        if require_index:
            return neighborhood, center, idx, center_idx
        else:
            return neighborhood, center

class PositionalEmbedding(nn.Module):
    def __init__(self, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)#[2^0,2^1,...,2^(n-1)]
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, f)

        Outputs:
            out: (B, 2*f*N_freqs+f)
        """
        out = [x]
        for freq in self.freq_bands:#[2^0,2^1,...,2^(n-1)]
            for func in self.funcs:
                out += [func(freq*x)]
        return torch.cat(out, -1)
    
class Adapter(nn.Module):
    def __init__(self,
                 embed_dims,
                 reduction_dims,
                 drop_rate_adapter=0.1
                ):
        super(Adapter, self).__init__()
    
        self.embed_dims = embed_dims
        self.super_reductuion_dim = reduction_dims
        
        self.dropout = nn.Dropout(p=drop_rate_adapter)

        if self.super_reductuion_dim > 0:
            self.layer_norm = nn.LayerNorm(self.embed_dims)
            self.scale = nn.Linear(self.embed_dims, 1)
            self.ln1 = nn.Linear(self.embed_dims, self.super_reductuion_dim)
            self.activate = nn.GELU()
            self.ln2 = nn.Linear(self.super_reductuion_dim, self.embed_dims)

            self.init_weights()
        
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                nn.init.normal_(m.bias, std=1e-6)


        self.apply(_init_weights)


    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        
        self.sampled_weight_0 = self.ln1.weight[:self.sample_embed_dim,:]
        self.sampled_bias_0 =  self.ln1.bias[:self.sample_embed_dim]

        self.sampled_weight_1 = self.ln2.weight[:, :self.sample_embed_dim]
        self.sampled_bias_1 =  self.ln2.bias


    def forward(self, x):
        x = self.layer_norm(x)
        # scale = F.relu(self.scale(x))
        scale = 0.7
        out = self.ln1(x)
        out = self.activate(out)
        out = self.dropout(out)
        out = self.ln2(out)

        return out*scale

    def calc_sampled_param_num(self):
    
        return  self.sampled_weight_0.numel() + self.sampled_bias_0.numel() + self.sampled_weight_1.numel() + self.sampled_bias_1.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
             total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        return total_flops


class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 384 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 384
        return feature_global.reshape(bs, g, self.encoder_channel) # [B, G, 384]


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., **kwargs):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                block_idx=i,
                **kwargs
                )
            for i in range(depth)])

    def forward(self, x, pos, **kwargs):
        block_depth=len(self.blocks)
        if kwargs.get('pretask_depth') and kwargs['path']=='pretask':
            block_depth = kwargs.get('pretask_depth')
        elif kwargs.get('rectify_depth') and kwargs['path']=='rectify':
            block_depth = kwargs.get('rectify_depth')
        for idx, block in enumerate(self.blocks):
            if idx == block_depth:
                break
            x = block(x + pos, **kwargs)
        return x

class PointNetSetAbstraction(nn.Module):
    def __init__(self, num_group, group_size, in_channel, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.group_divider = Group(num_group, group_size)
        self.num_group = num_group
        self.group_size = group_size
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B, N, C = xyz.shape
        neighborhood, center, idx, center_idx = self.group_divider(xyz.float(), require_index=True) #[B, G, n, 3]
        new_xyz = center.reshape((B, self.num_group, -1))
        new_points = points.reshape((B*N, -1))[idx].reshape((B, self.num_group, self.group_size, -1))

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, group_size, npoint/num_group]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0,2,1)
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp, interpolate_neighbors=16):
        super(PointNetFeaturePropagation, self).__init__()
        self.interpolate_neighbors = interpolate_neighbors
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, C] 64
            xyz2: sampled input points position data, [B, S, C] 1024
            points1: input points data, [B, N, D]
            points2: input points data, [B, S, D]
        Return:
            new_points: upsampled points data, [B, N, D']
        """
        
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :self.interpolate_neighbors], idx[:, :, :self.interpolate_neighbors]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-4)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, self.interpolate_neighbors, 1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0,2,1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = new_points.permute(0,2,1)
        return new_points


@MODELS.register_module()
class Point_MAE_unify(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE_unify] ', logger ='Point_MAE_unify')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.mask_ratio = config.transformer_config.mask_ratio
        self.depth = config.transformer_config.depth 
        self.num_heads = config.transformer_config.num_heads 
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.vis_num = 64 - int(self.mask_ratio*64)
        self.vis_short = 16
        self.cls_dim = config.cls_dim
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
            **self.config.prompter_config
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        # self.MAE_encoder = MaskTransformer(config)
        self.shape_pred = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim//2),
            nn.GELU(),
            nn.Linear(self.trans_dim//2, self.vis_short)
        )

        self.coarse_pred = nn.Sequential(
            nn.Linear(self.vis_short*self.vis_num, self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, 3 * int(64-self.vis_num))
        )
        self.predict_token_generator = nn.Sequential(
            nn.Linear(self.trans_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
            pretask_adapter=True,
            pretask_depth=4
        )

        print_log(f'divide point cloud into {self.num_group} x {self.group_size} points ...', logger ='Point_MAE_unify')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # prediction head
        self.dense_pred = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        self.rectify_prompter = RectifyPrompter(in_channels=3, out_channels=3, hidden_dimesion=self.trans_dim, embedding_level=4, num_group = 32, group_size = 16, top_center_dim=12)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )
        
        for layer in self.cls_head_finetune:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=5.0**0.5)

        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100
    
    def load_model_from_ckpt(self, bert_ckpt_path, logger = None):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", "").replace('_block',''): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger=logger)
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger=logger
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger=logger)
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger=logger
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger=logger)
        else:
            print_log('Training from scratch!!!', logger=logger)
            self.apply(self._init_weights)

    def forward(self, pts, label=None, completion_prompt=False, denoise=False, point_num=1024, **kwargs):
        B, _, _ = pts.shape

        if denoise:
            vis_center_grouper = Group(num_group=self.vis_num, group_size=16)
            vis_neighborhood, vis_center = vis_center_grouper(pts, require_index=False)
            vis_group_input_tokens = self.encoder(vis_neighborhood)
            # transformer
            pos = self.pos_embed(vis_center)
            vis_group_input_tokens = self.blocks(vis_group_input_tokens, pos, 
                                                 path='rectify', 
                                                 rectify_adapter=True, 
                                                 rectify_prompts=True, 
                                                 rectify_depth=self.config.prompter_config['rectify_depth'] 
                                                 )
            B, P, _ = pts.shape
            pred_vector = self.rectify_prompter(pts, vis_center, vis_group_input_tokens, require_shape_feature=False)
            pred_noise_score = torch.norm(pred_vector, p=2, dim=-1, keepdim=False)
            noise_idx = torch.argsort(pred_noise_score, dim=1, descending=True)
            # # recify
            pts = pts + pred_vector*0.2
            pts = torch.gather(pts, 1, noise_idx[:,-int(point_num*0.95):,None].expand(-1, -1, 3))
            visualize = False
            if visualize == True:
                import os
                task = 'completion'
                os.makedirs(f'./visualization/{task}', exist_ok=True)
                np.save(f'./visualization/{task}/pred_noise_score-{kwargs["batch_idx"]}', pred_noise_score.detach().cpu().numpy())
                np.save(f'./visualization/{task}/noise_idx-{kwargs["batch_idx"]}', noise_idx.detach().cpu().numpy())
                np.save(f'./visualization/{task}/denoise-pts-{kwargs["batch_idx"]}', pts.detach().cpu().numpy())
                # np.save(f'./visualization/{task}/rebuild_points-{kwargs["batch_idx"]}', rebuild_points.detach().cpu().numpy())

            recall = torch.mean(torch.sum(noise_idx[:,:-int(point_num*0.95)].detach()>point_num,dim=-1)/(P-point_num+0.1))
        
        if completion_prompt:
            vis_center_grouper = Group(num_group=self.vis_num, group_size=16)
            vis_neighborhood, vis_center = vis_center_grouper(pts, require_index=False)
            vis_group_input_tokens = self.encoder(vis_neighborhood)
    
            batch_size, seq_len, C = vis_group_input_tokens.size()
            x_vis = vis_group_input_tokens.reshape(batch_size, -1, self.trans_dim)
            # add pos embedding
            pos = self.pos_embed(vis_center)
            # transformer
            x_vis = self.blocks(x_vis, pos, 
                                path='pretask', 
                                pretask_adapter=True, 
                                pretask_prompts=True, 
                                pretask_depth=self.config.prompter_config['pretask_depth']
                                )
            x_vis = self.norm(x_vis)

            pos_emd_vis = self.decoder_pos_embed(vis_center).reshape(B, -1, self.trans_dim)
            # shape completion prompter
            shape_feature = self.shape_pred(x_vis).reshape(B, self.vis_short*self.vis_num)
            predict_center = self.coarse_pred(shape_feature).reshape(B, int(64-self.vis_num), 3)
            predict_token = self.predict_token_generator(x_vis) # +predict_center_pos
            pos_emd_mask = self.decoder_pos_embed(predict_center).reshape(B, -1, self.trans_dim) #.detach()
            _,N,_ = pos_emd_mask.shape
            mask_token = self.mask_token.expand(B, N, -1)
            mask_token = propagate(predict_center, vis_center, mask_token, predict_token, de_neighbors=6)
            x_full = torch.cat([x_vis, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
            x_rec = self.MAE_decoder(x_full, pos_full, N, pretask_adapter=True, path='pretask')

            B, M, C = x_rec.shape
            relative_rebuild_points = self.dense_pred(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B, M, -1, 3)  # B M 1024
            rebuild_points = (relative_rebuild_points + predict_center.unsqueeze(-2)).reshape(B, -1, 3)

            sample_rebuild_points, _ = misc.fps(rebuild_points, point_num//4)
            pts = torch.cat([pts, sample_rebuild_points],dim=1).contiguous()
            if pts.shape[1] > point_num:
                pts = misc.fps(pts, point_num)[0]
            
            visualize = False
            if visualize == True:
                import os
                task = 'completion'
                os.makedirs(f'./visualization/{task}', exist_ok=True)
                np.save(f'./visualization/{task}/predict_coarse_center-{kwargs["batch_idx"]}', predict_center.detach().cpu().numpy())
                np.save(f'./visualization/{task}/pts-{kwargs["batch_idx"]}', pts.detach().cpu().numpy())
                np.save(f'./visualization/{task}/rebuild_points-{kwargs["batch_idx"]}', rebuild_points.detach().cpu().numpy())

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  #  B G C
        
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)
        # token_pos = self.pos_embed(self.prompt_cor)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        propagation_dict = {}
        if self.config.prompt_propagation_after:
            # level 2 centers for propagation
            level2_grouper = Group(num_group=self.num_group//2, group_size=8)
            _, center2, center1_idx, center2_idx = level2_grouper(center, require_index=True, gather_idx=self.config.gather_idx)
            propagation_dict = {
                'center1': center,
                'center1_idx': center1_idx,
                'center2': center2,
                'center2_idx': center2_idx,
                'gather_idx': self.config.gather_idx,
                'prompt_propagation_after': self.config.prompt_propagation_after
            }

        # transformer
        x = self.blocks(x, pos, 
                        path='downstream', 
                        downstream_adapter=True, 
                        downstream_prompts=True, 
                        classification=True, 
                        **propagation_dict)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret