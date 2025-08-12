import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from utils import misc
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
import numpy as np
from .build import MODELS
from .modules import square_distance, index_points

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data

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
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)#[2^0,2^1,...,2^(n)]
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

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp, interpolate_neighbors=6):
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

class Encoder(nn.Module):
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
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


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
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        # return x
        return feature_list

# segmentation finetune model
@MODELS.register_module()
class PointTransformer_seg(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims


        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=True),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv1d(64, 128, kernel_size=1, bias=True),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(0.2),
                                   )
        self.positional_embedding = PositionalEmbedding(12)
        # self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 75, mlp=[384*4, 1024], interpolate_neighbors=5)
        self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3, mlp=[384*4, 1024], interpolate_neighbors=5)
        # self.group_divider1 = Group(num_group=self.num_group*3, group_size=self.group_size//2)
        # self.propagation_1 = PointNetFeaturePropagation(in_channel=384 + 75, mlp=[384, 384], interpolate_neighbors=5)
        

        self.seg_head = nn.Sequential(
            nn.Conv1d(1024+128+384*6, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, self.cls_dim, 1)
        )

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        self.get_loss = get_loss().cuda()


    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, cls_label, label_points=None):
        if label_points is not None:
            B, N, _ = label_points.shape
        else:
            B, N, _ = pts.shape
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)

        group_input_tokens = self.encoder(neighborhood)  # B G N

        pos = self.pos_embed(center)
        # final input
        x = group_input_tokens
        # cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        # cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # transformer
        feature_list = self.blocks(x, pos)
        feature_list = [self.norm(x).contiguous() for x in feature_list]
        # x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        # pos = torch.cat((cls_pos, pos), dim=1)

        # x = self.blocks(x, pos)
        # cls_tokens = x[:, :1, :]
        # x = x[:, 1:, :]
        x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=-1) #1152
        x_max = torch.max(x,1)[0]
        x_avg = torch.mean(x,1)
        x_max_feature = x_max.view(B, -1).unsqueeze(-2).repeat(1, N, 1)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-2).repeat(1, N, 1)
        cls_label_one_hot = cls_label.view(B, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).transpose(-1,-2).repeat(1, N, 1) # 128
        x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), -1) #1152*2 + 64
        torch.cuda.empty_cache()

        # _, center1 = self.group_divider1(pts)
        # f_level_1 = self.propagation_1(center1, center, self.positional_embedding(center1), x)  
        # if label_points is not None:
        #     f_level_0 = self.propagation_0(label_points, center1, self.positional_embedding(label_points), f_level_1)    
        # else:
        #     f_level_0 = self.propagation_0(pts, center1, self.positional_embedding(pts), f_level_1)

        if label_points is not None:
            # f_level_0 = self.propagation_0(label_points, center, self.positional_embedding(label_points), x)
            f_level_0 = self.propagation_0(label_points, center, label_points, x)    
        else:
            # f_level_0 = self.propagation_0(pts, center, self.positional_embedding(pts), x)
            f_level_0 = self.propagation_0(pts, center, pts, x)

        x = torch.cat((f_level_0,x_global_feature), -1)
        x = self.seg_head(x.transpose(-1,-2))
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss