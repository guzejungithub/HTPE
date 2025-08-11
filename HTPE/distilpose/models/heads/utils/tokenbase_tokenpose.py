from functools import partial
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from timm.models.layers.weight_init import trunc_normal_
import math
from easydict import EasyDict
import copy
import numpy as np
MIN_NUM_PATCHES = 16
BN_MOMENTUM = 0.1

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        tmp_x,tok_attn,attn = self.fn(x, **kwargs)
        return tmp_x + x, tok_attn, attn
        #return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn, fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim*fusion_factor)
        self.fn = fn
    def forward(self, x, **kwargs):
        x,tok_attn,attn = self.fn(self.norm(x), **kwargs)
        return x,tok_attn,attn
        #return self.fn(self.norm(x), **kwargs)


class Residual_(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x,freqs_cis, **kwargs):
        tmp_x,tok_attn,attn = self.fn(x,freqs_cis, **kwargs)
        return tmp_x + x, tok_attn, attn
        #return self.fn(x, **kwargs) + x

class PreNorm_(nn.Module):
    def __init__(self, dim, fn, fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim*fusion_factor)
        self.fn = fn
    def forward(self, x,freqs_cis, **kwargs):
        x,tok_attn,attn = self.fn(self.norm(x),freqs_cis, **kwargs)
        return x,tok_attn,attn


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x),None,None

class Attention(nn.Module):
    """
    Self-attention Module
    """
    def __init__(self, dim, heads = 8, dropout = 0., num_keypoints=None, scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim//heads) ** -0.5 if scale_with_head else  dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.num_keypoints = num_keypoints

    def forward(self, x, freqs_cis , mask = None, return_tok=False):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = qkv.split(qkv.size(-1) // 3, dim=-1)
        
        q_temp, k_temp, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        # k = rearrange(k, 'b n (h d) -> b h n d', h = h)
        # v = rearrange(v, 'b n (h d) -> b h n d', h = h)
        q = q_temp.clone() 
        k = k_temp.clone()


        q, k = apply_rotary_emb(q_temp, k_temp, freqs_cis=freqs_cis)
        # q, k = apply_rotary_emb(q_temp, k_temp, freqs_cis=freqs_cis)

        # dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        dots = (q* self.scale) @ k.transpose(-2, -1)
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        if return_tok:
            # N = HW + J
            J = self.num_keypoints
            tok_attn = attn[:, :, :J, J:]                        # (B, H, J, HW)
            tok_attn = tok_attn.sum(1) / self.heads              # (B, J, HW), average all head 
            return out, tok_attn, attn
        else:
            return out,None,attn

class Transformer(nn.Module):
    """
    Vision-transformer Module
    """
    def __init__(self, dim, depth, heads, mlp_dim, dropout, num_keypoints=None, all_attn=False, scale_with_head=False, pruning_loc=[3,6,9]):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual_(PreNorm_(dim, Attention(dim, heads = heads, dropout = dropout,num_keypoints = num_keypoints, 
                                 scale_with_head=scale_with_head))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
        self.pruning_loc = pruning_loc

    def forward(self, x,freqs_cis,learnable_absolute, mask = None,pos=None, prune=False, keep_ratio=0.7, pass_1_pos=None):
        if len(x.shape) == 2:
            _, C = x.shape
            B = 1
        else:
            B, _, C = x.shape
            
        pos = pos.expand(B, -1, -1)
        attn_res = []
        
        # use the positional embedding from pass 1 in training 
        if pass_1_pos is not None:
            pos = pass_1_pos 
            
        for idx,(attn, ff) in enumerate(self.layers):
            # >>>>>>>>>> add patch embedding >>>>>>>>>>
            if idx>0 and self.all_attn:
                pos  = pos * learnable_absolute[idx]
                x[:,self.num_keypoints:] += pos # adding embedding within transformer

            # >>>>>>>>>> Attention layer >>>>>>>>>>
            if idx in self.pruning_loc and prune and keep_ratio < 1:
                x, tok_attn, x_attn = attn(x,freqs_cis[idx], mask=mask, return_tok=True) 
                joint_tok_copy = x[:, :self.num_keypoints]                           # (B, J, C)     save token

                B, _, num_patches = tok_attn.shape                          # num_patch = HW
                num_keep_node = math.ceil( num_patches * keep_ratio )       # K = HW * ratio

                # attentive token
                human_attn = tok_attn.sum(1)           # (B, HW)
                attentive_idx = human_attn.topk(num_keep_node, dim=1)[1]            # (B, K)        without gradient
                attentive_idx = attentive_idx.unsqueeze(-1).expand(-1, -1, C)       # (B, K, C)
                x_attentive = torch.gather(x[:, self.num_keypoints:], dim=1, index=attentive_idx)       # (B, N, C) -> (B, K, C)
                pos = torch.gather(pos, dim=1, index=attentive_idx)                                     # (B, N, C) -> (B, K, C)
                
                x = torch.cat([joint_tok_copy, x_attentive], dim=1)
                x,_,_ = ff(x)
                attn_res.append(x_attn)
            else:
                x,_,x_attn = attn(x,freqs_cis[idx], mask = mask)
                x,_,_ = ff(x)
                attn_res.append(x_attn)

        return x,attn_res,pos
    
def polar_(abs, angle):
    real = abs * torch.cos(angle)  # 计算实部
    imag = abs * torch.sin(angle)  # 计算虚部
    return torch.complex(real, imag)  # 组合成复数张量


def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor,freqs_keypoint: torch.Tensor,freqs_keypoint_y: torch.Tensor, t_keypoint: torch.Tensor,t_keypoint_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    M = t_keypoint.shape[0]
    depth = freqs.shape[1]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_keypoint_ =  (t_keypoint.unsqueeze(-1) @ freqs_keypoint[0].unsqueeze(-2) + freqs_keypoint[1].unsqueeze(-2).repeat(1,M,1)).view(depth, M, num_heads, -1).permute(0, 2, 1, 3)
        freqs_keypoint_y =  (t_keypoint_y.unsqueeze(-1) @ freqs_keypoint_y[0].unsqueeze(-2) + freqs_keypoint_y[1].unsqueeze(-2).repeat(1,M,1)).view(depth, M, num_heads, -1).permute(0, 2, 1, 3)  
        freqs_cis = polar_(torch.ones_like(freqs_x), freqs_x + freqs_y)
        freqs_cis_keypoint = polar_(torch.ones_like(freqs_keypoint_), freqs_keypoint_+freqs_keypoint_y)
        # freqs_cis_keypoint_y = polar_(torch.ones_like(freqs_keypoint_y), freqs_keypoint_y)
        freqs_cis_result =  torch.cat((freqs_cis_keypoint, freqs_cis), dim=2)         
    return freqs_cis_result

def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * math.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(math.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(math.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def init_t_xy(end_x: int, end_y: int):

    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.floor(torch.div(t, end_x)).float()
    return t_x, t_y   

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        
        
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xk_ = xk_.clone() * freqs_cis
    xq_ = xq_.clone() * freqs_cis
    xq_out = torch.view_as_real(xq_).flatten(3)
    xk_out = torch.view_as_real(xk_).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)





class Transformer_sd(nn.Module):
    """
    Vision-transformer Module
    """
    def __init__(self, dim, depth, heads, mlp_dim, dropout, num_keypoints=None, all_attn=False, scale_with_head=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual_(PreNorm_(dim, Attention(dim, heads = heads, dropout = dropout, 
                                 scale_with_head=scale_with_head))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, freqs_cis,mask = None,pos=None):
        res = []
        attn_res = []
        for idx,(attn, ff) in enumerate(self.layers):
            if idx>0 and self.all_attn:
                x[:,self.num_keypoints:] += pos

            #x = attn(x, mask = mask)
            x,_,x_attn = attn(x, freqs_cis[idx],mask = mask)
            x,_,_ = ff(x)
            res.append(x)
            attn_res.append(x_attn)
        
        return res,attn_res
    
class TokenPose_PPT_base(nn.Module):
    def __init__(self, *, feature_size, patch_size, num_keypoints, dim, depth,  heads, 
                 mlp_ratio, apply_init=False, apply_multi=True, heatmap_size=[64,48], 
                 channels = 3, dropout = 0., emb_dropout = 0., pos_embedding_type="learnable"):
        """
        TokenPose base head, heatmap-based prediction head.
        """
        super().__init__()
        assert isinstance(feature_size,list) and isinstance(patch_size,list), \
               'image_size and patch_size should be list'
        assert feature_size[0] % patch_size[0] == 0 and \
               feature_size[1] % patch_size[1] == 0, \
               'Image dimensions must be divisible by the patch size.'
        num_patches = (feature_size[0] // (patch_size[0])) * (feature_size[1] // (patch_size[1]))
        patch_dim = channels * patch_size[0] * patch_size[1]
        assert pos_embedding_type in ['sine','learnable','sine-full']

        self.inplanes = 64
        self.patch_size = patch_size
        self.heatmap_size = heatmap_size
        hidden_heatmap_dim = heatmap_size[0] * heatmap_size[1] // 8
        heatmap_dim = heatmap_size[0] * heatmap_size[1]
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")
        mlp_dim = dim * mlp_ratio

        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        h,w = feature_size[0] // (self.patch_size[0]), feature_size[1] // ( self.patch_size[1])
        self._make_position_embedding(w, h, dim, pos_embedding_type)


        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        # transformer
        self.transformer_layer1= Transformer(dim, depth, heads, mlp_dim, dropout, 
                                            num_keypoints=num_keypoints, all_attn=self.all_attn, 
                                            scale_with_head=True)
        self.to_keypoint_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_heatmap_dim),
            nn.LayerNorm(hidden_heatmap_dim),
            nn.Linear(hidden_heatmap_dim, heatmap_dim)
        ) if (dim <= hidden_heatmap_dim*0.5 and apply_multi) else  nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, heatmap_dim)
        )
        trunc_normal_(self.keypoint_token, std=.02)
        if apply_init:
            self.apply(self._init_weights)

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_keypoints, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, feature, mask = None, ratio=1.0):
        p = self.patch_size
        # transformer
        x = rearrange(feature, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p[0], p2 = p[1])
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape

        keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b = b)
        if self.pos_embedding_type in ["sine","sine-full"] :
            x += self.pos_embedding[:, :n]
            x = torch.cat((keypoint_tokens, x), dim=1)
        else:
            x = torch.cat((keypoint_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + self.num_keypoints)]
        x = self.dropout(x)

        #x = self.transformer_layer1(x,mask,self.pos_embedding)
        x, attn_x, _ = self.transformer_layer1(x,mask,self.pos_embedding,prune=True, keep_ratio=ratio)
        y1 = EasyDict(
            vis_token = x[:, 0:self.num_keypoints],
            kpt_token = x[:, self.num_keypoints:]
        )

        kpt_token = x[:, 0:self.num_keypoints]
        vis_token = x[:, self.num_keypoints:]
        x = self.to_keypoint_token(kpt_token)
        x = self.mlp_head(x)
        x = rearrange(x,'b c (p1 p2) -> b c p1 p2',p1=self.heatmap_size[0],p2=self.heatmap_size[1])
        
        output = EasyDict(
            y1 = y1,
            vis_token=vis_token,
            kpt_token=kpt_token,
            pred=x,
            attn=attn_x
        )
        return output

class TokenPose_TB_base(nn.Module):
    def __init__(self, *, feature_size, patch_size, num_keypoints, dim, depth, heads, 
                 mlp_ratio, apply_init=False, apply_multi=True, heatmap_size=[64,48], 
                 channels = 3, dropout = 0., emb_dropout = 0., pos_embedding_type="learnable"):
        """
        TokenPose base head, heatmap-based prediction head.
        """
        super().__init__()
        assert isinstance(feature_size,list) and isinstance(patch_size,list), \
               'image_size and patch_size should be list'
        assert feature_size[0] % patch_size[0] == 0 and \
               feature_size[1] % patch_size[1] == 0, \
               'Image dimensions must be divisible by the patch size.'
        num_patches = (feature_size[0] // (patch_size[0])) * (feature_size[1] // (patch_size[1]))
        patch_dim = channels * patch_size[0] * patch_size[1]
        assert pos_embedding_type in ['sine','learnable','sine-full']

        self.inplanes = 64
        self.patch_size = patch_size
        self.heatmap_size = heatmap_size
        hidden_heatmap_dim = heatmap_size[0] * heatmap_size[1] // 8
        heatmap_dim = heatmap_size[0] * heatmap_size[1]
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")
        mlp_dim = dim * mlp_ratio

        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        h,w = feature_size[0] // (self.patch_size[0]), feature_size[1] // ( self.patch_size[1])
        self._make_position_embedding(w, h, dim, pos_embedding_type)


        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)





        self.compute_cis = partial(compute_mixed_cis, num_heads= heads)
        
        freqs = []
        for i in range(depth): 
            freqs.append(
                init_random_2d_freqs(dim=dim//heads, num_heads=heads, theta=100.0)
            )
        freqs = torch.stack(freqs, dim=1).view(2, depth, -1)

        self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
        self.freqs_keypoint = nn.Parameter(torch.rand(2, depth, dim//2) , requires_grad=True)
        self.freqs_keypoint_y = nn.Parameter(torch.rand(2, depth, dim//2) , requires_grad=True)
        self.learnable_absolute = nn.Parameter(torch.ones(depth), requires_grad=True)      
        t_x, t_y = init_t_xy(end_x = heatmap_size[0] // patch_size[0], end_y = heatmap_size[1] // patch_size[1])
        t_keypoint = torch.tensor([0, -1, 1,-2,2,-3,3,-4,4,-5,5,-6,6,-7,7,-8,8]).float()
        t_keypoint_y = torch.tensor([2.5, 1, -1,2,-2,6,-6,8,-8,7,-7,4,-4,5,-5,3,-3]).float()
        self.register_buffer('freqs_t_x', t_x)
        self.register_buffer('freqs_t_y', t_y)
        self.register_buffer('freqs_t_keypoint', t_keypoint)
        self.register_buffer('freqs_t_keypoint_y', t_keypoint_y)






        # transformer
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, 
                                       num_keypoints=num_keypoints, all_attn=self.all_attn, 
                                       scale_with_head=True)

        self.to_keypoint_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_heatmap_dim),
            nn.LayerNorm(hidden_heatmap_dim),
            nn.Linear(hidden_heatmap_dim, heatmap_dim)
        ) if (dim <= hidden_heatmap_dim*0.5 and apply_multi) else  nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, heatmap_dim)
        )
        trunc_normal_(self.keypoint_token, std=.02)
        if apply_init:
            self.apply(self._init_weights)

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_keypoints, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, feature, mask = None):
        p = self.patch_size
        # transformer
        x = rearrange(feature, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p[0], p2 = p[1])
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape

        keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b = b)
        if self.pos_embedding_type in ["sine","sine-full"] :
            x += self.pos_embedding[:, :n] * self.learnable_absolute[0]
            x = torch.cat((keypoint_tokens, x), dim=1)
        else:
            x = torch.cat((keypoint_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + self.num_keypoints)]
        x = self.dropout(x)

        if self.freqs_t_x.shape[0] != x.shape[1] - 17:
            t_x, t_y = init_t_xy(end_x = W // self.patch_size[0], end_y = H // self.patch_size[1])
            t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            

        else:
            t_x, t_y = self.freqs_t_x, self.freqs_t_y
            t_keypoint = self.freqs_t_keypoint
            t_keypoint_y = self.freqs_t_keypoint_y

        freqs_cis = self.compute_cis(self.freqs, t_x, t_y,self.freqs_keypoint, self.freqs_keypoint_y , t_keypoint,t_keypoint_y)
        learnable_absolute = self.learnable_absolute


        x, _, _ = self.transformer(x,freqs_cis,learnable_absolute, mask,self.pos_embedding)

        kpt_token = x[:, 0:self.num_keypoints]
        vis_token = x[:, self.num_keypoints:]
        x = self.to_keypoint_token(kpt_token)
        x = self.mlp_head(x)
        x = rearrange(x,'b c (p1 p2) -> b c p1 p2',p1=self.heatmap_size[0],p2=self.heatmap_size[1])
        
        output = EasyDict(
            vis_token=vis_token,
            kpt_token=kpt_token,
            pred=x
        )
        return output

class DistilPose_base(nn.Module):
    def __init__(self, *, feature_size, patch_size, num_keypoints, dim, depth, heads, 
                 mlp_ratio, apply_init=False, apply_multi=True, hidden_dim=384,
                 channels = 3, dropout = 0., emb_dropout = 0., pos_embedding_type="learnable", 
                 out_mode='all+', heatmap_size=[48, 64]):
        super().__init__()
        """
        DistilPose base head.
        Basically the same as TokenPose_base, regression-based prediction head
        """
        assert isinstance(feature_size,list) and isinstance(patch_size,list), \
               'image_size and patch_size should be list'
        assert feature_size[0] % patch_size[0] == 0 and \
               feature_size[1] % patch_size[1] == 0, \
               'Image dimensions must be divisible by the patch size.'
        num_patches = (feature_size[0] // (patch_size[0])) * (feature_size[1] // (patch_size[1]))
        patch_dim = channels * patch_size[0] * patch_size[1]
        assert pos_embedding_type in ['sine','learnable','sine-full']

        self.inplanes = 64
        self.patch_size = patch_size
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")
        self.heatmap_size = heatmap_size
        mlp_dim = dim * mlp_ratio

        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        h,w = feature_size[0] // (self.patch_size[0]), feature_size[1] // ( self.patch_size[1])
        self._make_position_embedding(w, h, dim, pos_embedding_type)

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        # transformer
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, 
                                       num_keypoints=num_keypoints, all_attn=self.all_attn, 
                                       scale_with_head=True)

        self.to_keypoint_token = nn.Identity()

        self.out_mode = out_mode
        if out_mode == '1sigma':
            out_dim = 4 # 2 coords, 1 score, 1 sigma
        elif out_mode == 'all':
            out_dim = 5 # 2 coords, 1 score, 2 sigma
        elif out_mode == 'score':
            out_dim = 3 # 2 coords, 1 score
        elif out_mode == 'dist':
            out_dim = 5 # 2 coords, 2 sigma, 1 rou
        elif out_mode == 'sigma':
            out_dim = 4 # 2 coords, 2 sigma
        else:
            out_dim = 2 # 2 coords

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        ) if (dim <= hidden_dim*0.5 and apply_multi) else  nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )
        trunc_normal_(self.keypoint_token, std=.02)
        if apply_init:
            self.apply(self._init_weights)

        self.to_pos = nn.Softplus()

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_keypoints, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, feature, mask = None):
        p = self.patch_size
        # transformer
        x = rearrange(feature, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p[0], p2 = p[1])
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape

        keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b = b)
        if self.pos_embedding_type in ["sine","sine-full"] :
            x += self.pos_embedding[:, :n]
            x = torch.cat((keypoint_tokens, x), dim=1)
        else:
            x = torch.cat((keypoint_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + self.num_keypoints)]
        x = self.dropout(x)

        x = self.transformer(x, mask,self.pos_embedding)

        kpt_token = x[:, 0:self.num_keypoints]
        vis_token = x[:, self.num_keypoints:]
        x = self.to_keypoint_token(kpt_token)
        x = self.mlp_head(x)
        if self.out_mode == '1sigma':
            output = EasyDict(
                vis_token=vis_token,
                kpt_token=kpt_token,
                pred=x[..., 0:2].sigmoid(),
                score=x[..., 2].unsqueeze(-1),
                sigma=self.to_pos(x[..., 3:4]),
            )
        elif self.out_mode == 'all':
            output = EasyDict(
                vis_token=vis_token,
                kpt_token=kpt_token,
                pred=x[..., 0:2].sigmoid(),
                score=x[..., 2].unsqueeze(-1),
                sigma=self.to_pos(x[..., 3:5])
            )
        elif self.out_mode == 'score':
            output = EasyDict(
                vis_token=vis_token,
                kpt_token=kpt_token,
                pred=x[..., 0:2].sigmoid(),
                score=x[..., 2].unsqueeze(-1)
            )
        elif self.out_mode == 'dist':
            output = EasyDict(
                vis_token=vis_token,
                kpt_token=kpt_token,
                pred=x[..., 0:2].sigmoid(),
                sigma=self.to_pos(x[..., 2:4]),
                rou=x[..., 4:].sigmoid()*2-1
            )
        elif self.out_mode == 'sigma':
            output = EasyDict(
                vis_token=vis_token,
                kpt_token=kpt_token,
                pred=x[..., 0:2].sigmoid(),
                sigma=self.to_pos(x[..., 2:4])
            )
        else:
            output = EasyDict(
                vis_token=vis_token,
                kpt_token=kpt_token,
                pred=x[..., 0:2].sigmoid()
            )
        return output

class SDPose(nn.Module):
    def __init__(self, *, feature_size, patch_size, num_keypoints, dim, depth,  heads, 
                 mlp_ratio, apply_init=False, apply_multi=True, heatmap_size=[64,48], 
                 channels = 3, dropout = 0., emb_dropout = 0., pos_embedding_type="learnable",
                 cycle_num=2):
        """
        SelfDistillPose base head, heatmap-based prediction head.
        """
        super().__init__()
        assert isinstance(feature_size,list) and isinstance(patch_size,list), \
               'image_size and patch_size should be list'
        assert feature_size[0] % patch_size[0] == 0 and \
               feature_size[1] % patch_size[1] == 0, \
               'Image dimensions must be divisible by the patch size.'
        num_patches = (feature_size[0] // (patch_size[0])) * (feature_size[1] // (patch_size[1]))
        patch_dim = channels * patch_size[0] * patch_size[1]
        assert pos_embedding_type in ['sine','learnable','sine-full']

        self.inplanes = 64
        self.patch_size = patch_size
        self.heatmap_size = heatmap_size
        hidden_heatmap_dim = heatmap_size[0] * heatmap_size[1] // 8
        heatmap_dim = heatmap_size[0] * heatmap_size[1]
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")
        mlp_dim = dim * mlp_ratio
        self.cycle_num = cycle_num

        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        h,w = feature_size[0] // (self.patch_size[0]), feature_size[1] // ( self.patch_size[1])
        self._make_position_embedding(w, h, dim, pos_embedding_type)


        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.to_pos = nn.Softplus()

        self.compute_cis = partial(compute_mixed_cis, num_heads= heads)
        
        freqs = []
        for i in range(depth): 
            freqs.append(
                init_random_2d_freqs(dim=dim//heads, num_heads=heads, theta=100.0)
            )
        freqs = torch.stack(freqs, dim=1).view(2, depth, -1)

        self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
        self.freqs_keypoint = nn.Parameter(torch.rand(2, depth, dim//2) , requires_grad=True)
        self.freqs_keypoint_y = nn.Parameter(torch.rand(2, depth, dim//2) , requires_grad=True)      
        t_x, t_y = init_t_xy(end_x = heatmap_size[0] // patch_size[0], end_y = heatmap_size[1] // patch_size[1])
        t_keypoint = torch.tensor([0, -1, 1,-2,2,-3,3,-4,4,-5,5,-6,6,-7,7,-8,8]).float()
        t_keypoint_y = torch.tensor([0, 1, -1,2,-2,6,-6,8,-8,7,-7,4,-4,5,-5,3,-3]).float()
        self.register_buffer('freqs_t_x', t_x)
        self.register_buffer('freqs_t_y', t_y)
        self.register_buffer('freqs_t_keypoint', t_keypoint)
        self.register_buffer('freqs_t_keypoint_y', t_keypoint_y)

        

        # transformer
        self.transformer_layer1= Transformer_sd(dim, depth, heads, mlp_dim, dropout, 
                                            num_keypoints=num_keypoints, all_attn=self.all_attn, 
                                            scale_with_head=True)
        self.to_keypoint_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_heatmap_dim),
            nn.LayerNorm(hidden_heatmap_dim),
            nn.Linear(hidden_heatmap_dim, heatmap_dim)
        ) if (dim <= hidden_heatmap_dim*0.5 and apply_multi) else  nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, heatmap_dim)
        )
        trunc_normal_(self.keypoint_token, std=.02)
        if apply_init:
            self.apply(self._init_weights)

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_keypoints, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, feature, mask = None):
        p = self.patch_size
        B, C, H, W = feature.shape

        # transformer
        x = rearrange(feature, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p[0], p2 = p[1])
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape

        keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b = b)
        if self.pos_embedding_type in ["sine","sine-full"] :
            x += self.pos_embedding[:, :n]
            x = torch.cat((keypoint_tokens, x), dim=1)
        else:
            x = torch.cat((keypoint_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + self.num_keypoints)]
        x = self.dropout(x)

        if self.freqs_t_x.shape[0] != x.shape[1] - 17:
            t_x, t_y = init_t_xy(end_x = W // self.patch_size[0], end_y = H // self.patch_size[1])
            t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            

        else:
            t_x, t_y = self.freqs_t_x, self.freqs_t_y
            t_keypoint = self.freqs_t_keypoint
            t_keypoint_y = self.freqs_t_keypoint_y

        freqs_cis = self.compute_cis(self.freqs, t_x, t_y,self.freqs_keypoint, self.freqs_keypoint_y , t_keypoint,t_keypoint_y)

        output_list = []

        for _ in range(self.cycle_num):
            x, attn = self.transformer_layer1(x,freqs_cis,mask,self.pos_embedding)
            x = x[-1]
            kpt_token = x[:, 0:self.num_keypoints]
            vis_token = x[:, self.num_keypoints:]
            tmp_res = self.to_keypoint_token(kpt_token)
            tmp_res = self.mlp_head(tmp_res)
            tmp_res = rearrange(tmp_res,'b c (p1 p2) -> b c p1 p2',p1=self.heatmap_size[0],p2=self.heatmap_size[1])
            output = EasyDict(
                vis_token=vis_token,
                kpt_token=kpt_token,
                pred=tmp_res,
                attn=attn[-1]
            )
            output_list.append(output)

        return output_list

class RLESDPose_base(nn.Module):
    def __init__(self, *, feature_size, patch_size, num_keypoints, dim, depth, heads, 
                 mlp_ratio, apply_init=False, apply_multi=True, hidden_dim=384,
                 channels = 3, dropout = 0., emb_dropout = 0., pos_embedding_type="learnable"):
        super().__init__()
        """
        DistilPose base head.
        Basically the same as TokenPose_base, regression-based prediction head
        """
        assert isinstance(feature_size,list) and isinstance(patch_size,list), \
               'image_size and patch_size should be list'
        assert feature_size[0] % patch_size[0] == 0 and \
               feature_size[1] % patch_size[1] == 0, \
               'Image dimensions must be divisible by the patch size.'
        num_patches = (feature_size[0] // (patch_size[0])) * (feature_size[1] // (patch_size[1]))
        patch_dim = channels * patch_size[0] * patch_size[1]
        assert pos_embedding_type in ['sine','learnable','sine-full']

        self.inplanes = 64
        self.patch_size = patch_size
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")
        mlp_dim = dim * mlp_ratio

        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        h,w = feature_size[0] // (self.patch_size[0]), feature_size[1] // ( self.patch_size[1])
        self._make_position_embedding(w, h, dim, pos_embedding_type)

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        # transformer
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, 
                                       num_keypoints=num_keypoints, all_attn=self.all_attn, 
                                       scale_with_head=True)

        self.to_keypoint_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4)
        ) if (dim <= hidden_dim*0.5 and apply_multi) else  nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4)
        )
        trunc_normal_(self.keypoint_token, std=.02)
        if apply_init:
            self.apply(self._init_weights)

        self.to_pos = nn.Softplus()

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_keypoints, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, feature, mask = None):
        p = self.patch_size
        # transformer
        x = rearrange(feature, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p[0], p2 = p[1])
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape

        keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b = b)
        if self.pos_embedding_type in ["sine","sine-full"] :
            x += self.pos_embedding[:, :n]
            x = torch.cat((keypoint_tokens, x), dim=1)
        else:
            x = torch.cat((keypoint_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + self.num_keypoints)]
        x = self.dropout(x)

        x, _ = self.transformer(x, mask,self.pos_embedding)

        kpt_token = x[:, 0:self.num_keypoints]
        vis_token = x[:, self.num_keypoints:]
        x = self.to_keypoint_token(kpt_token)
        x = self.mlp_head(x)
        output = EasyDict(
            vis_token=vis_token,
            kpt_token=kpt_token,
            pred=x
        )
        return output