from functools import partial

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


# attention related classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = (7,7,7)
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )
        
        # relative positional bias
        w1,w2,w3 = window_size
        # 初始化相对位置索引矩阵[2*H-1,2*W-1,2*D-1,num_heads]
        self.rel_pos_bias = nn.Embedding((2 * w1 - 1) *(2 * w2 - 1)*(2 * w3 - 1), self.heads)
        pos1 = torch.arange(w1)
        pos2 = torch.arange(w2)
        pos3 = torch.arange(w3)
        # 首先我们利用torch.arange和torch.meshgrid函数生成对应的坐标，[3,H,W,D] 然后堆叠起来，展开为一个二维向量，得到的是绝对位置索引。
        grid = torch.stack(torch.meshgrid(pos1, pos2, pos3, indexing = 'ij'))
        grid = rearrange(grid, 'c i j k -> (i j k) c')
        # 广播机制，分别在第一维，第二维，插入一个维度，进行广播相减，得到 3, whd*ww, whd*ww的张量
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...') 
        rel_pos[...,0] += w1 - 1
        rel_pos[...,1] += w2 - 1
        rel_pos[...,2] += w3 - 1
        # 做了个乘法操作，以进行区分,最后一维上进行求和，展开成一个一维坐标   a*x1 + b*x2 + c*x3  (a= hd b=d c =1) 
        rel_pos_indices = (rel_pos * torch.tensor([(2 *w2 - 1)*(2 *w3 - 1), (2 *w3 - 1), 1])).sum(dim = -1)
        
        # 注册为一个不参与网络学习的变量
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)
               

    def forward(self, x):
        batch, height, width, depth, window_height, window_width, window_depth ,_, device, h = *x.shape, x.device, self.heads
        # flatten
        x = rearrange(x, 'b x y z w1 w2 w3 d -> (b x y z) (w1 w2 w3) d')

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))
        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h (w1 w2 w3) d -> b w1 w2 w3 (h d)', w1 = window_height, w2 = window_width, w3 = window_depth)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y z) ... -> b x y z ...', x = height, y = width, z = depth)

class MaxViT_Block(nn.Module):
    def __init__(
        self,
        *,
        dim = 512,
        dim_head = 32,
        window_size = (8,8,8),
        dropout = 0.1,
    ):
        super().__init__()
        w1,w2,w3 = window_size

        self.net = nn.Sequential(
            Rearrange('b d (x w1) (y w2) (z w3) -> b x y z w1 w2 w3 d', w1 = w1, w2 = w2, w3 = w3),  # block-like attention
            PreNormResidual(dim, Attention(dim = dim, dim_head = dim_head, dropout = dropout, window_size = window_size)),
            PreNormResidual(dim, FeedForward(dim = dim, dropout = dropout)),
            Rearrange('b x y z w1 w2 w3 d -> b d (x w1) (y w2) (z w3)'),

            Rearrange('b d (w1 x) (w2 y) (w3 z) -> b x y z w1 w2 w3 d', w1 = w1, w2 = w2, w3 = w3),  # grid-like attention
            PreNormResidual(dim, Attention(dim = dim, dim_head = dim_head, dropout = dropout, window_size = window_size)),
            PreNormResidual(dim, FeedForward(dim = dim, dropout = dropout)),
            Rearrange('b x y z w1 w2 w3 d -> b d (w1 x) (w2 y) (w3 z)'),
            )

    def forward(self, x):
        x = self.net(x)
        return x
    

if __name__ == "__main__":

    maxvit_block = MaxViT_Block(
        dim = 256,                        # dimension of first layer, doubles every layer
        dim_head = 32,                    # dimension of attention heads, kept at 32 in paper
        window_size = (8,8,6),            # window size for block and grids
        dropout = 0.0                     # dropout
    )
    img = torch.randn(2, 256, 32, 32, 24)
    preds = maxvit_block(img) 
    print(preds.shape)

