from torch import nn
from functools import partial
import torch
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch
import math
import warnings

import torch.nn.functional as F
import linmutli as lm
from helpers import *
from perf import *

half_dist = 3
shifts = 49


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    #issuecomment-532968956 ... I've opted for
    See discussion: https://github.com/tensorflow/tpu/issues/494
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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
    def __init__(self, num_variants, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_variants = num_variants
        self.dim = dim

        self.Wkv = nn.Linear(dim, 2 * head_dim, bias=qkv_bias)
        self.Wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, num_layer):
        B, N, C = x.shape
        print("A", x.shape)
        # torch.Size([batch, #heads, #patches, embedding])
        q = self.Wq(x).reshape(B, N, self.num_heads, C //
                               self.num_heads).permute(0, 2, 1, 3)
        kv = self.Wkv(x).reshape(
            B, N, 2, C // self.num_heads).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]  # torch.Size([batch, #patches, embedding])

        attn = einsum('b h i d, b  j d -> b  h i j', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = einsum('b h i j, b j d -> b h i d', attn,
                   v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, num_feats=256, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0., share_kv=False):
        super().__init__()
        assert (
            dim % num_heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.num_feats = num_feats

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, num_feats)))
        if share_kv:
            self.proj_v = self.proj_k
        else:
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, num_feats)))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, nx=None, ny=None, num_layer=1):
        b, n, d = x.shape
        d_h, h, k = self.head_dim, self.num_heads, self.num_feats
        kv_len = n
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.scale * \
            self.query(x).reshape(b, n, h, d_h).transpose(1, 2)
        kv = self.kv(x).reshape(b, n, 2, d).permute(2, 0, 1, 3)
        # make torchscript happy (cannot use tensor as tuple)
        keys, values = kv[0], kv[1]

        # project keys and values along the sequence length dimension to k
        def proj_seq_len(args): return torch.einsum('bnd,nk->bkd', *args)
        kv_projs = (self.proj_k, self.proj_v)
        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values
        def merge_key_values(t): return t.reshape(b, k, -1, d_h).transpose(
            1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention
        attn = torch.einsum('bhnd,bhkd->bhnk', queries, keys)
        attn = (attn - torch.max(attn, dim=-1,
                keepdim=True)[0]).softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class AttentionModified(nn.Module):
    def __init__(self, num_variants, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_variants = num_variants
        self.dim = dim
        self.Wkv = nn.Linear(dim,  2 * head_dim, bias=qkv_bias)
        self.Wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, variants_patches, num_layer):
        # torch.Size([num_variants,batch_size,  num_patches, embedding_dim])
        num_variants, B, N, C = variants_patches.shape
        # assert num_variants == self.num_variants, f
        # "num variants ({num_variants}) doesn't match model ({self.num_variants})."

        # torch.Size([batch, #varaints , #patches, embedding])
        variants_patches = variants_patches.permute(1, 0, 2, 3)
        q = self.Wq(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(
            0, 2, 1, 3).unsqueeze(dim=-2)  # torch.Size([batch, #heads, #patches,1, embedding])
        kv = self.Wkv(variants_patches).reshape(
            B, num_variants, N, 2, C // self.num_heads).permute(3, 0, 2, 1, 4).unsqueeze(dim=2)

        # k:  torch.Size([batch,  #patches,  #varaints, embedding])            v:  torch.Size([batch, 1 ,   #patches,  #varaints, embedding])
        k, v = kv[0], kv[1]

        # attn:  torch.Size([batch, #heads, #patches, 1, #varaints ])
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).squeeze()
        x = x.transpose(1, 2).reshape(B, N, C)
        print(x.shape)
        x = self.proj(x)
        print(x.shape)
        x = self.proj_drop(x)

        return x


class Downsample(nn.Module):
    def __init__(self, dim, dim_out, seq_len):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = partial(nn.LayerNorm, eps=1e-6)(dim_out)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.img_size = seq_len ** 0.5
        self.dim_out = dim_out
        self.dim = dim

    def forward(self, x):
        B, embedding_dim, h, w = x.shape
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # 44*128, 32, 32, 192
        x = self.norm(x)  # TODO: layernorm on embedding dim?
        x = x.permute(0, 3, 1, 2)  # 44*128, 192, 32, 32
        x = self.pool(x)
        _, embedding_dim, h_new, w_new = x.shape

        x = x.reshape(B, embedding_dim, h_new, w_new)

        return x


class Transformer_first(nn.Module):
    def __init__(self, num_variants, num_patches, depth, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, proj_drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.num_patches = num_patches
        self.dim = dim

        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(
            torch.zeros(num_variants, num_patches, dim))
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.num_variants = num_variants
        self.embedding_dim = dim
        self.mlp_hidden_dim = mlp_hidden_dim

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(num_variants, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop) if not i == 0
                else AttentionModified(num_variants, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop),
                Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                    act_layer=act_layer, drop=proj_drop)
            ]))

        fill_wiith(self.pos_emb, std=.02)

    def forward(self, patches):
        num_variants, B, num_patches, embedding_dim = patches.shape
        assert num_variants == self.num_variants
        assert embedding_dim == self.embedding_dim
        patches = patches.transpose(0, 1)  # V,B,HW,D => B,V,HW,D
        # print(patches.shape)
        # print(self.pos_emb.shape) # V,HW,D
        pos_emb = self.pos_emb.unsqueeze(dim=0)  # 1,V,HW,D
        # print(pos_emb.shape)
        patches = patches + pos_emb
        patches = patches.transpose(0, 1)  # V,B,HW,D
        # print(patches.shape)
        x = patches[0]
        num_layer = 0
        for attn, mlp in self.layers:
            if num_layer == 0:
                x = x + self.drop_path(attn(self.norm1(x),
                                            self.norm1(patches), num_layer=num_layer))
            else:
                x = x + self.drop_path(attn(self.norm1(x),
                                            num_layer=num_layer))

            x = x + self.drop_path(mlp(self.norm2(x)))
            num_layer += 1
        # x = x.unsqueeze(dim=0)
        # patches = torch.cat((x, patches[1:]))
        return x


class Transformer(nn.Module):
    def __init__(self, num_variants, num_patches, depth, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, proj_drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.num_patches = num_patches
        self.dim = dim

        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.zeros(num_patches, dim))
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.num_variants = num_variants
        self.embedding_dim = dim
        self.mlp_hidden_dim = mlp_hidden_dim

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerformerSelfAttention(dim),
                Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                    act_layer=act_layer, drop=proj_drop)
            ]))

        fill_wiith(self.pos_emb, std=.02)

    def forward(self, x):
        B, num_patches,  embedding_dim = x.shape

        # assert  num_blocks == self.num_blocks
        assert embedding_dim == self.embedding_dim
        pos_emb = self.pos_emb.unsqueeze(dim=0)
        x = x + pos_emb
        num_layer = 0
        for attn, mlp in self.layers:
            print("TEST", x.shape)
            x = x + self.drop_path(attn(self.norm1(x)))
            x = x + self.drop_path(mlp(self.norm2(x)))
            num_layer += 1
        return x

# Making embedding for: Input as B,C,H,W and output as


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, stride_size=1, padding_size=1, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        # batch,3,32,32
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.projections = nn.ModuleList([])

        for _ in range(shifts):
            self.projections.append(
                nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                          stride=stride_size, padding=padding_size)
            )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, img):
        B, C, H, W = img.shape
        assert H == self.img_size[0] and W == self.img_size[1], "IMG SIZE FAIL"

        patches = []
        # the original one
        img_new = img  # B, C, H, W
        new_patch = self.norm(self.projections[0](
            img_new))  # B, D = 192 , H , W
        # print(new_patch.shape)
        new_patch = (new_patch.flatten(2))  # B, D, H*W
        new_patch = new_patch.transpose(1, 2)  # B, H*W, D
        # print(new_patch.shape)
        patches.append(new_patch)

        half_dist = 3

        paddings = []

        for i in range(-half_dist, half_dist+1):
            for j in range(-half_dist, half_dist+1):
                if i == 0 and j == 0:
                    continue
                paddings.append((i, j))

        for i in range(len(paddings)):
            cur_padding = paddings[i]
            # B, C, H, W (along H and W)
            img_new = torch.roll(img, shifts=cur_padding, dims=(2, 3))
            new_patch = self.norm(self.projections[i+1](img_new))
            new_patch = new_patch.flatten(2).transpose(1, 2)
            patches.append(new_patch)
            # each is B, H*W, D

        patches = torch.stack(patches)
        # dist^2 such so d^2, B, H*W, D
        # print(patches.shape)

        return patches


class TransformerMain(nn.Module):
    def __init__(self, *, scaling_factor=4, num_variants, image_size, patch_size, num_classes, embedding_dim, heads, blocks=4, num_layers_per_block, mlp_mult=4, channels=3, dim_head=64, qkv_bias=True, attn_drop=0.0,
                 proj_drop=0.0, stochastic_depth_drop=0.1, kernel_size=3, stride_size=1, padding_size=1):
        super().__init__()
        assert (image_size %
                patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        cnt = image_size // patch_size
        initial_num_blocks = cnt * cnt

        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.padding_size = padding_size
        self.blocks = blocks
        hierarchies = list(reversed(range(blocks)))
        layer_heads = heads
        # layer_dims = list(map(lambda t: t * dim, mults))

        layer_dims = embedding_dim
        self.start_embedding = embedding_dim[0][0]
        self.end_embedding = embedding_dim[-1][-1]
        # dim_pairs = zip(layer_dims[:-1], layer_dims[1:])
        num_blocks = (initial_num_blocks, initial_num_blocks,
                      initial_num_blocks, initial_num_blocks)

        seq_len = (initial_num_blocks, initial_num_blocks//(scaling_factor),
                   initial_num_blocks//(scaling_factor**2), initial_num_blocks//(scaling_factor**3))
        self.end_num_patches = seq_len[-1]

        self.to_patch_embedding = PatchEmbed(img_size=image_size, patch_size=self.kernel_size, stride_size=self.stride_size, padding_size=padding_size, in_chans=3,
                                             embed_dim=self.start_embedding)
        block_repeats = cast_tuple(num_layers_per_block, blocks)
        layer_heads = cast_tuple(layer_heads, blocks)
        num_blocks = cast_tuple(num_blocks, blocks)
        seq_lens = cast_tuple(seq_len, blocks)

        dim_pairs = cast_tuple(layer_dims, blocks)
        self.layers = nn.ModuleList([])

        for level, heads, (dim_in, dim_out), block_repeat, seq_len in zip(hierarchies, layer_heads, dim_pairs, block_repeats, seq_lens):

            is_last = level == 0
            depth = block_repeat
            is_first = level == (blocks-1)
            print("is first: ", is_first, "is last: ", is_last)
            self.layers.append(nn.ModuleList([
                Transformer(num_variants, seq_len, depth, dim_in, heads, mlp_mult, qkv_bias=qkv_bias, attn_drop=attn_drop,
                            drop_path=stochastic_depth_drop, proj_drop=proj_drop) if not is_first else
                Transformer_first(num_variants, seq_len, depth, dim_in, heads, mlp_mult, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                  drop_path=stochastic_depth_drop, proj_drop=proj_drop),
                Downsample(
                    dim_in, dim_out, seq_len) if not is_last else nn.Identity()
            ]))

        self.norm = partial(nn.LayerNorm, eps=1e-6)(self.end_embedding)
        self.mlp_head = nn.Linear(self.end_embedding, num_classes)
        self.apply(_init_vit_weights)

    def forward(self, img):
        # print("input size: ", img.shape)
        patches = self.to_patch_embedding(img)  # Va, B, HxW,D
        # print("x after embedding: ", patches.shape)
        num_hierarchies = len(self.layers)
        for level, (transformer, reduce_image_size) in zip(reversed(range(num_hierarchies)), self.layers):
            print("T", patches.shape)
            patches = transformer(patches)
            print("T", patches.shape)
            # T1: mean across all shifts

            if level > 0:
                grid_size = (
                    int(patches.shape[1]**0.5), int(patches.shape[1]**0.5))
                patches = to_image_plane(patches, grid_size, self.patch_size)
                patches = reduce_image_size(patches)
                patches = to_patches_plane(patches, self.patch_size)
        print("T", patches.shape)
        patches = self.norm(patches)
        patches_pool = torch.mean(patches, dim=(1))
        return self.mlp_head(patches_pool)


def to_patches_plane(x, patch_size):
    patch_size = (patch_size, patch_size)
    batch, depth, height, width = x.shape  # ([128, 192, 8, 8])
    x = x.reshape(batch, depth, height*width)  # 128, 192, 64
    x = x.permute(0, 2, 1)  # 128, 64, 192
    return x


def to_image_plane(x, grid_size, patch_size):
    batch, num_patches, depth = x.shape  # 128, 256, 192
    x = x.permute(0, 2, 1)  # 128, 192, 256
    x = x.reshape(batch, depth, grid_size[0], grid_size[1])

    return x


def _init_vit_weights(m, n: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(m, nn.Linear):
        fill_wiith(m.weight, std=.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        fill_wiith(m.weight, std=.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):

        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
