from functools import partial
import torch
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch
import math
import warnings

from torch.nn.init import _calculate_fan_in_and_fan_out
import torch.nn.functional as F

# helpers

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)

# LayerNorm = partial(nn.InstanceNorm2d, affine = True)

# classes
NUM_VARIANTS = 10

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
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



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')







class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
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
        self.dim = dim
        self.scale = qk_scale or head_dim ** -0.5
        self.num_variants = num_variants

        self.Wkv_odd = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.Wq_odd = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, num_layer):
        B, N, C = x.shape  # torch.Size([num_variants,batch_size,  num_patches, embedding_dim])

        q = self.Wq_odd(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # torch.Size([batch, #heads, #patches, embedding])
        kv = self.Wkv_odd(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0,3, 1, 4)
        k, v = kv[0], kv[1]  # torch.Size([batch, #patches, embedding])

        attn = einsum('b h i d, b h j d -> b  h i j', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = einsum('b h i j, b h j d -> b h i d', attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, N):
        # N - num tokens
        flops = 0
        # q
        flops += N * self.dim * self.dim
        # k,v
        flops += 2  * N * self.dim * self.dim
        # attn
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        # multiply by v
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # proj
        flops += N * self.dim * self.dim
        return flops


class Attention_variants(nn.Module):
    def __init__(self, num_variants, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_variants = num_variants
        self.dim = dim
        self.Wkv_even = nn.Linear(dim,  2* dim, bias=qkv_bias)

        # self.Wq = nn.ModuleList([])
        # for i in range(num_variants):
        self.Wq_even = nn.Linear(dim, dim, bias=qkv_bias)

        # self.Wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.ones = torch.ones_like(self.identity)
        # print(self.identity.is_cuda)

    def forward(self, x, variants_patches, num_layer):
        num_variants, B, N, C = variants_patches.shape  # torch.Size([num_variants,batch_size,  num_patches, embedding_dim])
        assert num_variants == self.num_variants, f
        "num variants ({num_variants}) doesn't match model ({self.num_variants})."
        variants_patches = variants_patches.permute(1, 0, 2, 3)  # torch.Size([batch, #varaints , #patches, embedding])
        q = self.Wq_even(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).unsqueeze(dim=-2) # torch.Size([batch, #heads, #patches,1, embedding])
        kv = self.Wkv_even(variants_patches).reshape(B, num_variants, N, 2, self.num_heads, C // self.num_heads).permute(3, 0, 4, 2, 1, 5)

        k, v = kv[0], kv[1]  # k:  torch.Size([batch,  #patches,  #varaints, embedding])            v:  torch.Size([batch, 1 ,   #patches,  #varaints, embedding])
        attn = (q @ k.transpose(-2, -1)) * self.scale  # attn:  torch.Size([batch, #heads, #patches, 1, #varaints ])
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).squeeze()
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # del new, attn_temp, indices, indices_v, indices_attn

        return x

    def flops(self, N):
        # N - num tokens
        flops = 0
        flops += N * self.dim * self.dim
        flops += 2 * self.num_variants * N * self.dim * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * self.num_variants
        flops += self.num_heads * N * self.num_variants * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops




class Reduce_image_size(nn.Module):
    def __init__(self, dim, dim_out, seq_len):
        super().__init__()
        self.conv =  nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm =  partial(nn.LayerNorm, eps=1e-6)(dim_out)
        self.pool = nn.MaxPool2d(3, stride = 2, padding = 1)
        self.dim_out = dim_out
        self.dim = dim
        self.img_size = seq_len ** 0.5
    def forward(self, x):
        B, embedding_dim, h, w = x.shape
        x = self.conv(x) # x of shape  128, 192, 32, 32
        x= x.permute(0,2, 3,1)
        x = self.norm(x)
        x = x.permute(0,3,1,2)
        x = self.pool(x)
        _, embedding_dim, h_new, w_new = x.shape
        x = x.reshape( B, embedding_dim, h_new, w_new)

        return x

    def flops(self):
        Ho, Wo = self.img_size, self.img_size
        # conv
        flops = Ho * Wo * self.dim_out * self.dim * 3 * 3
        # norm
        flops += Ho * Wo * self.dim_out
        # pool
        flops += Ho * Wo * self.dim_out
        return flops



class Transformer_first(nn.Module):
    def __init__(self, num_variants, num_patches, depth, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, proj_drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.num_patches = num_patches
        self.dim = dim
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.zeros(num_variants, num_patches, dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim

        self.num_variants = num_variants
        self.embedding_dim = dim

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                 Attention(num_variants, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop) if not i == 0
                                             else Attention_variants(num_variants, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop),
                Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)
            ]))

        trunc_normal_(self.pos_emb, std=.02)

    def forward(self, patches):
        num_variants, B, num_patches, embedding_dim = patches.shape
        assert num_variants == self.num_variants
        assert embedding_dim == self.embedding_dim
        patches = patches.transpose(0, 1)  # 128, 44, 64, 16, 192
        pos_emb = self.pos_emb.unsqueeze(dim=0)
        patches = patches + pos_emb
        patches = patches.transpose(0, 1)
        x = patches[0]
        num_layer = 0
        for attn, mlp in self.layers:
            if num_layer == 0:
                x = x + self.drop_path(attn(self.norm1(x), self.norm1(patches), num_layer=num_layer))
            else:
                x = x + self.drop_path(attn(self.norm1(x), num_layer=num_layer))

            x = x + self.drop_path(mlp(self.norm2(x)))
            num_layer += 1
        return x

    def flops(self):
        flops = 0
        for attn, mlp in self.layers:
            flops += attn.flops(self.num_patches)
            # norm
            flops += self.num_patches * self.dim
            # mlp
            flops += 2* self.num_patches * self.dim * self.mlp_hidden_dim
            # norm
            flops += self.num_patches * self.dim
        return flops



class Transformer(nn.Module):
    def __init__(self, num_variants, num_patches, depth, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, proj_drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.num_patches = num_patches
        self.dim = dim
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.zeros(num_patches, dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.num_variants = num_variants
        self.embedding_dim = dim
        self.mlp_hidden_dim = mlp_hidden_dim

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(num_variants, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop),
                Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)
            ]))

        trunc_normal_(self.pos_emb, std=.02)

    def forward(self, x):
        B, num_patches,  embedding_dim = x.shape
        # assert  num_blocks == self.num_blocks
        assert embedding_dim == self.embedding_dim
        pos_emb = self.pos_emb.unsqueeze(dim=0)
        x = x + pos_emb
        num_layer = 0
        for attn, mlp in self.layers:
            x = x + self.drop_path(attn(self.norm1(x), num_layer=num_layer))
            x = x + self.drop_path(mlp(self.norm2(x)))
            num_layer +=1
        return x

    def flops(self):
        flops = 0
        for attn, mlp in self.layers:
            # attn
            flops += attn.flops(self.num_patches)
            # norm
            flops += self.num_patches * self.dim
            # mlp
            flops += 2* self.num_patches * self.dim * self.mlp_hidden_dim
            # norm
            flops += self.num_patches * self.dim

        return flops


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=1, padding_size=1, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size,img_size)
        patch_size = (patch_size,patch_size)
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.projections = nn.ModuleList([])

        for _ in range(NUM_VARIANTS):
            self.projections.append(
                nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size, padding=padding_size)
            )

    def forward(self, img):
        B, C, H, W = img.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        #  (padding_left,padding_right, ,padding_top,padding_bottom)
        p2d1  = (1, 0, 0, 0)  # 3, 225,224
        p2d2 = (2, 0, 0, 0)  # 3, 226,224
        p2d3 = (3, 0, 0, 0)  # 3, 227,224
        p2d4 = (0, 0, 1, 0)  # 3, 224,225
        p2d5 = (0, 0, 2, 0)  # 3, 224,226
        p2d6 = (0, 0, 3, 0)  # 3, 224,227
        p2d7 = (1, 0, 1, 0)  # 3, 225,225
        p2d8 = (2, 0, 2, 0)  # 3, 226,226
        p2d9 = (3, 0, 3, 0)  # 3, 227,227
        paddings = [p2d1, p2d2, p2d3, p2d4, p2d5, p2d6, p2d7, p2d8, p2d9]

        patches = []
        img_new = img
        new_patch = self.norm(self.projections[0](img_new))
        new_patch = new_patch.flatten(2).transpose(1, 2)
        patches.append(new_patch)

        for i in range(len(paddings)):
            cur_padding = paddings[i]
            img_new = F.pad(img, cur_padding, mode='circular')
            new_patch = self.norm(self.projections[i+1](img_new))
            new_patch = new_patch.flatten(2).transpose(1, 2)
            patches.append(new_patch)


        patches = torch.stack(patches)

        return  patches

    def flops(self):
        Ho, Wo = self.grid_size
        for i in range(10):
            flops = Ho * Wo * self.embed_dim * self.in_chans * (4 * 4)
            flops += Ho * Wo * self.embed_dim
        return flops





class ShiftingTransformer(nn.Module):
    def __init__(self,*, scaling_factor, output_dir, num_variants, image_size,patch_size,num_classes,embedding_dim,heads,num_hierarchies,num_layers_per_block, mlp_mult = 4,channels = 3, dim_head = 64, qkv_bias=True,attn_drop=0.0,
                 proj_drop=0.0,stochastic_depth_drop = 0.1 , init_patch_embed_size =1, kernel_size=3, stride_size=1, padding_size=1):
        super().__init__()
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        fmap_size = image_size // patch_size
        len_pyramid = len(num_layers_per_block)
        input_size_after_patch = image_size // init_patch_embed_size
        initial_num_blocks = fmap_size * fmap_size

        assert input_size_after_patch % patch_size == 0
        down_sample_ratio = input_size_after_patch // patch_size

        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.padding_size= padding_size
        self.num_hierarchies = num_hierarchies
        self.num_classes = num_classes
        hierarchies = list(reversed(range(num_hierarchies)))
        layer_heads = heads
        # layer_dims = list(map(lambda t: t * dim, mults))

        layer_dims = embedding_dim
        self.start_embedding = embedding_dim[0][0]
        self.end_embedding = embedding_dim[-1][-1]
        # dim_pairs = zip(layer_dims[:-1], layer_dims[1:])
        num_blocks = (initial_num_blocks, initial_num_blocks, initial_num_blocks , initial_num_blocks)
        if num_hierarchies == 4:
            seq_len = (initial_num_blocks, initial_num_blocks//(scaling_factor), initial_num_blocks//(scaling_factor**2), initial_num_blocks//(scaling_factor**3))
        elif num_hierarchies == 3:
            seq_len = (initial_num_blocks, initial_num_blocks//(scaling_factor), initial_num_blocks//(scaling_factor**2))

        self.end_num_patches = seq_len[-1]
        self.to_patch_embedding = PatchEmbed(img_size=image_size, patch_size=self.kernel_size, stride_size=self.stride_size, padding_size=padding_size, in_chans=3,
                                             embed_dim=self.start_embedding)
        block_repeats = cast_tuple(num_layers_per_block, num_hierarchies)
        layer_heads = cast_tuple(layer_heads, num_hierarchies)
        num_blocks = cast_tuple(num_blocks, num_hierarchies)
        seq_lens = cast_tuple(seq_len, num_hierarchies)

        dim_pairs = cast_tuple(layer_dims,num_hierarchies)
        self.layers = nn.ModuleList([])

        # print("build pyramid: ")
        # print("levels:", hierarchies, "heds: ", layer_heads, "dim paors: ", dim_pairs, "block repeats: ", block_repeats, "num blocks: ", num_blocks)
        for level, heads, (dim_in, dim_out), block_repeat, seq_len in zip(hierarchies, layer_heads, dim_pairs, block_repeats,seq_lens):
            print("level: ", level, "heads: ", heads, "dim in dim out: ", (dim_in, dim_out),
                  "block repeat: ", block_repeat,"seq len: ", seq_len)
            with open(str(output_dir) + "/pyramid_parameters.txt", "a+") as text_file:
                text_file.write(" level: ")
                text_file.write(str(level))
                text_file.write(" heads: ")
                text_file.write(str(heads))
                text_file.write(" dim in: ")
                text_file.write(str(dim_in))
                text_file.write(" dim out: ")
                text_file.write(str(dim_out))
                text_file.write(" block repeat: ")
                text_file.write(str(block_repeat))
                text_file.write(" seq len: ")
                text_file.write(str(seq_len))
                text_file.write("\n")

            is_last = level == 0
            depth = block_repeat
            is_first = level == (num_hierarchies-1)
            print("is first: ", is_first, "is last: ", is_last)
            self.layers.append(nn.ModuleList([
                Transformer(num_variants, seq_len, depth, dim_in, heads, mlp_mult, qkv_bias=qkv_bias, attn_drop=attn_drop,
                            drop_path=stochastic_depth_drop, proj_drop=proj_drop) if not is_first else
                Transformer_first(num_variants, seq_len, depth, dim_in, heads, mlp_mult, qkv_bias=qkv_bias, attn_drop=attn_drop,
                            drop_path=stochastic_depth_drop, proj_drop=proj_drop),
                Reduce_image_size(dim_in, dim_out, seq_len) if not is_last else nn.Identity()
            ]))


        self.norm = partial(nn.LayerNorm, eps=1e-6)(self.end_embedding)
        self.mlp_head = nn.Linear(self.end_embedding, num_classes)
        self.apply(_init_vit_weights)
        text_file.close()


    def forward(self, img):
        patches = self.to_patch_embedding(img) # 44, 128, 256,192
        num_hierarchies = len(self.layers)
        for level, (transformer, reduce_image_size) in zip(reversed(range(num_hierarchies)), self.layers):
            patches = transformer(patches)

            if level > 0:
                grid_size = (int(patches.shape[1]**0.5), int(patches.shape[1]**0.5))
                patches = to_image_plane(patches, grid_size, self.patch_size)
                patches = reduce_image_size(patches)
                patches = to_patches_plane(patches, self.patch_size)
        patches = self.norm(patches)
        patches_pool = torch.mean(patches, dim=(1))
        return self.mlp_head(patches_pool)

    def flops(self):
        flops = 0
        flops += self.to_patch_embedding.flops()
        print(flops/ 1e9)
        for level, (transformer, reduce_image_size) in zip(reversed(range(len(self.layers))), self.layers):
            flops += transformer.flops()
            print(flops/ 1e9)
            if level > 0:
                flops += reduce_image_size.flops()
                print(flops/ 1e9)
        # last norm
        flops += self.end_embedding * self.end_num_patches
        print(flops/ 1e9)
        # MLP
        flops += self.end_embedding * self.num_classes
        print(flops/ 1e9)
        return flops



def to_patches_plane(x, patch_size):
    patch_size = (patch_size, patch_size)
    batch, depth, height, width = x.shape # ([128, 192, 8, 8])
    x = x.reshape( batch, depth, height*width) #  128, 192, 64
    x = x.permute(0,2,1) #  128, 64, 192
    return x

def to_image_plane(x, grid_size, patch_size):
    batch, num_patches, depth = x.shape #  128, 256, 192
    x = x.permute(0,2,1)  # 128, 192, 256
    x = x.reshape(batch, depth, grid_size[0], grid_size[1])

    return x


def _init_vit_weights(m, n: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    # print(" ")
    # print("initialization new")
    # print(m)
    if isinstance(m, nn.Linear):
        # print("initialize linear")
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        # print("initialize conv")
        # NOTE conv was left to pytorch default in my original init
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        # print("initialize norm")

        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

# if __name__ == '__main__':
#     image_size = 224
#     x = torch.rand(1, 3, image_size, image_size).cuda()
#     patch_size = 4
#     embedding_dim = ((64, 192), (192, 384), (384, 384))
#     heads = (2, 6, 12)
#     num_hierarchies = 3  # number of hierarchies
#     num_layers_per_block = (2, 2,10)  # the number of transformer blocks at each heirarchy, starting from the bottom
#     model = NesT(
#         scaling_factor=4,
#         output_dir=".",
#         num_variants=10,
#         image_size=224,
#         patch_size=4,
#         embedding_dim=embedding_dim,
#         heads=heads,
#         num_hierarchies=num_hierarchies,  # number of hierarchies
#         num_layers_per_block=num_layers_per_block,  # the number of transformer blocks at each heirarchy, starting from the bottom
#         num_classes=1000,
#         init_patch_embed_size =1,
#         kernel_size = 4,
#         stride_size = 4,
#         padding_size = 0
#     ).cuda()
#     model.eval()
#     flops = model.flops()
#     print(f"number of GFLOPs: {flops / 1e9}")
#     #
#     # througput
#     repetitions = 1000
#     total_time = 0
#     optimal_batch_size = 128 #TODO
#     dummy_input = torch.randn(optimal_batch_size, 3, 224,224, dtype=torch.float).cuda()
#     with torch.no_grad():
#         for rep in range(repetitions):
#             starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#             starter.record()
#             _ = model(dummy_input)
#             ender.record()
#             torch.cuda.synchronize()
#             curr_time = starter.elapsed_time(ender)/1000
#             total_time += curr_time
#     Throughput = (repetitions*optimal_batch_size)/total_time
#     print("final throughput:", Throughput)

