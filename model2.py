import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


class PreNormWithDropPath(nn.Module):
    def __init__(self, dim, fn, drop_path_rate):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fn = fn
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x, **kwargs):
        return self.drop_path(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GLU(dim, hidden_dim, nn.SiLU()),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim,
                 dropout, stochastic_depth=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])

        dpr = [x.item() for x in torch.linspace(
            0, stochastic_depth, depth)]

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormWithDropPath(embedding_dim, Attention(
                    dim=embedding_dim, num_heads=heads), drop_path_rate=dpr[i]),
                PreNormWithDropPath(embedding_dim, FeedForward(
                    dim=embedding_dim, hidden_dim=mlp_dim, dropout=dropout), drop_path_rate=dpr[i])
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size,
                 n_conv_layers,
                 n_input_channels,
                 n_output_channels,
                 in_planes,
                 ):    # filter size for in between convolutions
        super(Tokenizer, self).__init__()

        assert(len(in_planes) == n_conv_layers-1)

        stride = max(
            1, (kernel_size // 2) - 1)
        padding = max(
            1, (kernel_size // 2))
        pooling_kernel_size = 3
        pooling_stride = 2
        pooling_padding = 1

        n_filter_list = [n_input_channels]+in_planes+[n_output_channels]

        # first layer, middle ones of same n_conv_layers-2 times, last layer

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding),
                          bias=False    # add a learnable bias param
                          ),
                nn.ReLU(),  # activation
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding)
            )
                for i in range(n_conv_layers)
            ])

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels, height, width):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

# For image size 32
# keep embedding (word) dim as 128
# layer=2
# mlp ratio 1
# num heads 2
# n conv layers 2


class CCT(nn.Module):
    def __init__(self,
                 img_size,
                 embedding_dim,
                 num_layers,
                 num_heads,
                 mlp_ratio,
                 num_classes,
                 n_conv_layers,   # conv layer for tokeniser
                 n_input_channels=3,    # 3 for images
                 kernel_size=3,     # remaining values should suit most
                 dropout=0.,
                 *args, **kwargs):
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   n_conv_layers=n_conv_layers,
                                   in_planes=[64, 80, 96])

        self.transformer = Transformer(
            embedding_dim=embedding_dim, depth=num_layers,
            heads=num_heads,
            mlp_dim=int(embedding_dim * mlp_ratio),
            dropout=dropout)

        seq_len = self.tokenizer.sequence_length(
            n_input_channels, img_size, img_size)

        self.positional_emb = Parameter(torch.zeros(1, seq_len, embedding_dim),
                                        requires_grad=True)
        init.trunc_normal_(self.positional_emb, std=0.2)

        self.dropout = Dropout(p=dropout)

        # self.mlp_head = nn.Sequential(
        #     RMSNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )
        # these two below are for same task commented above
        self.norm = RMSNorm(embedding_dim)
        self.fc = Linear(embedding_dim, num_classes)

        self.attention_pool = Linear(embedding_dim, 1)
        # weights for different layers, how to pool

        # settng weights
        self.apply(self.init_weight)

    def forward(self, x):
        x = self.tokenizer(x)
        x += self.positional_emb
        x = self.dropout(x)
        x = self.transformer(x)

        x = self.norm(x)
        x = torch.matmul(F.softmax(self.attention_pool(
            x), dim=1).transpose(-1, -2), x).squeeze(-2)
        x = self.fc(x)

        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)
