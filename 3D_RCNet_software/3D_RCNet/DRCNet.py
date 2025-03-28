import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath
from einops.layers.torch import Rearrange


class unfold_3d(nn.Module):
    def __init__(self, kernel_size, stride, padd=[1, 1, 1], padd_mode='replicate'):
        super(unfold_3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padd = padd
        self.padd_mode = padd_mode

    def forward(self, x):
        x = F.pad(x, (self.padd[0], self.padd[0], self.padd[1], self.padd[1], self.padd[2], self.padd[2]),
                  mode=self.padd_mode)
        # x = F.ReflectionPad3d(x, [0, 0, padd[0], padd[1], padd[2]], mode=padd_mode)
        x = x.unfold(2, self.kernel_size[0], self.stride[0]) \
            .unfold(3, self.kernel_size[1], self.stride[1]) \
            .unfold(4, self.kernel_size[2], self.stride[2])
        # x = rearrange(x, 'b c h w d k1 k2 k3 -> b h w d (k1 k2 k3) c')
        x = rearrange(x, 'b c h w d k1 k2 k3 -> b (h w d) (k1 k2 k3) c')
        return x


class MLP_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(dim * 3),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim * 3, dim, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(dim)
        )

    def forward(self, x):
        return self.net(x)


class Rconv_3D(nn.Module):
    def __init__(self, dim, kernel_size=[3, 3, 3], stride=[1, 1, 1], heads=4):
        super(Rconv_3D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padd = [kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2]
        self.num_heads = heads

        self.proj = nn.Conv3d(dim, dim, kernel_size=1)

        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Sequential(
            nn.Conv3d(dim, dim * 3, kernel_size=1),
            nn.Conv3d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        )

        self.norm = nn.Sequential(
            Rearrange('b c h w d -> b (h w d) c'),
            # nn.LayerNorm(dim)
        )

        self.unfold_k = nn.Sequential(
            unfold_3d(kernel_size=self.kernel_size, stride=self.stride, padd=self.padd),
            # nn.LayerNorm(dim)
        )
        self.unfold_v = nn.Sequential(
            unfold_3d(kernel_size=self.kernel_size, stride=self.stride, padd=self.padd),
            # nn.LayerNorm(dim)
        )

    def forward(self, x):
        B, C, H, W, S = x.shape

        qkv = self.qkv(x).reshape(B, 3, C, H, W, S)
        q, k, v = qkv.unbind(1)
        q = self.norm(q)
        k = self.unfold_k(k)
        v = self.unfold_v(v)

        B, L, K, C = k.shape
        # q = q.reshape(B, self.num_heads, L, 1, -1)
        q = q.contiguous().view(B, self.num_heads, L, 1, -1)  # (B,head,(h*w*d),1,c/head)
        k = k.view(B, self.num_heads, L, K, -1)  # (B,head,(hwd),27,c/head)
        v = v.view(B, self.num_heads, L, K, -1)

        attn = q @ k.transpose(-2, -1)  # (B,head,(hwd),1,27)
        attn = (attn * self.scale).softmax(dim=-1)

        x = (attn @ v).transpose(1, 2)  # (B,head,(hwd),1,c/head)
        # x = torch.einsum('bhlxk,bhlkc->bhlxc', attn, v)
        x = x.reshape(B, L, C).transpose(-2, -1).reshape(B, C, H, W, S)  # B, n, C
        x = self.proj(x)
        return x


class Rconv_3D_Down(nn.Module):
    def __init__(self, dim, dim2, kernel_size=[3, 3, 3], stride=[1, 1, 1], heads=4):
        super(Rconv_3D_Down, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padd = [kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2]
        self.num_heads = heads

        self.proj = nn.Sequential(
            nn.Conv3d(dim, dim2, kernel_size=1)
        )

        self.q_down = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=[2, 2, 2], padding=[1, 1, 1])
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Sequential(
            nn.Conv3d(dim, dim * 3, kernel_size=1),
            nn.Conv3d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        )

        self.norm = nn.Sequential(
            Rearrange('b c h w d -> b (h w d) c'),
            # nn.LayerNorm(dim)
        )

        self.unfold_k = nn.Sequential(
            unfold_3d(kernel_size=self.kernel_size, stride=self.stride, padd=self.padd),
            # nn.LayerNorm(dim)
        )
        self.unfold_v = nn.Sequential(
            unfold_3d(kernel_size=self.kernel_size, stride=self.stride, padd=self.padd),
            # nn.LayerNorm(dim)
        )

    def forward(self, x):
        B, C, S, H, W = x.shape

        qkv = self.qkv(x).reshape(B, 3, C, S, H, W)
        q, k, v = qkv.unbind(1)
        q = q[:, :, ::2, ::2, ::2]
        # q = self.q_down(q)
        B, C, S_, H_, W_ = q.shape
        q = self.norm(q)
        k = self.unfold_k(k)
        v = self.unfold_v(v)
        B, L_, C = q.shape
        B, L, K, C = k.shape
        q = q.view(B, self.num_heads, L_, 1, -1)
        k = k.view(B, self.num_heads, L, K, -1)
        v = v.view(B, self.num_heads, L, K, -1)

        attn = torch.einsum('bhqxc,bhlkc->bhqxk', q, k)
        attn = (attn * self.scale).softmax(dim=-1)
        x = torch.einsum('bhqxk,bhlkc->bhqxc', attn, v)
        x = x.reshape(B, L_, C).transpose(-2, -1).reshape(B, C, S_, H_, W_)
        x = self.proj(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, heads, init_values=1e-4, drop_path=0.2):
        super().__init__()
        # self.layers = nn.ModuleList([])  #它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器

        self.norm1 = nn.BatchNorm3d(dim)
        self.norm2 = nn.BatchNorm3d(dim)

        # self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        # self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        # self.norm2 = nn.LayerNorm([100, 14, 14])
        self.mlp = nn.Sequential(
            MLP_Block(dim=dim),
            # Rearrange('B C S H W-> B S H W C')
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = nn.Sequential(
            Rconv_3D(
                dim, heads=heads),
            # Rearrange('B C S H W-> B S H W C')
        )

    def forward(self, x):
        # x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DownTransformer(nn.Module):
    def __init__(self, dim, dim2, heads, init_values=1e-4, drop_path=0.2):
        super().__init__()
        self.layers = nn.ModuleList([])  # 它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器

        self.norm1 = nn.BatchNorm3d(dim)
        self.norm2 = nn.BatchNorm3d(dim2)
        # self.gamma_1 = nn.Parameter(init_values * torch.ones(dim2), requires_grad=True)
        # self.gamma_2 = nn.Parameter(init_values * torch.ones(dim2), requires_grad=True)
        # self.norm2 = nn.LayerNorm([100, 14, 14])
        self.mlp = nn.Sequential(
            MLP_Block(dim=dim2),
            # Rearrange('B C S H W-> B S H W C')
        )
        self.drop_path = nn.Sequential(
            # Rearrange('B S H W C-> B C S H W'),
            DropPath(drop_path) if drop_path > 0. else nn.Identity()
        )
        self.path_mlp = nn.Sequential(
            Rearrange('B S H W C-> B C S H W'),
        )
        self.attn = Rconv_3D_Down(dim, dim2, heads=heads)

    def forward(self, x):
        # x = self.path_mlp(self.gamma_1 * self.attn(self.norm1(x)))
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm3d(dim)
        # self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
        #                             requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # x = x.permute(0, 2, 3, 4, 1) # (N, C, S, H, W) -> (N, S, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, S, H, W, C) -> (N, C, S, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=1, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[32, 64, 128, 256], drop_path_rate=0.05,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            # nn.Conv3d(in_chans, dims[0]//2, kernel_size=(2,1,1), stride=(2,1,1)),
            # nn.BatchNorm3d(dims[0]//2),
            # nn.Conv3d(dims[0]//2, dims[0], kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            # nn.BatchNorm3d(dims[0])
            nn.Conv3d(in_chans, dims[0], kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            nn.BatchNorm3d(dims[0])
            # LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            if i < 1:
                downsample_layer = nn.Sequential(
                    nn.BatchNorm3d(dims[i]),
                    # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv3d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                )
            else:
                downsample_layer = nn.Sequential(
                    nn.BatchNorm3d(dims[i]),
                    # Rconv_3D_Down(
                    #     dims[i], dims[i+1], heads=8, dropout=0.),
                    DownTransformer(dims[i], dims[i + 1], heads=8, init_values=1e-4,
                                    drop_path=0)
                )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # # model 3 和 4
        for i in range(4):
            if i < 2:
                stage = nn.Sequential(
                    *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            else:
                stage = nn.Sequential(
                    *[Transformer(dim=dims[i], heads=8, drop_path=dp_rates[cur + j],
                                  init_values=layer_scale_init_value) for j in range(depths[i])]
                )
            self.stages.append(stage)
            cur += depths[i]
        # --------------------------------------------------------------------------------------------------
        # 每个stage最后一个block加入vit
        # for i in range(4):
        #     self.stage = nn.Sequential()
        #     for j in range(depths[i]):
        #         if j < depths[i]-1:
        #             st = Block(dim=dims[i], drop_path=dp_rates[cur + j],
        #                         layer_scale_init_value=layer_scale_init_value)
        #         else:
        #             st = Transformer(dim=dims[i], heads=8, drop_path=dp_rates[cur + j],
        #                               init_values=layer_scale_init_value)
        #         self.stage.add_module(str(j), st)
        #     self.stages.append(self.stage)
        #     cur += depths[i]
        # -------------------------------------------------------------------------------------------------
        # 某一个stage最后一个block加入vit
        # for i in range(4):
        #     self.stage = nn.Sequential()
        #     if i == 3:
        #         for j in range(depths[i]):
        #             if j < depths[i]-1:
        #                 st = Block(dim=dims[i], drop_path=dp_rates[cur + j],
        #                             layer_scale_init_value=layer_scale_init_value)
        #             else:
        #                 st = Transformer(dim=dims[i], heads=8, drop_path=dp_rates[cur + j],
        #                                   init_values=layer_scale_init_value)
        #             self.stage.add_module(str(j), st)
        #     else:
        #         self.stage = nn.Sequential(
        #                  *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
        #                 layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
        #                 )
        #     self.stages.append(self.stage)
        #     cur += depths[i]
        # -------------------------------------------------------------------------------------------------------
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-3, -2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        # return F.log_softmax(x, dim=1)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


def HSIVit(**kwargs):
    model = ConvNeXt(**kwargs)
    return model


dict = {'HSIVit': HSIVit}

if __name__ == "__main__":
    input = torch.randn(32, 1, 200, 27, 27)
    conv = model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=15)
    print('sss')
    out = conv(input)
    print(out)