# miziha_torch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs):
        if self.drop_prob == 0. or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=inputs.dtype, device=inputs.device)
        random_tensor = torch.floor(random_tensor)
        output = inputs.div(keep_prob) * random_tensor
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_channels):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_channels, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        h, w = self.input_resolution
        b, _, c = x.shape
        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # 左上
        x1 = x[:, 1::2, 0::2, :]  # 左下
        x2 = x[:, 0::2, 1::2, :]  # 右上
        x3 = x[:, 1::2, 1::2, :]  # 右下
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(b, -1, 4 * c)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, dropout):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        nn.init.trunc_normal_(self.fc1.weight, std=.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.trunc_normal_(self.fc2.weight, std=.02)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def windows_partition(x, window_size):
    """
    将特征图分割成不重叠的窗口
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def windows_reverse(windows, window_size, H, W):
    """
    将窗口合并成特征图
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attention_dropout=0., dropout=0.):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置编码表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 获取相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # 初始化相对位置编码
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout=0.,
                 attention_dropout=0., droppath=0.):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attention_dropout=attention_dropout, dropout=dropout)

        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, dropout=dropout)

        # 计算注意力掩码
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = windows_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 分割成窗口
        x_windows = windows_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = windows_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # 逆循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinT(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout=0.,
                 attention_dropout=0., droppath=0., downsample=False):
        super().__init__()
        self.dim = in_channels
        self.out_channels = out_channels
        self.input_resolution = input_resolution

        self.blocks = nn.ModuleList()
        for i in range(2):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=in_channels, input_resolution=input_resolution,
                    num_heads=num_heads, window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    dropout=dropout, attention_dropout=attention_dropout,
                    droppath=droppath if not isinstance(droppath, list) else droppath[i]))

        self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)

        if downsample:
            self.downsample = PatchMerging(input_resolution, dim=in_channels, out_channels=out_channels)
        else:
            self.downsample = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, H * W, C)  # B H*W C

        for block in self.blocks:
            x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)
            B, _, C = x.shape
            x = x.view(B, self.input_resolution[0] // 2, self.input_resolution[1] // 2, C).permute(0, 3, 1, 2)
        else:
            x = x.view(B, self.input_resolution[0], self.input_resolution[1], C).permute(0, 3, 1, 2)
            x = self.cnn(x)
            
        return x


# 示例使用方法
if __name__ == "__main__":
    # 创建模型实例
    model = SwinT(
        in_channels=96,
        out_channels=192,
        input_resolution=(56, 56),
        num_heads=3,
        window_size=7,
        downsample=True
    )
    
    # 创建示例输入
    x = torch.randn(1, 96, 56, 56)
    
    # 前向传播
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")