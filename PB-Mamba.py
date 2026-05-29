import time
import math
from functools import partial
from typing import Optional, Callable
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from CASA import SCA_CASA # 加在头部
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MAconv(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


def generate_s_shape_indices(H, W, stripe_width, direction="lr", device="cuda"):
    """
    生成 S 形扫描路径的索引
    direction: "lr" (左上->右下), "rl" (右上->左下), "lr_t" (左下->右上), "rl_t" (右下->左上)
    """
    # 计算条纹数量
    num_stripes = W // stripe_width
    if W % stripe_width != 0:
        num_stripes += 1

    # 动态调整首尾条纹宽度（首尾条纹宽度减半）
    stripe_widths = [stripe_width] * num_stripes
    if num_stripes > 1:
        stripe_widths[0] = max(1, stripe_width // 2)  # 首条宽度减半
        stripe_widths[-1] = W - sum(stripe_widths[:-1])  # 调整最后一条宽度

    indices = torch.arange(H * W, dtype=torch.long, device=device).reshape(H, W)

    # 根据方向调整初始张量
    if direction == "rl":
        indices = torch.flip(indices, dims=[1])  # 水平翻转
    elif direction == "lr_t":
        indices = torch.flip(indices, dims=[0])  # 垂直翻转
    elif direction == "rl_t":
        indices = torch.flip(indices, dims=[0, 1])  # 水平+垂直翻转

    # 生成 S 形路径
    s_shape_indices = []
    for stripe_idx in range(num_stripes):
        start_col = sum(stripe_widths[:stripe_idx])
        end_col = start_col + stripe_widths[stripe_idx]
        stripe = indices[:, start_col:end_col]

        # 每隔一行反转（S 形路径）
        for row in range(H):
            if row % 2 == 1:
                stripe[row, :] = torch.flip(stripe[row, :], dims=[0])
        s_shape_indices.append(stripe.flatten())

    return torch.cat(s_shape_indices)

def sscan(inp, stripe_width, shift_len=0, d_state=16, d_inner=None, dt_rank=16, dt_min=0.001, dt_max=0.1):
    B, C, H, W = inp.shape
    L = H * W

    # 确保输入张量在 CUDA 上
    inp = inp.to("cuda")

    # 四向 S 形扫描路径
    directions = ["lr", "rl", "lr_t", "rl_t"]
    paths = []
    indices_list = []  # 保存索引
    for direction in directions:
        # 生成 S 形路径索引
        indices = generate_s_shape_indices(H, W, stripe_width, direction, device=inp.device)

        # 应用 Shift-Stripe 偏移
        if shift_len > 0:
            indices = torch.roll(indices, shifts=shift_len, dims=0)

        # 按照索引重排输入张量
        inp_flat = inp.reshape(B, C, -1)
        path = torch.index_select(inp_flat, dim=-1, index=indices)
        paths.append(path)
        indices_list.append(indices)

    # 组合 4 个方向的路径
    xs = torch.stack(paths, dim=1)  # (B, 4, C, L)

    # 组合 4 个方向的索引
    scan_ids = torch.stack(indices_list, dim=0)  # (4, L)

    # 初始化 SSM 参数（简化版）
    if d_inner is None:
        d_inner = C
    dt_rank = math.ceil(d_inner / 16) if dt_rank == "auto" else dt_rank

    # 动态调整 dt（基于 stripe_width）
    dt_scale = 1.0 / stripe_width  # 示例：条纹越窄，dt 越小
    dt = torch.exp(
        torch.rand(d_inner, device=inp.device) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    ) * dt_scale
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    dts = inv_dt.unsqueeze(0).unsqueeze(-1).expand(B, d_inner, L)

    # 初始化其他 SSM 参数，确保在 CUDA 上
    As = -torch.exp(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32, device=inp.device))).view(1, d_state).expand(d_inner, -1)
    Bs = torch.ones(B, d_state, L, device=inp.device)
    Cs = torch.ones(B, d_state, L, device=inp.device)
    Ds = torch.ones(d_inner, device=inp.device)

    # 应用 SSM 处理（selective_scan_fn）
    out_paths = []
    for k in range(4):
        path = xs[:, k, :, :].float()  # (B, C, L)
        out = selective_scan_fn(
            path, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=None,
            delta_softplus=True,
            return_last_state=False,
        )
        out_paths.append(out)

    # 组合 SSM 输出
    out_xs = torch.stack(out_paths, dim=1)  # (B, 4, C, L)
    return scan_ids, out_xs

def sscan_4d(inp, stripe_width, shift_len=0, d_state=16, d_inner=None, dt_rank=16, dt_min=0.001, dt_max=0.1):
    return sscan(inp, stripe_width, shift_len, d_state, d_inner, dt_rank, dt_min, dt_max)

def inverse_ids_generate(origin_ids, K=4):
    inverse_ids = torch.argsort(origin_ids, dim=-1)
    return inverse_ids

def mair_ids_generate(inp_shape, stripe_width=2, K=4):
    inp_b, inp_c, inp_h, inp_w = inp_shape
    # 确保 inp_idx 在 CUDA 上
    inp_idx = torch.arange(inp_h * inp_w, device="cuda").reshape(1, 1, inp_h, inp_w).float()
    xs_scan_ids, _ = sscan_4d(inp_idx, stripe_width, d_state=16, d_inner=1)
    xs_inverse_ids = inverse_ids_generate(xs_scan_ids, K=K)
    return xs_scan_ids, xs_inverse_ids

def mair_ids_scan(inp, xs_scan_ids, bkdl=False, K=4):
    B, C, H, W = inp.shape
    L = H * W
    xs_scan_ids = xs_scan_ids.reshape(K, L)

    y1 = torch.index_select(inp.reshape(B, 1, C, -1), -1, xs_scan_ids[0])
    y2 = torch.index_select(inp.reshape(B, 1, C, -1), -1, xs_scan_ids[1])
    y3 = torch.index_select(inp.reshape(B, 1, C, -1), -1, xs_scan_ids[2])
    y4 = torch.index_select(inp.reshape(B, 1, C, -1), -1, xs_scan_ids[3])

    if bkdl:
        inp_flatten = torch.cat((y1, y2, y3, y4), dim=1)
    else:
        inp_flatten = torch.cat((y1, y2, y3, y4), dim=1).reshape(B, 4, -1)
    return inp_flatten

def mair_ids_inverse(inp, xs_scan_ids, shape=None):
    B, K, _, L = inp.shape
    xs_scan_ids = xs_scan_ids.reshape(K, L)
    if not shape:
        y1 = torch.index_select(inp[:, 0, :], -1, xs_scan_ids[0]).reshape(B, -1, L)
        y2 = torch.index_select(inp[:, 1, :], -1, xs_scan_ids[1]).reshape(B, -1, L)
        y3 = torch.index_select(inp[:, 2, :], -1, xs_scan_ids[2]).reshape(B, -1, L)
        y4 = torch.index_select(inp[:, 3, :], -1, xs_scan_ids[3]).reshape(B, -1, L)
    else:
        B, C, H, W = shape
        y1 = torch.index_select(inp[:, 0, :], -1, xs_scan_ids[0]).reshape(B, -1, H, W)
        y2 = torch.index_select(inp[:, 1, :], -1, xs_scan_ids[1]).reshape(B, -1, H, W)
        y3 = torch.index_select(inp[:, 2, :], -1, xs_scan_ids[2]).reshape(B, -1, H, W)
        y4 = torch.index_select(inp[:, 3, :], -1, xs_scan_ids[3]).reshape(B, -1, H, W)
    return torch.cat((y1, y2, y3, y4), dim=1)


class SSAModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // 8)
        self.fc2 = nn.Linear(dim // 8, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        y = x.view(B, C, -1).mean(dim=-1)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(B, C, 1, 1)
        return x * y


class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2., dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random",
                 dt_scale=1.0, dt_init_floor=1e-4, dropout=0., conv_bias=True, bias=False, device=None, dtype=None,
                 stripe_width=2, shift_len=0, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.stripe_width = stripe_width
        self.shift_len = shift_len

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.ssa = SSAModule(self.d_inner)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        stripe_width = self.stripe_width if self.stripe_width <= min(H, W) else 2

        # 生成  扫描路径
        xs_scan_ids, xs_inverse_ids = mair_ids_generate(inp_shape=(1, 1, H, W), stripe_width=stripe_width, K=K)

        # 应用 扫描路径
        xs = mair_ids_scan(x, xs_scan_ids, bkdl=False, K=K)

        # 投影到低维空间
        x_dbl = F.conv1d(xs.reshape(B, -1, L), self.x_proj_weight.reshape(-1, self.d_inner, 1), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = F.conv1d(dts.reshape(B, -1, L), self.dt_projs_weight.reshape(K * self.d_inner, -1, 1), groups=K)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        y = mair_ids_inverse(out_y, xs_inverse_ids, shape=(B, -1, H, W))

        y1, y2, y3, y4 = torch.chunk(y, 4, dim=1)
        y = y1 + y2 + y3 + y4

        y = self.ssa(y)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups

    x = x.view(batch_size, height, width, groups, channels_per_group)

    x = torch.transpose(x, 3, 4).contiguous()

    x = x.view(batch_size, height, width, -1)

    return x


class SS_Conv_SSM(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim // 2)
        self.self_attention = SS2D(d_model=hidden_dim // 2, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

        self.conv33conv33conv11 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim // 2),
            MAconv(hidden_dim // 2, hidden_dim // 2, k=3, s=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            MAconv(hidden_dim // 2, hidden_dim // 2, k=3, s=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=1, stride=1),
            nn.ReLU()
        )

    def forward(self, input: torch.Tensor):
        input_left, input_right = input.chunk(2, dim=-1)
        x = self.drop_path(self.self_attention(self.ln_1(input_right)))
        input_left = input_left.permute(0, 3, 1, 2).contiguous()
        input_left = self.conv33conv33conv11(input_left)
        input_left = input_left.permute(0, 2, 3, 1).contiguous()
        output = torch.cat((input_left, x), dim=-1)
        output = channel_shuffle(output, groups=2)
        return output + input


class VSSLayer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SS_Conv_SSM(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSLayer_up(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SS_Conv_SSM(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 4, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)
        self.sca = SCA_CASA(in_channels=96, top_k=64)

        self.ape = False
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # if num_classes > 0:
        #     self.head = nn.Sequential(
        #         nn.Linear(self.num_features, self.num_features // 2),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(self.num_features // 2, num_classes)
        #     )
        # else:
        #     self.head = nn.Identity()

        self.apply(self._init_weights)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [1, 96, 56, 56]
        x = self.sca(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.forward_backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.head(x)
        return x


medmamba_t = VSSM(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=6).to("cuda")
medmamba_s = VSSM(depths=[2, 2, 8, 2], dims=[96, 192, 384, 768], num_classes=6).to("cuda")
medmamba_b = VSSM(depths=[2, 2, 12, 2], dims=[128, 256, 512, 1024], num_classes=6).to("cuda")

data = torch.randn(1, 3, 224, 224).to("cuda")

print(medmamba_t(data).shape)