import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

class KMeans:
    def __init__(self, n_clusters=4, max_iter=10, tol=1e-6, rng_seed=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.rng_seed = rng_seed

    def fit_predict(self, x):
        torch.manual_seed(self.rng_seed)
        N, C = x.shape
        idx = torch.randperm(N, device=x.device)[:self.n_clusters]
        centroids = x[idx].clone()
        for _ in range(self.max_iter):
            distances = torch.cdist(x, centroids)
            cluster_idx = torch.argmin(distances, dim=1)
            new_centroids = torch.stack([
                x[cluster_idx == k].mean(dim=0) if (cluster_idx == k).sum() > 0 else centroids[k]
                for k in range(self.n_clusters)
            ])
            if torch.all(torch.norm(new_centroids - centroids, dim=1) < self.tol):
                break
            centroids = new_centroids
        return cluster_idx, centroids

class CASA(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, num_k=4, residual=True, act='relu', mean_type='s', n_mlp=1):
        super(CASA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ks = ks
        self.num_k = num_k
        self.residual = residual
        self.mean_type = mean_type

        self.w = nn.Parameter(torch.Tensor(out_channels, in_channels, ks, ks))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

        self.kmeans = KMeans(n_clusters=num_k, max_iter=10, tol=1e-6, rng_seed=0)

        inner_dims = 32 * n_mlp
        if mean_type == 's':
            _in_c = in_channels
        elif mean_type == 'c':
            _in_c = ks * ks
        else:
            _in_c = in_channels + ks * ks

        self.get_kernel = nn.Sequential(
            nn.Linear(_in_c, inner_dims),
            nn.ReLU(inplace=True),
            nn.Linear(inner_dims, ks * ks),
            nn.Sigmoid()
        )

        self.get_bias = nn.Sequential(
            nn.Linear(_in_c, inner_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(inner_dims // 2, out_channels)
        )

        self.proj_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True) if act == 'relu' else nn.GELU()

    def feat_mean(self, x, mean_type):
        if mean_type == 's':
            return reduce(x, 'b c h w k1 k2 -> b (h w) c', 'mean')
        elif mean_type == 'c':
            return reduce(x, 'b c h w k1 k2 -> b (h w) (k1 k2)', 'mean')
        else:
            xs = reduce(x, 'b c h w k1 k2 -> b (h w) c', 'mean')
            xc = reduce(x, 'b c h w k1 k2 -> b (h w) (k1 k2)', 'mean')
            return torch.cat([xs, xc], dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_in = x
        x = self.proj_in(x)

        x_patches = F.unfold(x, kernel_size=self.ks, padding=self.ks // 2)
        L = x_patches.shape[-1]
        H = W = int(L ** 0.5)  # 假设是正方形图像块
        x_patches = rearrange(x_patches, 'b (c k1 k2) (h w) -> b c h w k1 k2', h=H, w=W, k1=self.ks, k2=self.ks)

        x_mean = self.feat_mean(x_patches, self.mean_type)
        cluster_idx, centroids = self.kmeans.fit_predict(x_mean[0])
        cluster_idx = cluster_idx.unsqueeze(0).expand(B, -1)
        centroids = centroids.unsqueeze(0).expand(B, -1, -1)

        out = torch.zeros(B, self.out_channels, H, W, device=x.device)

        for i in range(self.num_k):
            mask = cluster_idx.eq(i).float().view(B, 1, H, W)
            if mask.sum() == 0:
                continue
            centroid_i = centroids[:, i]
            weight_mod = self.get_kernel(centroid_i).view(B, 1, 1, self.ks * self.ks)
            weight = self.w.view(1, self.out_channels, self.in_channels, self.ks * self.ks) * weight_mod
            weight = weight.view(B * self.out_channels, self.in_channels, self.ks, self.ks)
            bias = self.get_bias(centroid_i)
            x_unfolded = F.unfold(x, kernel_size=self.ks, padding=self.ks // 2)
            x_unfolded = x_unfolded.view(B, self.in_channels, self.ks * self.ks, H * W)
            out_i = torch.einsum('b c k n, b o c k -> b o n', x_unfolded, weight.view(B, self.out_channels, self.in_channels, self.ks * self.ks))
            out_i = out_i.view(B, self.out_channels, H, W) + bias.view(B, self.out_channels, 1, 1)
            out += out_i * mask

        out = self.act(out)
        if self.residual and self.in_channels == self.out_channels:
            out = out + x_in

        return out

class GS_SCA_CASA(nn.Module):
    def __init__(self, in_channels, top_k=64, num_k=4):
        super(GS_SCA_CASA, self).__init__()
        self.in_channels = in_channels
        self.top_k = top_k

        self.f = CASA(in_channels, in_channels, ks=3, num_k=4, residual=False, act='relu', mean_type='s', n_mlp=1)
        self.g = CASA(in_channels, in_channels, ks=3, num_k=4, residual=False, act='relu', mean_type='s', n_mlp=1)
        self.h = CASA(in_channels, in_channels, ks=3, num_k=4, residual=True, act='relu', mean_type='s', n_mlp=1)

        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def mean_variance_norm(self, feat, eps=1e-5):
        N, C = feat.size()[:2]
        feat_var = feat.view(N, C, -1).var(dim=2, unbiased=False) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return (feat - feat_mean) / feat_std

    def forward(self, x):
        B, C, H, W = x.shape
        normed = self.mean_variance_norm(x)

        f_proj = self.f(normed).view(B, C, -1)
        g_proj = self.g(normed).view(B, C, -1)
        h_proj = self.h(x).view(B, C, -1)

        attention = torch.bmm(f_proj.permute(0, 2, 1), g_proj)

        k = min(self.top_k, attention.shape[-1])
        if k < 1:
            raise ValueError(f"top_k={self.top_k} is too small for feature size {H}x{W}")

        topk = torch.topk(attention, k, dim=-1)[0]
        threshold = topk[:, :, -1].unsqueeze(-1).expand_as(attention)
        sparse_mask = attention >= threshold
        attention = attention.masked_fill(~sparse_mask, float('-inf'))
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(h_proj, attention.permute(0, 2, 1)).view(B, C, H, W)
        out = self.out_conv(out)

        gate = self.gate_conv(x)
        fused = gate * out + (1 - gate) * x
        return fused