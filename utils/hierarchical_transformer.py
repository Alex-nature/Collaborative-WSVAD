import torch
import torch.nn as nn
from collections import OrderedDict
import math
import numpy as np

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LocalAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, window_size: int, dropout: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def create_local_attention_mask(self, seq_len, device):
        """创建局部注意力的mask矩阵，确保在正确的设备上"""
        mask = torch.ones(seq_len, seq_len, device=device) * float('-inf')
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 0
        return mask

    def forward(self, x):
        seq_len = x.size(0)
        attn_mask = self.create_local_attention_mask(seq_len, x.device)
        
        x_ln = self.ln_1(x)
        attn_out = self.attn(x_ln, x_ln, x_ln, attn_mask=attn_mask, need_weights=False)[0]
        x = x + attn_out
        
        x = x + self.mlp(self.ln_2(x))
        return x

class GlobalAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def forward(self, x):
        x_ln = self.ln_1(x)
        attn_out = self.attn(x_ln, x_ln, x_ln, need_weights=False)[0]
        x = x + attn_out
        
        x = x + self.mlp(self.ln_2(x))
        return x

class HierarchicalTransformer(nn.Module):
    def __init__(self, 
                 width: int,
                 local_layers: int,
                 global_layers: int,
                 heads: int,
                 window_size: int,
                 dropout: float = 0.1,
                 use_local_residual: bool = False,
                 use_global_residual: bool = False):
        super().__init__()
        
        self.use_local_residual = use_local_residual
        self.use_global_residual = use_global_residual
        
        # dropout层
        self.dropout = nn.Dropout(p=dropout)
        
        # 局部注意力层
        self.local_layers = nn.ModuleList([
            LocalAttentionBlock(width, heads, window_size, dropout)
            for _ in range(local_layers)
        ])
        
        # 全局注意力层
        self.global_layers = nn.ModuleList([
            GlobalAttentionBlock(width, heads, dropout)
            for _ in range(global_layers)
        ])

    def forward(self, x: torch.Tensor):
        # x shape: [seq_len, batch, dim]
        x = self.dropout(x)
        
        # 保存输入用于残差连接
        original_input = x
        
        # 先进行局部建模
        for layer in self.local_layers:
            x = layer(x)
        
        # 局部建模后的残差连接
        if self.use_local_residual:
            x = x + original_input
            
        # 保存局部建模结果用于全局残差连接
        local_output = x
            
        # 再进行全局建模
        for layer in self.global_layers:
            x = layer(x)
            
        # 全局建模后的残差连接
        if self.use_global_residual:
            x = x + local_output
            
        return x 