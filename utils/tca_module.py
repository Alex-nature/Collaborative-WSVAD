"""
Temporal Context Aggregation (TCA) Module
==========================================

这是一个用于视频异常检测的时序上下文聚合模块，可以独立使用在其他项目中。

主要特点:
- 结合全局和局部注意力机制
- 自适应学习全局/局部特征的权重
- 支持距离感知的邻接矩阵
- 支持滑动窗口掩码

引用:
如果使用此模块，请引用原始论文: PEL4VAD

作者: 从 PEL4VAD 项目提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy.spatial.distance import pdist, squareform


class DistanceAdj(nn.Module):
    """
    距离感知邻接矩阵 (使用scipy实现)
    
    根据时间步之间的距离生成邻接矩阵，距离越近的时间步权重越大。
    使用可学习的参数来调整距离衰减的速度。
    
    Args:
        sigma (float): 初始权重参数
        bias (float): 初始偏置参数
    """
    def __init__(self, sigma, bias):
        super(DistanceAdj, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor(1))
        self.b = nn.Parameter(torch.FloatTensor(1))
        self.w.data.fill_(sigma)
        self.b.data.fill_(bias)

    def forward(self, batch_size, seq_len, device='cuda'):
        """
        生成距离邻接矩阵 (使用scipy计算曼哈顿距离)
        
        Args:
            batch_size (int): batch大小
            seq_len (int): 序列长度
            device (str): 设备类型 'cuda' 或 'cpu'
            
        Returns:
            torch.Tensor: shape [batch_size, seq_len, seq_len]
        """
        # 使用scipy计算时间步之间的曼哈顿距离（cityblock）
        positions = np.arange(seq_len).reshape(-1, 1)
        dist = pdist(positions, 'cityblock')  # 计算成对距离
        dist = squareform(dist)  # 转换为方阵
        
        # 转换为torch tensor并移到指定设备
        dist = torch.from_numpy(dist).float().to(device)
        
        # 使用可学习参数调整距离权重
        dist = torch.exp(-torch.abs(self.w * dist ** 2 - self.b))
        
        # 扩展到batch维度
        dist = dist.unsqueeze(0).repeat(batch_size, 1, 1)
        
        return dist


class TCA(nn.Module):
    """
    时序上下文聚合模块 (Temporal Context Aggregation)
    
    该模块结合了全局注意力和局部注意力两种机制:
    - 全局注意力: 可以关注整个序列的所有时间步
    - 局部注意力: 只能关注滑动窗口内的时间步
    - 通过可学习的alpha参数自适应融合两者
    
    Args:
        d_model (int): 输入特征维度
        dim_k (int): Query和Key的维度
        dim_v (int): Value的维度
        n_heads (int): 注意力头的数量
        norm (bool): 是否使用power norm和L2归一化，默认False
        
    Input:
        x: shape [batch_size, seq_len, d_model]
        mask: shape [n_heads, batch_size, seq_len, seq_len]
        adj: shape [batch_size, seq_len, seq_len] 或 None
        
    Output:
        shape [batch_size, seq_len, d_model]
    """
    def __init__(self, d_model, dim_k, dim_v, n_heads, norm=False):
        super(TCA, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k  # same as dim_q
        self.n_heads = n_heads
        self.norm = norm

        # Q, K, V 线性投影
        self.q = nn.Linear(d_model, dim_k)
        self.k = nn.Linear(d_model, dim_k)
        self.v = nn.Linear(d_model, dim_v)
        self.o = nn.Linear(dim_v, d_model)

        # 缩放因子
        self.norm_fact = 1 / math.sqrt(dim_k)
        
        # 可学习的全局-局部融合参数
        self.alpha = nn.Parameter(torch.tensor(0.))
        
        self.act = nn.Softmax(dim=-1)

    def forward(self, x, mask, adj=None):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征 [batch_size, seq_len, d_model]
            mask (torch.Tensor): 局部窗口掩码 [n_heads, batch_size, seq_len, seq_len]
            adj (torch.Tensor, optional): 距离邻接矩阵 [batch_size, seq_len, seq_len]
            
        Returns:
            torch.Tensor: 输出特征 [batch_size, seq_len, d_model]
        """
        # 生成 Q, K, V 并重塑为多头形式
        Q = self.q(x).view(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        K = self.k(x).view(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        V = self.v(x).view(-1, x.shape[0], x.shape[1], self.dim_v // self.n_heads)

        # 计算注意力分数（全局）
        if adj is not None:
            # 加入距离邻接矩阵的先验知识
            g_map = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.norm_fact + adj
        else:
            g_map = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.norm_fact
        
        # 克隆一份用于局部注意力
        l_map = g_map.clone()
        
        # 应用掩码：将窗口外的位置设为极小值
        l_map = l_map.masked_fill_(mask.data.eq(0), -1e9)

        # 应用softmax归一化
        g_map = self.act(g_map)  # 全局注意力权重
        l_map = self.act(l_map)  # 局部注意力权重
        
        # 计算加权和
        glb = torch.matmul(g_map, V).view(x.shape[0], x.shape[1], -1)  # 全局特征
        lcl = torch.matmul(l_map, V).view(x.shape[0], x.shape[1], -1)  # 局部特征

        # 自适应融合全局和局部特征
        alpha = torch.sigmoid(self.alpha)
        tmp = alpha * glb + (1 - alpha) * lcl
        
        # 可选的归一化
        if self.norm:
            tmp = torch.sqrt(F.relu(tmp)) - torch.sqrt(F.relu(-tmp))  # power norm
            tmp = F.normalize(tmp)  # l2 norm
            
        # 输出投影
        tmp = self.o(tmp).view(-1, x.shape[1], x.shape[2])
        
        return tmp


def generate_sliding_window_mask(window_size, temporal_scale, batch_size, n_heads=1, device='cuda'):
    """
    生成滑动窗口掩码
    
    每个时间步只能关注其周围window_size范围内的时间步。
    
    Args:
        window_size (int): 窗口大小
        temporal_scale (int): 时间序列长度
        batch_size (int): batch大小
        n_heads (int): 注意力头数量，默认1
        device (str): 设备类型 'cuda' 或 'cpu'
        
    Returns:
        torch.Tensor: shape [n_heads, batch_size, temporal_scale, temporal_scale]
                     值为1表示可以关注，值为0表示不能关注
                     
    Example:
        >>> mask = generate_sliding_window_mask(3, 5, 2, 1, 'cpu')
        >>> print(mask[0, 0])  # 查看第一个样本的掩码
        tensor([[1., 1., 0., 0., 0.],
                [1., 1., 1., 0., 0.],
                [0., 1., 1., 1., 0.],
                [0., 0., 1., 1., 1.],
                [0., 0., 0., 1., 1.]])
    """
    m = torch.zeros((temporal_scale, temporal_scale))
    w_len = window_size
    
    # 为每个时间步j生成其可关注的窗口
    for j in range(temporal_scale):
        for k in range(w_len):
            # 计算窗口内的位置，并处理边界
            target_idx = j - w_len // 2 + k
            clamped_idx = min(max(target_idx, 0), temporal_scale - 1)
            m[j, clamped_idx] = 1.
    
    # 扩展到batch和多头维度
    m = m.repeat(n_heads, batch_size, 1, 1).to(device)
    
    return m


# ========== 项目适配的包装类 ==========

class TCATransformerEncoder(nn.Module):
    """
    TCA时序编码器 - 适配本项目的接口
    
    兼容原TransformerEncoder的接口，可以直接替换使用
    
    Args:
        width (int): 特征维度 (对应d_model)
        layers (int): TCA层数
        heads (int): 注意力头数
        dropout (float): Dropout比率
        window_size (int): 滑动窗口大小
        use_distance_adj (bool): 是否使用距离邻接矩阵
        gamma (float): 距离衰减参数
        bias (float): 距离偏置参数
        use_norm (bool): 是否在TCA中使用归一化
        
    Input:
        x: shape [seq_len, batch_size, width] (Transformer格式)
        
    Output:
        shape [seq_len, batch_size, width]
    """
    def __init__(self, width, layers, heads, dropout=0.1, 
                 window_size=9, use_distance_adj=True,
                 gamma=0.6, bias=0.2, use_norm=True):
        super(TCATransformerEncoder, self).__init__()
        
        self.width = width
        self.layers = layers
        self.heads = heads
        self.window_size = window_size
        self.use_distance_adj = use_distance_adj
        
        # 创建多层TCA
        self.tca_layers = nn.ModuleList([
            TCA(d_model=width, 
                dim_k=width, 
                dim_v=width, 
                n_heads=heads,
                norm=use_norm)
            for _ in range(layers)
        ])
        
        # Layer Normalization (每层后)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(width) for _ in range(layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 距离邻接矩阵（可选）
        if use_distance_adj:
            self.distance_adj = DistanceAdj(gamma, bias)
        else:
            self.distance_adj = None
        
        print(f"\n{'='*60}")
        print(f"TCA时序编码器初始化完成:")
        print(f"  - 层数: {layers}")
        print(f"  - 注意力头数: {heads}")
        print(f"  - 特征维度: {width}")
        print(f"  - 窗口大小: {window_size}")
        print(f"  - 距离邻接矩阵: {'启用' if use_distance_adj else '禁用'}")
        print(f"  - 归一化: {'启用' if use_norm else '禁用'}")
        print(f"  - Dropout: {dropout}")
        print(f"{'='*60}\n")
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: shape [seq_len, batch_size, width]
            
        Returns:
            shape [seq_len, batch_size, width]
        """
        seq_len, batch_size, width = x.shape
        device = x.device
        
        # 转换为TCA期望的格式 [batch, seq_len, width]
        x = x.permute(1, 0, 2)
        
        # 生成滑动窗口掩码 (只生成一次)
        mask = generate_sliding_window_mask(
            self.window_size, seq_len, batch_size, 
            self.heads, device
        )
        
        # 生成距离邻接矩阵 (如果启用)
        if self.distance_adj is not None:
            adj = self.distance_adj(batch_size, seq_len, device)
        else:
            adj = None
        
        # 通过多层TCA
        for i, (tca_layer, layer_norm) in enumerate(zip(self.tca_layers, self.layer_norms)):
            # TCA注意力
            residual = x
            x_tca = tca_layer(x, mask, adj)
            
            # 残差连接 + Dropout
            x = residual + self.dropout(x_tca)
            
            # Layer Normalization
            x = layer_norm(x)
        
        # 转回Transformer格式 [seq_len, batch, width]
        x = x.permute(1, 0, 2)
        
        return x

