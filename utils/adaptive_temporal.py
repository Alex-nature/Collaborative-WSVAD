"""
自适应时序建模模块
包含：
1. 特征变化率计算
2. 多尺度时间卷积
3. 自适应权重生成
4. 动态邻接矩阵构建
5. 图卷积层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def compute_change_rate(features):
    """
    计算相邻帧之间的特征变化率
    
    参数:
        features: [B, T, D] 或 [T, B, D] - 帧特征序列
        
    返回:
        delta: [B, T] - 变化率序列，第一帧设为0
        
    公式:
        cos_sim = (f_t · f_{t-1}) / (||f_t|| ||f_{t-1}||)
        delta_t = 1 - cos_sim
    """
    # 统一为 [B, T, D] 格式
    # 判断依据：特征维度通常在256-1024之间，而batch和时序长度通常<256
    # 如果最后一维 > 128，大概率已经是 [B, T, D] 或 [T, B, D]
    # 如果第一维远大于第二维（如256 > 4），则是 [T, B, D] 需要转换
    if features.dim() == 3:
        if features.shape[2] < 128:  # 最后一维太小，可能格式有误
            # 假设是 [B, D, T] 或其他错误格式，尝试调整
            if features.shape[1] > 128:  # 第二维可能是特征维
                features = features.permute(0, 2, 1)  # [B, D, T] -> [B, T, D]
        elif features.shape[0] > features.shape[1] * 2:  # T >> B，是 [T, B, D] 格式
            features = features.permute(1, 0, 2)  # [T, B, D] -> [B, T, D]
    
    B, T, D = features.shape
    
    # 归一化特征
    features_norm = F.normalize(features, p=2, dim=-1)  # [B, T, D]
    
    # 计算相邻帧的余弦相似度
    f_curr = features_norm[:, 1:, :]   # [B, T-1, D] 当前帧
    f_prev = features_norm[:, :-1, :]  # [B, T-1, D] 前一帧
    
    # 逐元素相乘再求和得到余弦相似度
    cos_sim = (f_curr * f_prev).sum(dim=-1)  # [B, T-1]
    
    # 变化率 = 1 - 余弦相似度
    delta = 1.0 - cos_sim  # [B, T-1]
    
    # 第一帧没有前驱，设为0或平均值
    delta_first = torch.zeros(B, 1, device=features.device, dtype=features.dtype)
    delta = torch.cat([delta_first, delta], dim=1)  # [B, T]
    
    return delta


class MultiScaleTemporalConv(nn.Module):
    """
    多尺度1D时间卷积模块
    使用不同kernel size (3, 5, 7) 捕获不同时间尺度的模式
    """
    
    def __init__(self, in_dim, out_dim, dropout=0.1):
        """
        参数:
            in_dim: 输入特征维度
            out_dim: 输出特征维度（每个分支）
            dropout: dropout比率
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 三个不同kernel size的卷积分支
        # kernel_size=3, padding=1 保持序列长度
        self.conv_k3 = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # kernel_size=5, padding=2 保持序列长度
        self.conv_k5 = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # kernel_size=7, padding=3 保持序列长度
        self.conv_k7 = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        参数:
            x: [B, T, D] 输入特征
            
        返回:
            y_k3, y_k5, y_k7: 三个分支的输出，每个为 [B, T, out_dim]
        """
        # Conv1d需要 [B, D, T] 格式
        x_conv = x.transpose(1, 2)  # [B, D, T]
        
        # 三路并行卷积
        y_k3 = self.conv_k3(x_conv).transpose(1, 2)  # [B, T, out_dim]
        y_k5 = self.conv_k5(x_conv).transpose(1, 2)  # [B, T, out_dim]
        y_k7 = self.conv_k7(x_conv).transpose(1, 2)  # [B, T, out_dim]
        
        return y_k3, y_k5, y_k7


class AdaptiveWeightGenerator(nn.Module):
    """
    自适应权重生成器
    根据当前帧特征和变化率，为多尺度卷积分支生成融合权重
    """
    
    def __init__(self, feature_dim, num_scales=3, hidden_dim=128):
        """
        参数:
            feature_dim: 输入特征维度
            num_scales: 尺度数量（默认3：k3, k5, k7）
            hidden_dim: MLP隐藏层维度
        """
        super().__init__()
        
        self.num_scales = num_scales
        
        # 输入: [特征向量, 变化率标量] -> 拼接后维度为 feature_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_scales)
        )
        
    def forward(self, features, delta):
        """
        参数:
            features: [B, T, D] 帧特征
            delta: [B, T] 变化率
            
        返回:
            weights: [B, T, num_scales] softmax归一化的权重
        """
        B, T, D = features.shape
        
        # 将delta扩展维度并拼接到特征
        delta_expanded = delta.unsqueeze(-1)  # [B, T, 1]
        mlp_input = torch.cat([features, delta_expanded], dim=-1)  # [B, T, D+1]
        
        # 通过MLP生成权重logits
        logits = self.mlp(mlp_input)  # [B, T, num_scales]
        
        # Softmax归一化，每帧的三个权重和为1
        weights = F.softmax(logits, dim=-1)  # [B, T, 3]
        
        return weights


class AdaptiveAdjacencyBuilder(nn.Module):
    """
    自适应邻接矩阵构建器
    根据变化率动态调整每个节点的时间窗口大小
    """
    
    def __init__(self, min_window=2, max_window=16):
        """
        参数:
            min_window: 最小窗口半径
            max_window: 最大窗口半径
        """
        super().__init__()
        
        self.min_window = min_window
        self.max_window = max_window
        
    def forward(self, delta, features=None, use_feature_sim=True):
        """
        根据变化率构建自适应邻接矩阵
        
        参数:
            delta: [B, T] 变化率序列
            features: [B, T, D] 特征（可选，用于计算特征相似度）
            use_feature_sim: 是否结合特征相似度
            
        返回:
            adj: [B, T, T] 邻接矩阵（已归一化）
            
        公式:
            w_t = round(min_w + (max_w - min_w) * (1 - delta_t / delta_max))
            连接条件: |i - j| <= w_t
        """
        B, T = delta.shape
        device = delta.device
        
        # 计算delta的最大值用于归一化
        delta_max = delta.max(dim=1, keepdim=True)[0] + 1e-6  # [B, 1]
        
        # 计算每帧的动态窗口大小
        # 变化小 -> delta小 -> 窗口大（平滑区域用大感受野）
        # 变化大 -> delta大 -> 窗口小（边界区域用小感受野保持精度）
        normalized_delta = delta / delta_max  # [B, T]
        window_size = self.min_window + (self.max_window - self.min_window) * (1.0 - normalized_delta)
        window_size = torch.round(window_size).long()  # [B, T]
        
        # 构建基于时间距离的邻接矩阵
        # 创建时间索引矩阵
        time_idx = torch.arange(T, device=device)  # [T]
        time_diff = torch.abs(time_idx.unsqueeze(0) - time_idx.unsqueeze(1))  # [T, T]
        
        # 初始化邻接矩阵
        adj = torch.zeros(B, T, T, device=device, dtype=torch.float32)
        
        # 对每个batch和每个时间步，根据其窗口大小设置邻接
        for b in range(B):
            for t in range(T):
                w_t = window_size[b, t].item()
                # 连接时间距离 <= w_t 的节点
                mask = time_diff[t] <= w_t
                adj[b, t, mask] = 1.0
        
        # 如果使用特征相似度进行加权
        if use_feature_sim and features is not None:
            # 计算特征余弦相似度
            features_norm = F.normalize(features, p=2, dim=-1)  # [B, T, D]
            # 相似度矩阵 [B, T, T]
            sim_matrix = torch.bmm(features_norm, features_norm.transpose(1, 2))
            sim_matrix = torch.clamp(sim_matrix, min=0.0)  # 只保留正相似度
            
            # 将相似度作为权重（仅在邻接存在的地方）
            adj = adj * sim_matrix
        
        # 添加自连接
        identity = torch.eye(T, device=device).unsqueeze(0).expand(B, -1, -1)
        adj = adj + identity
        
        # 行归一化（对称归一化也可）
        degree = adj.sum(dim=-1, keepdim=True) + 1e-6  # [B, T, 1]
        adj_normalized = adj / degree  # [B, T, T]
        
        return adj_normalized


class GraphConvLayer(nn.Module):
    """
    简单图卷积层
    实现公式: H' = σ(A H W)
    """
    
    def __init__(self, in_dim, out_dim, dropout=0.1, use_residual=True):
        """
        参数:
            in_dim: 输入特征维度
            out_dim: 输出特征维度
            dropout: dropout比率
            use_residual: 是否使用残差连接
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_residual = use_residual
        
        # 线性变换
        self.weight = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        # 残差投影（当维度不匹配时）
        if use_residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual_proj = None
            
        # 归一化和激活
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        """
        参数:
            x: [B, T, in_dim] 节点特征
            adj: [B, T, T] 邻接矩阵（已归一化）
            
        返回:
            out: [B, T, out_dim] 更新后的节点特征
        """
        # 线性变换
        h = self.weight(x)  # [B, T, out_dim]
        
        # 图卷积: A H
        h = torch.bmm(adj, h)  # [B, T, out_dim]
        h = h + self.bias
        
        # 残差连接
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(x)
            else:
                residual = x
            h = h + residual
        
        # 归一化和激活
        h = self.norm(h)
        h = F.relu(h, inplace=True)
        h = self.dropout(h)
        
        return h


class AdaptiveConvGCNTemporal(nn.Module):
    """
    自适应卷积+图卷积时序建模模块
    
    完整流程:
    1. 计算帧间变化率
    2. 多尺度时间卷积
    3. 根据变化率生成自适应融合权重
    4. 融合多尺度特征
    5. 构建自适应邻接矩阵
    6. 图卷积传播
    """
    
    def __init__(
        self,
        width,                    # 特征维度（必须与CLIP输出一致）
        tcn_out_dim=None,        # 时间卷积输出维度（默认与width相同）
        gcn_layers=2,            # 图卷积层数
        min_window=2,            # 最小窗口
        max_window=16,           # 最大窗口
        dropout=0.1,             # dropout比率
        use_feature_sim=True,    # 是否使用特征相似度
        weight_hidden_dim=128    # 权重生成器隐藏层维度
    ):
        super().__init__()
        
        self.width = width
        self.tcn_out_dim = tcn_out_dim if tcn_out_dim is not None else width
        self.gcn_layers = gcn_layers
        self.use_feature_sim = use_feature_sim
        
        # 1. 多尺度时间卷积
        self.multi_scale_conv = MultiScaleTemporalConv(
            in_dim=width,
            out_dim=self.tcn_out_dim,
            dropout=dropout
        )
        
        # 2. 自适应权重生成器
        self.weight_generator = AdaptiveWeightGenerator(
            feature_dim=width,
            num_scales=3,
            hidden_dim=weight_hidden_dim
        )
        
        # 3. 自适应邻接矩阵构建器
        self.adj_builder = AdaptiveAdjacencyBuilder(
            min_window=min_window,
            max_window=max_window
        )
        
        # 4. 图卷积层
        self.gcn_layers_list = nn.ModuleList()
        for i in range(gcn_layers):
            in_dim = self.tcn_out_dim if i == 0 else width
            out_dim = width  # 最终输出维度必须是width
            self.gcn_layers_list.append(
                GraphConvLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    use_residual=True
                )
            )
        
        # 5. 最终输出投影（确保维度）
        if self.tcn_out_dim != width:
            self.output_proj = nn.Linear(width, width)
        else:
            self.output_proj = nn.Identity()
            
    def forward(self, x):
        """
        参数:
            x: [T, B, D] 或 [B, T, D] 输入帧特征
            
        返回:
            out: [T, B, D] 或 [B, T, D] 输出特征（与输入格式一致）
        """
        # 记录原始格式
        # 判断：如果最后一维>128（特征维），第一维远大于第二维，则是[T,B,D]
        input_format_TBD = False
        if x.dim() == 3 and x.shape[2] > 128:  # 最后一维是特征维
            if x.shape[0] > x.shape[1] * 2:  # T >> B
                input_format_TBD = True
        
        # 统一转换为 [B, T, D]
        if input_format_TBD:
            x = x.permute(1, 0, 2)  # [T, B, D] -> [B, T, D]
        
        B, T, D = x.shape
        
        # === 步骤1: 计算特征变化率 ===
        delta = compute_change_rate(x)  # [B, T]
        
        # === 步骤2: 多尺度时间卷积 ===
        y_k3, y_k5, y_k7 = self.multi_scale_conv(x)  # 每个 [B, T, tcn_out_dim]
        
        # === 步骤3: 生成自适应融合权重 ===
        weights = self.weight_generator(x, delta)  # [B, T, 3]
        
        # === 步骤4: 融合多尺度特征 ===
        # 逐帧加权融合
        w_k3 = weights[:, :, 0:1]  # [B, T, 1]
        w_k5 = weights[:, :, 1:2]  # [B, T, 1]
        w_k7 = weights[:, :, 2:3]  # [B, T, 1]
        
        y_fused = w_k3 * y_k3 + w_k5 * y_k5 + w_k7 * y_k7  # [B, T, tcn_out_dim]
        
        # === 步骤5: 构建自适应邻接矩阵 ===
        adj = self.adj_builder(delta, y_fused, use_feature_sim=self.use_feature_sim)  # [B, T, T]
        
        # === 步骤6: 图卷积传播 ===
        h = y_fused
        for gcn_layer in self.gcn_layers_list:
            h = gcn_layer(h, adj)  # [B, T, width]
        
        # === 最终输出投影 ===
        out = self.output_proj(h)  # [B, T, width]
        
        # 恢复原始格式
        if input_format_TBD:
            out = out.permute(1, 0, 2)  # [B, T, D] -> [T, B, D]
        
        return out


# === 测试代码 ===
if __name__ == "__main__":
    # 测试参数
    batch_size = 4
    seq_len = 32
    feature_dim = 768  # CLIP ViT-B/16 输出维度
    
    # 创建模块
    model = AdaptiveConvGCNTemporal(
        width=feature_dim,
        gcn_layers=2,
        min_window=2,
        max_window=16,
        dropout=0.1
    )
    
    # 测试输入 [T, B, D]
    x = torch.randn(seq_len, batch_size, feature_dim)
    
    # 前向传播
    print("输入形状:", x.shape)
    out = model(x)
    print("输出形状:", out.shape)
    print("维度匹配:", out.shape == x.shape)
    
    # 测试 [B, T, D] 格式
    x_btd = torch.randn(batch_size, seq_len, feature_dim)
    out_btd = model(x_btd)
    print("\n[B,T,D]输入形状:", x_btd.shape)
    print("[B,T,D]输出形状:", out_btd.shape)
    print("维度匹配:", out_btd.shape == x_btd.shape)
    
    print("\n✓ 模块测试通过！")

