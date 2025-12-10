from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from utils.clip import clip
from utils.prompt_net import PromptLearner
from utils.tca_module import TCATransformerEncoder


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout: float = 0, attn_mask: torch.Tensor = None):
        super(TransformerEncoder, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, dropout, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class Model(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        visual_length: int,
        prompt_prefix: int,
        prompt_postfix: int,
        visual_width: int,
        visual_head: int,
        visual_layers: int,
        device: str,
        # TCA参数
        use_tca: bool = True,
        tca_window_size: int = 9,
        tca_dropout: float = 0.1,
        use_distance_adj: bool = True,
        tca_gamma: float = 0.6,
        tca_bias: float = 0.2,
        tca_norm: bool = True,
        # 双流架构开关
        use_dual_stream: bool = False
    ):
        super().__init__()

        # 保存基本参数
        self.visual_length = visual_length
        self.embed_dim = embed_dim
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device
        self.use_tca = use_tca
        self.use_dual_stream = use_dual_stream

        # 加载CLIP模型
        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        # 初始化视觉编码器 - 根据配置选择TCA或标准Transformer
        if use_tca:
            print("\n🌊 使用TCA (Temporal Context Aggregation) 时序编码器")
            self.temporal = TCATransformerEncoder(
                width=visual_width,        # 例如 512
                layers=visual_layers,
                heads=visual_head,
                dropout=tca_dropout,
                window_size=tca_window_size,
                use_distance_adj=use_distance_adj,
                gamma=tca_gamma,
                bias=tca_bias,
                use_norm=tca_norm
            )
        else:
            print("\n⚡ 使用标准Transformer时序编码器")
            self.temporal = TransformerEncoder(
                width=visual_width,
                layers=visual_layers,
                heads=visual_head
            )

        # 初始化提示学习器
        if use_dual_stream:
            # 双流模式：创建两个独立的PromptLearner（直接使用，不需要适配）
            self.actor_prompt_learner = PromptLearner()
            self.action_prompt_learner = PromptLearner()
            # 为了向后兼容，保留prompt_learner属性（但不会被使用）
            self.prompt_learner = None
        else:
            # 单流提示学习器（保持向后兼容）
            self.prompt_learner = PromptLearner()
            self.actor_prompt_learner = None
            self.action_prompt_learner = None

        # 初始化其他组件
        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.global_text_prompt_embeddings = nn.Embedding(77, self.embed_dim)
        self.dtype = self.clipmodel.dtype
        # 占位符 "X" 的 token id（clip.tokenize 输出中位置1）
        self.x_token_id = clip.tokenize("X")[0, 1].item()
        
        # 双流架构：不再需要投影层和融合模块，直接使用原始特征
        if use_dual_stream:
            print("\n🎭 启用双流架构 (Actor-Action Disentangled Prompting)")
            print("   - 直接使用TCA前后的原始特征（512维）")
        
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def encode_video(self, images, return_dual_stream=False):
        """
        编码视频特征
        
        Args:
            images: 输入视频特征 [batch, seq_len, dim]
            return_dual_stream: 是否返回双流特征（Actor和Action）
            
        Returns:
            如果 return_dual_stream=False: 返回单一特征 [batch, seq_len, dim]
            如果 return_dual_stream=True: 返回 (actor_features, action_features)
                - actor_features: TCA之前的特征（空间特征）[batch, seq_len, dim]
                - action_features: TCA之后的特征（时序特征）[batch, seq_len, dim]
        """
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        # Actor特征：TCA之前的特征（空间特征，未经过时序处理）
        actor_features = images.permute(1, 0, 2)  # [batch, seq_len, dim]
        
        # Action特征：TCA之后的特征（时序增强特征）
        x = self.temporal(images)
        action_features = x.permute(1, 0, 2)  # [batch, seq_len, dim]
        
        if return_dual_stream:
            return actor_features, action_features
        else:
            # 保持向后兼容，默认返回单一特征
            return action_features

    def get_tokenized_classnames(self, classes):
        if self.use_dual_stream:
            # 双流模板：前后各4个占位符，便于分别替换
            # 结构: [SOS] X X X X {class_name} X X X X . [EOS]
            prompts = [" ".join(["X"] * 4) + " " + name + " " + " ".join(["X"] * 4) + "." for name in classes]
        else:
            # 单流模板：保持原始的 4 个占位符
            # 结构: [SOS] X X X X {class_name} . [EOS]
            prompts = [" ".join(["X"] * 4) + " " + name + "." for name in classes]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = self.clipmodel.token_embedding(tokenized_prompts.to(self.device)).type(self.dtype)

        return embedding, tokenized_prompts

    def encode_text_prompt(self, text, visual, actor_features=None, action_features=None):
        """
        生成动态文本特征
        
        Args:
            text: 类别名称列表
            visual: 视觉特征（单流模式使用）[batch, seq_len, 512]
            actor_features: Actor特征（双流模式使用）[batch, seq_len, 512] - TCA之前的特征
            action_features: Action特征（双流模式使用）[batch, seq_len, 512] - TCA之后的特征
        """
        classes = [name.replace("_", " ") for name in text]
        class_tokens = torch.cat([clip.tokenize(p) for p in classes])
        class_tokens = class_tokens.to(self.device)
        with torch.no_grad():
            class_features = self.clipmodel.encode_text_original(class_tokens)
            class_features = class_features / class_features.norm(dim=-1, keepdim=True)

        context_embedding = class_features
        prompt_vectors, tokenized_prompts = self.get_tokenized_classnames(classes)

        # 根据是否使用双流架构选择不同的提示学习方式
        if self.use_dual_stream and actor_features is not None and action_features is not None:
            # 双流模式：分别使用两个独立的PromptLearner生成各自的context
            # actor_features: [batch, seq_len, 512] - TCA之前的特征（空间特征）
            # action_features: [batch, seq_len, 512] - TCA之后的特征（时序特征）
            # 分别经过原先单流的文本处理过程生成各自的context
            actor_context = self.actor_prompt_learner(context_embedding, actor_features)  # [1, 4, 512]
            action_context = self.action_prompt_learner(context_embedding, action_features)  # [1, 4, 512]
        else:
            # 单流模式：使用原始提示学习器
            context = self.prompt_learner(context_embedding, visual)
            actor_context = None
            action_context = None
        
        # 构建prompt_vectors
        if self.use_dual_stream and actor_context is not None and action_context is not None:
            # 双流模式：动态查找前/后4个"X"的位置，避免多词类名导致索引偏移
            token_ids = tokenized_prompts  # [num_classes, 77]
            x_mask = token_ids == self.x_token_id

            if torch.any(x_mask.sum(dim=1) < 8):
                raise RuntimeError("Tokenized prompt中占位符X数量不足，无法替换为actor/action context。")

            # 收集每个类别中"X"的索引，前4个用于actor，后4个用于action
            x_positions = x_mask.nonzero(as_tuple=False)  # [8*num_classes, 2] -> (cls_idx, pos)
            actor_indices = []
            action_indices = []
            num_classes = token_ids.shape[0]
            for cls_idx in range(num_classes):
                pos = x_positions[x_positions[:, 0] == cls_idx][:, 1]
                actor_indices.append(pos[:4])
                action_indices.append(pos[-4:])
            actor_indices = torch.stack(actor_indices, dim=0)   # [num_classes, 4]
            action_indices = torch.stack(action_indices, dim=0) # [num_classes, 4]

            # 准备context并写回对应位置
            actor_context_4 = actor_context[0, :4, :].unsqueeze(0).expand(num_classes, -1, -1)   # [B,4,512]
            action_context_4 = action_context[0, :4, :].unsqueeze(0).expand(num_classes, -1, -1) # [B,4,512]

            # scatter到对应token位置
            prompt_vectors = prompt_vectors.clone()
            actor_idx_exp = actor_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            action_idx_exp = action_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            prompt_vectors.scatter_(1, actor_idx_exp, actor_context_4)
            prompt_vectors.scatter_(1, action_idx_exp, action_context_4)
        else:
            # 单流模式：保持原始逻辑，使用4-token context替换4个X
            prompt_vectors = torch.cat(
                [
                    prompt_vectors[:, :1],
                    context[0].unsqueeze(0).expand(prompt_vectors.shape[0], -1, -1),
                    prompt_vectors[:, 1 + context.shape[1]:],
                ],
                dim=1,
            )

        # 生成动态文本特征
        text_features = self.clipmodel.encode_text(prompt_vectors, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features

    def forward(self, visual, text, lengths, is_training=True):
        if self.use_dual_stream:
            # ========== 双流架构 ==========
            # 1. 获取双流特征（不融合，直接使用）
            actor_features_raw, action_features_raw = self.encode_video(visual, return_dual_stream=True)
            # actor_features_raw: [batch, seq_len, 512] - TCA之前的特征（空间特征）
            # action_features_raw: [batch, seq_len, 512] - TCA之后的特征（时序特征）
            
            # 2. 使用Action特征（TCA之后的特征）计算logits（与单流模式保持一致）
            visual_features = action_features_raw  # [batch, seq_len, 512]
            visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
            
            # 3. 获取文本特征（直接使用原始特征，不进行投影）
            text_features = self.encode_text_prompt(text, None, actor_features_raw, action_features_raw)
            
            # 5. 扩展维度以匹配批次大小
            text_features = text_features.unsqueeze(0).expand(
                visual_features.shape[0], text_features.shape[0], text_features.shape[1])
            text_features = text_features.permute(0, 2, 1)
            
            # 6. 计算logits
            logits = visual_features_norm @ text_features.type(visual_features_norm.dtype)
            logits = logits * self.clipmodel.logit_scale.exp()
            
        else:
            # ========== 单流架构（保持向后兼容）==========
            # 1. 获取视觉特征
            visual_features = self.encode_video(visual)
            visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
            
            # 2. 获取文本特征
            text_features = self.encode_text_prompt(text, visual_features)
            
            # 3. 扩展维度以匹配批次大小
            text_features = text_features.unsqueeze(0).expand(
                visual_features.shape[0], text_features.shape[0], text_features.shape[1])
            text_features = text_features.permute(0, 2, 1)
            
            # 4. 计算logits
            logits = visual_features_norm @ text_features.type(visual_features_norm.dtype)
            logits = logits * self.clipmodel.logit_scale.exp()
        
        return logits
