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
        tca_norm: bool = True
    ):
        super().__init__()

        # 保存基本参数
        self.visual_length = visual_length
        self.embed_dim = embed_dim
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device
        self.use_tca = use_tca

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
        self.prompt_learner_pos = PromptLearner()

        # 初始化其他组件
        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.global_text_prompt_embeddings = nn.Embedding(77, self.embed_dim)
        self.dtype = self.clipmodel.dtype
        
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def encode_video(self, images):
        """
        编码视频特征
        
        Args:
            images: 输入视频特征 [batch, seq_len, dim]
            
        Returns:
            返回单一特征 [batch, seq_len, dim]
        """
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings
        
        # 经过时序编码器处理
        x = self.temporal(images)
        visual_features = x.permute(1, 0, 2)  # [batch, seq_len, dim]
        
        return visual_features

    def get_tokenized_classnames(self, classes):
        prompts = [" ".join(["X"] * 4) + " " + name + "." for name in classes]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = self.clipmodel.token_embedding(tokenized_prompts.to(self.device)).type(self.dtype)

        return embedding, tokenized_prompts

    def encode_text_prompt(self, text, visual):
        """
        生成动态文本特征
        
        Args:
            text: 类别名称列表
            visual: 视觉特征 [batch, seq_len, dim]
        """
        classes = [name.replace("_", " ") for name in text]
        class_tokens = torch.cat([clip.tokenize(p) for p in classes])
        class_tokens = class_tokens.to(self.device)
        with torch.no_grad():
            class_features = self.clipmodel.encode_text_original(class_tokens)
            class_features = class_features / class_features.norm(dim=-1, keepdim=True)

        context_embedding = class_features
        prompt_vectors, tokenized_prompts = self.get_tokenized_classnames(classes)

        # 使用提示学习器生成动态上下文
        context = self.prompt_learner_pos(context_embedding, visual)
        
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
        # ========== 单流架构 ==========
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
