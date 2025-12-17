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

        # 初始化两套提示学习器：actor 和 action
        self.prompt_learner_actor = PromptLearner()  # 用于TCA前的特征（actor）
        self.prompt_learner_action = PromptLearner()  # 用于TCA后的特征（action）

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
            actor_features: TCA前的特征 [batch, seq_len, dim]
            action_features: TCA后的特征 [batch, seq_len, dim]
        """
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images_with_pos = images.permute(1, 0, 2) + frame_position_embeddings
        
        # actor: TCA前的特征（仅位置编码）
        actor_features = images_with_pos.permute(1, 0, 2)  # [batch, seq_len, dim]
        
        # 经过时序编码器处理得到action
        x = self.temporal(images_with_pos)
        action_features = x.permute(1, 0, 2)  # [batch, seq_len, dim]
        
        return actor_features, action_features

    def get_tokenized_classnames(self, classes):
        prompts = [" ".join(["X"] * 4) + " " + name + " " + " ".join(["X"] * 4) + "." for name in classes]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = self.clipmodel.token_embedding(tokenized_prompts.to(self.device)).type(self.dtype)

        return embedding, tokenized_prompts

    def encode_text_prompt(self, text, actor, action):
        """
        生成动态文本特征
        
        Args:
            text: 类别名称列表
            actor: TCA前的特征 [batch, seq_len, dim]
            action: TCA后的特征 [batch, seq_len, dim]
        """
        classes = [name.replace("_", " ") for name in text]
        class_tokens = torch.cat([clip.tokenize(p) for p in classes])
        class_tokens = class_tokens.to(self.device)
        with torch.no_grad():
            class_features = self.clipmodel.encode_text_original(class_tokens)
            class_features = class_features / class_features.norm(dim=-1, keepdim=True)

        context_embedding = class_features
        prompt_vectors, tokenized_prompts = self.get_tokenized_classnames(classes)

        # 使用两套提示学习器分别生成actor和action的上下文
        context_actor = self.prompt_learner_actor(context_embedding, actor)
        context_action = self.prompt_learner_action(context_embedding, action)
        
        # actor context放在前半部分，action context放在后半部分
        # 模板格式：XXXX[class]XXXX，结构是：[SOS] + [前4个X] + [class] + [后4个X] + [.] + [EOS] + [PAD...]
        # context长度是4（n_ctx=4）
        n_ctx = context_actor.shape[1]  # context长度，通常是4
        prefix_end = 1 + n_ctx  # SOS(1) + 前4个X(4) = 5，即class tokens的起始位置
        
        # 对于每个class，分别处理其prompt
        result_vectors = []
        for i, cls in enumerate(classes):
            # 为当前class构建prompt并tokenize来确定结构
            test_prompt = " ".join(["X"] * 4) + " " + cls + " " + " ".join(["X"] * 4) + "."
            test_tokens = clip.tokenize(test_prompt)
            current_tokens = tokenized_prompts[i]  # [seq_len]
            
            # 找到EOS token的位置（49407是CLIP的EOS token ID）
            eos_positions = (current_tokens == 49407).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                eos_idx = eos_positions[0].item()
                # 结构：[SOS] + [前4个X] + [class] + [后4个X] + [.] + [EOS]
                # 所以：后4个X结束位置 = EOS位置 - 1（"."的位置）- 1 = EOS - 2
                # 但实际上，我们需要找到"."的位置
                # 简化：假设"."在EOS之前1个位置，后4个X在"."之前4个位置
                suffix_start = eos_idx - 1 - n_ctx  # "."位置 - 后4个X长度
                class_end = suffix_start  # class tokens的结束位置
            else:
                # 如果没有找到EOS，使用test_tokens的长度来计算
                # 假设有效长度是test_tokens的实际长度（去掉pad）
                valid_len = (test_tokens[0] != 0).sum().item()  # 非pad token的数量
                if valid_len > 0:
                    eos_idx = valid_len - 1
                    suffix_start = eos_idx - 1 - n_ctx
                    class_end = suffix_start if suffix_start > prefix_end else prefix_end + 4
                else:
                    # 默认值
                    class_end = prefix_end + 10  # 假设class tokens长度为10
                    suffix_start = class_end + n_ctx
            
            # 确保索引有效
            class_end = max(prefix_end, min(class_end, prompt_vectors.shape[1]))
            suffix_start = max(class_end, min(suffix_start, prompt_vectors.shape[1]))
            
            # 拼接：SOS + actor_context + class + action_context + 剩余
            parts = [
                prompt_vectors[i:i+1, :1],  # SOS (1)
                context_actor[0].unsqueeze(0),  # actor context (n_ctx)
                prompt_vectors[i:i+1, prefix_end:class_end],  # class tokens (可变)
                context_action[0].unsqueeze(0),  # action context (n_ctx)
                prompt_vectors[i:i+1, suffix_start:],  # 剩余tokens (. + EOS + PAD)
            ]
            
            # 验证长度匹配
            original_len = prompt_vectors.shape[1]
            new_len = sum(p.shape[1] for p in parts)
            if new_len != original_len:
                # 如果长度不匹配，调整最后一个部分
                diff = original_len - (new_len - parts[-1].shape[1])
                if diff > 0:
                    parts[-1] = prompt_vectors[i:i+1, suffix_start:suffix_start+diff]
                else:
                    # 如果还是不对，使用原始长度
                    parts[-1] = prompt_vectors[i:i+1, suffix_start:]
            
            prompt_vec = torch.cat(parts, dim=1)
            
            # 最终验证
            if prompt_vec.shape[1] != original_len:
                raise ValueError(
                    f"Prompt vector length mismatch for class {i} ({cls}): "
                    f"expected {original_len}, got {prompt_vec.shape[1]}"
                )
            
            result_vectors.append(prompt_vec)
        
        prompt_vectors = torch.cat(result_vectors, dim=0)

        # 生成动态文本特征
        text_features = self.clipmodel.encode_text(prompt_vectors, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features

    def forward(self, visual, text, lengths, is_training=True):
        # ========== 单流架构 ==========
        # 1. 获取视觉特征（actor和action）
        actor_features, action_features = self.encode_video(visual)
        # 使用action特征进行对齐（后续部分保持不变）
        action_features_norm = action_features / action_features.norm(dim=-1, keepdim=True)
        
        # 2. 获取文本特征（使用actor和action）
        text_features = self.encode_text_prompt(text, actor_features, action_features)
        
        # 3. 扩展维度以匹配批次大小
        text_features = text_features.unsqueeze(0).expand(
            action_features.shape[0], text_features.shape[0], text_features.shape[1])
        text_features = text_features.permute(0, 2, 1)
        
        # 4. 计算logits（使用action特征）
        logits = action_features_norm @ text_features.type(action_features_norm.dtype)
        logits = logits * self.clipmodel.logit_scale.exp()
        
        return logits
