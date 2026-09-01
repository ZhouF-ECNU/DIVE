from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn
from thop import profile

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


# implement attention module for v-v self-attention
class Attention(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., settings=''):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # original self-attention for the original path
        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)

        # replace k & q by v
        k = v
        q = k

        # self-attention, higher temperate for resnets performs better
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (attn).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        x_ori = self.proj_drop(self.proj(x_ori))
        return [x, x_ori]


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# Projection head for disentanglement
class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = QuickGELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        transformed_x = self.fc2(self.gelu(self.fc1(x)))

        return transformed_x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
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
        if isinstance(self.attn, Attention):
            x = x.transpose(0, 1)
            x, x_ori = self.attn(x)
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x, whole = False, ffn = False):
        # dual paths for blocks deeper than "d"
        
        if isinstance(self.attn, Attention):
            if isinstance(x, list):
                if not ffn:
                    x, x_ori = x
                    x_res = self.attention(self.ln_1(x_ori))
                    x_res, x_ori_res = x_res
                    x_ori += x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res # skip ffn for the new path
                    return [x, x_ori]
                else:
                    x, x_ori_1 = x
                    x_res = self.attention(self.ln_1(x_ori_1))
                    x_res, x_ori_res = x_res
                    x_ori = x_ori_1 +  x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res # skip ffn for the new path
                    x = x_res + x_ori_1
                    x = x + self.mlp(self.ln_2(x))
                    return [x, x_ori]
            # start of dual path
            else:
                x_res = self.attention(self.ln_1(x))
                if isinstance(x_res, list):
                     x_res, x_ori_res = x_res
                     x_ori = x + x_ori_res
                     x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                     x += x_res
                     return [x, x_ori]

        # singl path before "d"
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x

class ResidualAttentionBlock_learnable_token(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None,
            text_layer=False, i = 0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        
        self.i = i
        self.compound_prompt_nctx = design_details['learnabel_text_embedding_length']
        self.text_layer = text_layer
        if i == 0:
            self.first_layer = True
        else:
           self.first_layer = False
    
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if isinstance(self.attn, Attention):
            x = x.transpose(0, 1)
            x, x_ori = self.attn(x)
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs):

        # dual paths for blocks deeper than "d"
        if isinstance(self.attn, Attention):
            x = inputs[0]
            if isinstance(x, list):
                x, x_ori = x
                x_res = self.attention(self.ln_1(x_ori))
                x_res, x_ori_res = x_res
                x_ori += x_ori_res
                x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                x += x_res # skip ffn for the new path
                return [x, x_ori]

            # start of dual path
            else:
                x_res = self.attention(self.ln_1(x))
                if isinstance(x_res, list):
                    x_res, x_ori_res = x_res
                    x_ori = x + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res
                    return [x, x_ori]

        # singl path before "d"
        else:
            x = inputs[0]
            compound_prompts_deeper = inputs[1]
            counter = inputs[2]
            if not self.first_layer:
                # First check if the ith layer needs compound prompts or not
                if not (counter > len(compound_prompts_deeper) - 1):
                    # Appending the learnable tokens in different way
                    # x -> [77, NCLS, DIM]
                    # First remove the learnable tokens from previous layer
                    prefix = x[:1, :, :]
                    suffix = x[1 + self.compound_prompt_nctx:, :, :]
                    textual_context = compound_prompts_deeper[counter]
                    textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                    # Add the learnable tokens of this layer with the input, replaced by previous
                    # layer learnable tokens
                    x = torch.cat([prefix, textual_context, suffix], dim=0)
                    # Once done, update the counter, so that the next time, it does not use same learnable tokens
                    counter += 1
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return [x, compound_prompts_deeper, counter]

# Text-Visual Cross Attention
class TextVisualCrossAttention(nn.Module):
    def __init__(self, d_model, n_head, text_dim=768, dropout=0.1, gate_scale=0.01):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.gate = gate_scale
        
        self.text_proj = nn.Linear(text_dim, d_model)

    def forward(self, x, text_features):
        L_vis, B, D = x.shape
        
        text_features = text_features.to(x.dtype)
        k_v = self.text_proj(text_features) 

        if k_v.dim() == 2:
            M, _ = k_v.shape
            k = v = k_v.unsqueeze(1).expand(-1, B, -1)
        else:
            k = v = k_v

        q = self.ln1(x)
        k = self.ln2(k)
        v = self.ln2(v)
        
        attn_output, _ = self.attn(query=q, key=k, value=v)
        attn_output = self.ln3(attn_output)

        return x + self.gate * attn_output

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, need_weights: bool = False, design_details = None,
                 text_layer = False, use_cross_attn = False, cross_attn_text_dim = 768):
        super().__init__()
        self.width = width
        self.layers = layers
        self.text_layer = text_layer
        self.design_deatails = design_details

        self.use_cross_attn = use_cross_attn
        self.cross_attn_layers = nn.ModuleDict()
        self.cross_attn_indices = []

        if self.use_cross_attn and not self.text_layer:
            self.cross_attn_indices = [layers - 2 - i for i in range(4)]
            for idx in self.cross_attn_indices:
                self.cross_attn_layers[str(idx)] = TextVisualCrossAttention(width, heads, text_dim=cross_attn_text_dim, gate_scale=0.01)

        if self.text_layer and (design_details is not None):
            self.resblocks = nn.ModuleList([ResidualAttentionBlock_learnable_token(width, heads, attn_mask, design_details, text_layer, i=i) for i in range(layers)])
        else:
            self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask,) for i in range(layers)])

    def ori_CLIP_with_patch_forward(self, x, out_layers):
        idx = 0
        out_tokens = []
        for r in self.resblocks:
            idx += 1
            x = r(x)
            if idx in out_layers:
                if isinstance(x, list):
                   out_tokens.append(x[1].clone())
                else:
                   out_tokens.append(x.clone())
        return [x, x], out_tokens

    def DIVE_forward(self, x, out_layers, ffn, text_dict_features=None, n_visual_ctx=0):
        idx = 0
        out_tokens = []
        for i, r in enumerate(self.resblocks):
            idx += 1
            x = r(x, ffn = ffn)
            
            # Cross-Attention Fusion
            if self.use_cross_attn and (i in self.cross_attn_indices) and (text_dict_features is not None):
                if isinstance(x, list):
                    # x[0] shape: (L, B, D), L = 1(CLS) + HW(patches) + n_visual_ctx(visual prompts)
                    cls_token = x[0][0:1, :, :]  # (1, B, D)
                    
                    if n_visual_ctx > 0:
                        patch_tokens = x[0][1:-n_visual_ctx, :, :]
                        visual_prompts = x[0][-n_visual_ctx:, :, :]  # (n_visual_ctx, B, D)
                    else:
                        patch_tokens = x[0][1:, :, :]
                        visual_prompts = None
                    
                    # Cross-attention is applied only to patch tokens
                    fused_patches = self.cross_attn_layers[str(i)](patch_tokens, text_dict_features)
                    
                    # [CLS] + [fused patches] + [visual prompts]
                    if visual_prompts is not None:
                        fused_visual = torch.cat([cls_token, fused_patches, visual_prompts], dim=0)
                    else:
                        fused_visual = torch.cat([cls_token, fused_patches], dim=0)
                    
                    x = [fused_visual, x[1]]
                else:
                    cls_token = x[0:1, :, :]
                    
                    if n_visual_ctx > 0:
                        patch_tokens = x[1:-n_visual_ctx, :, :]
                        visual_prompts = x[-n_visual_ctx:, :, :]
                    else:
                        patch_tokens = x[1:, :, :]
                        visual_prompts = None
                    
                    fused_patches = self.cross_attn_layers[str(i)](patch_tokens, text_dict_features)
                    
                    if visual_prompts is not None:
                        x = torch.cat([cls_token, fused_patches, visual_prompts], dim=0)
                    else:
                        x = torch.cat([cls_token, fused_patches], dim=0)

            if idx in out_layers:
                if isinstance(x, list):
                   out_tokens.append(x[0].clone())
                else:
                   out_tokens.append(x.clone())
        return x, out_tokens

    def forward(self, x: torch.Tensor, out_layers = [6, 12, 18, 24], DPAM_layer = None, ffn = False, text_dict_features = None, n_visual_ctx = 0):
        if not self.text_layer:
            out_tokens = []
            if DPAM_layer is None:
                [x, x], out_tokens = self.ori_CLIP_with_patch_forward(x, out_layers)
                return [x, x], out_tokens
            else:
                x, out_tokens = self.DIVE_forward(x, out_layers, ffn, text_dict_features=text_dict_features, n_visual_ctx=n_visual_ctx)
                return x, out_tokens
        
        # Text Encoder
        elif self.design_deatails is None:
            for idx, r in enumerate(self.resblocks):
                x = r(x)
            if isinstance(x, list): return x[0]
            return x
        elif self.design_deatails is not None:
            for idx, r in enumerate(self.resblocks):
                x = r(x)
            if isinstance(x, list):
                return x[0]
            else:
                return x

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, need_weights=True, use_cross_attn=True, cross_attn_text_dim=output_dim)
        self.attn = None
        self.embed_dim = width
        self.num_heads = heads

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state_all() 
        else:
            cuda_rng_state = None
        np_rng_state = np.random.get_state()
        import random
        py_rng_state = random.getstate()

        self.state_projection = ProjectionHead(input_dim=output_dim, hidden_dim=1024, output_dim=output_dim)
        self.semantic_projection = ProjectionHead(input_dim=output_dim, hidden_dim=1024, output_dim=output_dim)

        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
             torch.cuda.set_rng_state_all(cuda_rng_state)
        np.random.set_state(np_rng_state)
        random.setstate(py_rng_state)

    @torch.no_grad()
    def DAPM_replace(self, DPAM_layer):
        if DPAM_layer is not None:
            for i in range(1, DPAM_layer):
                self.attn = Attention(self.embed_dim, self.embed_dim, self.num_heads, True)
                self.attn.qkv.weight.data = self.transformer.resblocks[-i].attn.in_proj_weight.clone()
                self.attn.qkv.bias.data = self.transformer.resblocks[-i].attn.in_proj_bias.clone()
                self.attn.proj.weight.data = self.transformer.resblocks[-i].attn.out_proj.weight.clone()
                self.attn.proj.bias.data = self.transformer.resblocks[-i].attn.out_proj.bias.clone()
                self.transformer.resblocks[-i].attn = self.attn

    def forward(self, x: torch.Tensor, features_list, ori_patch = False, proj_use = True, DPAM_layer = None, ffn = False, text_dict_features = None, shared_ctx = None):
        x = self.conv1(x) 
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1) 
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) 
        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 1) ** 0.5)

        if side != new_side:
            pos_embed_spatial = self.positional_embedding[1:, :]  # (side*side, width)
            pos_embed_spatial = pos_embed_spatial.reshape(side, side, x.shape[-1]).permute(2, 0, 1).unsqueeze(0)  # (1, width, side, side)
            pos_embed_spatial = torch.nn.functional.interpolate(
                pos_embed_spatial, 
                size=(new_side, new_side), 
                mode='bilinear',
                align_corners=False
            )

            pos_embed_spatial = pos_embed_spatial.squeeze(0).permute(1, 2, 0).reshape(new_side * new_side, x.shape[-1])  # (new_side*new_side, width)
            pos = torch.cat([self.positional_embedding[:1, :], pos_embed_spatial], dim=0)  # (new_side*new_side+1, width)
        else:
            pos = self.positional_embedding
        
        pos = pos.to(x.dtype)
        x = x + pos
        
        # Add visual prompts after positional embedding and before ln_pre
        if shared_ctx is not None:
            # shared_ctx shape: (n_visual_ctx, visual_dim) 例如 (4, 1024)
            visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).to(x.dtype)  # (B, 4, 1024)
            # [CLS token + patch tokens + visual prompts]
            x = torch.cat([x, visual_ctx], dim=1)  # (B, 1+HW+4, 1024)
        
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # If visual prompts are added, these tokens need to be excluded during cross-attention
        n_visual_ctx = shared_ctx.shape[0] if shared_ctx is not None else 0
        
        [x, x_ori], patch_tokens = self.transformer(x, features_list, DPAM_layer = DPAM_layer, ffn = ffn, 
                                                    text_dict_features = text_dict_features, 
                                                    n_visual_ctx = n_visual_ctx)
        
        if True:
            patch_token_list = []
            for patch_token in patch_tokens:
                # patch_token shape: (L, B, D), L = 1(CLS) + HW(patches) + n_visual_ctx(visual prompts)
                if n_visual_ctx > 0:
                    # Remove the last n_visual_ctx tokens which are visual prompts
                    patch_token_without_prompts = patch_token[:-n_visual_ctx, :, :]
                else:
                    patch_token_without_prompts = patch_token
                
                patch_token_proj = self.ln_post(patch_token_without_prompts.permute(1, 0, 2)) @ self.proj  # LND -> NLD
                patch_token_list.append(patch_token_proj)
            patch_tokens = patch_token_list

            # x_ori[0, :, :] shape: (B, D), corresponds to the [CLS] token after the last block
            global_features = x_ori[0, :, :] @ self.proj  # (B, output_dim)

            # disentanglement projection
            state_features = self.state_projection(global_features)
            semantic_features = self.semantic_projection(global_features) 

            return global_features+state_features, state_features, global_features+semantic_features, semantic_features, patch_tokens

        return x


class DIVE(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details = None
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(), text_layer=True, design_details=design_details
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
            
    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, feature_list = [], ori_patch = False, proj_use = True, DPAM_layer = None, ffn = False, text_dict_features = None, shared_ctx = None):
        return self.visual(image.type(self.dtype), feature_list, ori_patch = ori_patch, proj_use = proj_use, 
                          DPAM_layer = DPAM_layer, ffn = ffn, text_dict_features = text_dict_features, 
                          shared_ctx = shared_ctx)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_text_learn(self, prompts, tokenized_prompts, deep_compound_prompts_text = None, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = prompts + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        if deep_compound_prompts_text is None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, deep_compound_prompts_text, 0])
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
    
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text