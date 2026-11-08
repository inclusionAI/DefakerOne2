import os
import cv2
import json
import timm
import torch
import argparse
import numpy as np
import albumentations as A
import torch.nn.functional as F
from torch import nn
from peft import LoraConfig, get_peft_model
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from typing import Any, Dict, List, Optional

parser = argparse.ArgumentParser()
parser.add_argument('--seg_module', type=str, required=True) # e.g. preview/seg_head.pth
parser.add_argument('--input_image', type=str, required=True) # your_image.jpg
parser.add_argument('--save_dir', type=str, required=True) # output_dir
parser.add_argument('--vit_head_dir', type=str, required=True) # e.g. preview/vit_head
parser.add_argument('--mllm_dir', type=str, required=True) # e.g. preview
parser.add_argument('--img_size', type=int, default=1024) # fixed 1024
args = parser.parse_args()


def resize(input, size=None, scale_factor=None, mode='bilinear', align_corners=False):
    return F.interpolate(input, size=size, scale_factor=scale_factor,
                         mode=mode, align_corners=align_corners)


def build_norm_layer(norm_cfg, num_features):
    if norm_cfg is None:
        return None, None
    norm_type = norm_cfg.get('type', 'BN')
    requires_grad = norm_cfg.get('requires_grad', True)
    if norm_type in ['BN', 'SyncBN']:
        layer = nn.BatchNorm2d(num_features)
    elif norm_type == 'GN':
        layer = nn.GroupNorm(norm_cfg.get('num_groups', 32), num_features)
    else:
        raise NotImplementedError(f'Unsupported norm type: {norm_type}')
    for p in layer.parameters():
        p.requires_grad = requires_grad
    return ('bn' if norm_type in ['BN', 'SyncBN'] else 'gn'), layer


def build_activation_layer(act_cfg):
    if act_cfg is None:
        return None
    act_type = act_cfg.get('type', 'ReLU')
    inplace = act_cfg.get('inplace', True)
    if act_type == 'ReLU':
        return nn.ReLU(inplace=inplace)
    if act_type == 'LeakyReLU':
        return nn.LeakyReLU(act_cfg.get('negative_slope', 0.01), inplace=inplace)
    if act_type == 'GELU':
        return nn.GELU()
    raise NotImplementedError(f'Unsupported activation type: {act_type}')


def _extract_features(visual_output: Any) -> Optional[torch.Tensor]:
    if isinstance(visual_output, torch.Tensor):
        return visual_output
    pooler = getattr(visual_output, 'pooler_output', None)
    if isinstance(pooler, torch.Tensor):
        return pooler
    last_hidden = getattr(visual_output, 'last_hidden_state', None)
    if isinstance(last_hidden, torch.Tensor):
        return last_hidden
    if isinstance(visual_output, (tuple, list)) and visual_output:
        first = visual_output[0]
        if isinstance(first, torch.Tensor):
            return first
    return None


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias='auto', conv_cfg=None, norm_cfg=None,
                 act_cfg=dict(type='ReLU'), inplace=True,
                 order=('conv', 'norm', 'act')):
        super().__init__()
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        self.order = order
        if bias == 'auto':
            bias = not self.with_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.with_norm:
            self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            if act_cfg_.get('type', 'ReLU') not in ['Tanh', 'PReLU', 'Sigmoid', 'GELU']:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

    @property
    def norm(self):
        return None if self.norm_name is None else getattr(self, self.norm_name)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and self.with_norm and norm:
                x = self.norm(x)
            elif layer == 'act' and self.with_activation and activate:
                x = self.activate(x)
        return x


class PPM(nn.ModuleList):
    def __init__(self, pool_scales, in_channels, channels,
                 conv_cfg=None, norm_cfg=None,
                 act_cfg=dict(type='ReLU'), align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        for ps in pool_scales:
            self.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                ConvModule(in_channels, channels, 1,
                           conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            ))

    def forward(self, x):
        return [resize(p(x), size=x.shape[2:], mode='bilinear',
                       align_corners=self.align_corners) for p in self]


class UPerHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes,
                 pool_scales=(1, 2, 3, 6), dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 in_index=(0, 1, 2, 3), align_corners=False):
        super().__init__()
        self.in_channels = list(in_channels)
        self.channels = channels
        self.in_index = list(in_index)
        self.align_corners = align_corners

        self.psp_modules = PPM(pool_scales, self.in_channels[-1], channels,
                               conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                               act_cfg=act_cfg, align_corners=align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * channels,
            channels, 3, padding=1,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for c in self.in_channels[:-1]:
            self.lateral_convs.append(ConvModule(c, channels, 1, conv_cfg=conv_cfg,
                                                 norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False))
            self.fpn_convs.append(ConvModule(channels, channels, 3, padding=1,
                                             conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                             act_cfg=act_cfg, inplace=False))
        self.fpn_bottleneck = ConvModule(len(self.in_channels) * channels, channels,
                                         3, padding=1, conv_cfg=conv_cfg,
                                         norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None

    def psp_forward(self, inputs):
        x = inputs[-1]
        outs = [x] + self.psp_modules(x)
        return self.bottleneck(torch.cat(outs, dim=1))

    def forward(self, inputs):
        inputs = [inputs[i] for i in self.in_index]
        laterals = [lc(inputs[i]) for i, lc in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs))
        n = len(laterals)
        for i in range(n - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + resize(laterals[i],
                                                        size=laterals[i - 1].shape[2:], mode='bilinear',
                                                        align_corners=self.align_corners)
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(n - 1)]
        fpn_outs.append(laterals[-1])
        for i in range(n - 1, 0, -1):
            fpn_outs[i] = resize(fpn_outs[i], size=fpn_outs[0].shape[2:],
                                 mode='bilinear', align_corners=self.align_corners)
        feat = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        if self.dropout is not None:
            feat = self.dropout(feat)
        return self.conv_seg(feat)


class VitSegHead(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: Optional[int] = None,
                 out_dim: int = 2048,
                 init_bias: float = -1.0,
                 dropout: float = 0.0,
                 merge_size: int = 2):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim or in_dim)
        self.out_dim = int(out_dim)
        self.init_bias = float(init_bias)
        self.dropout_p = float(dropout or 0.0)
        self.merge_size = max(1, int(merge_size))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(self.in_dim, self.hidden_dim, kernel_size=1)
        self.act = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(self.hidden_dim, self.out_dim, kernel_size=1)
        nn.init.constant_(self.conv2.bias, self.init_bias)
        self._pending: List[Dict[str, Any]] = []
        self.last_logits: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'in_dim': self.in_dim,
            'hidden_dim': self.hidden_dim,
            'out_dim': self.out_dim,
            'init_bias': self.init_bias,
            'dropout': self.dropout_p,
            'merge_size': self.merge_size,
        }

    def _fold_to_chunks(self, feats: torch.Tensor, grid_thw: Optional[torch.Tensor]) -> Optional[List[torch.Tensor]]:
        if grid_thw is None:
            return None
        try:
            grid = grid_thw.detach().to('cpu').tolist()
        except Exception:
            return None
        m = self.merge_size
        chunks: List[torch.Tensor] = []
        offset = 0
        n_tokens = feats.shape[0]
        for row in grid:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                return None
            t, h, w = (int(v) for v in row)
            gh, gw = max(1, h // m), max(1, w // m)
            cnt = t * gh * gw
            if cnt <= 0 or offset + cnt > n_tokens:
                return None
            chunk = feats[offset:offset + cnt].reshape(t, gh, gw, -1)
            chunks.append(chunk.permute(0, 3, 1, 2).contiguous())
            offset += cnt
        if not chunks or offset != n_tokens:
            return None
        return chunks

    def to_fp32(self) -> 'VitSegHead':
        for module in (self.conv1, self.conv2):
            if module.weight.dtype != torch.float32:
                module.weight.data = module.weight.data.to(torch.float32)
            if module.bias is not None and module.bias.dtype != torch.float32:
                module.bias.data = module.bias.data.to(torch.float32)
        return self

    def forward(self, feats: torch.Tensor, grid_thw: Optional[torch.Tensor] = None) -> torch.Tensor:
        if feats.dim() == 3:
            feats = feats.reshape(-1, feats.shape[-1])
        x = feats.to(self.conv1.weight.dtype)
        chunks = self._fold_to_chunks(x, grid_thw)
        if chunks is None:
            chunks = [x.mean(dim=0, keepdim=True).reshape(1, -1, 1, 1)]
        pooled = [self.pool(c) for c in chunks]
        x = pooled[0] if len(pooled) == 1 else torch.cat(pooled, dim=0)
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        logits = x.flatten(1).float()
        self.last_logits = logits
        return logits


class VitHeadFeatures(nn.Module):
    def __init__(self, device):
        super().__init__()
        head_dir = args.vit_head_dir
        base_model = args.mllm_dir
        base = Qwen3_5ForConditionalGeneration.from_pretrained(base_model, torch_dtype=torch.bfloat16)
        self.visual = base.model.visual.to(device)
        del base
        self.processor = AutoProcessor.from_pretrained(head_dir).image_processor
        with open(os.path.join(head_dir, 'vit_bce_head_config.json'), encoding='utf-8') as f:
            config = json.load(f)
        if config['out_dim'] != 2048:
            raise ValueError('Segmentation fusion requires a 2048-channel ViT head.')
        self.head = VitSegHead(**config).to(device)
        self.head.load_state_dict(load_file(os.path.join(head_dir, 'vit_bce_head.safetensors')))
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, paths):
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        inputs = self.processor(images=images, return_tensors='pt')
        device = next(self.visual.parameters()).device
        pixels = inputs['pixel_values'].to(device=device, dtype=next(self.visual.parameters()).dtype)
        grid = inputs['image_grid_thw'].to(device)
        features = _extract_features(self.visual(pixels, grid_thw=grid))
        if features is None:
            raise RuntimeError('Qwen vision encoder returned no usable features.')
        vectors = F.relu(self.head(features, grid))
        if vectors.shape != (len(paths), 2048):
            raise ValueError(f'Expected one 2048-dimensional vector per image, got {tuple(vectors.shape)}')
        return vectors


class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model('vit_large_patch16_dinov3.lvd1689m', pretrained=True, features_only=True)
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["attn.qkv"],
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        self.vit = get_peft_model(self.vit, lora_config)
        self.decode_head = UPerHead(
            in_channels=(256, 512, 1024, 2048), channels=512, num_classes=2,
            pool_scales=(1, 2, 3, 6), dropout_ratio=0.1, conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'), in_index=(0, 1, 2, 3), align_corners=False)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1), nn.ReLU(),
                          nn.UpsamplingBilinear2d(scale_factor=2.0),
                          nn.Conv2d(512, 256, 3, 1, 1),
                          nn.UpsamplingBilinear2d(scale_factor=2.0)),
            nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1),
                          nn.UpsamplingBilinear2d(scale_factor=2.0)),
            nn.Identity(),
            nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(1024, 2048, 3, 1, 1))
        ])

    def forward(self, x, vit_features):
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            x = self.vit.model.model.forward_intermediates(
                x, [5, 11, 17, 23], return_prefix_tokens=True, norm=True)[1]
        with torch.amp.autocast(device_type='cuda', enabled=False):
            features = [self.convs[i](x[i][0].float()) for i in range(4)]
            if vit_features.shape != features[-1].shape[:2]:
                raise ValueError('ViT head vectors must match the last feature map batch and channels.')
            features[-1] = features[-1] + vit_features.to(features[-1])[:, :, None, None]
            return self.decode_head(features)


def build_transform(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        A.ToTensorV2(),
    ])


def run_infer():
    device = torch.device('cuda:0')
    model = ViT().to(device)

    state_dict = torch.load(args.seg_module, map_location='cpu')
    new_sd = {k.replace('module.', '', 1) if k.startswith('module.') else k: v
              for k, v in state_dict.items()}
    model.load_state_dict(new_sd)
    model.eval()
    vit_head = VitHeadFeatures(device)

    img = cv2.imread(args.input_image)
    if img is None:
        raise FileNotFoundError(f'Cannot read input image: {args.input_image}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    transform = build_transform(args.img_size)
    img_tensor = transform(image=img)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor, vit_head([args.input_image]))
        pred = F.interpolate(
            logits.float(), size=(h, w), mode='bilinear', align_corners=False
        ).squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)

    pred = (pred * 255).astype(np.uint8)

    os.makedirs(args.save_dir, exist_ok=True)
    save_name = os.path.splitext(os.path.basename(args.input_image))[0] + '.png'
    save_path = os.path.join(args.save_dir, save_name)
    cv2.imwrite(save_path, pred)
    print(f'Prediction saved: {save_path}')


if __name__ == '__main__':
    run_infer()
