from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision import models


@dataclass
class WalkStats:
    id_mean: float
    id_std: float
    lum_mean: float
    lum_std: float
    composite_min: float
    composite_max: float


class ChunkedClipEncoder(nn.Module):

    def __init__(self, model_name: str):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
        self.encoder.requires_grad_(False).eval()
        self.hidden_size = self.encoder.config.hidden_size

    @torch.no_grad()
    def forward(self, prompts: Sequence[str], device: torch.device) -> Tensor:
        max_content = self.tokenizer.model_max_length - 2
        outputs: List[Tensor] = []
        for prompt in prompts:
            ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
            chunks = ids.split(max_content) or [ids]
            pooled = []
            for chunk in chunks:
                encoded = self.tokenizer.prepare_for_model(
                    chunk.tolist(),
                    add_special_tokens=True,
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                if input_ids.ndim == 1:
                    input_ids = input_ids.unsqueeze(0)
                attention = encoded.get("attention_mask")
                if attention is not None and attention.ndim == 1:
                    attention = attention.unsqueeze(0)
                result = self.encoder(
                    input_ids=input_ids,
                    attention_mask=None if attention is None else attention.to(device),
                )
                pooled.append(result.pooler_output)
            outputs.append(torch.stack(pooled).mean(0).squeeze(0))
        return torch.stack(outputs)


class FiLM(nn.Module):
    def __init__(self, text_dim: int, channels: int):
        super().__init__()
        self.affine = nn.Linear(text_dim, 2 * channels)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: Tensor, text: Tensor) -> Tensor:
        gamma, beta = self.affine(text).chunk(2, dim=1)
        return (1 + gamma[:, :, None, None]) * x + beta[:, :, None, None]


class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, text_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(),
        )
        self.film = FiLM(text_dim, out_ch)

    def forward(self, x: Tensor, text: Tensor) -> Tensor:
        return self.film(self.net(x), text)


class SemanticFiLMUNet(nn.Module):
    def __init__(
        self, in_classes: int, out_classes: int, clip_dim: int, text_dim: int = 512, base: int = 64
    ):
        super().__init__()
        self.text_projection = nn.Sequential(
            nn.Linear(clip_dim, text_dim), nn.SiLU(), nn.Linear(text_dim, text_dim)
        )
        self.e1, self.e2, self.e3 = (
            Block(in_classes, base, text_dim),
            Block(base, base * 2, text_dim),
            Block(base * 2, base * 4, text_dim),
        )
        self.mid = Block(base * 4, base * 8, text_dim)
        self.d3, self.d2, self.d1 = (
            Block(base * 12, base * 4, text_dim),
            Block(base * 6, base * 2, text_dim),
            Block(base * 3, base, text_dim),
        )
        self.head = nn.Conv2d(base, out_classes, 1)

    def forward(self, control: Tensor, clip_embedding: Tensor) -> Tensor:
        t = self.text_projection(clip_embedding)
        e1 = self.e1(control, t)
        e2 = self.e2(F.max_pool2d(e1, 2), t)
        e3 = self.e3(F.max_pool2d(e2, 2), t)
        m = self.mid(F.max_pool2d(e3, 2), t)
        d3 = self.d3(
            torch.cat(
                [F.interpolate(m, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1
            ),
            t,
        )
        d2 = self.d2(
            torch.cat(
                [F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1
            ),
            t,
        )
        d1 = self.d1(
            torch.cat(
                [F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1
            ),
            t,
        )
        return self.head(d1)


def dice_loss(
    logits: Tensor, target: Tensor, classes: int, eps: float = 1e-6, ignore_index=255
) -> Tensor:
    valid = target != ignore_index
    probs = logits.softmax(1) * valid[:, None]
    onehot = F.one_hot(target.masked_fill(~valid, 0), classes).permute(0, 3, 1, 2).float()
    onehot = onehot * valid[:, None]
    intersection = (probs * onehot).sum((0, 2, 3))
    denominator = (probs + onehot).sum((0, 2, 3))
    return 1 - ((2 * intersection + eps) / (denominator + eps)).mean()


def semantic_indicators(
    probs: Tensor, road_class: int, landuse_classes: Sequence[int], tile_area_km2: float
) -> Tuple[Tensor, Tensor]:

    probs = probs.float()
    road = probs[:, road_class : road_class + 1]
    kernel = road.new_tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])[None, None]
    degree = F.conv2d(road, kernel, padding=1)

    junctions = (road * torch.sigmoid(8.0 * (degree - 2.5))).sum((1, 2, 3))
    intersection_density = junctions / tile_area_km2
    shares = probs[:, landuse_classes].mean((2, 3))
    shares = shares / shares.sum(1, keepdim=True).clamp_min(1e-8)
    lum = -(shares * shares.clamp_min(1e-8).log()).sum(1) / math.log(len(landuse_classes))
    return intersection_density, lum


def walkability_1_20(indensity: Tensor, lum: Tensor, stats: WalkStats) -> Tensor:
    composite = (indensity - stats.id_mean) / max(stats.id_std, 1e-8) + (
        lum - stats.lum_mean
    ) / max(stats.lum_std, 1e-8)
    normalized = (composite - stats.composite_min) / max(
        stats.composite_max - stats.composite_min, 1e-8
    )
    return 1 + 19 * normalized.clamp(0, 1)


class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        self.blocks = nn.ModuleList([features[:4], features[4:9], features[9:16], features[16:23]])
        self.requires_grad_(False).eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        x = ((prediction + 1) / 2 - self.mean) / self.std
        y = ((target + 1) / 2 - self.mean) / self.std
        loss = x.new_zeros(())
        for block in self.blocks:
            x, y = block(x), block(y)
            loss = loss + F.l1_loss(x, y)
        return loss / len(self.blocks)


@torch.no_grad()
def encode_clip_sequences(
    tokenizer, text_encoder: nn.Module, prompts: Sequence[str], device: torch.device
) -> Tensor:

    max_content = tokenizer.model_max_length - 2
    prompt_embeddings = []
    for prompt in prompts:
        raw_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
        chunk_embeddings = []
        for chunk in raw_ids.split(max_content):
            encoded = tokenizer.prepare_for_model(
                chunk.tolist(),
                add_special_tokens=True,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            chunk_embeddings.append(text_encoder(input_ids)[0])
        prompt_embeddings.append(torch.stack(chunk_embeddings).mean(0).squeeze(0))
    return torch.stack(prompt_embeddings)


def read_labels(path, palette, size, classes, id_map=None, ignore_index=None, ignore_colors=()):
    with Image.open(path) as image:
        image = image.resize((size, size), Image.Resampling.NEAREST)
        array = np.asarray(image)
        if image.mode == "P":
            array = np.asarray(image.convert("RGB"))
    if array.ndim == 3:
        rgb = array[..., :3]
        labels = np.full(rgb.shape[:2], -1, dtype=np.int64)
        for class_id, color in palette.items():
            labels[np.all(rgb == color, axis=-1)] = class_id
        if ignore_index is not None:
            for color in ignore_colors:
                labels[np.all(rgb == color, axis=-1)] = ignore_index
        if (labels < 0).any():
            raise ValueError(f"Unknown mask colors in {path}; supply an explicit crosswalk")
    else:
        labels = array.astype(np.int64)
        if id_map is not None:
            mapped = np.full(labels.shape, -1, dtype=np.int64)
            for source_id, class_id in id_map.items():
                mapped[labels == source_id] = class_id
            if ignore_index is not None:
                mapped[labels == ignore_index] = ignore_index
            if (mapped < 0).any():
                raise ValueError(f"Unmapped mask values in {path}")
            labels = mapped
    valid = labels != ignore_index if ignore_index is not None else np.ones(labels.shape, bool)
    if (labels[valid] < 0).any() or (labels[valid] >= classes).any():
        raise ValueError(f"Mask class outside [0, {classes - 1}]: {path}")
    return torch.from_numpy(labels.copy()).long()


def colorize(labels, palette):
    colors = torch.tensor(
        [palette[i] for i in range(len(palette))], device=labels.device, dtype=torch.float32
    )
    valid = labels != 255
    if ((labels[valid] < 0) | (labels[valid] >= len(palette))).any():
        raise ValueError("Invalid semantic labels")
    rgb = colors[labels.masked_fill(~valid, 0)]
    rgb[~valid] = rgb.new_tensor([241, 240, 235])
    return rgb.movedim(-1, -3) / 255.0


def collate(samples):
    return {
        key: (
            [sample[key] for sample in samples]
            if isinstance(samples[0][key], str)
            else torch.stack([sample[key] for sample in samples])
        )
        for key in samples[0]
    }


def semantic_loss(logits, batch, stats, config, device):
    target = batch["semantic"].to(device)
    valid = target != config["ignore_index"]
    ce = F.cross_entropy(
        logits.float(), target, ignore_index=config["ignore_index"], reduction="sum"
    ) / valid.sum().clamp_min(1)
    dice = dice_loss(logits.float(), target, config["num_classes"])
    density, mix = semantic_indicators(
        logits.float().softmax(1),
        config["road_class"],
        config["landuse_classes"],
        config["tile_area_km2"],
    )
    walk = F.smooth_l1_loss(walkability_1_20(density, mix, stats), batch["walkability"].to(device))
    total = sum(w * loss for w, loss in zip(config["semantic_weights"], (ce, dice, walk)))
    return {"total": total, "cross_entropy": ce, "dice": dice, "walkability": walk}


def amp(device, enabled=True):
    return torch.autocast(
        device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda"
    )
