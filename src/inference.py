from pathlib import Path
import gc
import numpy as np
from PIL import Image
import torch

from . import core
from .config import validate
from .records import Records, prompt


def load_semantic(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    c = state["config"]
    validate(c)
    clip = core.ChunkedClipEncoder(c["pretrained_model"]).to(device).eval()
    model = (
        core.SemanticFiLMUNet(
            c["num_controls"], c["num_classes"], clip.hidden_size, c["text_dim"], c["base_channels"]
        )
        .to(device)
        .eval()
    )
    model.load_state_dict(state["model"])
    model.requires_grad_(False)
    return model, clip, c, core.WalkStats(**state["walk_stats"])


@torch.inference_mode()
def semantic_maps(rows, root, path, out, device):
    out = Path(out)
    (out / "semantic").mkdir(parents=True, exist_ok=True)
    model, clip, c, stats = load_semantic(path, device)
    samples = Records(rows, root, c, 1, targets=False)
    for sample in samples:
        text = clip([sample["prompt"]], device)
        with core.amp(device):
            labels = model(sample["control"][None].to(device), text).argmax(1)[0]
        Image.fromarray(labels.cpu().numpy().astype(np.uint8)).save(
            out / "semantic" / f"{sample['tile_id']}.png"
        )
    return c, stats


@torch.inference_mode()
def satellite_images(rows, root, path, semantic_config, out, device, seed_value=None):
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
        UNet2DConditionModel,
    )

    state = torch.load(path, map_location="cpu", weights_only=False)
    c = state["config"]
    validate(c)
    validate(semantic_config)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if c["semantic_palette"] != semantic_config["semantic_palette"]:
        raise ValueError("Semantic palettes differ between stages")
    unet = UNet2DConditionModel.from_pretrained(c["pretrained_model"], subfolder="unet")
    control = ControlNetModel.from_unet(unet)
    control.load_state_dict(state["model"])
    del state, unet
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        c["pretrained_model"],
        controlnet=control,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.float().requires_grad_(False).eval()
    pipe.unet.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.controlnet.requires_grad_(False).eval()
    out = Path(out)
    (out / "generated").mkdir(exist_ok=True)
    for row in rows:
        labels = torch.tensor(
            np.array(Image.open(out / "semantic" / f"{row['tile_id']}.png"), copy=True),
            device=device,
        ).long()[None]
        condition = core.colorize(labels, semantic_config["semantic_palette"])
        if c["variant"] == "without_semantics":
            condition.zero_()
        with core.amp(device):
            text = core.encode_clip_sequences(
                pipe.tokenizer,
                pipe.text_encoder,
                [prompt(row["prompt_structured"], c["variant"])],
                device,
            ).to(dtype)
            negative = core.encode_clip_sequences(
                pipe.tokenizer, pipe.text_encoder, [""], device
            ).to(dtype)
            latent = pipe(
                prompt_embeds=text,
                negative_prompt_embeds=negative,
                image=condition,
                num_inference_steps=c["inference_steps"],
                guidance_scale=c["guidance_scale"],
                controlnet_conditioning_scale=c["control_scale"],
                generator=torch.Generator(device=device).manual_seed(
                    c["seed"] if seed_value is None else seed_value
                ),
                output_type="latent",
            ).images
        image = pipe.vae.decode(latent.float() / pipe.vae.config.scaling_factor).sample
        image = (
            ((image[0].float().clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) * 127.5)
            .round()
            .astype(np.uint8)
        )
        Image.fromarray(image).save(out / "generated" / f"{row['tile_id']}.png")
    del pipe
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


class Evaluator:
    def __init__(self, path, device, classes):
        self.device, self.classes = device, classes
        try:
            self.model = torch.jit.load(path, map_location=device).eval()
        except RuntimeError:
            from torchvision.models.segmentation import deeplabv3_resnet50

            state = torch.load(path, map_location="cpu", weights_only=False)
            self.model = deeplabv3_resnet50(
                weights=None, weights_backbone=None, num_classes=classes, aux_loss=False
            ).to(device)
            self.model.load_state_dict(state.get("model", state))
            self.model.eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, image):
        x = (
            torch.tensor(np.array(image, copy=True), device=self.device)
            .permute(2, 0, 1)[None]
            .float()
            / 255
        )
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        output = self.model((x - mean) / std)
        output = output["out"] if isinstance(output, dict) else output
        if output.shape[1] != self.classes:
            raise ValueError("Evaluator label order or channel count differs")
        output = torch.nn.functional.interpolate(
            output, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return output.argmax(1)[0].cpu().numpy()
