import torch
import torch.nn.functional as F
from . import core


class SemanticStage:
    def __init__(self, config, device, stats):
        self.config, self.device, self.stats = config, device, stats
        self.clip = core.ChunkedClipEncoder(config["pretrained_model"]).to(device).eval()
        self.model = core.SemanticFiLMUNet(
            config["num_controls"],
            config["num_classes"],
            self.clip.hidden_size,
            config["text_dim"],
            config["base_channels"],
        ).to(device)

    def loss(self, batch):
        text = self.clip(batch["prompt"], self.device)
        with core.amp(self.device):
            logits = self.model(batch["control"].to(self.device), text)
        return core.semantic_loss(logits, batch, self.stats, self.config, self.device)


class SatelliteStage:
    def __init__(self, config, device):
        from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        self.config, self.device = config, device
        source = config["pretrained_model"]
        self.tokenizer = CLIPTokenizer.from_pretrained(source, subfolder="tokenizer")
        self.text = CLIPTextModel.from_pretrained(source, subfolder="text_encoder").to(device)
        self.vae = AutoencoderKL.from_pretrained(source, subfolder="vae").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(source, subfolder="unet").to(device)
        for module in (self.text, self.vae, self.unet):
            module.requires_grad_(False).eval()
        self.model = ControlNetModel.from_unet(self.unet).to(device)
        self.model.enable_gradient_checkpointing()
        self.scheduler = DDPMScheduler.from_pretrained(source, subfolder="scheduler")
        if self.scheduler.config.prediction_type != "epsilon":
            raise ValueError("The method requires epsilon-prediction SD 1.5")
        self.perceptual = (
            core.VGGPerceptual().to(device).eval() if config["satellite_weights"][2] else None
        )

    def loss(self, batch):
        images = batch["satellite"].to(self.device)
        condition = core.colorize(
            batch["semantic"].to(self.device), self.config["semantic_palette"]
        )
        if self.config["variant"] == "without_semantics":
            condition = torch.zeros_like(condition)
        with torch.no_grad():
            text = core.encode_clip_sequences(
                self.tokenizer, self.text, batch["prompt"], self.device
            )
            latent = self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor
            noise = torch.randn_like(latent)
            timestep = torch.randint(
                self.scheduler.config.num_train_timesteps, (len(images),), device=self.device
            )
            noisy = self.scheduler.add_noise(latent, noise, timestep)
        with core.amp(self.device):
            down, middle = self.model(
                noisy,
                timestep,
                encoder_hidden_states=text,
                controlnet_cond=condition,
                return_dict=False,
            )
            predicted = self.unet(
                noisy,
                timestep,
                encoder_hidden_states=text,
                down_block_additional_residuals=down,
                mid_block_additional_residual=middle,
            ).sample
        diffusion = F.mse_loss(predicted.float(), noise)
        alpha = self.scheduler.alphas_cumprod.to(self.device)[timestep, None, None, None]
        clean = (noisy - (1 - alpha).sqrt() * predicted.float()) / alpha.sqrt()

        decoded = self.vae.decode(clean / self.vae.config.scaling_factor).sample
        image = F.l1_loss(decoded, images)
        perceptual = self.perceptual(decoded, images) if self.perceptual else decoded.new_zeros(())
        total = sum(
            w * loss
            for w, loss in zip(self.config["satellite_weights"], (diffusion, image, perceptual))
        )
        return {"total": total, "diffusion": diffusion, "image_l1": image, "perceptual": perceptual}
