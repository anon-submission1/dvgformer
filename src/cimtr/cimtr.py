"""
This code is adapted from https://github.com/Mathux/TMR
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from einops import pack, unpack, repeat, reduce, rearrange, einsum
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, FloatTensor, LongTensor
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput
from src.cimtr.actor import ACTORStyleEncoder, ACTORStyleDecoder
from src.cimtr.losses import KLLoss, InfoNCE_with_filtering


def length_to_mask(length: List[int], device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


class CImTrConfig(PretrainedConfig):
    model_type = "cimtr"
    is_composition = True

    def __init__(
        self,
        traj_repr: str = 'state',  # state or action
        motion_option: str = 'local',
        temporal_downsample: int = 5,
        vision_backbone: str = 'dinov2_vits14_reg',
        img_transformer_enc: bool = True,
        vae: bool = True,
        factor: Optional[float] = 1.0,
        sample_mean: Optional[bool] = False,
        latent_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_size: int = 1024,
        loss_coef_i2t_recons: float = 1.0,
        loss_coef_t2t_recons: float = 1.0,
        loss_coef_emb_sim: float = 1.0e-5,
        loss_coef_kl: float = 1.0e-5,
        loss_coef_contrast: float = 0.1,
        temperature: float = 0.1,
        threshold_selfsim: float = 0.6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.traj_repr = traj_repr
        self.motion_option = motion_option
        self.temporal_downsample = temporal_downsample
        self.fps = 15 // temporal_downsample
        self.action_fps = 15
        self.vision_backbone = vision_backbone
        self.img_transformer_enc = img_transformer_enc
        self.image_resolution = (168, 294)
        self.image_featmap_shape = (5, 9)
        self.vae = vae
        self.factor = factor
        self.sample_mean = sample_mean
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_size = ff_size
        self.loss_coef_i2t_recons = loss_coef_i2t_recons
        self.loss_coef_t2t_recons = loss_coef_t2t_recons
        self.loss_coef_emb_sim = loss_coef_emb_sim
        self.loss_coef_kl = loss_coef_kl
        self.loss_coef_contrast = loss_coef_contrast
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim
        pass


@dataclass
class CImTrOutput(ModelOutput):
    '''
    A class to store the output of the CImTr.
    '''
    loss: Optional[torch.FloatTensor] = None
    img_latents: Optional[torch.FloatTensor] = None
    traj_latents: Optional[torch.FloatTensor] = None
    img_decodes: Optional[torch.FloatTensor] = None
    traj_decodes: Optional[torch.FloatTensor] = None


class CImTr(PreTrainedModel):
    """
    Code highly adapated from:
    TEMOS: Generating diverse human motions
    from textual descriptions
    Find more information about the model on the following website:
    https://mathis.petrovich.fr/temos

    Args:
        config: CImTrConfig
            Configuration class with all the parameters for the model.
    """

    config_class = CImTrConfig

    def __init__(
        self,
        config: CImTrConfig,
    ) -> None:
        super().__init__(config)

        self.traj_repr = config.traj_repr
        self.temporal_downsample = config.temporal_downsample
        self.img_transformer_enc = config.img_transformer_enc
        self.vae = config.vae

        if config.traj_repr == 'state':
            input_dim = 7
        elif config.traj_repr == 'action':
            input_dim = 6
        else:
            raise ValueError(
                "traj_repr should be either 'state' or 'action'")

        self.traj_projector = nn.Sequential(
            # 1) the only *strided* layer: kernel=5, stride=5 ⇒ T → T/5
            nn.Conv1d(input_dim, config.latent_dim,
                      kernel_size=self.temporal_downsample,
                      stride=self.temporal_downsample),
            nn.GELU(),
            # 2–3) two depth‑wise layers (stride 1) for richer features
            nn.Conv1d(config.latent_dim, config.latent_dim,
                      kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.latent_dim, config.latent_dim,
                      kernel_size=3, padding=1),
        )

        self.traj_unprojector = nn.Sequential(
            # mirror the two stride‑1 convs
            nn.ConvTranspose1d(config.latent_dim, config.latent_dim,
                               kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(config.latent_dim, config.latent_dim,
                               kernel_size=3, padding=1),
            nn.GELU(),
            # final *upsampling* layer: kernel=5, stride=5 ⇒ T/5 → T
            # For exact inversion we leave padding=0, output_padding=0
            nn.ConvTranspose1d(config.latent_dim, input_dim,
                               kernel_size=self.temporal_downsample,
                               stride=self.temporal_downsample)
        )

        self.traj_encoder = ACTORStyleEncoder(
            num_feats=config.latent_dim,
            vae=config.vae,
            latent_dim=config.latent_dim,
            ff_size=config.ff_size,
            num_layers=config.n_layers,
            num_heads=config.n_heads,
        )
        self.traj_decoder = ACTORStyleDecoder(
            num_feats=config.latent_dim,
            latent_dim=config.latent_dim,
            ff_size=config.ff_size,
            num_layers=config.n_layers,
            num_heads=config.n_heads,
        )

        self.image_backbone = torch.hub.load(
            'facebookresearch/dinov2', config.vision_backbone)
        self.image_backbone.eval()
        for param in self.image_backbone.parameters():
            param.requires_grad = False

        if self.img_transformer_enc:
            self.image_projector = nn.Sequential(
                nn.Linear(384, config.latent_dim),
                nn.GELU(),
                nn.Linear(config.latent_dim, config.latent_dim),
            )

            self.image_encoder = ACTORStyleEncoder(
                num_feats=config.latent_dim,
                vae=config.vae,
                latent_dim=config.latent_dim,
                ff_size=config.ff_size,
                num_layers=config.n_layers,
                num_heads=config.n_heads,
            )
        else:
            self.image_encoder = nn.Sequential(
                nn.Linear(384, self.config.latent_dim),
                nn.GELU(),
                nn.Linear(self.config.latent_dim, self.config.latent_dim),
                nn.GELU(),
                nn.Linear(self.config.latent_dim,
                          self.config.latent_dim * (2 if self.vae else 1)),
            )

        # losses
        self.reconstruction_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.latent_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.kl_loss_fn = KLLoss()
        self.contrastive_loss_fn = InfoNCE_with_filtering(
            temperature=config.temperature,
            threshold_selfsim=config.threshold_selfsim)

    def encode(
        self,
        inputs,
        masks,
        modality: str,
        sample_mean: Optional[bool] = None,
        factor: Optional[float] = None,
        return_distribution: bool = False,
    ):
        sample_mean = self.config.sample_mean if sample_mean is None else sample_mean
        factor = self.config.factor if factor is None else factor

        # Encode the inputs
        if modality == "image":
            if masks is not None:
                encoded = self.image_encoder(inputs, masks)
            else:
                encoded = self.image_encoder(inputs)
            encoded = encoded.view(
                inputs.shape[0], -1, self.config.latent_dim)
        elif modality == "traj":
            encoded = self.traj_encoder(inputs, masks)
        else:
            raise ValueError("Modality not recognized.")

        # Sampling
        if self.vae:
            dists = encoded.unbind(1)
            mu, logvar = dists
            if sample_mean:
                latent_vectors = mu
            else:
                # Reparameterization trick
                std = logvar.exp().pow(0.5)
                eps = std.data.new(std.size()).normal_()
                latent_vectors = mu + factor * eps * std
        else:
            dists = None
            (latent_vectors,) = encoded.unbind(1)

        if return_distribution:
            return latent_vectors, dists

        return latent_vectors

    # Forward: X => trajectories
    def forward(
        self,
        images: FloatTensor,
        states: FloatTensor,
        actions: FloatTensor,
        lengths: Optional[List[int]] = None,
        action_labels: Optional[torch.FloatTensor] = None,
        next_state_labels: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[Tensor] = None,
        sample_mean: Optional[bool] = None,
        factor: Optional[float] = None,
        **kwargs,
    ) -> CImTrOutput:
        """
        Forward pass of the model.
        Args:
            images: [B, 1, C, H, W] images to encode.
            states: [B, L, N, 7] of the camera trajectory.
            actions: [B, L, N, 6] of the camera trajectory.
            lengths: [B] lengths of the sequences.
            action_labels: labels for the actions (unused).
            next_state_labels: labels for the next states (unused).
            attention_mask: [B, L] attention mask for the sequences.
            sample_mean: sample the mean vector instead of random sampling (optional).
            factor: a scaling factor for sampling the VAE (optional).
            **kwargs: additional arguments.
        Returns:
            CImTrOutput: output of the model.
        """

        B, L, N, C = states.shape
        if N != self.config.temporal_downsample:
            assert N == 1, "N should be 1 or self.config.temporal_downsample"
            L = L // self.config.temporal_downsample
            lengths = (lengths / self.config.temporal_downsample).long()
            time_steps = np.arange(0, L * self.config.temporal_downsample).reshape(
                L, self.config.temporal_downsample)
            states = states.flatten(1, 2)[:, time_steps]
            actions = actions.flatten(1, 2)[:, time_steps]
            B, L, N, C = states.shape

        B, _, _, H, W = images.shape
        if self.traj_repr == 'state':
            trajs = states.flatten(1, 2)
        elif self.traj_repr == 'action':
            trajs = actions.flatten(1, 2)
        else:
            raise ValueError(
                "traj_repr should be either 'state' or 'action'")

        if attention_mask is None:
            masks = length_to_mask(lengths, device=images.device)
        else:
            masks = attention_mask.bool()

        # encode the image and decode into trajectories
        image_outputs = self.image_backbone.forward_features(
            images.flatten(0, 1))
        if self.img_transformer_enc:
            patch_tokens = image_outputs['x_norm_patchtokens']
            patch_size = self.image_backbone.patch_size
            h, w = H // patch_size, W // patch_size
            patch_tokens = rearrange(
                patch_tokens, 'b (h w) c -> b c h w', h=h, w=w)
            patch_tokens = F.adaptive_avg_pool2d(
                patch_tokens, self.config.image_featmap_shape)
            patch_tokens = rearrange(
                patch_tokens, 'b c h w -> b (h w) c')
            image_feat = torch.cat([image_outputs['x_norm_clstoken'][:, None],
                                    patch_tokens], dim=1)
            image_feat = self.image_projector(image_feat)
            B, L, C = image_feat.shape
            image_masks = torch.ones(
                [B, L], dtype=bool, device=image_feat.device)
        else:
            image_feat = image_outputs['x_norm_clstoken']
            image_masks = None
        img_latents, img_dists = self.encode(
            image_feat, image_masks, 'image',
            sample_mean=sample_mean, factor=factor, return_distribution=True
        )
        img_decodes = self.traj_decoder(img_latents, masks)
        img_decodes = self.traj_unprojector(
            img_decodes.transpose(1, 2)).transpose(1, 2)  # B, L, C -> B, C, L for 1D conv

        # encode the trajectories and decode into trajectories
        traj_feat = self.traj_projector(
            trajs.transpose(1, 2)).transpose(1, 2)  # B, L, C -> B, C, L for 1D conv
        traj_latents, traj_dists = self.encode(
            traj_feat, masks, 'traj',
            sample_mean=sample_mean, factor=factor, return_distribution=True
        )
        traj_decodes = self.traj_decoder(traj_latents, masks)
        traj_decodes = self.traj_unprojector(
            traj_decodes.transpose(1, 2)).transpose(1, 2)  # B, L, C -> B, C, L for 1D conv

        # reconstruction loss
        # image -> motion
        i2t_recons_loss = self.reconstruction_loss_fn(img_decodes, trajs)
        # motion -> motion
        t2t_recons_loss = self.reconstruction_loss_fn(traj_decodes, trajs)
        # VAE loss
        if self.vae:
            # Create a centred normal distribution to compare with
            # logvar = 0 -> std = 1
            ref_mus = torch.zeros_like(traj_dists[0])
            ref_logvar = torch.zeros_like(traj_dists[1])
            ref_dists = (ref_mus, ref_logvar)

            kl_loss = (
                self.kl_loss_fn(img_dists, traj_dists) +  # image to motion
                self.kl_loss_fn(traj_dists, img_dists) +  # motion to image
                self.kl_loss_fn(traj_dists, ref_dists) +  # motion
                self.kl_loss_fn(img_dists, ref_dists)  # image
            )
        else:
            kl_loss = 0.0
        # latent loss
        latent_loss = self.latent_loss_fn(img_latents, traj_latents)
        # CLaTr: contrastive loss
        contrast_loss = self.contrastive_loss_fn(
            img_latents, traj_latents, image_outputs['x_norm_clstoken'])
        loss = (
            self.config.loss_coef_i2t_recons * i2t_recons_loss +
            self.config.loss_coef_t2t_recons * t2t_recons_loss +
            self.config.loss_coef_emb_sim * latent_loss +
            self.config.loss_coef_kl * kl_loss +
            self.config.loss_coef_contrast * contrast_loss
        )

        return CImTrOutput(
            loss=loss,
            img_latents=F.normalize(img_latents, dim=-1),
            traj_latents=F.normalize(traj_latents, dim=-1),
            img_decodes=img_decodes,
            traj_decodes=traj_decodes,
        )


def main():
    from torch.utils.data import DataLoader
    from src.data.drone_path_seq_dataset import DronePathSequenceDataset, collate_fn_video_drone_path_dataset

    config = CImTrConfig(traj_repr='state', temporal_downsample=1)
    model = CImTr(config)

    # model = CImTr.from_pretrained(
    #     'logs/CImTr-trans-AL-lr0.001b256ep10-losscontrast1--swift-offset')

    dataset = DronePathSequenceDataset('youtube_drone_videos',
                                       'dataset_mini.h5',
                                       split_name='trainval',
                                       image_option='init',
                                       fps=model.config.fps,
                                       action_fps=model.config.action_fps,
                                       resolution=model.config.image_resolution,
                                       motion_option=model.config.motion_option,
                                       speed_scale=False,
                                       )
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                            collate_fn=collate_fn_video_drone_path_dataset,
                            num_workers=0, drop_last=True)
    batch = next(iter(dataloader))

    output = model(**batch)
    print(output.loss)
    pass


if __name__ == '__main__':
    main()
    pass
