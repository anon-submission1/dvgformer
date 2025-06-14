from typing import Optional, Tuple, Union, Dict, Sequence, List
from functools import partial
from dataclasses import dataclass
import matplotlib.pyplot as plt
from einops import pack, unpack, repeat, reduce, rearrange, einsum
import numpy as np
from transforms3d.quaternions import qinverse, qconjugate, qmult, qnorm, quat2mat, mat2quat, quat2axangle, axangle2quat, nearly_equivalent
from transforms3d.euler import euler2quat, quat2euler, euler2mat, mat2euler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
import transformers
from transformers import PreTrainedModel, GPT2Model, AutoModel, Dinov2Backbone, DepthAnythingForDepthEstimation, set_seed
from transformers.utils import ModelOutput
from src.models.config_dvgformer import DVGFormerConfig
from src.data.state_action_conversion import state_avg, state_std, action_avg, action_std, get_states_actions, reverse_states_actions
from src.utils.pytorch3d_rotation_conversion import quaternion_to_matrix, matrix_to_quaternion, euler_angles_to_matrix, matrix_to_euler_angles
from src.utils.padding import concated_seq_to_instances, padded_seq_to_instances, padding


class QualityTokenizer(nn.Module):
    '''
    A class to extract quality tokens.
    '''

    def __init__(self, config):
        '''
        Args:
            config (DVGFormerConfig): the configuration for the model
        '''
        super().__init__()
        self.config = config
        self.num_bins = config.num_quantile_bins
        self.quality_option = config.quality_option
        if self.quality_option == 'mlp':
            self.embed_quality = nn.Sequential(
                nn.Linear(1, config.hidden_size), nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
        elif self.quality_option == 'embed':
            self.embed_quality = nn.Embedding(
                self.num_bins, config.hidden_size)
        else:
            raise ValueError(
                f'Quality option {config.quality_option} not supported.')

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): [batch_size]
        Returns:
            token (torch.Tensor): [batch_size, hidden_size]
        '''
        if self.quality_option == 'mlp':
            dtype = self.embed_quality[0].weight.dtype
            token = self.embed_quality(
                x.unsqueeze(-1).to(dtype) / self.num_bins)
        else:
            token = self.embed_quality(x)
        return token


class ImageTokenizer(nn.Module):
    '''
    A class to extract image tokens based on a vision backbone and TokenLearner, following Robotics Transformer.
    '''

    def __init__(self, config):
        '''
        Args:
            config (DVGFormerConfig): the configuration for the model
        '''
        super().__init__()
        self.use_depth = config.use_depth
        self.vision_backbone = config.vision_backbone
        self.image_featmap_shape = config.image_featmap_shape
        self.hidden_size = config.hidden_size

        # backbone
        # dinov2_vits14_reg
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', config.vision_backbone)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        # same as LLaVA, two-layer MLP after fixed vision backbone
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(config.image_featmap_shape),
            nn.Conv2d(config.vision_feat_dim, config.hidden_size, 1),
            nn.GELU(),
            nn.Conv2d(config.hidden_size, config.hidden_size, 1),
        )

        if config.use_depth:
            self.depth_model = DepthAnythingForDepthEstimation.from_pretrained(
                'depth-anything/Depth-Anything-V2-Small-hf')
            self.depth_model.eval()
            for parameter in self.depth_model.parameters():
                parameter.requires_grad_(False)
            h, w = config.image_featmap_shape
            self.depth_feat = nn.Sequential(
                nn.AdaptiveAvgPool2d((h * 4, w * 4)),
                nn.Conv2d(1, config.hidden_size, 3, 2, 1),
                nn.GELU(),
                nn.Conv2d(config.hidden_size, config.hidden_size, 3, 2, 1),
                nn.GELU(),
                nn.Conv2d(config.hidden_size, config.hidden_size, 3, 1, 1),
            )
        else:
            self.depth_model = None
            self.depth_feat = None

    def _extract_backbone_features(self, images):
        '''
        Only extract image features from the backbone. No learnable parameters.
        Args:
            images (torch.Tensor): [batch_size, num_channels, height, width]
        Returns:
            feature (torch.Tensor): [batch_size, hidden_size, height, width]
        '''
        B, C, H, W = images.shape
        output = self.backbone.forward_features(images)
        feat = output['x_norm_patchtokens']
        patch_size = self.backbone.patch_size
        h, w = H // patch_size, W // patch_size
        feature = rearrange(feat, 'b (h w) c -> b c h w', h=h, w=w)
        return feature

    def forward(self, images):
        '''
        Args:
            images (torch.Tensor): [batch_size, num_channels, height, width]
        Returns:
            image_tokens (torch.Tensor): [batch_size, n_token_image, hidden_size]
        '''
        original_feature = self._extract_backbone_features(images)
        feature = self.bottleneck(original_feature)
        if self.use_depth:
            outputs = self.depth_model(images)
            disparity = outputs.predicted_depth
            depth_feature = self.depth_feat(disparity.unsqueeze(1))
            feature = feature + depth_feature
        image_tokens = rearrange(feature, 'b c h w -> b (h w) c')
        return image_tokens


@dataclass
class DVGFormerOutput(ModelOutput):
    '''
    A class to store the output of the DVGFormerModel.
    '''
    loss: Optional[torch.FloatTensor] = None
    drone_type_preds: Optional[torch.FloatTensor] = None
    action_preds: Optional[torch.FloatTensor] = None
    stop_preds: Optional[torch.FloatTensor] = None
    next_state_preds: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class DVGFormerModel(PreTrainedModel):
    """
    Drone VideoGraphy Transformer Model.
    """

    config_class = DVGFormerConfig
    _supports_flash_attn_2 = True  # dummy value to pass the check in the base class

    def __init__(self, config):
        '''
        Args:
            config (DVGFormerConfig): the configuration for the model
        '''
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_data_frames = config.max_data_frames
        self.max_model_frames = config.max_model_frames
        self.n_future_frames = config.n_future_frames
        self.n_future_steps = config.n_future_frames // config.fps_downsample

        # begin of action token
        self.boa_token_embed = nn.Parameter(
            torch.randn(config.hidden_size))
        # padding token
        self.pad_token_embed = nn.Parameter(
            torch.randn(config.hidden_size))

        # number of tokens for describing the entire sequence
        self.n_token_noise = config.n_token_noise
        self.n_token_quality = config.n_token_quality
        self.n_token_drone_type = config.n_token_drone_type
        self.n_token_init_image = config.n_token_init_image
        self.n_token_prepend = config.n_token_prepend
        # fps downsample ratio
        self.fps_downsample = config.fps_downsample
        self.n_action_every_image = config.n_action_every_image
        self.per_token_preds = config.per_token_preds
        # number of tokens for describing one frame
        self.n_token_state = config.n_token_state
        self.n_token_image = config.n_token_image
        self.n_token_boa = config.n_token_boa
        self.n_token_action = config.n_token_action
        self.n_token_one_frame = config.n_token_one_frame
        # number of tokens to predict
        self.n_token_predict = config.n_token_predict

        # token_types: 0 for predicting nothing, 1 for predicting both next_state and action pred
        # within-frame positional embedding
        self.in_frame_pe = nn.Embedding(
            config.n_token_one_frame, config.hidden_size // 2)
        # cross-frame positional embeddings
        self.cross_frame_pe = nn.Embedding(
            config.max_model_frames // config.fps_downsample,
            config.hidden_size // 2)
        self.prepend_pe = nn.Parameter(torch.randn(
            self.n_token_prepend, config.hidden_size))

        self.execute_option = config.execute_option
        self.motion_option = config.motion_option
        self.image_option = config.image_option

        # tokens for the entire sequence
        self.embed_quality = QualityTokenizer(config)
        self.embed_drone_type = nn.Embedding(
            2, config.hidden_size)  # 0: non-fpv, 1: fpv
        # tokens for each frame
        self.embed_img = ImageTokenizer(config)
        self.embed_state = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            # nn.LayerNorm(config.hidden_size)
        )
        self.embed_action = nn.Sequential(
            nn.Linear(config.action_dim * config.per_token_preds,
                      config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            # nn.LayerNorm(config.hidden_size)
        )
        self.embed_ln = nn.LayerNorm(config.hidden_size)

        self.transformer = GPT2Model(config.gpt2_config)
        # set the original positional embeddings to zero
        self.transformer.wpe.weight.data.zero_()
        # turn off requires_grad for the original positional embeddings
        self.transformer.wpe.requires_grad_(False)

        # binary classification for drone type, non-fpv vs fpv
        # take image features at t=0
        self.predict_drone_type = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # b, c, h, w -> b, c, 1, 1
            nn.Flatten(start_dim=1),  # b, c, 1, 1 -> b, c
            nn.Linear(config.vision_feat_dim, config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, 1),
        )
        # predictions for each frame
        self.predict_action = nn.Linear(
            config.hidden_size, config.action_dim * config.per_token_preds)
        self.predict_stop = nn.Linear(
            config.hidden_size, 1)  # binary classification for end-of-seq
        # auxiliary predictions
        self.predict_next_state = nn.Linear(
            config.hidden_size, config.state_dim * config.per_token_preds)
        # self.predict_future_action = nn.Linear(
        #     config.hidden_size, config.action_dim * self.n_future_steps)

        self.stop_loss = partial(torchvision.ops.sigmoid_focal_loss,
                                 alpha=config.focal_alpha)
        self.drone_type_loss = nn.BCEWithLogitsLoss(reduction='none')

        # Initialize weights and apply final processing
        self.post_init()

    def _tokenize(
        self,
        time_steps: torch.LongTensor,
        images: torch.FloatTensor,
        states: torch.FloatTensor,
        actions: torch.FloatTensor,
        seq_length: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        noise_embed: Optional[torch.LongTensor] = None,
        quality: Optional[torch.LongTensor] = None,
        drone_type: Optional[torch.LongTensor] = None,
        intrinsics: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ):
        """
        Convert input data to token embeddings and add within/across frame positional embeddings.
        (noise_embed, quality, drone_type, i_1, s_1, a_1, i_2, s_2, a_2, ...)
        Args:
            time_steps (torch.LongTensor): [batch_size, padded_length]
            images (torch.FloatTensor): [batch_size, padded_length, num_channels, height, width]
            states (torch.FloatTensor): [batch_size, padded_length, n_action_every_image, state_dim]
            actions (torch.FloatTensor): [batch_size, padded_length, n_action_every_image, action_dim]
            seq_length (torch.LongTensor): [batch_size]
            attention_mask (torch.FloatTensor): [batch_size, padded_length]
            noise_embed (torch.FloatTensor): [batch_size, hidden_size]
            quality (torch.LongTensor): [batch_size]
            drone_type (torch.LongTensor): [batch_size]
            intrinsics (torch.FloatTensor): [batch_size, 3, 3]
            past_key_values (Tuple[Tuple[torch.FloatTensor]]): past key values for the transformer
        Returns:
            stacked_embeds (torch.FloatTensor): [batch_size, stacked_length, hidden_size]
            stacked_attn_mask (torch.FloatTensor): [batch_size, stacked_length]
            position_ids (torch.LongTensor): [batch_size, stacked_length]
            within_frame_pos (torch.LongTensor): [batch_size, stacked_length]
            token_types (torch.LongTensor): [batch_size, stacked_length]
        """
        device = images.device
        b, l = states.shape[:2]
        n = self.n_token_one_frame

        # embedding for each frame
        _images = padded_seq_to_instances(images, seq_length)
        _images = torch.cat(_images, dim=0)
        _states = padded_seq_to_instances(states, seq_length)
        _states = torch.cat(_states, dim=0)
        _actions = padded_seq_to_instances(actions, seq_length)
        _actions = torch.cat(_actions, dim=0)
        stacked_len, steps, _ = _states.shape
        if intrinsics is not None:
            _intrinsics = intrinsics.float()[:, None].repeat(1, l, 1, 1)
            _intrinsics = padded_seq_to_instances(_intrinsics, seq_length)
            _intrinsics = torch.cat(_intrinsics, dim=0)
        else:
            _intrinsics = None

        # image embeddings
        if self.image_option == 'init':
            # only use the first image for the prepend condition
            if drone_type is not None and noise_embed is not None:
                init_image_embeds = self.embed_img(
                    images[:, 0].to(self.dtype))
            else:
                init_image_embeds = None
            image_embeds = torch.zeros([b, l, 0, self.hidden_size],
                                       device=device, dtype=self.dtype)
        elif self.image_option == 'all':
            # use all images for the sequence
            init_image_embeds = None
            image_embeds = self.embed_img(
                _images.to(self.dtype))
            image_embeds = concated_seq_to_instances(
                image_embeds, seq_length)
            image_embeds = padding(image_embeds, self.config.pad_side,
                                   self.config.pad_token_value, l)
        elif self.image_option == 'none':
            # no image embeddings
            init_image_embeds = None
            image_embeds = torch.zeros([b, l, 0, self.hidden_size],
                                       device=device, dtype=self.dtype)
        else:
            raise ValueError(
                f'Image option {self.image_option} not supported.')

        # state embeddings
        if self.n_token_state > 0:
            state_embeds = self.embed_state(_states.to(self.dtype))
            state_embeds = concated_seq_to_instances(
                state_embeds, seq_length)
            state_embeds = padding(state_embeds, self.config.pad_side,
                                   self.config.pad_token_value, l)
        else:
            state_embeds = torch.zeros([b, l, 0, self.hidden_size],
                                       device=device, dtype=self.dtype)

        # action embeddings
        if self.n_token_action > 0:
            action_embeds = self.embed_action(_actions.to(self.dtype).view(
                stacked_len, -1, self.per_token_preds * self.config.action_dim))
            action_embeds = concated_seq_to_instances(
                action_embeds, seq_length)
            action_embeds = padding(action_embeds, self.config.pad_side,
                                    self.config.pad_token_value, l)
        else:
            action_embeds = torch.zeros([b, l, 0, self.hidden_size],
                                        device=device, dtype=self.dtype)

        boa_embeds = self.boa_token_embed.repeat(b, l, 1)

        # attention mask
        stacked_attn_mask = repeat(attention_mask, 'b l -> b (l n)',
                                   n=n).to(self.dtype)

        # stack token embeddings within each time step
        stacked_embeds = torch.cat(
            [state_embeds[:, :, :self.n_token_state],
             image_embeds[:, :, :self.n_token_image],
             boa_embeds[:, :, None][:, :, :self.n_token_boa],
             action_embeds[:, :, :self.n_token_predict]],
            dim=2)
        # tokens for t=0, 1, ...
        stacked_embeds = rearrange(stacked_embeds, 'b l n c -> b (l n) c')
        # replace the tokens outside the sequence with padding tokens
        stacked_embeds[~stacked_attn_mask.bool()[:, -l * n:]] = \
            self.pad_token_embed
        # token_types: 0 for predicting nothing, 1 for predicting both next_state and action pred
        token_types = [
            torch.zeros([b, l, self.n_token_state],
                        device=device, dtype=torch.long),
            torch.zeros([b, l, self.n_token_image], device=device,
                        dtype=torch.long),
            torch.ones([b, l, self.n_token_boa], device=self.device,
                       dtype=torch.long),  # boa
        ]
        if self.n_token_action > 0:
            token_types.extend([
                torch.ones([b, l, self.n_token_predict - 1], device=self.device,
                           dtype=torch.long),  # action 0~n-2
                torch.zeros([b, l, 1], device=self.device,
                            dtype=torch.long)  # last action n-1
            ])
        token_types = torch.cat(token_types, dim=2)
        if self.n_token_one_frame == 1 and self.n_token_boa == 0:
            token_types[:, :, -1] = 1
        token_types = rearrange(token_types, 'b l n -> b (l n)')

        # across-frame positional embeddings
        position_ids = repeat(time_steps, 'b l -> b (l n)', n=n)
        # within-frame positional embeddings
        within_frame_pos = torch.arange(n, dtype=torch.long, device=device)
        within_frame_pos = repeat(within_frame_pos, 'n -> b (l n)', b=b, l=l)

        # only prepend quality and drone type tokens for input_embeds if needed
        noise_embed = noise_embed.to(device).to(
            self.dtype)[:, None] if noise_embed is not None else None
        quality_embed = self.embed_quality(
            quality)[:, None] if quality is not None else None
        drone_type_embed = self.embed_drone_type(
            drone_type)[:, None] if drone_type is not None else None
        for embed, n_token in zip([noise_embed, quality_embed, drone_type_embed, init_image_embeds],
                                  [self.n_token_noise, self.n_token_quality, self.n_token_drone_type, self.n_token_init_image]):
            if embed is None or n_token == 0:
                continue
            stacked_embeds = torch.cat([embed[:, :n_token],
                                        stacked_embeds], dim=1)
            position_ids = torch.cat([torch.ones([b, n_token], dtype=torch.long, device=device
                                                 ) * self.config.ignore_value,
                                      position_ids], dim=1)
            within_frame_pos = torch.cat([torch.ones([b, n_token], dtype=torch.long, device=device
                                                     ) * self.config.ignore_value,
                                          within_frame_pos], dim=1)
            # token_types: 0 for predicting nothing, 1 for predicting both next_state and action pred
            token_types = torch.cat([torch.zeros([b, n_token], dtype=torch.long, device=device),
                                     token_types], dim=1)
            # attention mask always include the prepended tokens
            stacked_attn_mask = torch.cat([torch.ones([b, n_token], dtype=self.dtype, device=device),
                                           stacked_attn_mask], dim=1)
        if past_key_values is not None:
            stacked_attn_mask = torch.cat([torch.ones([b, past_key_values[0][0].shape[2]],
                                                      dtype=self.dtype, device=device),
                                           stacked_attn_mask[:, -l * n:]], dim=1)
        return stacked_embeds, stacked_attn_mask, position_ids, within_frame_pos, token_types

    def _init_frame_pred(
        self,
        images: torch.FloatTensor,
    ):
        """
        Predict the drone type based on the image at t=0.
        Args:
            time_steps (torch.LongTensor): [batch_size, padded_length]
            images (torch.FloatTensor): [batch_size, padded_length, num_channels, height, width]
        Returns:
            drone_type_preds (torch.FloatTensor): [batch_size]
        """
        # estimate the drone type
        init_imgs = images[:, 0]
        init_img_feat = self.embed_img._extract_backbone_features(
            init_imgs.to(self.dtype))  # b, l, c
        drone_type_preds = self.predict_drone_type(init_img_feat)  # b, 1
        return drone_type_preds

    def forward(
        self,
        noise_embed: Optional[torch.LongTensor] = None,
        quality: Optional[torch.LongTensor] = None,
        drone_type: Optional[torch.LongTensor] = None,
        intrinsic: Optional[torch.FloatTensor] = None,
        time_steps: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        seq_length: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        within_frame_pos: Optional[torch.LongTensor] = None,
        token_types: Optional[torch.LongTensor] = None,
        drone_type_labels: Optional[torch.FloatTensor] = None,
        next_state_labels: Optional[torch.FloatTensor] = None,
        action_labels: Optional[torch.FloatTensor] = None,
        stop_labels: Optional[torch.FloatTensor] = None,
        # future_action_labels: Optional[torch.FloatTensor] = None,
        depth_labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor], DVGFormerOutput]:
        '''
        Forward pass of the model given inputs.
        There are two options for the forward pass:
        1. Training: the model predicts the next token based on inputs of
                     (noise_embed, quality, drone_type, intrinsic, time_steps, images, states, actions, seq_length, attention_mask).
                     Usually not used with past_key_values.
                     To calculate the loss, provide (drone_type_labels, next_state_labels, action_labels, stop_labels, depth_labels).
        2. Generation: the model predicts the next self.n_action_every_image actions based inputs of
                       (inputs_embeds, position_ids, within_frame_pos, token_types, attention_mask).
                       Usually used together with past_key_values.
                       Loss calculated outside of this function.

        Args:
            noise_embed (torch.LongTensor): [batch_size, hidden_size]
            quality (torch.LongTensor): [batch_size]
            drone_type (torch.LongTensor): [batch_size]
            intrinsic (torch.FloatTensor): [batch_size, 3, 3]
            time_steps (torch.LongTensor): [batch_size, padded_length]
            images (torch.FloatTensor): [batch_size, padded_length, 3, height, width]
            states (torch.FloatTensor): [batch_size, padded_length, n_action_every_image, state_dim]
            actions (torch.FloatTensor): [batch_size, padded_length, n_action_every_image, action_dim]
            seq_length (torch.LongTensor): [batch_size]
            past_key_values (Tuple[Tuple[torch.Tensor]]): the past key values for the transformer
            attention_mask (torch.FloatTensor): [batch_size, padded_length] / [batch_size, stacked_length]
            inputs_embeds (torch.FloatTensor): [batch_size, stacked_length, hidden_size]
            position_ids (torch.LongTensor): [batch_size, stacked_length]
            within_frame_pos (torch.LongTensor): [batch_size, stacked_length]
            token_types (torch.LongTensor): [batch_size, stacked_length]
            drone_type_labels (torch.LongTensor): [batch_size]
            next_state_labels (torch.FloatTensor): [batch_size, padded_length, n_action_every_image, state_dim]
            action_labels (torch.FloatTensor): [batch_size, padded_length, n_action_every_image, action_dim]
            stop_labels (torch.FloatTensor): [batch_size, padded_length]
            future_action_labels (torch.FloatTensor): [batch_size, padded_length, n_step, action_dim]
            depth_labels (torch.FloatTensor): [batch_size, padded_length, height, width]
            use_cache (bool): whether to use cache for the transformer
            output_hidden_states (bool): whether to output hidden states
            output_attentions (bool): whether to output attentions
        Returns:
            DVGFormerOutput: the output of the model
        '''
        # bool indicators
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # L: stacked_length (padded_length * n_token_one_frame)
        # l: padded_length
        if inputs_embeds is not None:
            device = inputs_embeds.device
            b, L = inputs_embeds.shape[:2]
            L = L - self.n_token_prepend if L > self.n_token_prepend else L
            l = int(np.ceil(L / self.n_token_one_frame))
            if L % self.n_token_one_frame != 0:
                # only predict the last frame (for generation)
                l = min(l, 1)
        else:
            device = states.device
            b, l = states.shape[:2]
            L = l * self.n_token_one_frame

        if inputs_embeds is None:
            # for training
            # only include the quality and drone type tokens at the beginning of the sequence
            if past_key_values is not None:
                quality = None
                drone_type = None
            # predict drone type
            if images.shape[1]:
                drone_type_preds = self._init_frame_pred(images)
            else:
                drone_type_preds = None

            inputs_embeds, attention_mask, position_ids, within_frame_pos, token_types = self._tokenize(
                time_steps, images, states, actions, seq_length, attention_mask,
                noise_embed, quality, drone_type, intrinsic, past_key_values)
        else:
            # for generation
            assert attention_mask is not None
            assert position_ids is not None
            assert within_frame_pos is not None
            assert token_types is not None

            # if the input is inputs_embeds, drone_type_preds should have been calculated elsewhere
            drone_type_preds = None

        inputs_embeds = self.embed_ln(inputs_embeds)
        # Note: we have kept the positional embeddings to 0 in the originial GPT2 model
        # token-level & frame-level positional embedding
        prepend_length = (within_frame_pos ==
                          self.config.ignore_value).sum(dim=1)[0].item()
        inputs_embeds[:, :prepend_length] += self.prepend_pe[:prepend_length]
        inputs_embeds[:, prepend_length:] += torch.stack(
            [self.cross_frame_pe(position_ids[:, prepend_length:]),
             self.in_frame_pe(within_frame_pos[:, prepend_length:])], dim=-1).flatten(-2)

        # sanity check
        inputs_embeds = inputs_embeds.to(self.dtype)
        past_key_values = None if past_key_values is None else tuple(
            tuple(pkv.to(self.dtype) for pkv in pkvs) for pkvs in past_key_values)
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=torch.zeros_like(position_ids),
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs[0]

        # token_types: 0 for predicting nothing, 1 for predicting both next_state and action pred
        # predict next state
        pred_next_state_pos = token_types == 1
        if pred_next_state_pos.any():
            next_state_preds = self.predict_next_state(
                hidden_states[pred_next_state_pos])
            next_state_preds = next_state_preds.view(
                [b, l, -1, self.config.state_dim])
            # unnormalize the state: tvec and qvec (global)
            _state_avg = torch.tensor(
                state_avg, dtype=torch.float32).to(device)
            _state_std = torch.tensor(
                state_std, dtype=torch.float32).to(device)
            _next_state_preds_unnorm = next_state_preds.float() * _state_std + _state_avg
            # make sure the quaternion is normalized
            tvecs, _qvecs = _next_state_preds_unnorm[..., :3], \
                _next_state_preds_unnorm[..., 3:]
            qvecs = F.normalize(_qvecs, dim=-1)
            next_state_preds_unnorm = torch.cat(
                [tvecs, qvecs], dim=-1)
            # normalize the next state predictions
            next_state_preds = (next_state_preds_unnorm -
                                _state_avg) / _state_std
        else:
            next_state_preds = None
        # predict next action
        pred_action_pos = token_types == 1
        if pred_action_pos.any():
            action_preds = self.predict_action(hidden_states[pred_action_pos])
            action_preds = action_preds.view(
                [b, l, -1, self.config.action_dim])
            # make sure the action preds is not affected by action downsample
            action_preds *= self.config.action_downsample
        else:
            action_preds = None
        # predict end-of-sequence
        # last token in frame
        pred_stop_pos = within_frame_pos == self.n_token_one_frame - 1
        if pred_stop_pos.any():
            stop_preds = self.predict_stop(hidden_states[pred_stop_pos])
            stop_preds = rearrange(stop_preds, '(b l) 1 -> b l', b=b)
            # future_action_preds = self.predict_future_action(
            #     hidden_states[pred_stop_pos])
            # future_action_preds = rearrange(future_action_preds, '(b l) (n c) -> b l n c',
            #                                 b=b, n=self.n_future_steps, c=self.config.action_dim)
        else:
            stop_preds = None
            # future_action_preds = None

        loss = None
        if stop_labels is not None:
            # drone type
            if drone_type_preds is not None and drone_type_labels is not None:
                init_idx = time_steps[:, 0] == 0
                loss_drone_type = (self.drone_type_loss(
                    drone_type_preds[:, 0], drone_type_labels.to(self.dtype)) * init_idx).mean()
            else:
                loss_drone_type = 0
            # sequence
            loss_next_state = F.l1_loss(next_state_preds, next_state_labels.to(self.dtype),
                                        reduction='none').mean(dim=[2, 3]) if next_state_preds is not None else 0
            if action_preds is not None:
                # make sure the action loss has the same norm for dense or sparse prediction
                # also clamp the values to minimize the effect of outliers
                _action_preds = (
                    action_preds / self.config.action_downsample).clamp(-5, 5)
                _action_labels = (
                    action_labels / self.config.action_downsample).clamp(-5, 5)
                loss_action = F.l1_loss(_action_preds, _action_labels.to(self.dtype),
                                        reduction='none').mean(dim=[2, 3])
            else:
                loss_action = 0
            loss_stop = self.stop_loss(stop_preds, stop_labels.to(self.dtype),
                                       reduction='none')
            # future_action_mask = (future_action_labels !=
            #                       self.config.ignore_value).all(dim=-1, keepdim=True)
            # loss_future_action = ((future_action_preds - future_action_labels.to(self.dtype)).abs() *
            #                       future_action_mask).mean(dim=[2, 3]) / self.fps_downsample

            sequence_mask = stop_labels != self.config.ignore_value
            seq_loss = (loss_next_state * self.config.loss_coef_state +
                        loss_action * self.config.loss_coef_action +
                        loss_stop * self.config.loss_coef_stop
                        # + loss_future_action * self.config.loss_coef_future
                        ) * sequence_mask
            loss = (loss_drone_type * self.config.loss_coef_drone_type +
                    seq_loss.mean())
        return DVGFormerOutput(
            loss=loss,
            drone_type_preds=drone_type_preds,
            action_preds=action_preds,
            stop_preds=stop_preds,
            next_state_preds=next_state_preds,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _exec_actions(
        self,
        states: torch.FloatTensor,
        actions: torch.FloatTensor,
    ):
        '''
        Execute the states and actions to get the next state.
        Args:
            states (torch.FloatTensor): [batch_size, l, n_action_every_image, state_dim]
            actions (torch.FloatTensor): [batch_size, l, n_action_every_image, action_dim]
        Returns:
            next_states (torch.FloatTensor): [batch_size, l, n_action_every_image, state_dim]
        '''

        device = states.device

        # l: padded_length for images & states
        b, l, n = states.shape[:3]

        # get next state based on the predicted action and its conversion
        # unnormalize the state: tvec and qvec (global)
        _states_unnorm = states.cpu().numpy() * state_std + state_avg
        # unnormalize the action: v and omega (global/local)
        _actions_unnorm = actions.cpu().numpy() * action_std + action_avg
        next_tvecs, next_qvecs = np.zeros([b, l, n, 3]), np.zeros([b, l, n, 4])
        for i in range(b):
            for j in range(l):
                next_tvecs[i][j], next_qvecs[i][j], _, _ = reverse_states_actions(
                    _states_unnorm[i][j],
                    _actions_unnorm[i][j],
                    motion_option=self.motion_option
                )
        next_states = np.concatenate(
            [next_tvecs, next_qvecs], axis=-1)
        # normalize the next states
        next_states = (next_states - state_avg) / state_std
        next_states = torch.tensor(
            next_states, dtype=torch.float32).to(device)

        return next_states

    def _get_actions(
            self,
            states: torch.FloatTensor,
            next_states: torch.FloatTensor,
    ):
        '''
        Get the actions based on the states and next states.
        Args:
            states (torch.FloatTensor): [batch_size, l, n_action_every_image, state_dim]
            next_states (torch.FloatTensor): [batch_size, state_dim]
        Returns:
            actions (torch.FloatTensor): [batch_size, action_dim]
        '''
        b, l, n = states.shape[:3]
        assert n == 1, 'please make sure n_token_predict==1 and n_action_every_image==1'

        _states_unnorm = states.cpu().numpy() * state_std + state_avg
        _next_states_unnorm = next_states.cpu().numpy() * state_std + state_avg
        _tvecs, _qvecs = _states_unnorm[..., :3], _states_unnorm[..., 3:]
        _qvecs /= np.linalg.norm(_qvecs, axis=-1, keepdims=True)
        _next_tvecs, _next_qvecs = _next_states_unnorm[..., :3], \
            _next_states_unnorm[..., 3:]
        _next_qvecs /= np.linalg.norm(_next_qvecs, axis=-1, keepdims=True)
        tvecs = np.concatenate(
            [_tvecs, _next_tvecs], axis=2).reshape(b, l * 2, 3)
        qvecs = np.concatenate(
            [_qvecs, _next_qvecs], axis=2).reshape(b, l * 2, 4)
        # get the actions based on the states and next states
        actions_unnorm = []
        for i in range(b):
            _states, _actions = get_states_actions(
                tvecs[i], qvecs[i], self.motion_option)
            actions_unnorm.append(_actions[::2])
        actions_unnorm = np.stack(actions_unnorm, axis=0).reshape(b, l, 1, -1)
        actions = (actions_unnorm - action_avg) / action_std
        actions = torch.tensor(actions, dtype=torch.float32).to(states.device)

        return actions

    def expand_actions(
        self,
        noise_embed: Optional[torch.LongTensor] = None,
        quality: Optional[torch.LongTensor] = None,
        drone_type: Optional[torch.LongTensor] = None,
        intrinsic: Optional[torch.FloatTensor] = None,
        time_steps: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        seq_length: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        gt_forcing: bool = False,
    ) -> DVGFormerOutput:
        '''
        Expand the actions for one (current) pair of image & state (for generation).

        Args:
            noise_embed (torch.LongTensor): [batch_size, hidden_size]
            quality (torch.LongTensor): [batch_size]
            drone_type (torch.LongTensor): [batch_size]
            intrinsic (torch.FloatTensor): [batch_size, 3, 3]
            time_steps (torch.LongTensor): [batch_size, padded_length]
            images (torch.FloatTensor): [batch_size, padded_length, num_channels, height, width]
            states (torch.FloatTensor): [batch_size, padded_length, n_action_every_image, state_dim]
            actions (torch.FloatTensor): [batch_size, padded_length, n_action_every_image, action_dim]
            seq_length (torch.LongTensor): [batch_size]
            past_key_values (Tuple[Tuple[torch.Tensor]]): the past key values for the transformer
            attention_mask (torch.FloatTensor): [batch_size, padded_length]
        Returns:
            DVGFormerOutput: the output of the model
        '''
        device = images.device

        # l: padded_length for images & states
        b, l = states.shape[:2]
        # only compute embeddings for the last frame
        l = 1
        # past_key_values: list of config.n_layer elements,
        # each a two-tuple of key, value of shape [batch_size, num_head, seq_length, C]
        # remove previous frames as they should be included in the past_key_values
        time_steps = time_steps[:, -l:]
        images = images[:, -l:]
        states = states[:, -l:]
        actions = actions[:, -l:]
        seq_length = torch.ones([b], dtype=torch.long) * l

        # only include prepend tokens for the first frame
        drone_type_preds = None
        if past_key_values is None:
            if drone_type is None:
                drone_type_preds = self._init_frame_pred(images)
                drone_type = (drone_type_preds[:, 0] > 0).type(torch.long)
            pass
        else:
            quality = None
            drone_type = None
            noise_embed = None

        inputs_embeds, attention_mask, position_ids, within_frame_pos, token_types = self._tokenize(
            time_steps, images, states, actions, seq_length, attention_mask.to(
                self.dtype), noise_embed, quality, drone_type, intrinsic, past_key_values)
        # remove the last several terms for prediction
        end_idx = inputs_embeds.shape[1] - self.n_token_predict
        mask_end_idx = attention_mask.shape[1] - self.n_token_predict
        inputs_embeds = inputs_embeds[:, :end_idx]

        # from transformers.generation.utils import GenerationMixin
        model_inputs = {'inputs_embeds': inputs_embeds,
                        'attention_mask': attention_mask[:, :mask_end_idx],
                        'position_ids': position_ids[:, :end_idx],
                        'within_frame_pos': within_frame_pos[:, :end_idx],
                        'token_types': token_types[:, :end_idx],
                        'past_key_values': past_key_values,
                        'use_cache': True,
                        }

        # predict the next self.n_action_every_image actions & next_states
        # t=0,1,2,...
        action_preds = torch.zeros([b, 1, self.n_action_every_image, self.config.action_dim],
                                   device=device, dtype=self.dtype)
        # t=1,2,3,...
        next_state_preds = torch.zeros([b, 1, self.n_action_every_image, self.config.state_dim],
                                       device=device, dtype=self.dtype)
        for i in range(self.n_token_predict + 1):
            outputs = self.forward(**model_inputs)
            model_inputs['past_key_values'] = outputs.past_key_values
            if self.execute_option == 'action':
                # t=0,1,2,3,4
                if i < self.n_token_predict or self.n_token_action == 0:
                    assert outputs.action_preds is not None
                    # action
                    if self.per_token_preds == 1:
                        action_preds[:, :, i] = outputs.action_preds[:, :, 0]
                    else:
                        action_preds = outputs.action_preds
                    # which action to use as condition for the next generation iteration
                    if gt_forcing:
                        executed_actions = actions
                    else:
                        executed_actions = action_preds
            elif self.execute_option == 'next_state':
                # next_state
                assert outputs.next_state_preds is not None
                next_state_preds = outputs.next_state_preds
                action_preds = self._get_actions(
                    states, next_state_preds.detach())
            else:
                raise NotImplementedError(
                    f'Execution option {self.execute_option} not implemented')
            # stop
            if outputs.stop_preds is not None:
                stop_preds = outputs.stop_preds
            # next token
            if self.n_token_action == 1 and i < self.n_token_predict:
                next_embeds = self.embed_action(
                    executed_actions.to(self.dtype).view(
                        b, l, -1, self.per_token_preds * self.config.action_dim)[:, -1, [i]])
            if i == self.n_token_predict:
                break

            # update the input kwargs for next generation iteration
            seq_idx = -self.n_token_predict + i
            inputs_embeds = torch.cat([inputs_embeds, next_embeds], dim=1)
            # [b, 1]
            model_inputs['inputs_embeds'] = next_embeds
            model_inputs['position_ids'] = position_ids[:, [seq_idx]]
            model_inputs['within_frame_pos'] = within_frame_pos[:, [seq_idx]]
            model_inputs['token_types'] = token_types[:, [seq_idx]]
            # [b, seq_length]
            attn_end_idx = seq_idx + 1
            model_inputs['attention_mask'] = (attention_mask[:, :attn_end_idx]
                                              if attn_end_idx < 0 else attention_mask)

        pass
        return DVGFormerOutput(
            drone_type_preds=drone_type_preds,
            action_preds=action_preds,
            stop_preds=stop_preds,
            next_state_preds=next_state_preds,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def main():
    import tqdm
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from torchvision.utils import make_grid
    import torchvision.transforms as T
    from transformers import set_seed
    from src.models.config_dvgformer import DVGFormerConfig
    from src.data.drone_path_seq_dataset import DronePathSequenceDataset, collate_fn_video_drone_path_dataset

    # torch.inverse multi-threading RuntimeError: lazy wrapper should be called at most once
    # https://github.com/pytorch/pytorch/issues/90613#issuecomment-1817307008
    torch.inverse(torch.ones((1, 1), device="cuda:0"))

    set_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = True

    config = DVGFormerConfig(
        fps=3,
        execute_option='next_state',
        # motion_option='global',
        # prediction_option='one-shot',
        action_option='sparse',
        image_option='none',
        # max_model_frames=30,
        n_token_state=1,
        n_token_action=0,
        n_token_boa=0,
        # attn_implementation='flash_attention_2'
    )
    # return

    dataset = DronePathSequenceDataset(
        'youtube_drone_videos',
        'dataset_mini.h5',
        drone_types=[1],
        fps=config.fps,
        action_fps=config.action_fps,
        max_model_frames=config.max_model_frames,
        image_option=config.image_option,
        motion_option=config.motion_option,
        resolution=config.image_resolution,
        num_quantile_bins=config.num_quantile_bins,
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False,
                            collate_fn=collate_fn_video_drone_path_dataset,
                            num_workers=0)

    device = 'cuda'
    model = DVGFormerModel(config).to(device)
    model.eval()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f'The model has {count_parameters(model.transformer):,} trainable parameters')

    # batch = next(iter(dataloader))
    total_loss = 0
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        batch = {key: value.to(device)
                 for key, value in batch.items()
                 #  if key != 'quality'
                 }
        # depth_loss = depth_model(**batch)

        out = model(**batch, output_attentions=True)
        if config._attn_implementation != 'flash_attention_2':
            attns = [attn.mean(dim=1)[1].float().detach().cpu().numpy()
                     for attn in out.attentions]
            # plot the attention map for the first and the last layer
            plt.figure(figsize=(20, 10))  # Increase the plot size
            plt.subplot(1, 2, 1)
            plt.imshow(attns[0], vmin=0, vmax=0.1)
            plt.subplot(1, 2, 2)
            plt.imshow(attns[-1], vmin=0, vmax=0.1)
            # Save the figure at a higher quality
            plt.savefig('attn.png', dpi=300)
        total_loss += out.loss
        if i == 0:
            break
    print('loss: ', total_loss / (i + 1))
    seqence_lvl_keys = ['noise_embed', 'quality', 'drone_type', 'intrinsic']

    # return
    partition_length = 2
    batch_pt0 = {key: value[:, :partition_length] if len(value.shape) > 1 else value for key, value in batch.items()
                 if key != 'seq_length' and key != 'intrinsic' and key != 'attention_mask' and 'label' not in key
                 and 'quality' not in key and 'drone_type' not in key and 'noise' not in key}
    batch_pt0['attention_mask'] = batch['attention_mask'][:,
                                                          :partition_length]
    batch_pt0['seq_length'] = torch.ones_like(
        batch['seq_length']) * partition_length
    batch_pt0['intrinsic'] = batch['intrinsic']
    batch_pt0['quality'] = batch['quality']
    batch_pt0['drone_type'] = batch['drone_type']
    batch_pt0['noise_embed'] = batch['noise_embed']
    # T.ToPILImage()(make_grid(rearrange(batch_pt0['images'],
    #                                    'b l c h w -> (b l) c h w'), normalize=True)
    #                ).save('frames.jpg')
    batch_pt1 = {key: value[:, partition_length:] for key, value in batch.items()
                 if key != 'seq_length' and key != 'intrinsic' and key != 'attention_mask' and 'label' not in key
                 and 'quality' not in key and 'drone_type' not in key and 'noise' not in key}
    batch_pt1['attention_mask'] = batch['attention_mask']
    batch_pt1['seq_length'] = (
        batch['seq_length'] - batch_pt0['seq_length']).clip(0)
    batch_pt1['intrinsic'] = batch['intrinsic']
    # T.ToPILImage()(make_grid(rearrange(batch_pt1['images'],
    #                                    'b l c h w -> (b l) c h w'), normalize=True)
    #                ).save('frames.jpg')

    out = model(**batch)
    # settings for auto-regressive generation
    model_kwargs = {'output_attentions': False,
                    'output_hidden_states': True,
                    'use_cache': True}
    out_pt0 = model(**batch_pt0, **model_kwargs)
    out_pt1 = model(**batch_pt1,
                    past_key_values=out_pt0.past_key_values,
                    **model_kwargs)
    print('two part difference 0: ',
          torch.max((out_pt0.action_preds -
                     out.action_preds[:, :partition_length]).norm(dim=-1)).item())
    print('two part difference 1: ',
          torch.max((out_pt1.action_preds -
                     out.action_preds[:, partition_length:]).norm(dim=-1)).item())

    seq_length = max(batch['seq_length'])
    batch_pt = {'noise_embed': batch['noise_embed'],
                'quality': batch['quality'],
                'drone_type': batch['drone_type'],
                'intrinsic': batch['intrinsic'],
                }
    out_gen = {
        'drone_type_preds': [],
        'action_preds': [],
        'stop_preds': [],
        'next_state_preds': [],
    }
    for t in range(seq_length):
        batch_pt.update({key: value[:, [t]] for key, value in batch.items()
                         if key != 'seq_length' and key != 'past_key_values' and key != 'attention_mask' and key not in seqence_lvl_keys and 'label' not in key})
        batch_pt['attention_mask'] = batch['attention_mask'][:, :t + 1]
        batch_pt['seq_length'] = torch.ones_like(batch['seq_length']) * (t + 1)
        outputs = model.expand_actions(**batch_pt, gt_forcing=True)
        if t == 0 and batch_pt['drone_type'] is None:
            batch_pt['drone_type'] = (outputs.drone_type_preds > 0).long()
        batch_pt['past_key_values'] = outputs.past_key_values
        for key in out_gen.keys() & outputs.keys():
            if key == 'drone_type_preds' and t == 0:
                out_gen[key] == outputs[key]
            out_gen[key].append(outputs[key])

    for key in out_gen.keys():
        if out_gen[key] and key != 'drone_type_preds':
            out_gen[key] = torch.cat(out_gen[key], dim=1)
        else:
            pass
    print('iterative difference action: ',
          torch.max((out_gen['action_preds'][batch['attention_mask'].bool()] -
                     out['action_preds'][batch['attention_mask'].bool()]).norm(dim=-1)).item())
    print('iterative difference next_state: ',
          torch.max((out_gen['next_state_preds'][batch['attention_mask'].bool()] -
                     out['next_state_preds'][batch['attention_mask'].bool()]).norm(dim=-1)).item())
    print('iterative difference stop: ',
          torch.max((out_gen['stop_preds'][batch['attention_mask'].bool()] -
                     out['stop_preds'][batch['attention_mask'].bool()])).item())

    return

    import time
    import copy

    seq_length = model.max_data_frames // model.fps_downsample
    batch = {
        'noise_embed': batch['noise_embed'][[0]],
        'quality': batch['quality'][[0]],
        'drone_type': batch['drone_type'][[0]],
        'intrinsic': batch['intrinsic'][[0]],
        'images': torch.randn([1, seq_length, 3, dataset.resolution[0], dataset.resolution[1]], device=device),
        'states': torch.randn([1, seq_length, model.n_action_every_image, model.config.state_dim], device=device),
        'actions': torch.randn([1, seq_length, model.n_action_every_image, model.config.action_dim], device=device),
        'seq_length': torch.ones([1], dtype=torch.long) * seq_length,
        'time_steps': torch.arange(seq_length, device=device, dtype=torch.long)[None],
        'attention_mask': torch.ones([1, seq_length], device=device),
    }

    t0 = time.time()
    for repeat in range(10):
        batch_pt = {'noise_embed': batch['noise_embed'],
                    'quality': batch['quality'],
                    'drone_type': batch['drone_type'],
                    'intrinsic': batch['intrinsic'],
                    }

        chunk_offset = 0
        chunk_size = model.max_model_frames // model.fps_downsample
        chunk_step = 1  # chunk_size // 2

        for t in range(seq_length):
            batch_pt.update({key: value[:, [t]] for key, value in batch.items()
                            if key != 'seq_length' and key != 'past_key_values' and key != 'attention_mask' and key not in seqence_lvl_keys})
            batch_pt['attention_mask'] = \
                batch['attention_mask'][:, chunk_offset:t + 1]
            batch_pt['seq_length'] = \
                torch.ones_like(batch['seq_length']) * (t + 1 - chunk_offset)
            batch_pt['time_steps'] = batch_pt['time_steps'] - chunk_offset
            with torch.no_grad():
                outputs = model.expand_actions(**batch_pt)
            batch_pt['past_key_values'] = outputs.past_key_values

            # determine if need chunking
            if (t + 1 - chunk_offset) == chunk_size and t + 1 < seq_length:
                chunk_offset += chunk_step
                # use batch_pt for getting the past_key_values
                batch_pt.update({key: value[:, chunk_offset:chunk_offset + (chunk_size - chunk_step)]
                                 for key, value in batch.items()
                                 if key != 'seq_length' and key != 'past_key_values' and key not in seqence_lvl_keys})
                batch_pt['seq_length'] = \
                    torch.ones_like(batch['seq_length']) * \
                    (chunk_size - chunk_step)
                batch_pt['time_steps'] = batch_pt['time_steps'] - chunk_offset
                batch_pt['past_key_values'] = None
                with torch.no_grad():
                    outputs = model(**batch_pt)
                # only include last chunk_offset frames
                batch_pt['past_key_values'] = outputs.past_key_values

    t1 = time.time()
    print('inference speed', model.config.max_model_frames, t1 - t0)

    pass


if __name__ == '__main__':
    main()
