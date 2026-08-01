import torch
import torch.nn as nn
from .dino_backbone import DINOv2VisionBackbone
from .mamba_block import RoboMambaSequenceEncoder
from .act_decoder import CVAEEncoder, ACTActionDecoder

class DINOv2RoboMambaACTPolicy(nn.Module):
    """
    Unified Policy Architecture:
      - Vision Backbone: Pre-trained DINOv2 (Meta AI) with spatial pooling
      - Sequence Model: RoboMamba (Selective State Space Model)
      - Action Decoder: ACT (Action Chunking with Transformers + CVAE)

    Implementation follows the original repos:
      - DINOv2: facebookresearch/dinov2 (forward_features -> x_norm_patchtokens)
      - ACT: tonyzhaozh/act (CVAE encoder with CLS token, sinusoidal pos, decoder with
        additional_pos_embed for [latent, proprio], single linear action_head)
      - Mamba: state-spaces/mamba (selective scan with input-dependent dt, B, C)
    """

    def __init__(
        self,
        dino_model="dinov2_vits14",
        freeze_dino=True,
        d_model=512,
        n_mamba_layers=4,
        state_dim=6,
        action_dim=6,
        chunk_size=16,
        latent_dim=32,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # 1. Vision Backbone (DINOv2)
        self.vision = DINOv2VisionBackbone(
            model_name=dino_model,
            d_model=d_model,
            freeze=freeze_dino,
            pool_spatial=True,
        )

        # 2. Joint State Embedding (for Mamba sequence input)
        self.state_encoder = nn.Linear(state_dim, d_model)

        # 3. Spatiotemporal Sequence Encoder (Mamba SSM)
        self.mamba = RoboMambaSequenceEncoder(d_model=d_model, n_layers=n_mamba_layers)

        # 4. CVAE Encoder (training only — discarded at inference)
        self.cvae = CVAEEncoder(
            action_dim=action_dim,
            state_dim=state_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            chunk_size=chunk_size,
        )

        # 5. ACT Action Decoder (with latent + proprio as additional memory tokens)
        self.decoder = ACTActionDecoder(
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            latent_dim=latent_dim,
            state_dim=state_dim,
        )

    def forward(self, images, state, actions=None):
        """
        Input:
            images: Multi-camera RGB tensor [B, Num_Cams, 3, 224, 224]
            state: Normalized robot joint state [B, state_dim]
            actions: (Training only) Ground-truth action chunk [B, chunk_size, action_dim]
        Output:
            pred_actions: Predicted action chunk [B, chunk_size, action_dim]
            mu: CVAE mean (None during inference)
            logvar: CVAE log variance (None during inference)
        """
        B = images.shape[0]

        # --- Observation Encoding ---

        # Extract visual tokens using DINOv2: [B, Num_Visual_Tokens, d_model]
        vis_tokens = self.vision(images)

        # Encode joint state: [B, 1, d_model]
        state_token = self.state_encoder(state).unsqueeze(1)

        # Concatenate visual tokens + state token along sequence dimension
        seq_tokens = torch.cat([vis_tokens, state_token], dim=1)  # [B, Seq_Len, d_model]

        # Pass through Mamba SSM for spatiotemporal encoding
        memory = self.mamba(seq_tokens)  # [B, Seq_Len, d_model]

        # --- Action Decoding ---

        if actions is not None:
            # Training: CVAE encoder produces latent z from (actions, state)
            z, mu, logvar = self.cvae(actions, state)
            pred_actions = self.decoder(memory, z, state)
            return pred_actions, mu, logvar
        else:
            # Inference: z = 0 (mean of standard Gaussian prior, matching original ACT)
            z = torch.zeros(B, self.latent_dim, device=images.device)
            pred_actions = self.decoder(memory, z, state)
            return pred_actions, None, None
