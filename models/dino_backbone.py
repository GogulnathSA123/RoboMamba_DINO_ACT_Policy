import warnings
import torch
import torch.nn as nn

class DINOv2VisionBackbone(nn.Module):
    """Pre-trained DINOv2 visual backbone for dense spatial patch representation."""

    def __init__(self, model_name="dinov2_vits14", d_model=512, freeze=True, pool_spatial=True):
        super().__init__()
        self.model_name = model_name
        self.pool_spatial = pool_spatial

        print(f"Loading pre-trained DINOv2 backbone '{model_name}'...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=".*dinov2.*")
            self.dinov2 = torch.hub.load("facebookresearch/dinov2", model_name)
        
        self.raw_embed_dim = self.dinov2.embed_dim  # 384 for ViT-Small, 768 for ViT-Base

        if freeze:
            print("Freezing DINOv2 parameters for fast training and low GPU memory footprint.")
            for param in self.dinov2.parameters():
                param.requires_grad = False

        # Projection layer: DINOv2 embedding dimension -> Mamba/ACT d_model
        self.projection = nn.Linear(self.raw_embed_dim, d_model)

    def forward(self, x):
        """
        Input:
            x: Camera tensor of shape [B, Num_Cams, 3, 224, 224]
        Output:
            visual_tokens: Projected visual tokens of shape [B, Num_Cams * Tokens_Per_Cam, d_model]
        """
        B, N_cams, C, H, W = x.shape
        x_flat = x.view(B * N_cams, C, H, W)

        if self.training and not any(p.requires_grad for p in self.dinov2.parameters()):
            with torch.no_grad():
                features = self.dinov2.forward_features(x_flat)
        else:
            features = self.dinov2.forward_features(x_flat)

        # Extract patch tokens: [B * N_cams, 256, Embed_Dim]
        patch_tokens = features["x_norm_patchtokens"]

        if self.pool_spatial:
            # Spatial pooling (16x16 grid -> 4x4 grid) to optimize Mamba sequence latency
            B_N, N_patches, E = patch_tokens.shape
            grid_size = int(N_patches ** 0.5)  # 16 for 224x224 input
            patches_2d = patch_tokens.view(B_N, grid_size, grid_size, E).permute(0, 3, 1, 2)
            pooled_2d = nn.functional.adaptive_avg_pool2d(patches_2d, (4, 4))  # 16 tokens per cam
            tokens = pooled_2d.permute(0, 2, 3, 1).reshape(B_N, 16, E)
        else:
            tokens = patch_tokens

        # Project embedding dimension: [B * N_cams, Tokens, d_model]
        projected = self.projection(tokens)

        # Reshape to [B, N_cams * Tokens, d_model]
        tokens_per_cam = projected.shape[1]
        visual_tokens = projected.view(B, N_cams * tokens_per_cam, -1)

        return visual_tokens
