"""
Large Motion Video Autoencoding with Cross-modal Video VAE
- https://arxiv.org/pdf/2412.17805

CV-VAE: A Compatible Video VAE for Latent
Generative Video Models
- https://arxiv.org/pdf/2405.20279

> 15:05

# Regularization
- Assume enc_2d(x) -> z ~ p^i(z)
    - Where p^i(z) is the distribution of latents produced by the 2D image encoder
    - Joint distribution of video frames: p^i(Z) = \prodsum_{k}^{t+1} p^i (z_k)
        - or...Z ~ p^v(Z)
    - "want to map p^i(z) -> p^v(Z)...neither have an analytic formulation"
- Alignment b/w image and video encoders
    - \tild{X^{v}_{i}} =  D_v ( E_i ( \phi(X) ) )
        - Reconstructed video X
        - \phi(X) -> 
"""

import torch
import torch.nn as nn


class PretrainedEncoder2D(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   
    
    def forward(self, x: torch.Tensor) -> None:
        """
        :x: image [H, W, C]
        """
        pass
    

class Encoder3D(nn.Module):
    """
    Map input video X to latent code.
    - e(X) -> Z
    
    Definitions
    ---
    :rho_s: H/h = W/w # compression rate (space)
    :rho_t: T/t       # compression rate (time)
    
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)    
    
    def forward(self, X: torch.Tensor):
        """
        Args
        ---
        :x: image tensor [B, H, W, C]
        :X: video tensor [B, T, H, W, C]
        
        Returns
        ---
        :z: image latent [T, H, W, C]
        :Z: video latent [T, H, W, C]
        """
        
        if len(X.shape) == 4:
            # process as image
            pass
        elif len(X.shape) == 5:
            # process as video
            pass
        else:
            raise Exception(f"Invalid input shape: {X.shape}")


class PretrainedDecoder2D(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   
    
    def forward(self, Z: torch.Tensor) -> None: 
        pass 


class Decoder3D(nn.Module):
    """
    Map latent Z to original video X.
    - e^-1(Z) -> X
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
    
    def forward(self, Z: torch.Tensor):
        """
        Args
        ---
        :z: image latent [B, H, W, C]
        :Z: video latent [B, T, H, W, C]
        
        Returns
        ---
        :x: image tensor [T, H, W, C]
        :X: video tensor [T, H, W, C]
        """
        
        if len(Z.shape) == 4:
            # process as image
            pass
        elif len(Z.shape) == 5:
            # process as video
            pass
        else:
            raise Exception(f"Invalid input shape: {Z.shape}")