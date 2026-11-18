from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import os
import math

__all__ = ['ResNetPart', 'resnet18part', 'resnet34part', 'resnet50part', 'resnet101part',
           'resnet152part']

    
class PatchGenerator(nn.Module):
    """
    Patch Generator for Spatial Transformer Network (STN)
    Lightweight 2-layer MLP with bounded residual affine parameterization (P2)
    Generates non-singular canonical horizontal stripe regions with bounded residual tuning
    """
    def __init__(self):
        super(PatchGenerator, self).__init__()

        # Lightweight localization network: 2048 -> 256 -> 18 (P2)
        # Replaces 75.5M parameter Conv2d with ~529k parameter MLP
        self.fc_loc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3 * 2 * 3),
        )
        
        # Zero-init last layer weights & biases so it starts exactly at canonical anchor
        nn.init.zeros_(self.fc_loc[3].weight)
        nn.init.zeros_(self.fc_loc[3].bias)
        
        # Non-singular canonical horizontal stripe anchor [3, 2, 3] (det = 1/3 > 0, P2)
        # Part 0 (top 1/3): s_x=1.0, s_y=1/3, t_x=0.0, t_y=-2/3
        # Part 1 (mid 1/3): s_x=1.0, s_y=1/3, t_x=0.0, t_y=0.0
        # Part 2 (bot 1/3): s_x=1.0, s_y=1/3, t_x=0.0, t_y=+2/3
        anchor = torch.tensor([
            [[1.0, 0.0, 0.0],
             [0.0, 1.0/3.0, -2.0/3.0]],
            [[1.0, 0.0, 0.0],
             [0.0, 1.0/3.0, 0.0]],
            [[1.0, 0.0, 0.0],
             [0.0, 1.0/3.0, 2.0/3.0]]
        ], dtype=torch.float32)
        self.register_buffer('anchor', anchor)
        
        # Perturbation bounds gamma: [3, 2, 3]
        # Restricts scale, shear/rotation, and translation to physically valid ranges
        gamma = torch.tensor([
            [[0.15, 0.05, 0.10],
             [0.05, 0.10, 0.15]],
            [[0.15, 0.05, 0.10],
             [0.05, 0.10, 0.15]],
            [[0.15, 0.05, 0.10],
             [0.05, 0.10, 0.15]]
        ], dtype=torch.float32)
        self.register_buffer('gamma', gamma)

    def forward(self, x):
        """
        Forward pass to generate bounded transformation parameters
        Args:
            x: Input feature map [B, 2048, H, W] or pooled feature [B, 2048]
        Returns:
            theta: Bounded transformation parameters [B, 3, 2, 3]
        """
        if x.dim() == 4:
            f = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        else:
            f = x.view(x.size(0), -1)
            
        delta = self.fc_loc(f).view(-1, 3, 2, 3)
        theta = self.anchor.unsqueeze(0) + self.gamma.unsqueeze(0) * torch.tanh(delta)
        return theta

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule for diffusion process
    Args:
        timesteps: Number of diffusion steps
        s: Small constant to prevent beta from being too small
    Returns:
        betas: Noise schedule parameters
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffusionPatchGenerator(nn.Module):
    """
    Diffusion-based Patch Generator
    Score-Conditioned Denoiser (P1) and Accelerated Analytic 5-Step DDIM (P5)
    Conditioned on [theta_t, f_g, emb(t)] to break the 1.0 MSE variance floor.
    """
    def __init__(self, num_steps=1000):
        super(DiffusionPatchGenerator, self).__init__()
        self.num_steps = num_steps
        
        # Initialize diffusion process parameters with cosine schedule
        betas = cosine_beta_schedule(num_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        
        # Register buffers for diffusion process
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Coefficients for forward diffusion and analytic reverse
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # Score-Conditioned Denoiser Architecture (P1)
        # Input: [theta_t (18), f_g (2048), t_emb (512)] = 2578
        input_dim = 18 + 2048 + 512
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 18)
        )
        
        # Zero-init last layer weights & bias for identity baseline
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
    def get_timestep_embedding(self, t):
        """Sinusoidal time encoding for timesteps t"""
        half_dim = 512 // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def predict_x0(self, x_t, noise_pred, t):
        """Analytic reconstruction of x_0 from x_t and predicted noise epsilon"""
        sqrt_recip_alpha = self.sqrt_recip_alphas_cumprod[t].view(-1, 1)
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1)
        return sqrt_recip_alpha * x_t - sqrt_recipm1 * noise_pred

    def forward(self, f_g, t=None, theta=None):
        """
        Forward pass for training denoiser (P1)
        Args:
            f_g: Global feature representation [B, 2048]
            t: Timestep tensor [B] or None (sampled uniformly)
            theta: Anchor transformation parameters [B, 3, 2, 3] or [B, 18]
        Returns:
            noise_pred, noisy_theta, noise, pred_theta_0
        """
        B = f_g.size(0)
        if theta is not None:
            theta_flat = theta.view(B, 18)
            if t is None:
                t = torch.randint(0, self.num_steps, (B,), device=f_g.device).long()
            elif isinstance(t, int):
                t = torch.full((B,), t, device=f_g.device, dtype=torch.long)
            elif t.dim() == 0:
                t = t.unsqueeze(0).expand(B).long()
            elif t.size(0) == 1:
                t = t.expand(B).long()
            else:
                t = t.long()

            noise = torch.randn_like(theta_flat)
            noisy_theta = self.q_sample(x_start=theta_flat, t=t, noise=noise)

            t_emb = self.get_timestep_embedding(t)
            # Concatenate [theta_t, f_g, t_emb] (P1: score condition)
            h = torch.cat([noisy_theta, f_g, t_emb], dim=1)
            noise_pred = self.net(h)
            pred_theta_0 = self.predict_x0(noisy_theta, noise_pred, t)

            return (noise_pred.view(B, 3, 2, 3),
                    noisy_theta.view(B, 3, 2, 3),
                    noise.view(B, 3, 2, 3),
                    pred_theta_0.view(B, 3, 2, 3))
        else:
            return self.ddim_sample(f_g)

    @torch.no_grad()
    def ddim_sample(self, f_g, num_steps=5, theta_init=None):
        """
        Accelerated Analytic 5-Step DDIM Sampling (P5)
        Deterministic reverse trajectory (eta=0) conditioned on [theta_t, f_g, emb(t)].
        Eliminates index-scrambling theta_renewal and reduces latency by 200x.
        """
        B = f_g.size(0)
        times = torch.linspace(0, self.num_steps - 1, steps=num_steps, dtype=torch.long, device=f_g.device)
        reversed_times = times.flip(0)  # e.g., [999, 749, 499, 249, 0]

        if theta_init is not None:
            theta_flat = theta_init.view(B, 18)
            t_start = reversed_times[0]
            t_batch = torch.full((B,), t_start, device=f_g.device, dtype=torch.long)
            noise = torch.randn_like(theta_flat)
            theta = self.q_sample(theta_flat, t_batch, noise=noise)
        else:
            theta = torch.randn(B, 18, device=f_g.device, dtype=torch.float32)

        for k in range(len(reversed_times)):
            t = reversed_times[k]
            t_batch = torch.full((B,), t, device=f_g.device, dtype=torch.long)
            t_emb = self.get_timestep_embedding(t_batch)
            h = torch.cat([theta, f_g, t_emb], dim=1)
            noise_pred = self.net(h)
            pred_x0 = self.predict_x0(theta, noise_pred, t_batch)

            if k == len(reversed_times) - 1:
                theta = pred_x0
            else:
                t_prev = reversed_times[k + 1]
                t_prev_batch = torch.full((B,), t_prev, device=f_g.device, dtype=torch.long)
                sqrt_alpha_prev = self.sqrt_alphas_cumprod[t_prev_batch].view(B, 1)
                sqrt_one_minus_alpha_prev = self.sqrt_one_minus_alphas_cumprod[t_prev_batch].view(B, 1)
                # Deterministic DDIM update (eta=0)
                theta = sqrt_alpha_prev * pred_x0 + sqrt_one_minus_alpha_prev * noise_pred

        return theta.view(B, 3, 2, 3)


class ResNetPart(nn.Module):
    """
    ResNet with part features by uniform partitioning
    Combines global and local features for person/vehicle re-identification
    """
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, num_parts=3, num_classes=0):
        super(ResNetPart, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.eval_fusion = True  # P6: Test-time feature fusion enabled by default
        
        # Construct base (pretrained) resnet
        if depth not in ResNetPart.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNetPart.__factory[depth](pretrained=pretrained)
        
        # Modify stride in last layer for better feature resolution
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        self.num_parts = num_parts
        self.num_classes = num_classes
        
        # Base network for feature extraction
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
            
        # Global and local feature pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.rap = nn.AdaptiveAvgPool2d((self.num_parts, 1))  # Regional average pooling

        # Global feature classifier
        self.bnneck = nn.BatchNorm1d(2048)
        init.constant_(self.bnneck.weight, 1)
        init.constant_(self.bnneck.bias, 0)
        self.bnneck.bias.requires_grad_(False)

        self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        init.normal_(self.classifier.weight, std=0.001)

        # Part feature classifiers
        for i in range(self.num_parts):
            name = 'bnneck' + str(i)
            setattr(self, name, nn.BatchNorm1d(2048))
            init.constant_(getattr(self, name).weight, 1)
            init.constant_(getattr(self, name).bias, 0)
            getattr(self, name).bias.requires_grad_(False)

            name = 'classifier' + str(i)
            setattr(self, name, nn.Linear(2048, self.num_classes, bias=False))
            
        # Initialize patch generation modules
        self.patch_proposal = PatchGenerator()
        self.diffusion_patch = DiffusionPatchGenerator()
        
        if not pretrained:
            self.reset_params()
    
    def forward(self, x):
        x = self.base(x)
        f_g_raw = self.gap(x).view(x.size(0), -1)
        f_g = self.bnneck(f_g_raw)
        
        if self.training is False:
            f_g_norm = F.normalize(f_g, p=2, dim=1)
            if not getattr(self, 'eval_fusion', True):
                return f_g_norm

            # P6: Test-time feature fusion of global + DDIM-refined parts
            initial_theta = self.patch_proposal(f_g_raw)
            theta = self.diffusion_patch.ddim_sample(f_g, num_steps=5, theta_init=initial_theta)
            
            f_p = []
            for i in range(self.num_parts):
                stripe = theta[:, i, :, :].float()
                grid = F.affine_grid(stripe, x.size(), align_corners=False)
                f_p_i = F.grid_sample(x, grid, align_corners=False)
                f_p_i = self.gap(f_p_i).view(x.size(0), -1)
                f_p_i = getattr(self, 'bnneck' + str(i))(f_p_i)
                f_p.append(F.normalize(f_p_i, p=2, dim=1))
                
            f_p_mean = torch.stack(f_p, dim=1).mean(dim=1)
            f_p_mean = F.normalize(f_p_mean, p=2, dim=1)
            f_fused = torch.cat([f_g_norm, 0.5 * f_p_mean], dim=1)
            f_fused = F.normalize(f_fused, p=2, dim=1)
            return f_fused

        logits_g = self.classifier(f_g)
        
        # Training phase:
        initial_theta = self.patch_proposal(f_g_raw)
        noise_pred, noisy_theta, noise, pred_theta_0 = self.diffusion_patch(f_g, theta=initial_theta)
        
        # Generate patch features using noisy theta
        f_p = []
        for i in range(self.num_parts):
            stripe = noisy_theta[:, i, :, :].float()
            grid = F.affine_grid(stripe, x.size(), align_corners=False)
            f_p.append(F.grid_sample(x, grid, align_corners=False))
            
        logits_p = []
        fs_p = []
        
        for i in range(self.num_parts):
            f_p_i = self.gap(f_p[i]).view(f_p[i].size(0), -1)
            f_p_i = self.bnneck(f_p_i)
            f_p_i = getattr(self, 'bnneck' + str(i))(f_p_i)
            fs_p.append(f_p_i)
            logits_p_i = getattr(self, 'classifier' + str(i))(f_p_i)
            logits_p.append(logits_p_i)
            
        fs_p = torch.stack(fs_p, dim=-1)
        logits_p = torch.stack(logits_p, dim=-1)

        return f_g, fs_p, logits_g, logits_p, noise_pred, noisy_theta, noise, initial_theta, pred_theta_0

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        # Ensure STN and Diffusion output heads remain zero-initialized
        nn.init.zeros_(self.patch_proposal.fc_loc[3].weight)
        nn.init.zeros_(self.patch_proposal.fc_loc[3].bias)
        nn.init.zeros_(self.diffusion_patch.net[-1].weight)
        nn.init.zeros_(self.diffusion_patch.net[-1].bias)
    
    def extract_all_features(self, x):
        x = self.base(x)
        f_g_raw = self.gap(x).view(x.size(0), -1)
        f_g = self.bnneck(f_g_raw)
        f_g = F.normalize(f_g, p=2, dim=1)
        
        # Get initial theta and run fast 5-step DDIM refinement (P5)
        initial_theta = self.patch_proposal(f_g_raw)
        theta = self.diffusion_patch.ddim_sample(f_g, num_steps=5, theta_init=initial_theta)
        
        # Generate patch features using refined theta
        f_p = []
        for i in range(self.num_parts):
            stripe = theta[:, i, :, :].float()
            grid = F.affine_grid(stripe, x.size(), align_corners=False)
            f_p_i = F.grid_sample(x, grid, align_corners=False)
            f_p_i = self.gap(f_p_i).view(f_p_i.size(0), -1)
            f_p_i = getattr(self, 'bnneck' + str(i))(f_p_i)
            f_p_i = F.normalize(f_p_i, p=2, dim=1)
            f_p.append(f_p_i)
            
        fs_p = torch.stack(f_p, dim=-1)
        
        return f_g, fs_p
        

def resnet18part(**kwargs):
    return ResNetPart(18, **kwargs)


def resnet34part(**kwargs):
    return ResNetPart(34, **kwargs)


def resnet50part(**kwargs):
    return ResNetPart(50, **kwargs)


def resnet101part(**kwargs):
    return ResNetPart(101, **kwargs)


def resnet152part(**kwargs):
    return ResNetPart(152, **kwargs)
