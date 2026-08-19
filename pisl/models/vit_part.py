import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import math
import timm

__all__ = ['vit_base_part']

class VitPatchGenerator(nn.Module):
    """Generates initial attention masks for parts directly from patch tokens."""
    def __init__(self, embed_dim=768, num_parts=3):
        super().__init__()
        self.num_parts = num_parts
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, num_parts)
        )
        # Initialize to uniform attention across patches roughly
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, patch_tokens):
        # patch_tokens: [B, N, D]
        masks = self.net(patch_tokens) # [B, N, num_parts]
        masks = masks.transpose(1, 2) # [B, num_parts, N]
        return masks

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffusionVitPatchGenerator(nn.Module):
    def __init__(self, embed_dim=768, num_parts=3, num_tokens=192, num_steps=1000):
        super().__init__()
        self.num_steps = num_steps
        self.num_parts = num_parts
        self.num_tokens = num_tokens
        self.mask_dim = num_parts * num_tokens
        
        betas = cosine_beta_schedule(num_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # We condition on the global feature (CLS token)
        input_dim = embed_dim + 512 # cls feature + time encoding
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.mask_dim)
        )
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()
        
        self.scale = 1.0
        
    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
               self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
               
    def get_timestep_embedding(self, t):
        half_dim = 256
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb
        
    def forward(self, f_g, t, masks=None):
        if isinstance(t, int):
            t = torch.tensor([t], device=f_g.device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.size(0) == 1:
            t = t.expand(f_g.size(0))
            
        t_emb = self.get_timestep_embedding(t)
        cond = torch.cat([f_g, t_emb], dim=1)
        
        if masks is not None:
            noise = torch.randn_like(masks)
            noisy_masks = self.q_sample(masks, t, noise)
            noise_pred = self.net(cond)
            # Re-shape predicted noise to match masks shape
            noise_pred = noise_pred.view(-1, self.num_parts, self.num_tokens)
            return noise_pred, noisy_masks, noise
        else:
            noise_pred = self.net(cond)
            return noise_pred.view(-1, self.num_parts, self.num_tokens)

    def prepare_diffusion_concat(self, gt_masks):
        t = torch.randint(0, self.num_steps, (gt_masks.size(0),), device=gt_masks.device).long()
        noise = torch.randn_like(gt_masks)
        x = self.q_sample(gt_masks, t, noise)
        return x, noise, t

    @torch.no_grad()
    def ddim_sample(self, f_g, num_steps=50):
        # Simplified DDIM sampling
        batch_size = f_g.size(0)
        masks = torch.randn(batch_size, self.num_parts, self.num_tokens, device=f_g.device)
        
        step_size = self.num_steps // num_steps
        timesteps = list(reversed(range(0, self.num_steps, step_size)))
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=f_g.device, dtype=torch.long)
            noise_pred = self.forward(f_g, t_batch)
            
            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i+1]] if i < len(timesteps) - 1 else torch.tensor(1.0)
            
            pred_x0 = (masks - torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)
            dir_xt = torch.sqrt(1 - alpha_prev) * noise_pred
            masks = torch.sqrt(alpha_prev) * pred_x0 + dir_xt
            
        return masks

class VitPart(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, num_parts=3, num_classes=0, img_size=(256, 128)):
        super().__init__()
        self.num_parts = num_parts
        self.num_classes = num_classes
        
        # Load pre-trained ViT, telling it our custom image size to interpolate positional embeddings
        self.base = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        self.embed_dim = self.base.embed_dim
        
        # Calculate number of patch tokens
        self.patch_size = 16  # standard for vit_base_patch16
        num_tokens = (img_size[0] // self.patch_size) * (img_size[1] // self.patch_size)
        
        self.bnneck = nn.BatchNorm1d(self.embed_dim)
        init.constant_(self.bnneck.weight, 1)
        init.constant_(self.bnneck.bias, 0)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.embed_dim, self.num_classes, bias=False)
        init.normal_(self.classifier.weight, std=0.001)

        for i in range(self.num_parts):
            name = 'bnneck' + str(i)
            setattr(self, name, nn.BatchNorm1d(self.embed_dim))
            init.constant_(getattr(self, name).weight, 1)
            init.constant_(getattr(self, name).bias, 0)
            getattr(self, name).bias.requires_grad_(False)
            
            name = 'classifier' + str(i)
            setattr(self, name, nn.Linear(self.embed_dim, self.num_classes, bias=False))
            
        self.patch_proposal = VitPatchGenerator(self.embed_dim, num_parts)
        self.diffusion_patch = DiffusionVitPatchGenerator(self.embed_dim, num_parts, num_tokens)
        
        if not pretrained:
            self.reset_params()

    def forward_features(self, x):
        # Extract features from timm ViT
        x = self.base.patch_embed(x)
        x = self.base._pos_embed(x)
        x = self.base.norm_pre(x)
        x = self.base.blocks(x)
        x = self.base.norm(x)
        return x # [B, 193, 768] (1 CLS + 192 patch tokens)

    def forward(self, x):
        features = self.forward_features(x)
        f_g = features[:, 0] # CLS token
        patch_tokens = features[:, 1:] # Patch tokens [B, 192, 768]
        
        f_g_norm = self.bnneck(f_g)
        
        if not self.training:
            f_g_norm = F.normalize(f_g_norm)
            return f_g_norm

        logits_g = self.classifier(f_g_norm)
        
        if self.training:
            initial_masks = self.patch_proposal(patch_tokens) # [B, num_parts, 192]
            # Normalize masks via softmax for weighted sum pooling
            attn_weights = F.softmax(initial_masks, dim=-1)
            
            noisy_masks, noise, t = self.diffusion_patch.prepare_diffusion_concat(initial_masks)
            noise_pred = self.diffusion_patch(f_g, t)
            
            f_p = []
            noisy_attn = F.softmax(noisy_masks, dim=-1)
            for i in range(self.num_parts):
                part_feat = (patch_tokens * noisy_attn[:, i:i+1, :].transpose(1, 2)).sum(dim=1)
                f_p.append(part_feat)
        else:
            masks = self.diffusion_patch.ddim_sample(f_g)
            attn_weights = F.softmax(masks, dim=-1)
            
            f_p = []
            for i in range(self.num_parts):
                part_feat = (patch_tokens * attn_weights[:, i:i+1, :].transpose(1, 2)).sum(dim=1)
                f_p.append(part_feat)
                
        logits_p = []
        fs_p = []
        
        for i in range(self.num_parts):
            f_p_i = f_p[i]
            f_p_i = getattr(self, 'bnneck' + str(i))(f_p_i)
            fs_p.append(f_p_i)
            logits_p_i = getattr(self, 'classifier' + str(i))(f_p_i)
            logits_p.append(logits_p_i)
            
        fs_p = torch.stack(fs_p, dim=-1)
        logits_p = torch.stack(logits_p, dim=-1)

        if self.training:
            return f_g_norm, fs_p, logits_g, logits_p, noise_pred, noisy_masks, noise, initial_masks
        else:
            return f_g_norm, fs_p, logits_g, logits_p

    def extract_all_features(self, x):
        features = self.forward_features(x)
        f_g = features[:, 0]
        patch_tokens = features[:, 1:]
        
        f_g_norm = self.bnneck(f_g)
        f_g_norm = F.normalize(f_g_norm)
        
        # Inference phase
        masks = self.diffusion_patch.ddim_sample(f_g)
        attn_weights = F.softmax(masks, dim=-1)
        
        f_p = []
        for i in range(self.num_parts):
            part_feat = (patch_tokens * attn_weights[:, i:i+1, :].transpose(1, 2)).sum(dim=1)
            f_p_i = getattr(self, 'bnneck' + str(i))(part_feat)
            f_p_i = F.normalize(f_p_i)
            f_p.append(f_p_i)
            
        fs_p = torch.stack(f_p, dim=-1)
        
        return f_g_norm, fs_p

def vit_base_part(**kwargs):
    return VitPart(model_name='vit_base_patch16_224', **kwargs)
