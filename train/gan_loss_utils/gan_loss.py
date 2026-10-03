from gan_loss_utils.sd_utils import get_x0_from_noise, DummyNetwork, NoOpContext
from diffusers import UNet2DConditionModel, DDIMScheduler
from gan_loss_utils.sd_unet_forward import classify_forward
import torch.nn.functional as F
import torch.nn as nn
import torch
import types

cls_pred_branch = nn.Sequential(
        nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1), # 32x32 -> 16x16 
        nn.GroupNorm(num_groups=32, num_channels=1280),
        nn.SiLU(),
        nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1), # 16x16 -> 8x8 
        nn.GroupNorm(num_groups=32, num_channels=1280),
        nn.SiLU(),
        nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1), # 8x8 -> 4x4
        nn.GroupNorm(num_groups=32, num_channels=1280),
        nn.SiLU(),
        nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=4, padding=0), # 4x4 -> 1x1
        nn.GroupNorm(num_groups=32, num_channels=1280),
        nn.SiLU(),
        nn.Conv2d(kernel_size=1, in_channels=1280, out_channels=1, stride=1, padding=0), # 1x1 -> 1x1
    )

cls_pred_branch.requires_grad_(True)

def compute_distribution_matching_loss(latents, fake, real, timesteps):
    with torch.no_grad():
        original_latents = latents
        pred_fake_x0 = fake
        pred_real_x0 = real

        p_real = (latents - pred_real_x0)
        p_fake = (latents - pred_fake_x0)

        grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True) 
        grad = torch.nan_to_num(grad)

    loss = 0.5 * F.mse_loss(original_latents.float(), (original_latents-grad).detach().float(), reduction="mean")
    return loss

def compute_cls_logits(pred_fake_x0):
        # we are operating on the VAE latent space, no further normalization needed for now 
        rep = pred_fake_x0

        # we only use the bottleneck layer 
        rep = rep[-1].float()
        logits = cls_pred_branch(rep).squeeze(dim=[2, 3])
        return logits
            
def compute_generator_clean_cls_loss(pred_fake_x0):
        pred_realism_on_fake_with_grad = compute_cls_logits(pred_fake_x0)
        loss = F.softplus(-pred_realism_on_fake_with_grad).mean()
        return loss 

def compute_guidance_clean_cls_loss(
        self, real_image, fake_image, 
        real_text_embedding, fake_text_embedding,
        real_unet_added_conditions=None, 
        fake_unet_added_conditions=None
    ):
    pred_realism_on_real = self.compute_cls_logits(
        real_image.detach(), 
        text_embedding=real_text_embedding,
        unet_added_conditions=real_unet_added_conditions
    )
    pred_realism_on_fake = self.compute_cls_logits(
        fake_image.detach(), 
        text_embedding=fake_text_embedding,
        unet_added_conditions=fake_unet_added_conditions
    )

    log_dict = {
        "pred_realism_on_real": torch.sigmoid(pred_realism_on_real).squeeze(dim=1).detach(),
        "pred_realism_on_fake": torch.sigmoid(pred_realism_on_fake).squeeze(dim=1).detach()
    }

    classification_loss = F.softplus(pred_realism_on_fake).mean() + F.softplus(-pred_realism_on_real).mean()
    loss_dict = {
        "guidance_cls_loss": classification_loss
    }
    return loss_dict, log_dict

def compute_loss_fake(noisy_latents, fake_x0_pred, text_embedding, uncond_embedding, unet_added_conditions=None, uncond_unet_added_conditions=None):
    # epsilon prediction loss 
    loss_fake = torch.mean(
        (fake_noise_pred.float() - noise.float())**2
    )

    loss_dict = {
        "loss_fake_mean": loss_fake,
    }

    fake_log_dict = {
        "faketrain_latents": latents.detach().float(),
        "faketrain_noisy_latents": noisy_latents.detach().float(),
        "faketrain_x0_pred": fake_x0_pred.detach().float()
    }

    return loss_dict, fake_log_dict