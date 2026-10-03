import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

import functools

def calculate_adaptive_weight(nll_loss, g_loss, last_layer=None, discriminator_weight=0):
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    d_weight = d_weight * discriminator_weight
    return d_weight

def hinge_d_loss_with_exemplar_weights(logits_real, logits_fake, weights):
    assert weights.shape[0] == logits_real.shape[0] == logits_fake.shape[0]
    loss_real = torch.mean(F.relu(1. - logits_real), dim=[1,2,3])
    loss_fake = torch.mean(F.relu(1. + logits_fake), dim=[1,2,3])
    loss_real = (weights * loss_real).sum() / weights.sum()
    loss_fake = (weights * loss_fake).sum() / weights.sum()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def l1(x, y):
    return torch.abs(x-y)

def l2(x, y):
    return torch.pow((x-y), 2)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        # spectral norm
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, 64, 4, stride=2, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1))
        self.conv3 = spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1))
        self.conv4 = spectral_norm(nn.Conv2d(256, 512, 4, stride=2, padding=1))
        self.conv5 = spectral_norm(nn.Conv2d(512, 1, 4, stride=1, padding=0))
        
        # normalization
        self.norm2 = nn.InstanceNorm2d(128)
        self.norm3 = nn.InstanceNorm2d(256)
        self.norm4 = nn.InstanceNorm2d(512)
        
        # initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
                    
    def get_features(self, x, low_ref):
        input_cat = torch.cat([x, low_ref], dim=1)
        feat1 = F.leaky_relu(self.conv1(input_cat), 0.2)
        feat2 = F.leaky_relu(self.norm2(self.conv2(feat1)), 0.2)
        feat3 = F.leaky_relu(self.norm3(self.conv3(feat2)), 0.2)
        feat4 = F.leaky_relu(self.norm4(self.conv4(feat3)), 0.2)
        out = self.conv5(feat4)
        return out, [feat1, feat2, feat3, feat4]

    def compute_feature_matching_loss(self, real_feats, fake_feats):
        feat_loss = 0
        for real_feat, fake_feat in zip(real_feats, fake_feats):
            feat_loss += F.l1_loss(fake_feat, real_feat.detach())
        return feat_loss
    
    def forward(self, x, low_ref):
        return self.get_features(x, low_ref)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
