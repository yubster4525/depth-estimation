import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Sequence, List
from models.unet import UNet, create_unet_model

class MDECUNet(nn.Module):
    """UNet model adapted for MDEC benchmark interface.
    
    This model extends the UNet architecture to provide multi-scale outputs
    as required by the MDEC benchmark framework.
    """
    
    def __init__(self, 
                 n_channels=3, 
                 out_scales=(0, 1, 2, 3), 
                 bilinear=True, 
                 base_channels=64,
                 use_virtual_stereo=False,
                 mask_name=None,
                 num_ch_mask=None):
        """
        Initialize the MDEC-compatible UNet model.
        
        Args:
            n_channels (int): Number of input channels (RGB=3)
            out_scales (Union[int, Sequence[int]]): Output scales, where 0 is full resolution
            bilinear (bool): Whether to use bilinear upsampling
            base_channels (int): Number of base channels in UNet
            use_virtual_stereo (bool): Whether to use virtual stereo prediction
            mask_name (str, optional): Name of mask type to use
            num_ch_mask (int, optional): Number of channels for mask prediction
        """
        super(MDECUNet, self).__init__()
        self.n_channels = n_channels
        self.out_scales = [out_scales] if isinstance(out_scales, int) else out_scales
        self.bilinear = bilinear
        self.use_virtual_stereo = use_virtual_stereo
        self.mask_name = mask_name
        self.num_ch_mask = num_ch_mask
        
        # Standard UNet backbone 
        self.inc = UNet.inc
        self.down1 = UNet.down1
        self.down2 = UNet.down2
        self.down3 = UNet.down3
        self.down4 = UNet.down4
        self.up1 = UNet.up1
        self.up2 = UNet.up2
        self.up3 = UNet.up3
        self.up4 = UNet.up4
        
        # Instead of using UNet directly, we recreate its components
        factor = 2 if bilinear else 1
        
        # Initial double convolution
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder (downsampling)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*8, base_channels*8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True)
        )
        
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*8, base_channels*16 // factor, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*16 // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*16 // factor, base_channels*16 // factor, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*16 // factor),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling blocks
        # Scale 3 (1/8 resolution)
        self.up1 = nn.Module()
        self.up1.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else \
                      nn.ConvTranspose2d(base_channels*16 // factor, base_channels*8 // factor, kernel_size=2, stride=2)
        
        self.up1.conv = nn.Sequential(
            nn.Conv2d(base_channels*16 // factor + base_channels*8, base_channels*8 // factor, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*8 // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*8 // factor, base_channels*8 // factor, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*8 // factor),
            nn.ReLU(inplace=True)
        )
        
        # Scale 2 (1/4 resolution)
        self.up2 = nn.Module()
        self.up2.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else \
                      nn.ConvTranspose2d(base_channels*8 // factor, base_channels*4 // factor, kernel_size=2, stride=2)
        
        self.up2.conv = nn.Sequential(
            nn.Conv2d(base_channels*8 // factor + base_channels*4, base_channels*4 // factor, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*4 // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4 // factor, base_channels*4 // factor, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*4 // factor),
            nn.ReLU(inplace=True)
        )
        
        # Scale 1 (1/2 resolution)
        self.up3 = nn.Module()
        self.up3.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else \
                      nn.ConvTranspose2d(base_channels*4 // factor, base_channels*2 // factor, kernel_size=2, stride=2)
        
        self.up3.conv = nn.Sequential(
            nn.Conv2d(base_channels*4 // factor + base_channels*2, base_channels*2 // factor, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*2 // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2 // factor, base_channels*2 // factor, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*2 // factor),
            nn.ReLU(inplace=True)
        )
        
        # Scale 0 (full resolution)
        self.up4 = nn.Module()
        self.up4.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else \
                      nn.ConvTranspose2d(base_channels*2 // factor, base_channels, kernel_size=2, stride=2)
        
        self.up4.conv = nn.Sequential(
            nn.Conv2d(base_channels*2 // factor + base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale output heads
        out_ch = 1
        if self.use_virtual_stereo:
            out_ch = 2  # Mono + stereo channels
            
        self.outc = nn.ModuleDict({
            'outc_0': nn.Conv2d(base_channels, out_ch, kernel_size=1),
            'outc_1': nn.Conv2d(base_channels*2 // factor, out_ch, kernel_size=1),
            'outc_2': nn.Conv2d(base_channels*4 // factor, out_ch, kernel_size=1),
            'outc_3': nn.Conv2d(base_channels*8 // factor, out_ch, kernel_size=1)
        })
        
        # Mask prediction if needed
        if self.mask_name and self.num_ch_mask:
            self.mask = nn.ModuleDict({
                'mask_0': nn.Conv2d(base_channels, num_ch_mask, kernel_size=1),
                'mask_1': nn.Conv2d(base_channels*2 // factor, num_ch_mask, kernel_size=1),
                'mask_2': nn.Conv2d(base_channels*4 // factor, num_ch_mask, kernel_size=1),
                'mask_3': nn.Conv2d(base_channels*8 // factor, num_ch_mask, kernel_size=1)
            })
        
        # Activation
        self.sigmoid = nn.Sigmoid()
    
    def _upsample_add(self, x, y):
        """Upsample x and add it to y.
        
        Args:
            x (Tensor): Tensor to upsample
            y (Tensor): Tensor to add to
            
        Returns:
            Tensor: Upsampled and added tensor
        """
        _, _, h, w = y.size()
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True) + y
    
    def forward(self, x):
        """
        Forward pass for MDEC UNet model.
        
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            Dict: Dictionary containing disp, depth_feats, and optionally mask predictions
        """
        # Encoder
        x1 = self.inc(x)         # Scale 0 (full res)
        x2 = self.down1(x1)      # Scale 1 (1/2)
        x3 = self.down2(x2)      # Scale 2 (1/4)
        x4 = self.down3(x3)      # Scale 3 (1/8)
        x5 = self.down4(x4)      # Scale 4 (1/16)
        
        # Store encoder features
        enc_feat = [x1, x2, x3, x4, x5]
        
        # Decoder with skip connections
        # Scale 3 (1/8)
        x_up1 = self.up1.up(x5)
        diff_y = x4.size()[2] - x_up1.size()[2]
        diff_x = x4.size()[3] - x_up1.size()[3]
        x_up1 = F.pad(x_up1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        dec3 = self.up1.conv(torch.cat([x4, x_up1], dim=1))
        
        # Scale 2 (1/4)
        x_up2 = self.up2.up(dec3)
        diff_y = x3.size()[2] - x_up2.size()[2]
        diff_x = x3.size()[3] - x_up2.size()[3]
        x_up2 = F.pad(x_up2, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        dec2 = self.up2.conv(torch.cat([x3, x_up2], dim=1))
        
        # Scale 1 (1/2)
        x_up3 = self.up3.up(dec2)
        diff_y = x2.size()[2] - x_up3.size()[2]
        diff_x = x2.size()[3] - x_up3.size()[3]
        x_up3 = F.pad(x_up3, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        dec1 = self.up3.conv(torch.cat([x2, x_up3], dim=1))
        
        # Scale 0 (full res)
        x_up4 = self.up4.up(dec1)
        diff_y = x1.size()[2] - x_up4.size()[2]
        diff_x = x1.size()[3] - x_up4.size()[3]
        x_up4 = F.pad(x_up4, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        dec0 = self.up4.conv(torch.cat([x1, x_up4], dim=1))
        
        # Store decoder features at each scale
        dec_feat = {
            0: dec0,
            1: dec1,
            2: dec2,
            3: dec3
        }
        
        # Output at multiple scales
        disp = {}
        for s in self.out_scales:
            if s in dec_feat:
                disp[s] = self.sigmoid(self.outc[f'outc_{s}'](dec_feat[s]))
        
        # Create output dict
        out = {'disp': disp, 'depth_feats': enc_feat}
        
        # Add mask prediction if needed
        if self.mask_name and self.num_ch_mask:
            mask = {}
            for s in self.out_scales:
                if s in dec_feat:
                    if self.mask_name == 'explainability':
                        mask[s] = torch.sigmoid(self.mask[f'mask_{s}'](dec_feat[s]))
                    elif self.mask_name == 'uncertainty':
                        mask[s] = F.relu(self.mask[f'mask_{s}'](dec_feat[s]))
                    else:
                        mask[s] = self.mask[f'mask_{s}'](dec_feat[s])
            out['mask'] = mask
        
        # Handle virtual stereo
        if self.use_virtual_stereo:
            out['disp_stereo'] = {k: v[:, 1:] for k, v in out['disp'].items()}
            out['disp'] = {k: v[:, :1] for k, v in out['disp'].items()}
        
        return out


def create_mdec_unet_model(pretrained=False, weights_path=None, **kwargs):
    """
    Create an MDEC-compatible UNet model instance.
    
    Args:
        pretrained (bool): If True, load weights from weights_path
        weights_path (str): Path to pretrained weights
        **kwargs: Additional arguments to pass to MDECUNet constructor
    
    Returns:
        MDECUNet: Initialized MDEC-compatible UNet model
    """
    model = MDECUNet(**kwargs)
    
    if pretrained and weights_path:
        # Try to load pretrained weights
        try:
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            
            # Try loading from a standard UNet model - needs mapping of weights
            try:
                standard_unet = create_unet_model(pretrained=True, weights_path=weights_path)
                
                # Map standard UNet weights to MDEC UNet
                # This is a simplistic approach and may not work for all cases
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in standard_unet.state_dict().items() 
                                  if k in model_dict and model_dict[k].shape == v.shape}
                
                # Update the model with pretrained weights
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print(f"Loaded partial weights from standard UNet model")
            except Exception as e2:
                print(f"Warning: Could not map standard UNet weights: {e2}")
    
    return model


if __name__ == "__main__":
    # Test the model
    model = create_mdec_unet_model(n_channels=3, out_scales=(0, 1, 2, 3), base_channels=64)
    
    # Input tensor
    x = torch.randn(2, 3, 256, 256)
    
    # Forward pass
    output = model(x)
    
    # Print output shapes
    print("Model output keys:", output.keys())
    print("Disparity scales:", output['disp'].keys())
    
    for k, v in output['disp'].items():
        print(f"Scale {k} shape: {v.shape}")
    
    print("Encoder features shapes:")
    for i, f in enumerate(output['depth_feats']):
        print(f"Feature {i} shape: {f.shape}")