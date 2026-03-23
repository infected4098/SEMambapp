import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layernorm import RMSNorm
from functools import partial
from .frequencyglp import FrequencyGLP
from .FAN import FANFFN, FANFFN_gate_freq, FANFFN_gate_channel
from .encdec import DenseEncoder, MagDecoder, PhaseDecoder


def compute_auto_padding_2d(stride_t=2, stride_f=2, kernel_t=3, kernel_f=3):
    """
    Compute padding (pad_t, pad_f) for Conv2d with given kernel and stride
    so that the output size is consistent (integer) for both even and odd inputs.

    Args:
        stride_t (int): stride in time dimension (default=2)
        stride_f (int): stride in freq dimension (default=2)
        kernel_t (int): kernel size in time dimension
        kernel_f (int): kernel size in freq dimension

    Returns:
        (pad_t, pad_f): tuple of int paddings
    """
    # The general rule for stride 2 is:
    # padding = floor((kernel_size - stride) / 2)
    pad_t = (kernel_t - stride_t) // 2
    pad_f = (kernel_f - stride_f) // 2

    return pad_t, pad_f



# github: https://github.com/state-spaces/mamba/blob/9127d1f47f367f5c9cc49c73ad73557089d02cb8/mamba_ssm/models/mixer_seq_simple.py
def create_block(
    d_model, cfg, expand=None, layer_idx=0, rms_norm=True, fused_add_norm=False, residual_in_fp32=False, 
    ):
    d_state = cfg['model_cfg']['d_state'] # 16
    d_conv = cfg['model_cfg']['d_conv'] # 4
    if expand is None:
        expand = cfg['model_cfg']['expand'] # 4
    norm_epsilon = cfg['model_cfg']['norm_epsilon'] # 0.00001

    mixer_cls = partial(Mamba, layer_idx=layer_idx, d_state=d_state, d_conv=d_conv, expand=expand)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon
    )
    block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            )
    block.layer_idx = layer_idx
    return block

class MambaBlock(nn.Module):
    def __init__(self, in_channels, cfg, expand=None):
        super(MambaBlock, self).__init__()
        n_layer = 1
        self.forward_blocks  = nn.ModuleList( create_block(in_channels, cfg, expand=expand) for i in range(n_layer))
        self.backward_blocks = nn.ModuleList( create_block(in_channels, cfg, expand=expand) for i in range(n_layer))
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
            )
        )

    def forward(self, x):
        x_forward, x_backward = x.clone(), torch.flip(x, [1])
        resi_forward, resi_backward = None, None

        # Forward
        for layer in self.forward_blocks:
            x_forward, resi_forward = layer(x_forward, resi_forward)
        y_forward = (x_forward + resi_forward) if resi_forward is not None else x_forward

        # Backward
        for layer in self.backward_blocks:
            x_backward, resi_backward = layer(x_backward, resi_backward)
        y_backward = torch.flip((x_backward + resi_backward), [1]) if resi_backward is not None else torch.flip(x_backward, [1])

        return torch.cat([y_forward, y_backward], -1)



class SEMambapp_bottleneck(nn.Module):
    """
    Temporal-Frequency Mamba block for sequence modeling.
    
    Attributes:
    cfg (Config): Configuration for the block.
    time_mamba (MambaBlock): Mamba block for temporal dimension.
    freq_mamba (MambaBlock): Mamba block for frequency dimension.
    tlinear (ConvTranspose1d): ConvTranspose1d layer for temporal dimension.
    flinear (ConvTranspose1d): ConvTranspose1d layer for frequency dimension.
    """
    def __init__(self, cfg):
        super(SEMambapp_bottleneck, self).__init__()
        self.cfg = cfg

        self.hid_feature = cfg['model_cfg']['hid_feature']
        
        # Initialize Mamba blocks
        
        self.unet_expansion = cfg['model_cfg']['unet_expansion']

        # Initialize ConvTranspose1d layers
        self.features = [self.hid_feature, int(self.hid_feature*self.unet_expansion), int(self.hid_feature*self.unet_expansion**2)]
        # Downsample
        self.downsamples = nn.ModuleList()
        self.downsamples.append(nn.Identity())
        self.downsamples.append(
            nn.Sequential(
                nn.Conv2d(self.features[0], self.features[1], (3, 4), stride=(1, 2), padding=compute_auto_padding_2d(1,2,3,4)),
                nn.InstanceNorm2d(self.features[1], affine=True),
                nn.PReLU(self.features[1])
            )
        )
        self.downsamples.append(
        nn.Sequential(
            nn.Conv2d(self.features[1], self.features[2], (3, 4), stride=(1, 2), padding=compute_auto_padding_2d(1,2,3,4)),
            nn.InstanceNorm2d(self.features[2], affine=True),
            nn.PReLU(self.features[2])
        )     
        )

        self.time_mambas = nn.ModuleList()
        self.time_mambas.extend( [MambaBlock(in_channels=self.features[i], cfg=cfg) for i in range(3)] )
        #self.freq_mambas = MambaBlock(in_channels=self.features[0], cfg=cfg)
        #self.flinears = nn.ConvTranspose1d(self.features[0]*2, self.features[0], 1, stride=1)
        #self.freq_trans = nn.ModuleList()
        #self.freq_trans.extend( [PreNorm(self.features[i], CEA(self.features[i], cfg)) for i in range(1,3)] )
        self.freq_ffns = nn.ModuleList()
        self.freq_ffns.extend( [FrequencyGLP(*feat, 2) for feat in [(100,self.features[0]), (50,self.features[1]), (25,self.features[2])]])

        self.freq_layernorm = nn.ModuleList()
        self.freq_layernorm.extend( [nn.LayerNorm(self.features[i]) for i in range(3)])

        self.channel_ffns = nn.ModuleList()
        self.channel_ffns.extend([FANFFN_gate_channel(self.features[i], 2) for i in range(3)])


        self.tlinears = nn.ModuleList()
        self.tlinears.extend( [nn.ConvTranspose1d(self.features[i]*2, self.features[i], 1, stride=1) for i in range(3)] )

        self.upsamples = nn.ModuleList()
        self.upsamples.append(nn.Sequential(nn.ConvTranspose2d(self.features[2], self.features[1], (3, 4), stride=(1, 2), padding=(1, 1), output_padding=(0, 0)),             
                              nn.InstanceNorm2d(self.features[1], affine=True),
            nn.PReLU(self.features[1])))
        self.upsamples.append(nn.Sequential(nn.ConvTranspose2d(self.features[1], self.features[0], (3, 4), stride=(1, 2), padding=(1, 1), output_padding=(0, 0)),             
                              nn.InstanceNorm2d(self.features[0], affine=True),
            nn.PReLU(self.features[0])))
        self.gates = nn.ModuleList()
        self.gates.append(nn.Sequential(nn.Conv2d(self.features[1]*2, self.features[1], kernel_size=1), nn.GELU(), LayerNorm(self.features[1], data_format="channels_first")))
        self.gates.append(nn.Sequential(nn.Conv2d(self.features[0]*2, self.features[0], kernel_size=1), nn.GELU(), LayerNorm(self.features[0], data_format="channels_first")))
        #self.gates.append(GatedFusion(self.features[1]*2, self.features[1]))
        #self.gates.append(GatedFusion(self.features[0]*2, self.features[0]))


    def forward(self, x):
        """
        Forward pass of the TFMamba block.
        
        Parameters:
        x (Tensor): Input tensor with shape (batch, channels, time, freq).
        
        Returns:
        Tensor: Output tensor after applying temporal and frequency Mamba blocks.
        """
        b, c, t, f = x.size()
        x_level1 = self.downsamples[0](x) # [B, C, T, F//2]
        x_level2 = self.downsamples[1](x_level1) # [B, EC, T, F//4]
        x_level3 = self.downsamples[2](x_level2) # [B, EC**2, T, F//8]  

        b, c, t, f = x_level1.size()
        x_level1 = x_level1.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x_level1 = self.tlinears[0]( self.time_mambas[0](x_level1).permute(0,2,1) ).permute(0,2,1) + x_level1 # [BF, T, C]
        x_level1 = x_level1.view(b, f, t, c).permute(0,2, 1, 3).contiguous().view(b*t, f, c)
        x_level1 = self.freq_layernorm[0](x_level1)
        x_level1 = x_level1.view(b, t, f, c).permute(0,3,1,2) # [B, C, T, F]
        x_level1 = self.freq_ffns[0](x_level1) + x_level1
        x_level1 = self.channel_ffns[0](x_level1) + x_level1
        res = x_level1

        b_ds, c_ds, t_ds, f_ds = x_level2.size()
        x_level2 = x_level2.permute(0, 3, 2, 1).contiguous().view(b_ds*f_ds, t_ds, c_ds)
        x_level2 = self.tlinears[1]( self.time_mambas[1](x_level2).permute(0,2,1) ).permute(0,2,1) + x_level2
        x_level2 = x_level2.view(b_ds, f_ds, t_ds, c_ds).permute(0, 2, 1, 3).contiguous().view(b_ds*t_ds, f_ds, c_ds)
        x_level2 = self.freq_layernorm[1](x_level2)
        x_level2 = x_level2.view(b_ds, t_ds, f_ds, c_ds).permute(0, 3, 1, 2) # [B, EC, T, F//4]
        x_level2 = self.freq_ffns[1](x_level2) + x_level2
        x_level2 = self.channel_ffns[1](x_level2) + x_level2


        b_ds, c_ds, t_ds, f_ds = x_level3.size()
        x_level3 = x_level3.permute(0, 3, 2, 1).contiguous().view(b_ds*f_ds, t_ds, c_ds) # [BF, T, C]
        x_level3 = self.tlinears[2]( self.time_mambas[2](x_level3).permute(0,2,1) ).permute(0,2,1) + x_level3
        x_level3 = x_level3.view(b_ds, f_ds, t_ds, c_ds).permute(0, 2, 1, 3).contiguous().view(b_ds*t_ds, f_ds, c_ds) # [BT, F//8, E**2C]
        x_level3 = self.freq_layernorm[2](x_level3)
        x_level3 = x_level3.view(b_ds, t_ds, f_ds, c_ds).permute(0, 3, 1, 2) # [B, E**2 C, T//4, F//8] 
        x_level3 = self.freq_ffns[2](x_level3) + x_level3 
        x_level2 = self.channel_ffns[2](x_level2) + x_level2
        # hierachical upsampling
        x_us_2 = self.upsamples[0](x_level3) # [B, EC, T//2, F//4]

        x_ds = torch.cat([x_level2, x_us_2], dim=1) # [B, 2EC, T//2, F//4]
        x_ds = self.gates[0](x_ds) # [B, EC, T//2, F//4]

        x_us = self.upsamples[1](x_ds) # [B, C, T, F//2]

        x = torch.cat([x_level1, x_us], dim=1) # [B, 2C, T, F//2]
        x = self.gates[1](x) # [B, C, T, F//2]

        x = res + x

        return x






class SEMambapp(nn.Module):
    """
    SEMamba model for speech enhancement using Mamba blocks.
    
    This model uses a dense encoder, multiple Mamba blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """
    def __init__(self, cfg):
        """
        Initialize the SEMamba model.
        
        Args:
        - cfg: Configuration object containing model parameters.
        """
        super(SEMambapp, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = cfg['model_cfg']['num_tfmamba'] if cfg['model_cfg']['num_tfmamba'] is not None else 4  # default tfmamba: 4

        # Initialize dense encoder
        self.dense_encoder = DenseEncoder(cfg)
        # abl_gloco_v11_timenoreduce_MambaFFN_gloco_nonorm 일 때에는 
        #self.TSMamba = nn.ModuleList([MambaFFNBlock_gloco_timenoreduce_real(cfg, single=True) for _ in range(self.num_tscblocks)])
        
        # Initialize Mamba blocks
        self.TSMamba = nn.ModuleList([SEMambapp_bottleneck(cfg) for _ in range(self.num_tscblocks)]) 
        #self.FAN = nn.ModuleList([FANFFN(cfg['model_cfg']['stft_hid_feature'], 2) for _ in range(self.num_tscblocks-2)])
        #self.TSMamba_loc = nn.ModuleList([TFMambaBlock(cfg, single=True) for _ in range(self.num_tscblocks//2)])
        # Initialize decoders
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg, single=True)

    def forward(self, noisy_mag, noisy_pha):
        """
        Forward pass for the SEMamba model.
        
        Args:
        - noisy_audio (torch.Tensor): Noisy audio waveform input tensor [B, 1, T].
        - noisy_mag (torch.Tensor): Noisy magnitude input tensor [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase input tensor [B, F, T].
        
        Returns:
        - denoised_mag (torch.Tensor): Denoised magnitude tensor [B, F, T].
        - denoised_pha (torch.Tensor): Denoised phase tensor [B, F, T].
        - denoised_com (torch.Tensor): Denoised complex tensor [B, F, T, 2].
        """
        # Reshape inputs

        # STFT feature extraction
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]

        # Encode input
        x = self.dense_encoder(x)
        # Maybe add long residual here? 
        for i in range(len(self.TSMamba)):
            # res = x
            # x = self.TFConvNeXt[i](x) # Residual connection inside
            # x = self.TSMamba[i](x) # Residual connection inside
            # x = x + res  # Long residual connection

            #res = x
            x = self.TSMamba[i](x) # Residual connection inside
            #if i not in [0, len(self.TSMamba)-1]:
            #    x = self.FAN[i-1](x)
            #x = self.TSMamba_loc[i](x) # Residual connection inside
            #x = x + res  # Long residual connection

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(x), 'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(x), 'b c t f -> b f t c').squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)),
            dim=-1
        )

        return denoised_mag, denoised_pha, denoised_com
