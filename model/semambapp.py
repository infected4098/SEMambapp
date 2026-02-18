






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
        self.mask_decoder = MagDecoder_mapping_learn(cfg)
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
