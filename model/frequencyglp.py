import torch
import torch.nn as nn
from .FAN import FANFFN_gate_freq

class LocalFrequencyMix(nn.Module):
    """     
    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)  
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.act3 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, T, F]

        b, c, t, f = x.size()
        x = x.permute(0,2,1,3).contiguous().view(b*t, c, f)  # [BT, F, C]

        residual = x
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = residual + x

        x = x.view(b, t, c, f).permute(0,2,1,3)  # [B, C, T, F]

        return x



class FrequencyGLP(nn.Module):
    """
    FANLayer: The layer used in FAN (https://arxiv.org/abs/2410.02675).
    code: https://github.com/YihongDong/FAN/blob/main/FANLayer.py
    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        p_ratio (float): The ratio of output dimensions used for cosine and sine parts (default: 0.25).
        activation (str or callable): The activation function to apply to the g component. If a string is passed,
            the corresponding activation from torch.nn.functional is used (default: 'gelu').
        use_p_bias (bool): If True, include bias in the linear transformations of p component (default: True). 
            There is almost no difference between bias and non-bias in our experiments.
    """
    
    def __init__(self, input_dim, channel, expansion):
        super(FrequencyGLP, self).__init__()
        
        # Ensure the p_ratio is within a valid range
        
        self.input_dim = input_dim
        self.channel = channel
        self.expansion = expansion
        self.expansion_dim = int(input_dim * expansion)

        self.global_branch = FANFFN_gate_freq(self.input_dim, self.expansion)
        self.local_branch = LocalFrequencyMix(self.channel)
        self.linear = nn.Conv2d(self.channel*2, self.channel, kernel_size=1)

    def forward(self, src):
        # src.shape = B, C, T, F

        global_output = self.global_branch(src)
        local_output = self.local_branch(src)
        output = torch.cat([global_output, local_output], dim=1) # [B, 2C, T, F]  # Concatenate along the feature dimension
        output = self.linear(output)

        return output
