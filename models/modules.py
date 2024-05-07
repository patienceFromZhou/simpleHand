
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import DropPath, Mlp

from .position_embedding import build_position_encoding



def remap_uv(feat: Tensor, uv_coord: Tensor) -> Tensor:
    """
    args:
        feat; [N, C, H, W]
        uv_coord: [N, J, 2] range ~ [0, 1]
    return:
        select_feat: [N, J, C]
    """
    uv_coord = torch.clamp((uv_coord - 0.5) * 2, -1, 1) # [N, J, 2], range ~ [-1, 1]
    uv_coord = uv_coord.unsqueeze(2) # [N, J, 1, 2]
    select_feat = F.grid_sample(feat, uv_coord, align_corners=True)  # [N, C, J, 1]
    select_feat = select_feat.permute((0, 2, 1, 3))
    select_feat = select_feat[:, :, :, 0]
    return select_feat


class MlpEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)
        self.encoder = Mlp(out_channels, out_channels * 4, out_channels, norm_layer=nn.LayerNorm)
        
    def forward(self, x):
        x = self.proj(x)
        x = self.encoder(x)
        return x
    
    
class IdentityBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        dim_out: int, 
        mlp_ratio: float = 4.0,
        dropout: float = 0,        
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        ) :
        super().__init__()
        
        self.dim = dim
        self.dim_out = dim_out
        
        self.norm1 = norm_layer(dim_out)
        self.token_mixer = nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim != self.dim_out:
            x = self.proj(x)

        # x_norm = self.norm1(x)
        # x = x + self.drop_path(self.token_mixer(x_norm))
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    
class AttentionBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        dim_out: int, 
        mlp_ratio: float = 4.0,
        nhead: int = 4,
        dropout: float = 0,        
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        pre_norm: bool = False
        ) :
        super().__init__()
        
        self.dim = dim
        self.dim_out = dim_out
        self.pre_norm = pre_norm
        
        if self.pre_norm:
            self.norm1 = norm_layer(dim_out)
        else:
            self.norm1 = nn.Identity()
        self.token_mixer = nn.MultiheadAttention(dim_out, nhead, dropout=dropout, batch_first=True)

        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim != self.dim_out:
            x = self.proj(x)

        x_norm = self.norm1(x)
        x = x + self.drop_path(self.token_mixer(x_norm, x_norm, x_norm)[0])
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, expansion_ratio=2,
        act1_layer=StarReLU, act2_layer=nn.Identity, 
        bias=False, kernel_size=(7, 7), padding=(3, 3),
        **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x
    
class SepConvBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        dim_out: int, 
        mlp_ratio: float = 4.0,
        kernel_size = 7,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        ) :
        super().__init__()
        
        self.dim = dim
        self.dim_out = dim_out
        
        self.norm1 = norm_layer(dim_out)
        self.token_mixer = SepConv(dim_out, kernel_size=(7, 1), padding=(3, 0))

        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim != self.dim_out:
            x = self.proj(x)

        x_norm = self.norm1(x)
        
        x_mix = x_norm.unsqueeze(2) #[ N, J, 1, C]
        x_mix = self.token_mixer(x_mix)
        x_mix = x_mix.squeeze(2)
        
        x = x + self.drop_path(x_mix)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x    
    
    
class LinearUpsample(nn.Module):
    def __init__(self, node_num_in, node_num_out):
        super().__init__()
        self.upsample = nn.Linear(node_num_in, node_num_out)
        
    def forward(self, x):
        x = x.permute(0, 2, 1) # [N, j_in, c_out] ->  [N, c_out, j_in]
        x = self.upsample(x) # [N, c_out, j_in] -> [N, c_out, j_out]
        x = x.permute(0, 2, 1) # [N, c_out, j_out] -> [N, j_out, c_out]
        return x


class MeshHead(nn.Module):
    def __init__(
        self, 
        in_channels,
        depths = [1, 1, 1],
        token_nums = [21, 256, 512],
        dims = [256, 128, 64],
        block_types = [AttentionBlock, AttentionBlock, AttentionBlock],
        first_prenorms = [False, True, True],
        dropout = 0.1,
        ):
        super().__init__()
        
        assert len(depths) == len(token_nums)
        assert len(token_nums) == len(dims)
        assert len(depths) == 3
        
        deconv_dim = 256

        self.in_channels = in_channels
        self.deconv = self.build_deconv_layer(1, [deconv_dim], [4, 4])
        
        self.proj_1 = nn.Linear(deconv_dim, dims[0])
        self.encoder_1 = self.build_encoder(dims[0], block_types[0], depths[0], dropout, first_prenorm=first_prenorms[0])
        self.upsample_1 = LinearUpsample(token_nums[0], token_nums[1])
        

        self.proj_2 = nn.Linear(dims[0], dims[1])
        self.encoder_2 = self.build_encoder(dims[1], block_types[1], depths[1], dropout, first_prenorm=first_prenorms[1])

        self.upsample_2 = LinearUpsample(token_nums[1], token_nums[2])
        
        self.proj_3 = nn.Linear(dims[1], dims[2])
        self.encoder_3 = self.build_encoder(dims[2], block_types[2], depths[2], dropout, first_prenorm=first_prenorms[2])

        self.upsample_3 = LinearUpsample(token_nums[2], 778)               
        
        self.pred_final = nn.Linear(dims[2], 3)
        
        self.pos_embedding_abs = build_position_encoding(hidden_dim=deconv_dim)
        
        self.pos_emb_1 = nn.Parameter(torch.zeros(1, token_nums[0], deconv_dim))
        self.pos_emb_2 = nn.Parameter(torch.zeros(1, token_nums[1], dims[0]))
        self.pos_emb_3 = nn.Parameter(torch.zeros(1, token_nums[2], dims[1]))
        
        nn.init.trunc_normal_(self.pos_emb_1, std=0.02)
        nn.init.trunc_normal_(self.pos_emb_2, std=0.02)
        nn.init.trunc_normal_(self.pos_emb_3, std=0.02)
        
    def build_encoder(self, dim, block_layer, depth=1, dropout=0.1, first_prenorm=True):
        blocks = []
        for i in range(depth):
            if i == 0:
                first_prenorm = first_prenorm
            else:
                first_prenorm = True
            block = block_layer(dim, dim, drop_path=dropout, dropout=dropout, pre_norm=first_prenorm)
            blocks.append(block)
        return nn.Sequential(*blocks)
    
    def _get_deconv_cfg(self, deconv_kernel):

        
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 0
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')
        return deconv_kernel, padding, output_padding
        

    def build_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            decov = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=planes, kernel_size=kernel, stride=4, padding=padding, output_padding=output_padding, bias=False)
            layers.append(decov)
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)
        
        

    def forward(self, feat: Tensor, uv_coords: Tensor) -> Tensor:
        """
        args:
            feat: [N, in_channels, H, W]
            uv_coords: [N, 21, 2]
        """
        feat = self.deconv(feat)
        
        # print(feat.shape)
        
        b, c, h, w = feat.shape
        
        feat_pos = self.pos_embedding_abs(b, h, w, feat.device)
        
        feat = feat_pos + feat

        uv_coords = uv_coords.reshape(-1, 21, 2)
        N, J, _ = uv_coords.shape
        select_feat = remap_uv(feat, uv_coords)  # [N, 21, latent_channels]
        token_features = select_feat.reshape(N, J, -1)

        token_features += self.pos_emb_1
        token_features = self.proj_1(token_features)
        token_features = self.encoder_1(token_features)
        token_features = self.upsample_1(token_features)
        
        token_features += self.pos_emb_2
        token_features = self.proj_2(token_features)        
        token_features = self.encoder_2(token_features)
        token_features = self.upsample_2(token_features)
        
        token_features += self.pos_emb_3
        token_features = self.proj_3(token_features)                
        token_features = self.encoder_3(token_features)
        token_features = self.upsample_3(token_features)                
        
        vertices = self.pred_final(token_features)

        return vertices