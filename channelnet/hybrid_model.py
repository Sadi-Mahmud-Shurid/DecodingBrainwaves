import torch
import torch.nn as nn
import math
from transformers import PreTrainedModel

from .hybrid_config import HybridEEGModelConfig
from .layers import *  # Import existing ChannelNet layers


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class FeaturesExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.temporal_block = TemporalBlock(
            config.in_channels,
            config.temp_channels,
            config.num_temp_layers,
            config.temporal_kernel,
            config.temporal_stride,
            config.temporal_dilation_list,
            config.input_width,
        )

        self.spatial_block = SpatialBlock(
            config.temp_channels * config.num_temp_layers,
            config.out_channels,
            config.num_spatial_layers,
            config.spatial_stride,
            config.input_height,
        )

        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualBlock(
                        config.out_channels * config.num_spatial_layers,
                        config.out_channels * config.num_spatial_layers,
                    ),
                    ConvLayer2D(
                        config.out_channels * config.num_spatial_layers,
                        config.out_channels * config.num_spatial_layers,
                        config.down_kernel,
                        config.down_stride,
                        0,
                        1,
                    ),
                )
                for i in range(config.num_residual_blocks)
            ]
        )

        self.final_conv = ConvLayer2D(
            config.out_channels * config.num_spatial_layers,
            config.out_channels,
            config.down_kernel,
            1,
            0,
            1,
        )

    def forward(self, x):
        """Extract CNN features from EEG input"""
        out = self.temporal_block(x)
        out = self.spatial_block(out)

        if len(self.res_blocks) > 0:
            for res_block in self.res_blocks:
                out = res_block(out)

        out = self.final_conv(out)
        return out


class HybridChannelNetModel(PreTrainedModel):
    
    config_class = HybridEEGModelConfig
    
    def __init__(self, config: HybridEEGModelConfig):
        super().__init__(config=config)
        
        
        self.encoder = FeaturesExtractor(config=config)
        
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, config.in_channels, config.input_height, config.input_width)
            dummy_output = self.encoder(dummy_input)
            batch_size, C, H, W = dummy_output.shape
        
       
        encoding_size = C * H * W
        
        if config.use_transformer:
            
            
            self.cnn_to_transformer = nn.Linear(seq_feature_dim, config.transformer_dim)
            
            
            self.pos_encoder = PositionalEncoding(
                config.transformer_dim, 
                max_len=W  # Use actual width as max sequence length
            )
            
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.transformer_dim,
                nhead=config.n_heads,
                dim_feedforward=config.transformer_dim * 4,
                dropout=config.transformer_dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-norm architecture (more stable)
            )
            
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.n_transformer_layers
            )
            
            
            self.projector = nn.Sequential(
                nn.Linear(config.transformer_dim, config.embedding_size),
                nn.LayerNorm(config.embedding_size)
            )
        else:
            
            self.projector = nn.Linear(encoding_size, config.embedding_size)
        
        
        self.classifier = nn.Linear(config.embedding_size, config.num_classes)
        
        self.config = config

    def forward(self, x):
        
        # Step 1: CNN feature extraction
        cnn_features = self.encoder(x)  # [batch, C, H, W]
        
        batch_size = x.size(0)
        
        if self.config.use_transformer:
            # Get dimensions
            _, C, H, W = cnn_features.shape
            
            # Reshape to sequence format: [batch, W, C*H]
            # W represents the temporal dimension (time steps)
            # C*H represents the features at each time step
            seq_features = cnn_features.permute(0, 3, 1, 2)  # [batch, W, C, H]
            seq_features = seq_features.contiguous().view(batch_size, W, C * H)  # [batch, W, C*H]
            
            # Project to transformer dimension
            seq_features = self.cnn_to_transformer(seq_features)  # [batch, W, transformer_dim]
            
            # Add positional encoding
            seq_features = self.pos_encoder(seq_features)
            
            # Transformer processing
            transformer_out = self.transformer(seq_features)  # [batch, W, transformer_dim]
            
            # Global pooling (aggregate across time/sequence)
            pooled = transformer_out.mean(dim=1)  # [batch, transformer_dim]
            
            # Project to embedding space
            emb = self.projector(pooled)  # [batch, embedding_size]
        else:
            # Direct projection (ChannelNet mode)
            cnn_features_flat = cnn_features.view(batch_size, -1)
            emb = self.projector(cnn_features_flat)
        
        # Classification
        cls = self.classifier(emb)  # [batch, num_classes]

        return emb, cls