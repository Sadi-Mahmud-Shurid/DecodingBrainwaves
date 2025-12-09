from transformers import PretrainedConfig


class HybridEEGModelConfig(PretrainedConfig):
    """
    Configuration class for Hybrid CNN-Transformer EEG encoder.
    
    This extends the ChannelNet architecture by adding Transformer layers
    for better global context modeling while maintaining compatibility with
    the existing training pipeline.
    """
    
    model_type = "hybrid_eeg_channelnet"

    def __init__(
        self,
        # Original ChannelNet parameters (maintained for compatibility)
        in_channels=1,
        temp_channels=10,
        out_channels=50,
        num_classes=40,
        embedding_size=512,
        input_width=440,
        input_height=128,
        temporal_dilation_list=None,
        temporal_kernel=(1, 33),
        temporal_stride=(1, 2),
        num_temp_layers=4,
        num_spatial_layers=4,
        spatial_stride=(2, 1),
        num_residual_blocks=4,
        down_kernel=3,
        down_stride=2,
        # NEW: Transformer parameters
        transformer_dim=256,        # Transformer hidden dimension
        n_heads=8,                  # Number of attention heads
        n_transformer_layers=4,     # Number of transformer encoder layers
        transformer_dropout=0.1,    # Dropout for transformer
        use_transformer=True,       # Whether to use transformer (can disable for ablation)
        **kwargs
    ):
        if temporal_dilation_list is None:
            temporal_dilation_list = [(1, 1), (1, 2), (1, 4), (1, 8), (1, 16)]

        super().__init__(**kwargs)
        
        # Original ChannelNet parameters
        self.in_channels = in_channels
        self.temp_channels = temp_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.input_width = input_width
        self.input_height = input_height
        self.temporal_dilation_list = temporal_dilation_list
        self.temporal_kernel = temporal_kernel
        self.temporal_stride = temporal_stride
        self.num_temp_layers = num_temp_layers
        self.num_spatial_layers = num_spatial_layers
        self.spatial_stride = spatial_stride
        self.num_residual_blocks = num_residual_blocks
        self.down_kernel = down_kernel
        self.down_stride = down_stride
        
        # Transformer parameters
        self.transformer_dim = transformer_dim
        self.n_heads = n_heads
        self.n_transformer_layers = n_transformer_layers
        self.transformer_dropout = transformer_dropout
        self.use_transformer = use_transformer
        
        # Validation
        assert transformer_dim % n_heads == 0, \
            f"transformer_dim ({transformer_dim}) must be divisible by n_heads ({n_heads})"