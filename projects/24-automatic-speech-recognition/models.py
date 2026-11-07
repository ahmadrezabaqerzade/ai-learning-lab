import torch
from torch import nn
import math


class CNN2DFeatureExtractor(nn.Module):
    """2D CNN for feature extraction from spectrograms."""
    def __init__(self, planes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, planes[0], kernel_size=9, stride=(1, 1), padding=4, bias=False),
            nn.BatchNorm2d(planes[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(planes[1]),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(planes[1], planes[2], kernel_size=3, stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(planes[2]),
            nn.ReLU(inplace=True)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)
        self.layer4 = nn.Sequential(
            nn.Conv2d(planes[2], planes[3], kernel_size=3, stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(planes[3]),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=(2, 2), padding=1)

    def forward(self, x):
        """Forward pass through the CNN feature extractor."""
        x = self.layer1(x)
        x = self.maxpool1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.maxpool2(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to the input."""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ASRModel(nn.Module):
    """Automatic Speech Recognition (ASR) model using CNN and Transformer."""
    def __init__(self, vocab, d_model, num_encoder_layer, num_decoder_layer, nhead, dim_forward, dropout,
                 vocab_size, planes=[8, 16, 32, 64], device='cpu'):
        super(ASRModel, self).__init__()
        self.d_model = d_model
        self.device = device
        self.vocab = vocab

        # CNN feature extractor
        self.feature_extractor = CNN2DFeatureExtractor(planes=planes)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.vocab['='])

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layer, num_decoder_layers=num_decoder_layer,
            dim_feedforward=dim_forward, dropout=dropout, activation=nn.functional.relu
        )

        # Classification head
        self.cls_head = nn.Linear(self.d_model, vocab_size)

        # Initialize weights
        self.init_weights()
        self.init_weights_e()

    def init_weights(self):
        """Initialize weights for the classification head."""
        initrange = 0.1
        self.cls_head.bias.data.zero_()
        self.cls_head.weight.data.uniform_(-initrange, initrange)

    def init_weights_e(self):
        """Initialize weights for the embedding layer."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, specs, tokens):
        """Forward pass through the ASR model."""
        # Extract features from spectrogram
        y = self.feature_extractor(specs)
        batch_size, num_channels, freq_bins, seq_len = y.shape

        # Reshape to match embedding dimension
        y = nn.functional.adaptive_avg_pool2d(y, output_size=(int(self.d_model / num_channels), seq_len))
        y = y.reshape(batch_size, -1, seq_len)
        y = y.permute(2, 0, 1)  # (seq_len, batch, d_model)

        # Embed tokens and add positional encoding
        tokens = self.embedding(tokens) * math.sqrt(self.d_model)
        tokens = tokens.permute(1, 0, 2)  # (seq_len, batch, d_model)
        tokens = self.pos_encoder(tokens)

        # Generate mask for transformer
        tokens_mask = nn.Transformer.generate_square_subsequent_mask(len(tokens)).to(self.device)

        # Pass through transformer
        out = self.transformer(y, tokens, tgt_mask=tokens_mask)
        out = out.permute(1, 0, 2)  # (batch, seq_len, d_model)

        # Classification head
        out = self.cls_head(out)
        return out