import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPT2Config:
    vocab_size: int = 10000
    seq_len: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_embed: int = 100
    f_expand: int = 4
    dropout: int = 0.1

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.n_embed = config.n_embed
        self.n_head  = config.n_head
        self.head_size = self.n_embed//self.n_head

        self.kqv_proj = nn.Linear(self.n_embed, self.n_embed*3, bias=False)
        self.c_proj   = nn.Linear(self.n_embed, self.n_embed, bias=False)
        self.c_proj.residual = True

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.kqv_proj(x).view(B, T, 3*self.n_head, self.head_size).transpose(1, 2).chunk(3, dim = -3)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.n_embed = config.n_embed
        self.expand_size = int(config.f_expand * self.n_embed)

        self.up_proj  = nn.Linear(self.n_embed, self.expand_size, bias=False)
        self.down_prj = nn.Linear(self.expand_size, self.n_embed, bias=False)
        self.down_prj.residual = True

    def forward(self, x):
        return self.down_prj(F.gelu(self.up_proj(x)))

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super(DecoderBlock, self).__init__()
        self.n_embed = config.n_embed
        self.dropout = nn.Dropout(config.dropout)
        self.ln1 = nn.LayerNorm(self.n_embed)
        self.mha = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(self.n_embed)
        self.ffn = FeedForward(config)

    def forward(self, x):
        y = x + self.dropout(self.mha(self.ln1(x)))
        y = y + self.dropout(self.ffn(self.ln2(y)))
        return y

class GPT2(nn.Module):
    def __init__(self, config):
        super(GPT2, self).__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embed)
        self.wpe = nn.Embedding(config.seq_len, config.n_embed)
        self.decoders = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.wte.weight
        self.apply(self._initweights)
    def _initweights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'residual'):
                std *= (2*self.config.n_layer)**-0.5
            nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean = 0.0, std = std)
        pass
    def forward(self, x):
        B, T = x.shape
        y = self.wte(x) + self.wpe(torch.arange(T).to(x.device))
        for decoder in self.decoders:
            y = decoder(y)
        y = self.ln(y)
        y = self.lm_head(y)
        return y