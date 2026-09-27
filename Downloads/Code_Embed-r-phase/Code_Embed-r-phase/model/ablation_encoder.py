import torch
import torch.nn.functional as F
from torch import nn
from model.embeddings import EmbeddingLayer
from model.pooling import MaskedMeanPooling, CLSPooling
from model.transformer import TransformerEncoder
from model.dual_encoder import DualEncoder

class AblationBaseEncoder(nn.Module):
    def __init__(self, vocab_size=16000, d_model=256, n_layers=4, n_heads=8, d_ff=1024, max_seq_len=256, dropout=0.1, pooling_strategy="mean"):
        super().__init__()
        self.d_model = d_model
        self.embeddings = EmbeddingLayer(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len, dropout=dropout)
        self.encoder = TransformerEncoder(n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.pooling = MaskedMeanPooling() if pooling_strategy == "mean" else CLSPooling()
        self.proj_linear = nn.Linear(d_model, d_model, bias=False)
        self.proj_ln = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask=None):
        h = self.embeddings(input_ids)
        h = self.encoder(h, attention_mask=attention_mask)
        z = self.pooling(h, attention_mask=attention_mask)
        z = self.proj_ln(self.proj_linear(z))
        return F.normalize(z, p=2.0, dim=1)

    def get_num_params(self): 
        return sum(p.numel() for p in self.parameters()), sum(p.numel() for p in self.parameters() if p.requires_grad)

class AblationDualEncoder(DualEncoder):
    def __init__(self, vocab_size=16000, d_model=256, n_layers=4, n_heads=8, d_ff=1024, max_seq_len=256, dropout=0.1, pooling_strategy="mean"):
        super().__init__()
        self.code_encoder = AblationBaseEncoder(vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout, pooling_strategy)
        self.text_encoder = AblationBaseEncoder(vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout, pooling_strategy)
    
    def encode_code(self, input_ids, attention_mask=None): return self.code_encoder(input_ids, attention_mask=attention_mask)
    def encode_text(self, input_ids, attention_mask=None): return self.text_encoder(input_ids, attention_mask=attention_mask)
    def get_num_params(self): return sum(p.numel() for p in self.parameters()), sum(p.numel() for p in self.parameters() if p.requires_grad)
