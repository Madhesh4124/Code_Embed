"""Pre-LN Transformer Block and Encoder Stack.

Implemented using raw PyTorch primitives according to CodeEmbed architectural rules.
Pre-LN (Pre-Layer Normalization) provides stable gradient flow during training
and eliminates the need for aggressive warmups or learning rate hacks.
"""

import torch
from torch import nn

from model.attention import MultiHeadSelfAttention


class FeedForwardNetwork(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation.

    Computes:
        FFN(x) = Dropout(Linear_2(Dropout(GELU(Linear_1(x)))))

    Args:
        d_model: Input and output feature dimension (default: 256).
        d_ff: Hidden expansion dimension (default: 1024).
        dropout: Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        d_model: int = 256,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        # Linear followed by residual add and LayerNorm: bias=False according to Rules.md
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for position-wise FFN.

        Args:
            x: FloatTensor of shape (B, L, D).

        Returns:
            FloatTensor of shape (B, L, D).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class PreLNTransformerBlock(nn.Module):
    """Transformer Encoder Block with Pre-Layer Normalization (Pre-LN).

    Computes:
        x_norm1 = LayerNorm1(x)
        x = x + Dropout(Attention(x_norm1, mask))
        x_norm2 = LayerNorm2(x)
        x = x + Dropout(FFN(x_norm2))

    Args:
        d_model: Model dimension (default: 256).
        n_heads: Number of attention heads (default: 8).
        d_ff: Feed-forward expansion dimension (default: 1024).
        dropout: Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout, bias=False
        )
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for Pre-LN block.

        Args:
            x: Input tensor of shape (B, L, D).
            attention_mask: Optional padding mask of shape (B, L).

        Returns:
            Output tensor of shape (B, L, D).
        """
        # 1. Pre-LN Self-Attention with residual connection
        x_norm1 = self.ln1(x)
        attn_out, _ = self.attn(x_norm1, attention_mask=attention_mask)
        x = x + self.dropout1(attn_out)

        # 2. Pre-LN Feed-Forward with residual connection
        x_norm2 = self.ln2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out

        return x


class TransformerEncoder(nn.Module):
    """Stack of Pre-LN Transformer Blocks with final Layer Normalization.

    Args:
        n_layers: Number of transformer blocks to stack (default: 4).
        d_model: Dimensionality of model (default: 256).
        n_heads: Number of attention heads (default: 8).
        d_ff: Hidden dimension in feed-forward networks (default: 1024).
        dropout: Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        n_layers: int = 4,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PreLNTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        # In Pre-LN architectures, a final LayerNorm is required after all blocks
        self.final_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through all stacked transformer blocks.

        Args:
            hidden_states: Input tensor of shape (B, L, D).
            attention_mask: Padding mask of shape (B, L).

        Returns:
            Encoded representations of shape (B, L, D).
        """
        x = hidden_states
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return self.final_ln(x)
