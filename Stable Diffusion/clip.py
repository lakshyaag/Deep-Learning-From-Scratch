import torch
from attention import SelfAttention
from torch import nn


class CLIPEmbedding(nn.Module):
    """
    This class defines the embedding layer for the CLIP model.
    It includes token embeddings and position embeddings.
    """

    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_token, n_embd))

    def forward(self, tokens: torch.LongTensor) -> torch.LongTensor:
        # tokens: B S

        # B S ->  B S E
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):
    """
    This class defines a single layer of the CLIP model.
    It consists of a self-attention mechanism followed by a feed-forward neural network.
    """

    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)

        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B S E

        # ---------------------------------------------------- #
        # SELF ATTENTION
        residue = x

        # B S E -> B S E
        x = self.layernorm_1(x)
        # B S E -> B S E
        x = self.attention(x, causal_mask=True)

        # B S E -> B S E
        x += residue

        # ---------------------------------------------------- #
        # FEED FORWARD
        residue = x

        # B S E -> B S E
        x = self.layernorm_2(x)
        # B S E -> B S E*4
        x = self.linear_1(x)

        # QuickGELU activation function
        x = x * torch.sigmoid(1.702 * x)

        # B S E*4 -> B S E
        x = self.linear_2(x)

        # B S E -> B S E
        x += residue

        return x


class CLIP(nn.Module):
    """
    This class defines the CLIP model.
    CLIP connects images and text with embedding and transformer layers
    """

    def __init__(self):
        super().__init__()

        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([CLIPLayer(12, 768) for _ in range(12)])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.LongTensor:
        # tokens: B S
        tokens = tokens.type(torch.long)

        # B S -> B S E
        state = self.embedding(tokens)

        for layer in self.layers:
            # B S E -> B S E
            state = layer(state)

        # B S E -> B S E
        output = self.layernorm(state)

        return output
