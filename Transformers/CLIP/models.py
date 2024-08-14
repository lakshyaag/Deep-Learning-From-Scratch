import torch
import torchvision
from torch import nn
from transformers import AutoModel
from typing import Optional
from torch.nn import functional as F


class ImageEncoder(nn.Module):
    """
    Image encoder model using ResNet50 with the classification head removed.

    Args:
    - trainable (bool): Whether to make the model trainable.
    - debug (bool): Whether to print debug information.
    """

    def __init__(self, trainable: bool = False, debug: bool = False):
        super().__init__()

        self.model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )

        # Remove the classification head
        self.model.fc = nn.Identity()

        for param in self.model.parameters():
            param.requires_grad = trainable

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._debug_print(x, "[IMG ENCODER] Input")
        x = self.model(x)
        self._debug_print(x, "[IMG ENCODER] Output")

        return x


class TextEncoder(nn.Module):
    """
    Text encoder model using a pre-trained transformer model.

    Args:
    - text_model (str): Pre-trained transformer model to use.
    - trainable (bool): Whether to make the model trainable.
    - debug (bool): Whether to print debug information.
    """

    def __init__(self, text_model: str, trainable: bool = False, debug: bool = False):
        super().__init__()

        self.model = AutoModel.from_pretrained(text_model)

        for param in self.model.parameters():
            param.requires_grad = trainable

        self.debug = debug

        self.cls_token_idx = 0

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self._debug_print(x, "[TEXT ENCODER] Input")
        x = self.model(input_ids=x, attention_mask=attn_mask)
        x = x.last_hidden_state[:, self.cls_token_idx, :]
        self._debug_print(x, "[TEXT ENCODER] Output")
        return x


class Projection(nn.Module):
    """
    Projection head for the CLIP model. Projects the input to the embedding dimension and applies a fully connected layer.

    Args:
    - d_model (int): Model dimension (output of the text and image encoders).
    - d_embd (int): Embedding dimension (shared between the text and image encoders).
    - dropout (float): Dropout rate
    - debug (bool): Whether to print debug information
    """

    def __init__(
        self, d_model: int, d_embd: int, dropout: float = 0.0, debug: bool = False
    ):
        super().__init__()

        self.projection = nn.Linear(d_model, d_embd)
        self.fc = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_embd, d_embd),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_embd)

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._debug_print(x, "[PROJECTION] Input")
        projected = self.projection(x)
        self._debug_print(projected, "[PROJECTION] Projected")
        x = self.fc(projected)
        self._debug_print(x, "[PROJECTION] FC")
        x += projected
        x = self.layer_norm(x)
        self._debug_print(x, "[PROJECTION] Output")
        return x


class CLIP(nn.Module):
    """
    Contrastive Language-Image Pre-training (CLIP) model. Combines the image and text encoders with a projection head.

    Args:
    - d_img_model (int): Emebdding dimension for the image encoder model.
    - d_text_model (int): Emebdding dimension for the text encoder model.
    - d_embd (int): Embedding dimension for the projection head.
    - temp (float): Temperature parameter for the contrastive loss.
    - trainable (bool): Whether to make the model trainable.
    - debug (bool): Whether to print debug information.
    """

    def __init__(
        self,
        text_model: str,
        d_img_model: int,
        d_text_model: int,
        d_embd: int,
        temp: float = 0.07,
        dropout: float = 0.0,
        trainable: bool = False,
        debug: bool = False,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(trainable=trainable, debug=debug)
        self.text_encoder = TextEncoder(
            text_model=text_model, trainable=trainable, debug=debug
        )

        self.image_projection = Projection(
            d_model=d_img_model, d_embd=d_embd, dropout=dropout, debug=debug
        )
        self.text_projection = Projection(
            d_model=d_text_model, d_embd=d_embd, dropout=dropout, debug=debug
        )

        self.temperature = nn.Parameter(torch.tensor(temp))
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._debug_print(image, "[CLIP] Image Input")
        self._debug_print(input_ids, "[CLIP] Text Input")

        # Extract feature representations of each modality
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(input_ids, attn_mask)

        self._debug_print(image_features, "[CLIP] Image Features")
        self._debug_print(text_features, "[CLIP] Text Features")

        # Joint multimodal embedding space
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        self._debug_print(image_embeddings, "[CLIP] Image Embeddings")
        self._debug_print(text_embeddings, "[CLIP] Text Embeddings")

        # Scaled pairwise cosine similarity between image and text embeddings
        logits = (image_embeddings @ text_embeddings.T) / self.temperature
        self._debug_print(logits, "[CLIP] Logits")

        image_similarity = image_embeddings @ image_embeddings.T
        text_similarity = text_embeddings @ text_embeddings.T

        # Build the targets for the contrastive loss (positive samples are on the diagonal)
        targets = F.softmax(
            (image_similarity + text_similarity) / 2 * self.temperature, dim=-1
        )

        # Calculate the contrastive loss and return the mean
        image_loss = (-targets.T * self.log_softmax(logits.T)).sum(1)
        text_loss = (-targets * self.log_softmax(logits)).sum(1)

        loss = (image_loss + text_loss) / 2.0

        return loss.mean()
