import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection


class CLIPClassifier(nn.Module):
    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        num_classes=6,
        dropout=0.2,
        freeze_backbone=False,
    ):
        super().__init__()

        # Pretrained CLIP Vision Encoder
        self.backbone = CLIPVisionModelWithProjection.from_pretrained(
            model_name
        )

        feature_dim = self.backbone.config.projection_dim

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

        # Backbone Freeze 여부
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):

        outputs = self.backbone(
            pixel_values=pixel_values
        )

        image_feature = outputs.image_embeds

        logits = self.classifier(image_feature)

        return logits