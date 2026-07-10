# models/clip.py

import torch
import open_clip
from PIL import Image


class CLIPModel:
    def __init__(
        self,
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path, class_names):

        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

        text = self.tokenizer(class_names).to(self.device)

        with torch.no_grad():

            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (
                image_features @ text_features.T
            ).softmax(dim=-1)

        idx = similarity.argmax().item()

        return class_names[idx], similarity.squeeze().cpu().numpy()