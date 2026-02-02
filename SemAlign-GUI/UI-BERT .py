import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from timm.models.vision_transformer import VisionTransformer


class UIBERT(nn.Module):
    """
    UI-BERT: BERT-based model for UI understanding with visual features
    Based on: Bai, X., et al. (2021). UI-BERT: Learning Generic Multimodal Representations for UI Understanding.
    """

    def __init__(self, visual_dim=768, text_dim=768, hidden_dim=512, num_heads=8):
        super(UIBERT, self).__init__()

        # Visual encoder (ViT)
        self.visual_encoder = VisionTransformer(
            img_size=224, patch_size=16, in_chans=3,
            embed_dim=visual_dim, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1
        )

        # Text encoder (BERT)
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_proj = nn.Linear(768, text_dim)

        # Multi-modal fusion
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj_fusion = nn.Linear(text_dim, hidden_dim)

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=0.1, batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, image, text_input_ids, text_attention_mask):
        # Visual feature extraction
        visual_features = self.visual_encoder.forward_features(image)  # [B, 197, D_vis]
        cls_visual = visual_features[:, 0]  # [B, D_vis]

        # Text feature extraction
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        cls_text = text_outputs.last_hidden_state[:, 0]  # [B, D_text]
        cls_text = self.text_proj(cls_text)

        # Project to common space
        visual_proj = self.visual_proj(cls_visual).unsqueeze(1)  # [B, 1, D_hidden]
        text_proj = self.text_proj_fusion(cls_text).unsqueeze(1)  # [B, 1, D_hidden]

        # Cross-modal attention
        attended_visual, _ = self.cross_attention(
            query=visual_proj,
            key=text_proj,
            value=text_proj
        )

        attended_text, _ = self.cross_attention(
            query=text_proj,
            key=visual_proj,
            value=visual_proj
        )

        # Concatenate and classify
        combined = torch.cat([
            attended_visual.squeeze(1),
            attended_text.squeeze(1)
        ], dim=1)

        output = self.classifier(combined)

        return output

    def extract_visual_features(self, image):
        with torch.no_grad():
            features = self.visual_encoder.forward_features(image)
        return features

    def extract_text_features(self, text_input_ids, text_attention_mask):
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            )
        return outputs.last_hidden_state


# Training script for UI-BERT
class UI_BERT_Trainer:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.criterion = nn.BCELoss()

    def preprocess_text(self, texts):
        """Tokenize text inputs"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        return encoded

    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            images1, images2, texts, labels = batch

            # Tokenize texts
            encoded_texts = self.preprocess_text(texts)

            # Move to device
            images1 = images1.to(self.device)
            images2 = images2.to(self.device)
            input_ids = encoded_texts['input_ids'].to(self.device)
            attention_mask = encoded_texts['attention_mask'].to(self.device)
            labels = labels.float().to(self.device)

            # Forward pass
            optimizer.zero_grad()

            # Process reference image
            outputs1 = self.model(images1, input_ids, attention_mask)

            # Process target image
            outputs2 = self.model(images2, input_ids, attention_mask)

            # Compute change probability (difference)
            change_prob = torch.abs(outputs1 - outputs2)

            # Compute loss
            loss = self.criterion(change_prob, labels.unsqueeze(1))
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        predictions = []
        ground_truth = []

        with torch.no_grad():
            for batch in dataloader:
                images1, images2, texts, labels = batch

                encoded_texts = self.preprocess_text(texts)

                images1 = images1.to(self.device)
                images2 = images2.to(self.device)
                input_ids = encoded_texts['input_ids'].to(self.device)
                attention_mask = encoded_texts['attention_mask'].to(self.device)

                outputs1 = self.model(images1, input_ids, attention_mask)
                outputs2 = self.model(images2, input_ids, attention_mask)

                change_prob = torch.abs(outputs1 - outputs2)

                predictions.extend(change_prob.cpu().numpy())
                ground_truth.extend(labels.numpy())

        return predictions, ground_truth