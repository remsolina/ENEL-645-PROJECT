import torch
import torch.nn as nn

class ViT_FER(nn.Module):
    def __init__(self, img_size=48, patch_size=6, emb_dim=128, num_heads=8, num_layers=4, num_classes=7):
        super(ViT_FER, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size
        self.patch_embed = nn.Linear(self.patch_dim, emb_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp_head = nn.Linear(emb_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (batch, 1, 48, 48)
        B = x.size(0)

        # Patchify
        patches = x.unfold(2, 6, 6).unfold(3, 6, 6)
        patches = patches.contiguous().view(B, self.num_patches, -1)  # (B, 64, 36)

        tokens = self.patch_embed(patches)  # (B, 64, emb_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, emb_dim)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 65, emb_dim)

        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]

        tokens = tokens.transpose(0, 1)  # (65, B, emb_dim)
        encoded = self.transformer(tokens)  # (65, B, emb_dim)

        cls_out = encoded[0]  # (B, emb_dim)
        logits = self.mlp_head(cls_out)  # (B, 7)

        return logits
