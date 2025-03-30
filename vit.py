import torch
import torch.nn as nn

class ViT_FER(nn.Module):
    def __init__(self, img_size=48, patch_size=6, emb_dim=128, num_heads=8, num_layers=4, num_classes=7):
        super(ViT_FER, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2  # e.g., 64 patches
        self.patch_dim = patch_size * patch_size          # 6x6=36 (for 1 channel)
        # Linear projection of flattened patches to embedding dim
        self.patch_embed = nn.Linear(self.patch_dim, emb_dim)
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, emb_dim))
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=emb_dim*4, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Classification head
        self.mlp_head = nn.Linear(emb_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # initialize positional embeddings
        nn.init.normal_(self.cls_token, std=0.02)        # initialize class token

    def forward(self, x):
        # x shape: (batch, 1, 48, 48)
        B = x.size(0)
        # Divide image into patches and flatten
        patches = x.unfold(2, 6, 6).unfold(3, 6, 6)       # shape: (B, 1, 8, 8, 6, 6)
        patches = patches.contiguous().view(B, self.num_patches, -1)  # (B, 64, 36)
        # Linear patch embedding
        tokens = self.patch_embed(patches)               # (B, 64, 128)
        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)    # (B, 1, 128)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 65, 128)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]  # add positional embedding
        # Transpose for transformer: (sequence_length, batch, embed_dim)
        tokens = tokens.transpose(0, 1)  # shape: (65, B, 128)
        encoded = self.transformer(tokens)  # (65, B, 128)
        cls_out = encoded[0]               # (B, 128) => encoded class token
        logits = self.mlp_head(cls_out)    # (B, 7)
        return logits

model = ViT_FER()  # or CNN_RNN_FER()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)              # forward pass
        loss = criterion(outputs, labels)    # compute loss
        loss.backward()                      # backpropagate
        optimizer.step()                     # update weights
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    correct = total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.3f}, Val Loss={avg_val_loss:.3f}, Val Acc={val_acc:.3f}")
    # (Save best model based on val_acc)
