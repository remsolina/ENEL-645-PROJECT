import torch
import torch.nn as nn

class CNN_RNN_FER(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN_RNN_FER, self).__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 x 24 x 24

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64 x 12 x 12
        )

        # Recurrent classifier
        self.lstm = nn.LSTM(input_size=64*12, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, 1, 48, 48)
        out = self.features(x)  # (B, 64, 12, 12)

        B, C, H, W = out.size()
        seq = out.permute(0, 2, 1, 3).reshape(B, H, C*W)  # (B, 12, 768)

        lstm_out, _ = self.lstm(seq)  # (B, 12, 128)
        final_feat = lstm_out[:, -1, :]
        logits = self.fc(final_feat)
        return logits

model = CNN_RNN_FER()   # or ViT_FER()
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
