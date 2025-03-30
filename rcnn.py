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
