import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)


# 2. Define NoProp model components
class DenoiseBlock(nn.Module):
    """Single denoising block u_θ"""

    def __init__(self, input_dim=5, hidden_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for noisy label
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, z):
        combined = torch.cat([x, z], dim=1)
        return self.net(combined)


class NoPropModel(nn.Module):
    def __init__(self, T=5, feature_dim=5, hidden_dim=10):
        super().__init__()
        self.T = T
        self.blocks = nn.ModuleList([DenoiseBlock(feature_dim, hidden_dim) for _ in range(T)])
        self.final_layer = nn.Linear(1, 1)

        # Cosine noise schedule
        self.alphas = torch.cos(torch.linspace(0, torch.pi / 2, T + 1)) ** 2
        self.snr = self.alphas / (1 - self.alphas)

    def forward(self, x, train=True):
        if train:
            # During training: sample random time step
            t = torch.randint(1, self.T + 1, (1,)).item()
            alpha_t = self.alphas[t]

            # Add noise to labels
            noise = torch.randn_like(y_train) * torch.sqrt(1 - alpha_t)
            z_t = alpha_t ** 0.5 * y_train + noise

            # Predict clean label
            pred = self.blocks[t - 1](x, z_t)
            return torch.sigmoid(pred)
        else:
            # Inference: denoising chain
            z = torch.randn_like(y_test)  # Start from noise
            for t in reversed(range(self.T)):
                alpha_t = self.alphas[t + 1]
                pred = self.blocks[t](x, z)
                z = (alpha_t ** 0.5 * pred) + ((1 - alpha_t) ** 0.5) * torch.randn_like(z)
            return torch.sigmoid(self.final_layer(z))


if __name__ == "__main__":
    # 3. Training setup
    model = NoPropModel(T=5)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass with random time step sampling
        outputs = model(X_train, train=True)
        loss = criterion(outputs, y_train)

        # Add KL divergence (simplified)
        kl_loss = 0.5 * torch.sum(model.alphas[0] / (1 - model.alphas[0]))  # Simplified KL
        total_loss = loss + 0.1 * kl_loss  # η=0.1 from paper

        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(X_test, train=False)
                test_loss = criterion(preds, y_test)
                acc = ((preds > 0.5).float() == y_test).float().mean()
            print(f'Epoch {epoch}, Loss: {total_loss.item():.4f}, Test Acc: {acc:.4f}')

    # 5. Save model
    torch.save(model.state_dict(), 'noprop_model.pth')