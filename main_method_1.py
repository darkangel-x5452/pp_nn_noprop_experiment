import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Small MLP block for each time step
class DenoisingBlock(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim + embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

    def forward(self, z_prev, x):
        x_embed = torch.cat([z_prev, x], dim=1)
        return self.fc(x_embed)

if __name__ == "__main__":
    print("Starting")
    if not torch.cuda.is_available():
        raise SystemError(
            'GPU device not found. For fast training, please enable GPU. See section above for instructions.')


    print("Generate dummy tabular dataset")
    X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
    X = StandardScaler().fit_transform(X)
    y = y.astype(np.int64)

    print("Torch datasets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print("Configuration")
    T = 5  # Diffusion steps
    d_embed = 10
    alpha_schedule = torch.linspace(0.9, 0.1, T)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Embedding for binary classes (learnable)")
    class_embed = nn.Parameter(torch.randn(2, d_embed).to(device))  # ✅ THIS IS A LEAF

    output_layer = nn.Linear(d_embed, 2)

    print("Create T independent blocks")
    blocks = nn.ModuleList([DenoisingBlock(5, d_embed) for _ in range(T)]).to(device)
    output_layer = output_layer.to(device)
    # optimizer = torch.optim.Adam(list(blocks.parameters()) + [class_embed, output_layer.parameters()], lr=1e-3)
    optimizer = torch.optim.Adam(list(blocks.parameters()) + [class_embed] + list(output_layer.parameters()),
        lr=1e-3
    )


    def snr(t):  # SNR(t) = α / (1 - α)
        return alpha_schedule[t] / (1 - alpha_schedule[t])


    print("Training loop")
    for epoch in range(20):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            uy = class_embed[yb]
            loss = 0

            z_prev = None
            for t in range(T):
                alpha = alpha_schedule[t].to(device)
                noise = torch.randn_like(uy)
                zt = torch.sqrt(alpha) * uy + torch.sqrt(1 - alpha) * noise  # Sample from q(zt|y)
                if t == 0:
                    z_prev = zt
                    continue

                u_hat = blocks[t](z_prev, xb)
                denoise_loss = (snr(t) - snr(t - 1)) * F.mse_loss(u_hat, uy)
                loss += denoise_loss
                z_prev = zt

            print("Final classification prediction")
            logits = output_layer(z_prev)
            cls_loss = F.cross_entropy(logits, yb)
            loss += cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} - Loss: {loss.item():.4f}")


    print("Inference")
    def predict(loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                z = torch.randn(xb.size(0), d_embed).to(device)  # z0
                for t in range(T):
                    z = blocks[t](z, xb)
                logits = output_layer(z)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / total


    print("Train Accuracy:", predict(train_loader))
    print("Test Accuracy:", predict(test_loader))
