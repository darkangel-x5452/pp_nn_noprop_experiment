import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader, TensorDataset

# Generate synthetic data
n_samples = 1000
n_features = 5
X = torch.randn(n_samples, n_features)
y = torch.randint(0, 2, (n_samples,))

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Hyperparameters
T = 10
embedding_dim = 5
hidden_dim = 10
eta = 0.1
epochs = 100


# Define cosine noise schedule
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps
    x = torch.linspace(0, steps, steps + 1)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return alphas_cumprod


alpha_bar = cosine_beta_schedule(T)


# Embedding layer for binary classes
class EmbeddingLayer(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        nn.init.orthogonal_(self.embedding.weight)

    def forward(self, y):
        return self.embedding(y)





# Denoising network for each time step
class DenoisingNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, z_prev, x):
        x_flat = x.view(x.size(0), -1)
        z_prev_flat = z_prev.view(z_prev.size(0), -1)
        combined = torch.cat([x_flat, z_prev_flat], dim=1)
        h = torch.relu(self.fc1(combined))
        h = torch.relu(self.fc2(h))
        return self.fc3(h)


if __name__ == "__main__":
    embedding_layer = EmbeddingLayer(2, embedding_dim)
    networks = nn.ModuleList([DenoisingNet(n_features, hidden_dim) for _ in range(T)])
    classification_layer = nn.Linear(embedding_dim, 2)

    # Optimizer
    optimizer = optim.Adam(
        list(networks.parameters()) + list(embedding_layer.parameters()) + list(classification_layer.parameters()),
        lr=0.001,
        weight_decay=0.001
    )

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()

            # Embedding lookup
            uy = embedding_layer(y_batch)

            # Classification loss on z_T
            alpha_bar_T = alpha_bar[-1]
            z_T = torch.sqrt(alpha_bar_T) * uy + torch.randn_like(uy) * torch.sqrt(1 - alpha_bar_T)
            class_logits = classification_layer(z_T)
            class_loss = nn.CrossEntropyLoss()(class_logits, y_batch)

            # KL divergence term
            alpha_bar_0 = alpha_bar[0]
            z0_mean = torch.sqrt(alpha_bar_0) * uy
            z0_var = 1 - alpha_bar_0
            kl_loss = 0.5 * (z0_var + z0_mean.pow(2) - 1 - torch.log(z0_var)).sum(1).mean()

            # Denoising loss
            denoise_loss = 0
            for t in range(1, T + 1):
                alpha_prev = alpha_bar[t - 1]
                z_prev_mean = torch.sqrt(alpha_prev) * uy
                z_prev = z_prev_mean + torch.randn_like(uy) * torch.sqrt(1 - alpha_prev)

                u_hat = networks[t - 1](z_prev, x_batch)
                snr_diff = (alpha_bar[t] / (1 - alpha_bar[t])) - (alpha_bar[t - 1] / (1 - alpha_bar[t - 1]))
                mse = nn.MSELoss(reduction='sum')(u_hat, uy)
                denoise_loss += snr_diff * mse

            denoise_loss = denoise_loss * eta / 2

            # Total loss
            loss = class_loss + kl_loss + denoise_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}')

    # Save the model
    torch.save({
        'networks': networks.state_dict(),
        'embeddings': embedding_layer.state_dict(),
        'classifier': classification_layer.state_dict()
    }, 'noprop_model.pth')