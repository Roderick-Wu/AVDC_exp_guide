import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from nflows.flows import Flow
from nflows.transforms import (
    CompositeTransform,
    ReversePermutation,
    AffineCouplingTransform,
    ActNorm
)
from nflows.distributions import StandardNormal
from nflows.nn.nets import MLP

class mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.block(x)


class simple_classifier(nn.Module):
    def __init__(self, x_input_dim, hidden_dim, output_dim):
        super(simple_classifier, self).__init__()
        self.blocks = nn.Sequential(
            nn.Linear(x_input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, t):
        x = x.permute(0, 2, 1, 3, 4)
        x = torch.mean(x, axis=(-1, -2))  # Average over the last two dimensions (H, W)
        xt = torch.cat((x, t), dim=-1) # Shape: (B, F, C + 1)
        return self.blocks(xt)


class classifier(nn.Module):
    def __init__(self, x_input_dim, hidden_dim, output_dim):
        super(classifier, self).__init__()
        self.mlp = mlp(x_input_dim + 1, hidden_dim, output_dim)

    def forward(self, x, x2, t):
        t_encoded = t.unsqueeze(-1)
        x = x.mean(axis=(-1, -2))
        xt = torch.cat((x, t_encoded), dim=-1) # Shape: (B, C + 1)
        return nn.functional.softmax(self.mlp(xt), dim=-1)


class RNDDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        # Fixed target network
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Predictor network
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Freeze the target network
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            target_out = self.target(x)
        pred_out = self.predictor(x)
        error = F.mse_loss(pred_out, target_out, reduction='none').mean(dim=1)  # Per sample
        return error  # Higher = more OOD

    def train_step(self, x, optimizer):
        self.train()
        optimizer.zero_grad()
        with torch.no_grad():
            target_out = self.target(x)
        pred_out = self.predictor(x)
        loss = F.mse_loss(pred_out, target_out)
        loss.backward()
        optimizer.step()
        return loss.item()




class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t, dtype=None):
        if dtype is None:
            dtype = next(self.parameters(), torch.tensor([], device=t.device)).dtype

        device = t.device
        half = self.dim // 2
        t_float = t.float()

        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / half
        )
        args = t_float[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return emb.to(dtype)

class VideoFrameBinaryClassifier(nn.Module):
    def __init__(self, in_channels, height, width, hidden_dim=128, emb_dim=256, num_classes=2):
        super().__init__()
        self.embed = SinusoidalTimestepEmbedding(emb_dim)

        # conv layers
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

        # keep norms in float32 for stability
        self.norm1 = nn.LayerNorm([hidden_dim, height, width], elementwise_affine=False).to(torch.float32)
        self.norm2 = nn.LayerNorm([hidden_dim, height, width], elementwise_affine=False).to(torch.float32)

        # embedding projection
        self.emb_proj = nn.Linear(emb_dim, hidden_dim)

        # classifier
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, t):
        """
        x: (B, C, H, W) or (B, C, T, H, W)
        t: (B,) or (B, T) timesteps
        """
        if x.dim() == 5:  # video: (B, C, T, H, W)
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            t = t.reshape(B * T)

        # conv stack
        h = F.relu(self.conv1(x))
        h = self.norm1(h.to(torch.float32)).to(h.dtype)  # force norm in fp32
        h = F.relu(self.conv2(h))
        h = self.norm2(h.to(torch.float32)).to(h.dtype)

        # add timestep embedding
        emb = self.embed(t, dtype=h.dtype)
        emb = self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        h = h + emb

        # global average pool
        h = h.mean(dim=(2, 3))

        return self.fc(h)





# Fix for nflows provided mlp, use this custom one for now
class SimpleContextMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes=[128, 128], activation=nn.ReLU):
        super().__init__()
        layers = []
        sizes = [in_features] + hidden_sizes
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(activation())

        layers.append(nn.Linear(sizes[-1], out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x, context=None): # Ignore the context
        return self.net(x)



def create_logpzo_model(input_dim, hidden_dim=128, num_layers=5):
    def create_net(in_features, out_features):
        return SimpleContextMLP(
            in_features=in_features,
            out_features=out_features,
            hidden_sizes=[hidden_dim]
        )

    transforms = []
    for _ in range(num_layers):
        transforms.append(ActNorm(features=input_dim))
        transforms.append(ReversePermutation(features=input_dim))
        transforms.append(
            AffineCouplingTransform(
                mask=torch.arange(input_dim) % 2,  # Alternating mask
                transform_net_create_fn=create_net,
            )
        )

    transform = CompositeTransform(transforms)
    base_dist = StandardNormal([input_dim])
    return Flow(transform, base_dist)

