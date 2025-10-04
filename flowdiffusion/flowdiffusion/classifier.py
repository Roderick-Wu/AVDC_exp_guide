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
        x = x.mean(axis=(-1, -2))  # Average over the last two dimensions (H, W)
        xt = torch.cat((x, t), dim=-1) # Shape: (B, F, C + 1)
        return nn.functional.softmax(self.blocks(xt), dim=-1)


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


class VideoFrameTransformer(nn.Module):
    def __init__(self, in_channels, num_frames, height, width, embed_dim=256, num_heads=4, num_layers=2, num_classes=2):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Flatten each frame into a vector and project to embedding space
        self.frame_proj = nn.Linear(in_channels * height * width, embed_dim)

        # Positional encoding for frames (temporal order)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, embed_dim))

        # Transformer encoder (temporal attention across frames)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier head (per frame)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: (B, C, F, H, W)
        Returns: (B, F, num_classes)
        """
        B, C, F, H, W = x.shape

        # Rearrange frames: (B, F, C*H*W)
        x = x.permute(0, 2, 1, 3, 4).reshape(B, F, C*H*W)

        # Project to embedding space
        x = self.frame_proj(x)  # (B, F, embed_dim)

        # Add temporal positional encoding
        x = x + self.pos_embedding[:, :F, :]

        # Transformer encoder
        x = self.transformer(x)  # (B, F, embed_dim)

        # Frame-wise classification
        logits = self.classifier(x)  # (B, F, num_classes)

        return nn.functional.softmax(logits, dim=-1)

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

