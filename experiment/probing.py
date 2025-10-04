import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import wandb
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import imageio
from torch.optim import Adam
import torch.distributions as dist
from torch.distributed import barrier
from torch.utils.data.distributed import DistributedSampler


import json
from torchvision import transforms as T, utils
import datetime

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from accelerate.logging import get_logger

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

__version__ = "0.0"

from torch.utils.data.dataloader import default_collate

from pynvml import *

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt

class prober: # object for poking into the weights of the model
    def __init__(self, probe_dir):
        self.activations = {}
        self.probe_dir = probe_dir

        if probe_dir is None:
            return

        try:
            os.mkdir(probe_dir)
        except OSError as error:
            print(error)

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()

        return hook

    def get_activations_with_grad(self, name):
        def hook(model, input, output):
            self.activations[name] = output
        return hook

    def save_activations(self, activation_name, file_name=None, use_numpy=True):
        """
        Save the activations of a specific layer to a file.

        Args:
            activation_name (str): Name of the layer whose activations are to be saved.
            file_name (str): Name of the file to save the activations.
            use_numpy (bool): If True, saves as numpy array; otherwise saves as torch tensor.

        Returns:
            None
        """
        if activation_name not in self.activations:
            raise ValueError(f"Activation {activation_name} not found.")

        activation = self.activations[activation_name]
        os.makedirs(self.probe_dir, exist_ok=True)
        if use_numpy:
            np.save(f"{self.probe_dir}/{file_name}", activation.cpu().numpy())
        else:
            torch.save(activation, f"{self.probe_dir}/{file_name}")

    def hook_all_layers(self, model):
        hooks = {}
        def hook_fn(name):
            def fn(_, __, output):
                self.activations[name] = output.detach()
                hooks[name] = self.activations[name]
            return fn

        for name, module in model.named_modules():
            module.register_forward_hook(hook_fn(name))
        return hooks

    # def show_tsne(self, task, episode, t=None):
    #     #import pdb; pdb.set_trace()
    #     plt.figure(1)
    #     activations_all = self.activations[name].cpu().numpy()
    #     dims = activations_all.shape

    #     if len(dims) not in [3, 4, 5]:
    #         return

    #     for i in range(dims[0]):
    #         if len(dims) == 3:
    #             activations = torch.from_numpy(activations_all[i])
    #             activations = activations.reshape(-1, dims[1])
    #         elif len(dims) == 4:
    #             activations = torch.from_numpy(activations_all[i])
    #             activations = activations.permute(1, 2, 0)
    #             activations = activations.reshape(-1, dims[1])
    #         elif len(dims) == 5: # Usually (b c f h w)
    #             activations = torch.from_numpy(activations_all[i])
    #             activations = activations.permute(1, 2, 3, 0)
    #             activations = activations.reshape(-1, dims[1])
    #         # print(activations.shape)
    #         # pca = PCA(n_components=)
    #         # pca_result = pca.fit_transform(activations)
    #         tsne = TSNE(n_components=2, perplexity=min(100, activations.shape[0] // 5), random_state=42)
    #         tsne_result = tsne.fit_transform(activations)
    #         # tsne_result = pca.fit_transform(activations)
    #         plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
        
    #     plt.title(name + f" {str(dims)} {str(activations.shape)}")
    #     plt.grid(True)

    #     if t is None:
    #         plt.savefig(f"{self.image_dir}/{task}/{episode}/{task}_tsne.png", bbox_inches='tight', dpi=300)
    #     else:
    #         plt.savefig(f"{self.image_dir}/{task}/{episode}/{task}_tsne_{t}.png", bbox_inches='tight', dpi=300)

    #     plt.close()

    # def show_video(self, name, method, timestep=None):
    #     """
    #     Visualize the activations of a specific layer using t-SNE, UMAP, or PCA.

    #     Args:
    #         name (str): Name of the layer to visualize.
    #         method (str): "tsne", "umap", or "pca"
    #         timestep (int, optional): just a label for the gif

    #     Returns:
    #         None
    #     """
    #     assert name in self.activations, f"Layer {name} not found in activations."
    #     assert method in {"tsne", "umap", "pca"}, "method must be 'tsne', 'umap', or 'pca'"

    #     embeddings = self.activations[name]
    #     gif_path = f"{self.image_dir}/{name}_{method}_{timestep}.gif"

    #     if len(embeddings.shape) == 4:
    #         embeddings = rearrange(embeddings, 't c h w -> c t h w')
    #         self.visualize_video_embeddings(embeddings, method, gif_path)
    #     elif len(embeddings.shape) == 5:
    #         B, C, T, H, W = embeddings.shape
    #         for b in range(B):
    #             self.visualize_video_embeddings(embeddings[b], method, gif_path.replace(".gif", f"_{b}.gif"))

    # def visualize_video_embeddings(self, embeddings, method, gif_path, frame_limit=None, figsize_per_video=(8, 5)):
    #     """
    #     Visualize video embeddings using t-SNE, UMAP, or PCA and save as a GIF.

    #     Args:
    #         embeddings (torch.Tensor): shape (B, C, T, H, W)
    #         method (str): "tsne", "umap", or "pca"
    #         gif_path (str): path to save the resulting GIF
    #         timestep (int, optional): just a label for the gif
    #         frame_limit (int, optional): max number of frames to visualize per video

    #     Returns:
    #         None
    #     """

    #     assert method in {"tsne", "umap", "pca"}, "method must be 'tsne', 'umap', or 'pca'"

    #     #import pdb; pdb.set_trace()

    #     embeddings = embeddings.detach().cpu()

    #     if len(embeddings.shape) == 4:
    #         C, T, H, W = embeddings.shape
    #         B = 1
    #     else:
    #         raise ValueError(f"Unsupported shape {embeddings.shape}. Expected 4D tensor.")

    #     if frame_limit:
    #         T = min(T, frame_limit)

    #     reducer = {
    #         "tsne": TSNE(n_components=2, init='pca', random_state=42),
    #         "pca": PCA(n_components=2),
    #         "umap": UMAP(n_components=2, random_state=42)
    #     }[method]

    #     images = []

    #     with TemporaryDirectory() as tmpdir:
    #         for t in range(T):
    #             fig, ax = plt.subplots(1, B, figsize=(figsize_per_video[0], figsize_per_video[1]))

    #             frame_feats = embeddings[:, t, :, :]  # shape (C, H, W)
    #             feats = frame_feats.permute(1, 2, 0).reshape(-1, C).numpy()  # shape (H*W, C)
    #             reduced = reducer.fit_transform(feats)

    #             x, y = reduced[:, 0], reduced[:, 1]
    #             sc = ax.scatter(x, y, c=np.linspace(0, 1, H*W), cmap='viridis', s=5)
    #             ax.set_title(f"Frame {t}, {method.upper()} Visualization ({embeddings.shape} to {reduced.shape})")
    #             ax.axis('off')
    #             ax.set_xlabel("Dim 1")
    #             ax.set_ylabel("Dim 2")

    #             frame_path = os.path.join(tmpdir, f"frame_{t:03d}.png")
    #             plt.tight_layout()
    #             plt.savefig(frame_path)
    #             images.append(imageio.imread(frame_path))
    #             plt.close()

    #         imageio.mimsave(gif_path, images, fps=2, loop=1)
