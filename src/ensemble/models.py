import typing
from typing import Tuple, Any, List, Dict, Optional
import torch
import pathlib as pb
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig, ListConfig

from taming.models.vqgan import GumbelVQ, VQModel

from .generate import DiffusersModel
from .vqgan import VQGAN
from .blocks import Model


class DiffusionEnsemble(Model):
  def __init__(self,
               diffusers: List[str],
               vqgan_config_path: pb.Path,
               vqgan_ckpt_path: Optional[pb.Path] = None,
               num_inference_steps: int | List[int] = 50,
               verbose: bool = True) -> None:
    super().__init__(verbose)

    # Create the inner models
    self.diffusers = DiffusersModel(diffusers, num_inference_steps, verbose)
    self.vqgan = VQGAN(vqgan_config_path, vqgan_ckpt_path, verbose)

  def compose(self, x: torch.Tensor, coef: torch.Tensor, at_layer: str = 'postquant'):
    """Summary

    Args:
        x (torch.Tensor): Tensor of size (N, M, C, H, W) where M is the number of
        diffusion models used in the ensemble.
        coef (torch.Tensor): Tensor of N positive coefficients that should sum to 1.
        at_layer (str): The layer which should be used to compose the M elements.
        Can be one of: 'preconv' | 'prequant' | 'postquant'.
    """
    at_layer_options = ['preconv', 'prequant', 'postquant']
    assert x.ndim == 5, f'invalid input tensor size, expected 5 but got: {x.ndim}'
    assert x.shape[1] == len(self.diffusers.repos), f'diffusers count mismatch, got: {x.shape[1]}'
    assert at_layer in at_layer_options, f'invalid at_layer, got: {at_layer}'
    assert torch.abs(coef.sum() - 1.0) <= 1e-6, f'the coef array must sum up to 1.0, got: {coef.sum()}'
    assert coef.shape[0] == x.shape[1], f'mismatch coef shape with number of diffusion models, expected {x.shape[1]}, got: {coef.shape[0]}'
    N, M, C, H, W = x.size()

    # Batch processing N x M
    h = x.reshape((N * M, C, H, W))

    # Encode the images up to a specific layer
    if at_layer == 'preconv':
      h = self.vqgan.encode_to_preconv(h)
    elif at_layer == 'prequant':
      h = self.vqgan.encode_to_prequant(h)
    elif at_layer == 'postquant':
      h, _, _ = self.vqgan.encode(h)

    # Compose the latent vectors
    h = h.reshape((N, M, *h.shape[1:]))
    coef = coef.reshape((1, M, 1, 1, 1))
    comp = torch.sum(coef * h, dim=1)

    # Decode the composed vectors
    if at_layer == 'preconv':
      h = self.vqgan.decode_from_preconv(comp)
    elif at_layer == 'prequant':
      h = self.vqgan.decode_from_prequant(comp)
    elif at_layer == 'postquant':
      h = self.vqgan.decode(comp)

    return h

