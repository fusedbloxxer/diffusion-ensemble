import pathlib as pb
import clip
from clip.model import CLIP
import torch
from torch import Tensor
import typing
from typing import Callable
import kornia as K
import kornia.augmentation as KA
import numpy as np

from .blocks import Model


class ImageEncoderCLIP(Model):
  def __init__(self, clip_path: pb.Path,  verbose: bool = True) -> None:
    super().__init__(verbose)

    # Internal model from OpenAI
    self.clip_model, _ = clip.load('ViT-B/32', download_root=clip_path)
    n_px = self.clip_model.visual.input_resolution
    self.preprocess: Callable[[Tensor], Tensor] = lambda x: clip_preprocess(x, n_px)

  def forward(self, target: Tensor, pred: Tensor) -> Tensor:
    """Receive two tensors of the same size (N, C, H, W) """
    assert np.equal(target.size(), pred.size()).all(), 'tensor sizes differ! target: {}, pred: {}'.format(target.size(), pred.size())

    # Apply standard preprocessing similar to OpenAI
    target, pred = self.preprocess(target), self.preprocess(pred)
    images: Tensor = torch.vstack([target, pred])
    features: Tensor = self.clip_model.encode_image(images)

    # Recover the features
    split_features = torch.vsplit(features, 2)
    target_features = split_features[0]
    pred_features = split_features[1]

    # Compute the loss
    loss: Tensor = torch.nn.functional.mse_loss(pred_features, target_features, reduction='none')
    return loss.mean(dim=-1)


def clip_preprocess(images: Tensor, n_px: int) -> Tensor:
  """Excpect a tensor of images of size (N, C, H, W) or (C, H, W)"""
  return KA.ImageSequential(
    KA.Resize((n_px, n_px), resample=K.constants.Resample.BICUBIC),
    KA.CenterCrop((n_px, n_px)),
    KA.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                 std=(0.26862954, 0.26130258, 0.27577711))
  )(images)

