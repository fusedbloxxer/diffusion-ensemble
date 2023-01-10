import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.core.composition import Compose


def postprocess(images: torch.Tensor, size: int = 256) -> torch.Tensor:
  """Expects the images as (N, C, H, W) with values in [-1, 1].
  Swaps channels: (N, H, W, C), maps images in [0, 1] and applies postprocessing."""
  C, H, W = images[0].shape
  min_side = np.min([H, W])
  ratio = min_side / size

  transform = A.Compose([
    A.Resize(height=round(ratio * H), width=round(ratio * W), interpolation=cv2.INTER_LINEAR),
  ])

  processed = []
  for image in images:
    image = image.permute((1, 2, 0)).contiguous()
    processed_image = postprocess_image(image.numpy(), transform)
    processed.append(processed_image.unsqueeze(0))

  return torch.vstack(processed)


def postprocess_image(image: np.ndarray, t: Compose) -> torch.Tensor:
  # Adjust the output to [-1, 1]
  image = np.clip(image, a_min=-1.0, a_max=1.0)

  # Map from [-1, 1] to [0, 1]
  image = (image + 1.) / 2.

  # Resize the image
  image = t(image=image)['image']

  # Back to tensor
  return torch.tensor(image, dtype=torch.float32)


def preprocess(images: torch.Tensor, size: int = 256) -> torch.Tensor:
  """Expects the images as (N, H, W, C) with values in [0, 1].
  Applies preprocessing, maps images in [-1, 1] and swaps channels: (N, C, H, W)"""
  H, W, C = images[0].shape
  min_side = np.min([H, W])
  ratio = size / min_side

  transform = A.Compose([
    A.Resize(height=round(ratio * H), width=round(ratio * W), interpolation=cv2.INTER_AREA),
  ])

  processed = []
  for image in images:
    image = preprocess_image(image.numpy(), transform)
    image = image.permute((2, 0, 1)).contiguous()
    image = image.unsqueeze(0)
    processed.append(image)

  return torch.vstack(processed)


def preprocess_image(image: np.ndarray, t: Compose) -> torch.Tensor:
  # Map from [0, 1] to [-1, 1]
  image = 2. * image - 1

  # Resize the image
  image = t(image=image)['image']

  # Back to tensor
  return torch.tensor(image, dtype=torch.float32)

