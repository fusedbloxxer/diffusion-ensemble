import typing
from typing import Tuple, Any, List, Dict, Optional, Callable
import torch
from torch import Tensor
import pathlib as pb
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from omegaconf import OmegaConf, DictConfig, ListConfig
from matplotlib import pyplot as plt
from IPython import display
import kornia as K
import kornia.augmentation as KA

from taming.modules.losses.lpips import LPIPS

from .losses import ImageEncoderCLIP
from .generate import DiffusersModel
from .vqgan import VQGAN
from .blocks import Model


class DiffusionEnsemble(Model):
  def __init__(self,
               diffusers: List[str],
               vqgan_config_path: pb.Path,
               vqgan_ckpt_path: Optional[pb.Path] = None,
               clip_path: pb.Path | None = None,
               num_inference_steps: int | List[int] = 50,
               verbose: bool = True) -> None:
    super().__init__(verbose)

    # Create the inner models
    self.diffusers = DiffusersModel(diffusers, num_inference_steps, verbose)
    self.vqgan = VQGAN(vqgan_config_path, vqgan_ckpt_path, verbose)

    # According to: https://github.com/richzhang/PerceptualSimilarity
    # Expects RGB images normalized to [-1, 1]
    self.perceptual_loss = LPIPS().eval()

    # According to: https://github.com/openai/CLIP
    # Expects unnormalized RGB images in [0.0, 1.0]
    self.clip_image_loss = ImageEncoderCLIP(clip_path=clip_path)

  def compose(self,
              x: torch.Tensor,
              coef: torch.Tensor,
              at_layer: str = 'postquant') -> torch.Tensor:
    """Compose images in latent space and reconstruct them.

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
    # assert torch.abs(coef.sum() - 1.0) <= 1e-6, f'the coef array must sum up to 1.0, got: {coef.sum()}'
    assert coef.shape[0] == x.shape[1], f'mismatch coef shape with number of diffusion models, expected {x.shape[1]}, got: {coef.shape[0]}'
    N, M, C, H, W = x.size()

    # Batch processing N x M
    h = x.reshape((N * M, C, H, W))

    # Encode the images up to a specific layer
    latent_vector = self.vqgan.encode_to_layer(h, at_layer)

    # Compose the latent vectors
    latent_vector = latent_vector.reshape((N, M, *latent_vector.shape[1:]))
    coef = coef.reshape((1, M, 1, 1, 1))
    comp = torch.sum(coef * latent_vector, dim=1)

    # Decode the composed vectors
    return self.vqgan.decode_from_layer(comp, at_layer)

  def forward(self,
              x: torch.Tensor,
              coef: torch.Tensor,
              at_layer: str = 'prequant',
              n_steps: int = 10_000,
              learn_rate: float = 15e-2,
              latent_from: str = 'noise',
              decay: Tuple[float, float, float] = (100, 0.9, 2e-4),
              composition: str = 'bar') -> Tuple[torch.Tensor, torch.Tensor]:
    """Summary

    Args:
        x (torch.Tensor): Tensor of size (N, M, C, H, W) where M is the number of
        diffusion models used in the ensemble.
        coef (torch.Tensor): Tensor of N coefficients.
        at_layer (str): The layer which should be used to compose the M elements.
        Can be one of: 'preconv' | 'prequant' | 'postquant'.
        latent_from (str): The latent vector can be obtained either through
        composing the input images with coef or by using noise as starting point.
        Can be one of: 'noise' | 'input'.
        composition (str): Indicate the type of operation to perform when
        mixing the images.
        Can be one of: 'bar' | 'rvec' (baricentric | vector).
    """
    latent_from_options = ['noise', 'input']
    at_layer_options = ['preconv', 'prequant', 'postquant']
    loss_fun_options = ['clip', 'perceptual']
    assert latent_from in latent_from_options, f'invalid latent_vector origin, got: {latent_from}'
    assert x.ndim == 5, f'invalid input tensor size, expected 5 but got: {x.ndim}'
    assert x.shape[1] == len(self.diffusers.repos), f'diffusers count mismatch, got: {x.shape[1]}'
    assert at_layer in at_layer_options, f'invalid at_layer, got: {at_layer}'
    assert coef.shape[0] == x.shape[1], f'mismatch coef shape with number of diffusion models, expected {x.shape[1]}, got: {coef.shape[0]}'
    assert composition in ['bar', 'rvec'], f'mismatch composition type, got: {composition}'
    N, M, C, H, W = x.size()

    # Compute latent_vector size
    l_N: int = N
    l_C: int = self.vqgan.model.encoder.z_channels
    l_H: int = H // 2 ** self.vqgan.model.encoder.ch_mult[-1]
    l_W: int = W // 2 ** self.vqgan.model.encoder.ch_mult[-1]

    if latent_from == 'input':
      # Batch processing N x M
      h = x.reshape((N * M, C, H, W))

      # Encode the images up to a specific layer
      latent_vector = self.vqgan.encode_to_layer(h, at_layer)

      # Resize the vectors
      latent_vector = latent_vector.reshape((N, M, *latent_vector.shape[1:]))

      # Compose the latent features
      if composition == 'bar':
        latent_vector = torch.sum(coef.reshape((1, M, 1, 1, 1)) * latent_vector, dim=1)
      elif composition == 'rvec':
        mask = torch.cat([torch.zeros(128), torch.ones(128)], dim=0).to(self.device, dtype=torch.int64).reshape((16, 16))
        mask: torch.Tensor = torch.nn.functional.one_hot(mask, M)
        mask: torch.Tensor = mask.permute((2, 0, 1))[None, :, None, ...]
        latent_vector = torch.sum(mask * latent_vector, dim=1)
    else:
      # Init latent_vector from noise
      latent_vector = torch.randn((l_N, l_C, l_H, l_W)).to(self.device)

    # Create the optimizer w.r.t. the latent vector
    latent_vector.requires_grad_(True)
    optim = torch.optim.Adam([latent_vector], lr=learn_rate, betas=(0.9, 0.999))
    sch_lr = torch.optim.lr_scheduler.StepLR(optim,
                                                     step_size=decay[0],
                                                     gamma=decay[1],
                                                     verbose=self.verbose)

    # Update the latent vector iteratively
    for e in range(n_steps):
      # Decode the composed vectors
      rec_imgs = self.vqgan.decode_from_layer(latent_vector, at_layer)

      # Normalize each output per-channel to [-1, 1]
      rec_imgs = rec_imgs.clamp(-1, 1)

      # Compute perceptual loss for the composed image against the model specific
      inp_imgs = x.reshape(N * M, C, H, W)
      rec_imgs = rec_imgs.unsqueeze(1).tile((1, M, 1, 1, 1)).reshape(N * M, C, H, W)

      # Augment the image crops
      p_loss: List[torch.Tensor] = []
      i_loss: List[torch.Tensor] = []
      crop_size: List[int] = [256, 64, 32]
      crop_nums: List[int] = [1, 8, 16]
      R: int = len(crop_nums)
      crop_imgs = self.crop_augment(inp_imgs,
                                                                 rec_imgs,
                                                                 crop_size,
                                                                 crop_nums,
                                                                 augment=False)

      # Use coefficients for each patch based on the inverse of their frequency
      patch_loss_coefs = torch.cat([torch.ones((crop_num,)) / crop_num for crop_num in crop_nums], dim=0)
      patch_loss_coefs = patch_loss_coefs.to(self.device)

      # Compute perceptual loss for all crop sizes
      for (input_crops, augmented_crops) in crop_imgs:
        c_loss: torch.Tensor = self.perceptual_loss(input_crops, augmented_crops)
        p_loss.append(c_loss.flatten(start_dim=0))

      # Compute the weighted perceptual loss
      p_loss: torch.Tensor = torch.cat(p_loss, dim=0).reshape((-1, N, M))
      p_loss = coef.tile((1, N, 1)) * p_loss
      p_loss = p_loss.sum(dim=2).T @ patch_loss_coefs.unsqueeze(1)

      # Compute clip loss for all crop sizes
      for (input_crops, augmented_crops) in crop_imgs:
        c_loss: torch.Tensor = self.clip_image_loss(input_crops, augmented_crops)
        i_loss.append(c_loss.flatten(start_dim=0))

      # Compute the weighted perceptual loss
      i_loss: torch.Tensor = torch.cat(i_loss, dim=0).reshape((-1, N, M))
      i_loss = coef.tile((1, N, 1)) * i_loss
      i_loss = i_loss.sum(dim=2).T @ patch_loss_coefs.unsqueeze(1)

      # Total loss
      loss = p_loss * 0.125 + i_loss * 1.0

      # Reset gradients
      optim.zero_grad()

      # Propagate the gradient loss over the latent vectors in the batch
      loss.backward()

      # Update the latent vector
      optim.step()
      sch_lr.step()

      # Display intermediary results
      if e % 50 == 0:
        display.clear_output()
        h = x.reshape((N * M, C, H, W))
        out_img = ((rec_imgs[0] + 1.) / 2).permute((1, 2, 0)).detach().cpu()
        in_imgs = ((h + 1.) / 2).permute((0, 2, 3, 1)).cpu()

        f, ax = plt.subplots(1, 3, figsize=(15, 15))
        ax[0].imshow(in_imgs[0])
        ax[1].imshow(in_imgs[1])
        ax[2].imshow(out_img)
        plt.title(f'[{e} / {n_steps}]: {p_loss.item()}, {sch_lr.get_lr()}')
        plt.show()
        print(loss.item())
        print(coef.flatten())
        print(p_loss.flatten())
    return rec_imgs, p_loss

  def crop_augment(self,
                   inp_imgs: torch.Tensor,
                   rec_imgs: torch.Tensor,
                   crop_size: List[int] = [32, 64, 128, 128],
                   crop_nums: List[int] = [32, 16,   8,   4],
                   augment: bool = True) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    crops: List[torch.Tensor] = []
    for i, (sz_crop, n_crops) in enumerate(zip(crop_size, crop_nums)):
      # Make use of crops on each image
      t_crop = KA.RandomCrop(size=(sz_crop,) * 2, same_on_batch=True)

      # Define augmentations to improve generalization
      if augment:
        t_aug = KA.ImageSequential(
          KA.RandomHorizontalFlip(p=0.5),
          KA.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
          KA.RandomGaussianNoise(mean=0, std=.5, p=0.5),
          KA.RandomPerspective(distortion_scale=0.25, p=0.5),
          KA.RandomAffine(degrees=(-15., 20.), translate=(.1, .25), scale=(.75, .95), p=0.5),
        )

      # Apply different cropping & aug across stacks
      inp_crop_stack: List[torch.Tensor] = []
      aug_crop_stack: List[torch.Tensor] = []
      for _ in range(n_crops):
        # Crops the same way across all images
        batched_imgs = torch.cat((inp_imgs, rec_imgs), dim=0)

        # Crop each image in the same location and augment them
        batched_crops: torch.Tensor = t_crop(batched_imgs)

        # Divide the batch to retrieve the pairs of images
        inp_crops, rec_crops = torch.split(batched_crops, 2, dim=0)

        # Augmentate the generated images
        if augment:
          aug_crops: torch.Tensor = t_aug(rec_crops)
        else:
          aug_crops: torch.Tensor = rec_crops

        # Save them along the way
        inp_crop_stack.append(inp_crops)
        aug_crop_stack.append(aug_crops)

      # Aggregate the crops of the same size
      inp_crops: torch.Tensor = torch.cat(inp_crop_stack)
      aug_crops: torch.Tensor = torch.cat(aug_crop_stack)

      # Save the crops
      crops.append((inp_crops, aug_crops))
    return crops

