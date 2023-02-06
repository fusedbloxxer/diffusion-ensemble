import pathlib as pb
from typing import List, Any, Optional, Dict, Callable, Tuple
import numpy as np
import PIL
import torch
from torch import Tensor
from PIL.Image import Image
from diffusers import DiffusionPipeline
import itertools
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import time
from datetime import datetime
import os

from .blocks import Model


class DiffusionModel(Model):
  def __init__(self,
               repo: str,
               num_inference_steps: int = 50,
               verbose: bool = True) -> None:
    super().__init__(verbose=verbose)

    # Internals
    self.repo = repo
    self.inference_steps = num_inference_steps
    self.pipeline = DiffusionPipeline.from_pretrained(self.repo)

  def forward(self,
              prompts: str | List[str],
              batch_size: int = 1,
              num_imgs_per_prompt: int | List[int] = 1) -> List[torch.Tensor]:
    # Args guarding
    if isinstance(prompts, str) and isinstance(num_imgs_per_prompt, list):
      raise Exception('A prompt cannot have multiple amounts of imgs')
    if isinstance(num_imgs_per_prompt, list) and len(num_imgs_per_prompt) != len(prompts):
      raise Exception('Need to provide num_imgs for all prompts')
    if isinstance(num_imgs_per_prompt, list) and min(num_imgs_per_prompt) <= 0:
      raise Exception(f'At least one image is needed per prompt! Actual: {min(num_imgs_per_prompt)}')
    if isinstance(num_imgs_per_prompt, int) and num_imgs_per_prompt <= 0:
      raise Exception(f'At least one image is needed per prompt! Actual: {num_imgs_per_prompt}')
    if isinstance(prompts, list) and len(prompts) == 0:
      raise Exception('At least one prompt must be provided')
    if isinstance(num_imgs_per_prompt, list):
      get_imgs_per_prompt = lambda i: num_imgs_per_prompt[i]
    else:
      get_imgs_per_prompt = lambda _: num_imgs_per_prompt

    # Process single prompt
    if isinstance(prompts, str):
      # Batch as many operations
      num_iters = num_imgs_per_prompt // batch_size
      num_last  = num_imgs_per_prompt %  batch_size

      # Use maximum batch-size for first num_iters steps
      images = []
      for _ in range(num_iters):
        iter_images = self.pipeline(prompts,
                                         output_type='npy',
                                         num_inference_steps=self.inference_steps,
                                         num_images_per_prompt=batch_size).images
        iter_images = torch.tensor(iter_images)
        images.append(iter_images)

      # Process the remainder
      if num_last != 0:
        iter_images = self.pipeline(prompts,
                                         output_type='npy',
                                         num_inference_steps=self.inference_steps,
                                         num_images_per_prompt=num_last).images
        iter_images = torch.tensor(iter_images)
        images.append(iter_images)

      # Stack tensors
      return [torch.vstack(images)]

    # Generate num_images per-prompt save them and repeat
    images = []
    tqdm_prompt = tqdm(prompts, disable=not self.verbose)
    for i, p in enumerate(tqdm_prompt):
      # Log progress
      tqdm_prompt.set_description(f'Prompt: {p}')

      # Generate images for the current prompt
      prompt_images = self.forward(p,
                                                 batch_size=batch_size,
                                                 num_imgs_per_prompt=get_imgs_per_prompt(i))

      # Retain the results
      images.extend(prompt_images)

    # Return a list with the generated images for each prompt
    return images

  def to(self, device: Optional[str | torch.device] = None) -> 'DiffusionModel':
    if device is None:
      return self
    if isinstance(device, str):
      device = torch.device(device)
    if not isinstance(device, torch.device):
      raise Exception('Invalid device type')
    super().to(device)
    self.pipeline = self.pipeline.to(device)
    return self


class DiffusersModel(Model):
  def __init__(self,
               repos: List[str],
               num_inference_steps: int | List[int] = 50,
               verbose: bool = True) -> None:
    super().__init__(verbose=verbose)

    # Args checking
    if isinstance(num_inference_steps, int):
      self.get_inference_steps = lambda _: num_inference_steps
    else:
      self.get_inference_steps = lambda i: num_inference_steps[i]

    # Internals
    self.repos = repos

  def forward(self,
              prompts: str | List[str] | List[List[str]],
              batch_size: int | List[int] = 1,
              num_imgs_per_prompt: int = 1,
              prompt_modifier: Callable[[str, int, str], str] = None) -> 'DiffusersOutput':
    # Args checking
    if isinstance(batch_size, int):
      get_batch_size: Callable[[int], int] = lambda _: batch_size
    else:
      get_batch_size: Callable[[int], int] = lambda i: batch_size[i]
    if prompt_modifier is None:
      prompt_modifier: Callable[[str, int, int, str], str] = dummy_prompt_changer
    if isinstance(num_imgs_per_prompt, list):
      raise Exception('num_imgs_per_prompt cannot be a list, all diffusers need to produce the same quantity!')
    if isinstance(prompts, list) and isinstance(prompts[0], list):
      get_prompts: Callable[[int], List[str]] = lambda model_index: [prompt_set[model_index]
                                                                     for prompt_set in prompts]
    elif isinstance(prompts, list):
      get_prompts: Callable[[int], List[str]] = lambda _: prompts
    elif isinstance(prompts, str):
      get_prompts: Callable[[int], List[str]] = lambda _: [prompts]

    # Generate all images sequentially for each model
    imgs: Dict[str, List[Tensor]] = {}
    tqdm_repos = tqdm(self.repos, disable=not self.verbose)
    for model_index, model_name in enumerate(tqdm_repos):
      # Log currently used model
      tqdm_repos.set_description(f'DiffusionModel: {model_name}')

      # Create model and move to device
      model = DiffusionModel(model_name, self.get_inference_steps(model_index)).to(self.device)

      # Customize the prompts for certain models
      model_prompts = get_prompts(model_index)
      model_prompts = [prompt_modifier(prompt, prompt_index, model_index, model_name)
                             for prompt_index, prompt in enumerate(model_prompts)]

      # Generate all prompts for the current model
      model_imgs = model.forward(model_prompts,
                                               batch_size=get_batch_size(model_index),
                                               num_imgs_per_prompt=num_imgs_per_prompt)

      # Free up memory
      del model

      # Save the results
      imgs[model_name] = model_imgs

    # Aggregate all results
    return DiffusersOutput(imgs, num_imgs_per_prompt, get_prompts, prompt_modifier)


class DiffusersOutput(object):
  def __init__(self,
               model_outputs: Dict[str, List[torch.Tensor]],
               num_imgs_per_prompt: int,
               get_prompts: Callable[[int], List[str]],
               prompt_modifier: Callable[[str, int, int, str], str] = None) -> None:
    super().__init__()

    # Args guarding
    if prompt_modifier is None:
      self.prompt_modifier: Callable[[str, int, str], str] = lambda prompt, _1, _2, _3: prompt
    else:
      self.prompt_modifier: Callable[[str, int, str], str] = prompt_modifier

    # Save the given components
    self.model_dict = model_outputs
    self.get_prompts = get_prompts
    self.model_names = list(model_outputs.keys())
    self.num_imgs_per_prompt = num_imgs_per_prompt

    # Get the size of the output
    first_element = list(model_outputs.values())[0]
    self.num_prompts = len(first_element)

    # Group model outputs by using the prompts
    self.prompts_: List[List[str]] = []
    self.images_: List[List[Tensor]] = []

    for prompt_index in range(self.num_prompts):
      prompt_set = []
      image_set = []

      for model_index, model_name in enumerate(self.model_names):
        # Get the list of prompts at pos_i for all models
        model_prompts = get_prompts(model_index)
        prompt = model_prompts[prompt_index]
        prompt = prompt_modifier(prompt, prompt_index, model_index, model_name)
        prompt_set.append(model_prompts[prompt_index])

        # Retrieve the associated images
        image: Tensor = model_outputs[model_name][prompt_index]
        image_set.append(image)

      self.prompts_.append(prompt_set)
      self.images_.append(image_set)

    # Provide helpful tensor output format
    self.images = torch.vstack([images for _, images in list(self)])
    self.prompts = self.prompts_

  def __getitem__(self, prompt_index: int | slice) -> Tuple[List[str], torch.Tensor]:
    """Stack all generated images for various prompts."""
    if isinstance(prompt_index, int):
      prompt_index = slice(prompt_index, prompt_index + 1)

    prompts = self.prompts_[prompt_index]
    images = self.images_[prompt_index]
    image_set = []

    for images_per_prompt in images:
      images_per_prompt_set = []

      # Stack the images in order along the num_imgs_per_prompt axis
      stacked_images_per_prompt = torch.vstack(images_per_prompt)

      # Perform strided gathering
      for jump_offset in range(self.num_imgs_per_prompt):
        images_per_prompt_i: Tensor = stacked_images_per_prompt[jump_offset::self.num_imgs_per_prompt]
        images_per_prompt_set.append(images_per_prompt_i)

      # (NumImgsPerPrompt x M x H x W x C)
      image_set.append(torch.stack(images_per_prompt_set, dim=0))

    # Stack on the prompt axis
    return prompts, torch.vstack(image_set)

  def __iter__(self) -> 'DiffusersOutput':
    self.prompt_index = 0
    return self

  def __next__(self) -> Tuple[List[str], torch.Tensor]:
    if self.prompt_index < self.num_prompts:
      prompt_output = self.__getitem__(self.prompt_index)
      self.prompt_index += 1
      return prompt_output
    else:
      raise StopIteration

  def show(self) -> None:
    # Find maximum number of cols & rows to be plotted
    max_cols: int = len(self.model_names)
    max_rows: int = self.num_prompts * self.num_imgs_per_prompt

    # Define plot
    f, ax = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(10, 10))

    def show_image(i: int, skip: int, stride: int, model: str, img: torch.Tensor, prompt: str,
                   nrows: int, ncols: int, ax: plt.Axes) -> None:
      # Obtain the current axis of the subplot
      if nrows > 1 and ncols > 1:
        c_axis = ax[i][skip + stride]
      elif nrows == 1 and ncols > 1:
        c_axis = ax[skip + stride]
      elif ncols == 1 and nrows > 1:
        c_axis = ax[i]
      else:
        c_axis = ax

      # Show the image and its prompt
      c_axis.imshow(img)
      c_axis.set_title(prompt, wrap=True)
      c_axis.set_xlabel(model, wrap=True)

    # Start plotting values
    # Iterate through each prompt set
    for i, (prompts, images) in enumerate(list(self)):
      N, M, H, W, C = images.shape

      # Iterate through the images generated for a single prompt
      for n in range(N):
        # Iterate through each model's images
        for m in range(M):
          model_name = self.model_names[m]
          image = images[n][m]
          prompt = prompts[0][m]
          show_image(N * i + n, 0, m, model_name, image, prompt, max_rows, max_cols, ax)

    f.tight_layout()
    plt.show()

  def save(self, root_dir: pb.Path, entry_dir: Optional[str] = None) -> None:
    raise NotImplemented('Need to work on this') # TODO: reuse generated images by saving them

    if entry_dir is None:
      timestamp = time.time()
      current_datetime = datetime.fromtimestamp(timestamp)
      datetime_now = current_datetime.strftime('%d_%B_%Y_%H_%M_%S')
      entry_dir = datetime_now

    # Prepent the root dir to the entry, make sure it exists
    entry_dir = root_dir / entry_dir
    if not entry_dir.exists():
      entry_dir.mkdir(parents=True, exist_ok=True)
    if not entry_dir.is_dir():
      raise Exception(f'{entry_dir} is not a directory')


def dummy_prompt_changer(prompt: str, prompt_index: int, model_index: int, model_name: str) -> str:
  return prompt

# landscape grup oameni, masina dog
# two faces