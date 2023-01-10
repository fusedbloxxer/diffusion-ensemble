import pathlib as pb
from typing import List, Any, Optional, Dict, Callable, Tuple
import numpy as np
import PIL
import torch
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
              prompt: str | List[str],
              batch_size: int = 1,
              num_imgs_per_prompt: int | List[int] = 1) -> List[torch.Tensor]:
    # Args guarding
    if isinstance(prompt, str) and isinstance(num_imgs_per_prompt, list):
      raise Exception('A prompt cannot have multiple amounts of imgs')
    if isinstance(num_imgs_per_prompt, list) and len(num_imgs_per_prompt) != len(prompt):
      raise Exception('Need to provide num_imgs for all prompts')
    if isinstance(prompt, list) and len(prompt) == 0:
      raise Exception('At least one prompt must be provided')
    if isinstance(num_imgs_per_prompt, list):
      get_imgs_per_prompt = lambda i: num_imgs_per_prompt[i]
    else:
      get_imgs_per_prompt = lambda _: num_imgs_per_prompt

    # Process single prompt
    if isinstance(prompt, str):
      # Batch as many operations
      num_iters = num_imgs_per_prompt // batch_size
      num_last  = num_imgs_per_prompt %  batch_size

      # Use maximum batch-size for first num_iters steps
      images = []
      for _ in range(num_iters):
        iter_images = self.pipeline(prompt,
                                    output_type='npy',
                                    num_inference_steps=self.inference_steps,
                                    num_images_per_prompt=batch_size).images
        iter_images = torch.tensor(iter_images)
        images.append(iter_images)

      # Process the remainder
      if num_last != 0:
        iter_images = self.pipeline(prompt,
                                    output_type='npy',
                                    num_inference_steps=self.inference_steps,
                                    num_images_per_prompt=num_last).images
        iter_images = torch.tensor(iter_images)
        images.append(iter_images)

      # Stack tensors
      return [torch.vstack(images)]

    # Generate num_images per-prompt save them and repeat
    images = []
    tqdm_prompt = tqdm(prompt, disable=not self.verbose)
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


class DiffusersOutput(object):
  def __init__(self,
               model_outputs: Dict[str, torch.Tensor | List[torch.Tensor]],
               prompts: List[str],
               prompt_modifier: Callable[[str, int, str], str] = None) -> None:
    super().__init__()

    # Args guarding
    if prompt_modifier is None:
      self.prompt_modifier = lambda prompt, prompt_index, model_index, model_name: prompt
    else:
      self.prompt_modifier = prompt_modifier

    # Save the given components
    self.model_dict = model_outputs
    self.prompts = prompts
    self.model_names = list(model_outputs.keys())

    # Get the size of the output
    first_element = list(model_outputs.values())[0]
    self.num_prompts = 1 if isinstance(first_element, torch.Tensor) else len(first_element)

    # Group model outputs by using the prompts
    values = []
    for model_name in self.model_names:
      values.append(itertools.product([model_name], self.model_dict[model_name]))
    self.prompt_outputs = list(zip(self.prompts, zip(*values)))

  def __getitem__(self, prompt_index: int | slice) -> torch.Tensor:
    """Stack all generated images for various prompts."""
    if isinstance(prompt_index, int):
      prompt_index = slice(prompt_index, prompt_index + 1)

    # Find all images from all models for each prompt
    imgs = []
    for _, prompt_imgs in self.prompt_outputs[prompt_index]:
      single_prompt_imgs = []

      # Fetch all model imgs for a prompt
      for _, model_imgs in prompt_imgs:
        single_prompt_imgs.append(model_imgs)

      # Group and save
      single_prompt_imgs = torch.vstack(single_prompt_imgs).unsqueeze(0)
      imgs.append(single_prompt_imgs)

    # Stack on the prompt axis
    return torch.vstack(imgs)

  def __iter__(self) -> 'DiffusersOutput':
    self.prompt_index = 0
    return self

  def __next__(self) -> Tuple[str, List[Tuple[str, torch.Tensor]]]:
    if self.prompt_index < self.num_prompts:
      prompt_output = self.prompt_outputs[self.prompt_index]
      self.prompt_index += 1
      return prompt_output
    else:
      raise StopIteration

  def save(self, root_dir: pb.Path, entry_dir: Optional[str] = None) -> None:
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

    # TODO: reuse generated images by saving them
    raise NotImplemented('Need to work on this')

  def show(self) -> None:
    # Find maximum number of cols & rows to be plotted
    max_cols, max_rows = 0, len(list(self))
    for i, (prompt, model_prompt) in enumerate(self):
      n_imgs_row = 0
      for j, (model_name, values) in enumerate(model_prompt):
        n_imgs_row += values.shape[0]
      if max_cols < n_imgs_row:
        max_cols = n_imgs_row

    # Define plot
    f, ax = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(15, 15))

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
    for i, (prompt, model_prompt) in enumerate(self):
      skip_cols = 0
      for j, (model_name, values) in enumerate(model_prompt):
        for offset, image in enumerate(values):
          current_prompt = self.prompt_modifier(prompt, i, j, model_name)
          show_image(i, skip_cols, offset, model_name, image, current_prompt, max_rows, max_cols, ax)

        # Add more jumpts
        skip_cols += values.shape[0]
    f.tight_layout()
    plt.show()


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
              prompt: str | List[str],
              batch_size: int | List[int] = 1,
              num_imgs_per_prompt: int | List[int] = 1,
              prompt_modifier: Callable[[str, int, str], str] = None) -> DiffusersOutput:
    # Args checking
    if isinstance(batch_size, int):
      get_batch_size = lambda _: batch_size
    else:
      get_batch_size = lambda i: batch_size[i]
    if prompt_modifier is None:
      prompt_modifier = lambda prompt, prompt_index, model_index, model_name: prompt
    if isinstance(prompt, str):
      prompt = [prompt]

    # Generate all images sequentially for each model
    imgs = {}
    tqdm_repos = tqdm(self.repos, disable=not self.verbose)
    for i, repo in enumerate(tqdm_repos):
      # Log currently used model
      tqdm_repos.set_description(f'DiffusionModel: {repo}')

      # Create model and move to device
      model = DiffusionModel(repo, self.get_inference_steps(i)).to(self.device)

      # Modify the prompts for certain models
      model_prompts = [prompt_modifier(p, j, i, repo) for j, p in enumerate(prompt)]

      # Generate all prompts for the current model
      model_imgs = model.forward(model_prompts,
                                 batch_size=get_batch_size(i),
                                 num_imgs_per_prompt=num_imgs_per_prompt)

      # Free up memory
      del model

      # Save the results
      imgs[repo] = model_imgs

    # Aggregate all results
    return DiffusersOutput(imgs, prompt, prompt_modifier)


def prompt_changer(p: str, p_i: int, m_i: int, m: str) -> str:
  if m_i == 0:
    return p
  elif m_i == 1:
    return p + ', ghibli style'
  else:
    return p

