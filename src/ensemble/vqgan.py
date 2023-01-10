import typing
from typing import Tuple, Any, List, Dict, Optional
import torch
import pathlib as pb
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig, ListConfig

from taming.models.vqgan import GumbelVQ, VQModel

from .blocks import Model


class VQGAN(Model):
  def __init__(self,
               config_path: pb.Path,
               ckpt_path: Optional[pb.Path] = None,
               verbose: bool = True) -> None:
    super().__init__(verbose)

    self.config = self.__load_config(config_path, self.verbose)
    self.model = self.__load_vqgan(self.config, ckpt_path, False)

  def quantize(self, z, temp=None, rescale_logits=False, return_logits=False):
    return self.model.quantize(z, temp, rescale_logits, rescale_logits)

  def encode_to_preconv(self, x: torch.Tensor) -> torch.Tensor:
    return self.model.encode_to_preconv(x)

  def encode_to_prequant(self, x: torch.Tensor) -> torch.Tensor:
    return self.model.encode_to_prequant(x)

  def encode(self, x):
    return self.model.encode(x)

  def decode_from_preconv(self, x: torch.Tensor) -> torch.Tensor:
    return self.model.decode_from_preconv(x)

  def decode_from_prequant(self, x: torch.Tensor) -> torch.Tensor:
    return self.model.decode_from_prequant(x)

  def decode_code(self, code_b):
    return self.model.decode_code(code_b)

  def decode(self, quant):
    return self.model.decode(quant)

  def forward(self, input):
    return self.model.forward(input)

  def __load_config(self, path_to_file: pb.Path, verbose: bool = True) -> (DictConfig | ListConfig):
    if not path_to_file.exists():
      raise Exception(f'{path_to_file} does not exist')
    if not path_to_file.is_file():
      raise Exception(f'{path_to_file} is not a file')

    # Load the yaml file from disk
    config = OmegaConf.load(path_to_file)

    if verbose:
      print(OmegaConf.to_yaml(config))

    return config

  def __load_vqgan(self, config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
      model = GumbelVQ(**config.model.params)
    else:
      model = VQModel(**config.model.params)
    if ckpt_path is not None:
      sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
      missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

