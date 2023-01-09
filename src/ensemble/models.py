import torch
import torch.nn as nn


class Model(nn.Module):
  def __init__(self, verbose: bool = True) -> None:
    super().__init__()

    # Internals
    self.verbose = verbose
    self.dummy = nn.Parameter(torch.empty(0))

  @property
  def device(self) -> torch.device:
    return self.dummy.device

