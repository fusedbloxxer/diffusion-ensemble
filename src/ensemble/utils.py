import typing
from typing import Any
import os
import zipfile
import urllib.request
import pathlib as pb
import shutil
import torch
import yaml
import omegaconf
from omegaconf import OmegaConf, DictConfig, ListConfig
from taming.models.vqgan import VQModel, GumbelVQ


def unzip_at_dir(zip_path: pb.Path, out_dir: pb.Path, exist_ok: bool = True) -> None:
  if not zip_path.exists():
    raise Exception(f'No zip file exists at: {zip_path}')

  if out_dir.exists() and len(os.listdir(out_dir)) != 0 and not exist_ok:
    raise Exception(f'Directory already filled: {out_dir}')

  if out_dir.exists() and len(os.listdir(out_dir)) != 0:
    return

  with zipfile.ZipFile(zip_path) as zip_file:
    if (bad_file := zip_file.testzip()) is not None:
      raise Exception(f'Bad zip file at: {zip_path / bad_file}')

    if not out_dir.exists():
      out_dir.mkdir(parents=True, exist_ok=True)

    zip_file.extractall(out_dir)


def download_file(url: str, file_path: pb.Path, exist_ok: bool = True) -> None:
  # Skip redownloading a file if it already exists
  if file_path.exists() and exist_ok:
    return
  elif not exist_ok:
    raise Exception(f'File already exists at: {file_path}')

  with urllib.request.urlopen(url) as file_response:
    # Create parent folder if not already present
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'wb') as file:
      # Save the GET response
      shutil.copyfileobj(file_response, file)

