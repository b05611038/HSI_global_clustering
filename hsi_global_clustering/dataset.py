import os
import glob
import json

import h5py
import numpy as np
import scipy.io

import torch

from typing import Optional, Dict, Tuple
from torch.utils.data import Dataset

from PIL import Image, ImageDraw


__all__ = ['JSONMATDataset']


class JSONMATDataset(Dataset):
    def __init__(
        self,
        mat_dir: str,
        data_key: str = 'cube',
        label_key: Optional[str] = 'label',
        json_dir: Optional[str] = None,
        label_dir: Optional[str] = None,
        class_to_index: Optional[Dict[str, int]] = None,
        transform: Optional[callable] = None,
        normalize: Optional[callable] = None,
        to_tensor: bool = True,
        full_loading: bool = False,
    ):
        """
        Dataset for loading HSI cubes from .mat files, with optional label reading.

        Args:
            mat_dir: directory containing .mat files.
            data_key: key in .mat dict for cube data.
            label_key: key in annotation JSON for label, if using json_dir.
            json_dir: directory with JSON annotations to rasterize onto cube.
            label_dir: directory with .pth label files matching mat filenames.
            class_to_index: mapping from class names to indices for JSON labels.
            transform: augmentation on (C, H, W)
            normalize: normalization on (C, H, W)
            to_tensor: whether to convert output to torch.Tensor
            full_loading: if True, preload all samples into memory.
        """
        self.mat_dir = mat_dir
        self.data_key = data_key
        self.label_key = label_key
        self.json_dir = json_dir
        self.label_dir = label_dir
        self.class_to_index = class_to_index
        self.transform = transform
        self.normalize = normalize
        self.to_tensor = to_tensor
        self.full_loading = full_loading

        # collect .mat files
        self.files = sorted(glob.glob(os.path.join(mat_dir, '*.mat')))
        if not self.files:
            raise ValueError(f"No .mat files found in '{mat_dir}'")

        # detect HDF5-based .mat
        try:
            with h5py.File(self.files[0], 'r'):
                self._use_h5 = True
        except Exception:
            self._use_h5 = False

        # only one annotation source allowed
        if self.json_dir and self.label_dir:
            raise ValueError("Provide only one of 'json_dir' or 'label_dir', not both.")
        if self.json_dir and not os.path.isdir(self.json_dir):
            raise ValueError(f"json_dir '{self.json_dir}' is not a valid directory")
        if self.label_dir and not os.path.isdir(self.label_dir):
            raise ValueError(f"label_dir '{self.label_dir}' is not a valid directory")

        # preload if requested
        self._data_list = None
        if self.full_loading:
            self._data_list = self.load_all()

    def __len__(self):
        if self._data_list is not None:
            return len(self._data_list)
        return len(self.files)

    def __getitem__(self, idx):
        # return from memory if preloaded
        if self._data_list is not None:
            return self._data_list[idx]

        path = self.files[idx]
        # 1) Load cube
        if self._use_h5:
            with h5py.File(path, 'r') as dd:
                arr = dd[self.data_key][()]
                mat_label = dd.get(self.label_key, None)
        else:
            dd = scipy.io.loadmat(path)
            arr = dd[self.data_key]
            mat_label = dd.get(self.label_key, None)

        # 2) Reshape to (C, H, W)
        if arr.ndim == 3 and arr.shape[2] < arr.shape[0]:
            arr = arr.transpose(2, 0, 1)
        elif arr.ndim != 3:
            raise RuntimeError(f'Expected 3D cube, got {arr.shape}')

        cube = torch.from_numpy(arr.astype('float32'))

        # 3) Build mask from JSON or PTH if provided
        label = None
        # case 1: load from .pth files
        if self.label_dir:
            base = os.path.splitext(os.path.basename(path))[0]
            label_path = os.path.join(self.label_dir, base + '.pth')
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found: '{label_path}'")
            label = torch.load(label_path)
            if not isinstance(label, torch.Tensor):
                raise TypeError(f"Loaded label must be a Tensor, got {type(label)}")
            if label.ndim != 2:
                raise ValueError(f"Label tensor must be 2D (H,W), got {label.ndim}D")
            # spatial dims must match
            if label.shape[0] != cube.shape[1] or label.shape[1] != cube.shape[2]:
                raise ValueError(
                    f"Label shape {label.shape} does not match cube spatial dims "
                    f"{(cube.shape[1], cube.shape[2])}"
                )

        # case 2: rasterize JSON polygons
        elif self.json_dir:
            base = os.path.splitext(os.path.basename(path))[0]
            json_path = os.path.join(self.json_dir, base + '.json')
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: '{json_path}'")
            with open(json_path, 'r') as f:
                ann = json.load(f)
            # create a blank label image
            H, W = cube.shape[1], cube.shape[2]
            img = Image.new('L', (W, H), 0)
            draw = ImageDraw.Draw(img)
            for obj in ann.get('shapes', []):
                cls = obj.get('label')
                cls_idx = self.class_to_index.get(cls, 0) if self.class_to_index else 0
                pts = [tuple(pt) for pt in obj.get('points', [])]
                draw.polygon(pts, fill=cls_idx)
            label = torch.from_numpy(np.array(img)).long()

        # 4) Fallback to MAT label
        if label is None and mat_label is not None:
            if mat_label.ndim == 2:
                label = mat_label.astype(np.int64)
            else:
                raise RuntimeError(f'Invalid mat label shape {mat_label.shape}')

        # 5) Convert to tensor and apply transforms
        if self.normalize:
            cube = self.normalize(cube)
        if self.transform:
            cube = self.transform(cube)
        if self.to_tensor and not torch.is_tensor(cube):
            cube = torch.from_numpy(cube)

        if label is not None:
            return cube, label

        return cube

    def load_all(self):
        """Eagerly load all data (warning: high RAM)."""
        return [self[i] for i in range(len(self.files))]


