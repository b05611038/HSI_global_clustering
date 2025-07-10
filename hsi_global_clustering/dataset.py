import os
import glob
import json

import numpy as np

import torch

from typing import Optional
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
        class_to_index: Optional[dict] = None,
        transform: Optional[callable] = None,
        normalize: Optional[callable] = None,
        to_tensor: bool = True,
    ):
        """
        Args:
            mat_dir: directory of .mat files
            data_key: key in .mat for the hyperspectral cube
            label_key: optional key in .mat for a 2D mask
            json_dir: directory of LabelMe JSONs (same basename as .mat)
            class_to_index: {'disease':1, 'healthy':0, …}
            transform: augmentation on (C,H,W)
            normalize: normalization on (C,H,W)
        """
        self.files = sorted(glob.glob(os.path.join(mat_dir, '*.mat')))
        self.data_key = data_key
        self.label_key = label_key
        self.json_dir = json_dir
        self.class_to_index = class_to_index or {}
        self.transform = transform
        self.normalize = normalize
        self.to_tensor = to_tensor

        # detect HDF5-based .mat
        try:
            import h5py
            with h5py.File(self.files[0], 'r'):
                self._use_h5 = True
        except:
            self._use_h5 = False

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        # 1) Load cube
        if self._use_h5:
            import h5py
            f = h5py.File(path, 'r')
            arr = f[self.data_key][()]
            mat_label = f.get(self.label_key, None)
            if mat_label is not None:
                mat_label = mat_label[()]
            f.close()
        else:
            import scipy.io
            dd = scipy.io.loadmat(path)
            arr = dd[self.data_key]
            mat_label = dd.get(self.label_key, None)

        # 2) Reshape to (C,H,W)
        if arr.ndim == 3 and arr.shape[2] < arr.shape[0]:
            arr = arr.transpose(2,0,1)
        elif arr.ndim != 3:
            raise RuntimeError(f'Expected 3D cube, got {arr.shape}')

        # 3) Build mask from JSON if provided
        mask = None
        if self.json_dir:
            base = os.path.basename(path).replace('.mat','.json')
            jpath = os.path.join(self.json_dir, base)
            if os.path.exists(jpath):
                data = json.load(open(jpath))
                H, W = arr.shape[1], arr.shape[2]
                img = Image.new('I',(W,H),0)
                draw = ImageDraw.Draw(img)
                for shape in data.get('shapes', []):
                    label = shape['label']
                    pts = shape['points']
                    lbl = self.class_to_index.get(label, 0)
                    poly = [(x,y) for x,y in pts]
                    draw.polygon(poly, fill=int(lbl))
                mask = np.array(img, dtype=np.int64)

        # 4) Fallback to MAT label
        if mask is None and mat_label is not None:
            if mat_label.ndim == 2:
                mask = mat_label.astype(np.int64)
            else:
                raise RuntimeError(f'Invalid mat label shape {mat_label.shape}')

        # 5) Convert to tensor and apply transforms
        cube = torch.from_numpy(arr.astype('float32'))
        if self.normalize:
            cube = self.normalize(cube)
        if self.transform:
            cube = self.transform(cube)
        if self.to_tensor and not torch.is_tensor(cube):
            cube = torch.from_numpy(cube)

        if mask is not None:
            mask_t = torch.from_numpy(mask)
            return cube, mask_t
        return cube

    def load_all(self):
        """Eagerly load all data (warning: high RAM)."""
        out = [self[i] for i in range(len(self))]
        return out

