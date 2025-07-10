import torch

import safetensors
from safetensors import safe_open
from safetensors.torch import save_file

__all__ = ['save_as_safetensors', 'load_safetensors']

def save_as_safetensors(tensors, filename):
    assert isinstance(tensors, dict)
    assert isinstance(filename, str)

    if not filename.endswith('.safetensors'):
        filename += '.safetensors'

    save_file(tensors, filename)
    return None

def load_safetensors(filename, device = 'cpu', extension_check = True):
    assert isinstance(filename, str)
    assert isinstance(device, (str, torch.device))
    assert isinstance(extension_check, bool)

    if extension_check:
        if not filename.endswith('.safetensors'):
            raise RuntimeError('File: {0} is not a .json file.'.format(filename))

    tensors = {}
    with safe_open(filename, framework = 'pt', device = device) as in_files:
        for key in in_files.keys():
            tensors[key] = in_files.get_tensor(key)

    return tensors
