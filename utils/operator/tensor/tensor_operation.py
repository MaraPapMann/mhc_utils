'''
@Desc: Tensor operations.
'''
import numpy as np
import torch
from typing import Any
from torch import Tensor
from matplotlib import pyplot as plt


def cat_tensors(tensor_a:Any, tensor_b:Tensor, dim:int=0) -> Tensor:
    if tensor_a is None:
        return tensor_b
    else:
        assert len(tensor_a.shape) == len(tensor_b.shape), 'The shape of Tensor A and B must be the same!'
        return torch.cat((tensor_a, tensor_b), dim)


def plot_image_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    """
    Plot a grid of images from a PyTorch tensor.

    Args:
        tensor (torch.Tensor): Tensor containing the images. Shape should be (N, C, H, W) where N is the number of images,
                               C is the number of channels (1 for grayscale, 3 for RGB), H is the height, and W is the width.
        nrow (int): Number of images in each row of the grid.
        padding (int): Padding between images in the grid.
        normalize (bool): If True, normalize the image to the range [0, 1].
        range (tuple): Range (min, max) to normalize the image to. If None, use the min and max of the tensor.
        scale_each (bool): If True, scale each image individually.
        pad_value (float): Value to use for padding.

    Returns:
        None
    """
    # Ensure the tensor is 4D (N, C, H, W)
    print(tensor.shape)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)  # Add a channel dimension for grayscale images
    elif tensor.dim() != 4:
        raise ValueError("Tensor must be 4D (N, C, H, W) or 3D (N, H, W) for grayscale images.")

    # Normalize the tensor if required
    if normalize:
        tensor = tensor.clone()  # Avoid modifying the original tensor
        if range is not None:
            min_val, max_val = range
        else:
            min_val, max_val = tensor.min(), tensor.max()
        if scale_each:
            for t in tensor:  # Scale each image individually
                t.sub_(t.min()).div_(t.max() - t.min()).mul_(max_val - min_val).add_(min_val)
        else:
            tensor.sub_(min_val).div_(max_val - min_val)

    # Convert the tensor to a numpy array and change the order of dimensions to (N, H, W, C)
    tensor = tensor.cpu().permute(0, 2, 3, 1).numpy()

    # Calculate the number of rows and columns in the grid
    ncol = int(np.ceil(len(tensor) / nrow))

    # Create the grid
    grid = np.full((ncol * (tensor.shape[1] + padding) - padding,
                    nrow * (tensor.shape[2] + padding) - padding,
                    tensor.shape[3]), pad_value, dtype=tensor.dtype)

    for i, img in enumerate(tensor):
        row = i // nrow
        col = i % nrow
        grid[row * (img.shape[0] + padding): row * (img.shape[0] + padding) + img.shape[0],
             col * (img.shape[1] + padding): col * (img.shape[1] + padding) + img.shape[1]] = img

    # Plot the grid
    print(tensor.shape)
    plt.imshow(grid, cmap='gray' if tensor.shape[3] == 1 else None)
    plt.axis('off')
    plt.show()