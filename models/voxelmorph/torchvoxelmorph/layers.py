import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):

        assert src.size(0) == flow.size(0), "Source and flow should have the same batch size"

        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()
        
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

def normalize_coords(deform_field):
    """Normalize displacement field coordinates to [-1, 1] for grid_sample.
    Args:
        deform_field: [B, 2, H, W], displacement field where
                      deform_field[:, 0, :, :] are displacements in x direction,
                      deform_field[:, 1, :, :] are displacements in y direction.
    Returns:
        normalized_field: [B, H, W, 2], normalized field suitable for grid_sample.
    """
    B, _, H, W = deform_field.size()
    # Create a mesh grid of absolute positions
    xx = torch.linspace(0, W - 1, W).type_as(deform_field).repeat(H, 1)
    yy = torch.linspace(0, H - 1, H).type_as(deform_field).repeat(W, 1).t()
    grid = torch.stack((xx, yy), dim=0)  # [2, H, W]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]

    # Add the displacement to the grid positions
    absolute_positions = grid + deform_field

    # Normalize the absolute positions to [-1, 1]
    normalized_field = torch.zeros_like(absolute_positions)
    normalized_field[:, 0, :, :] = 2.0 * absolute_positions[:, 0, :, :] / (W - 1) - 1
    normalized_field[:, 1, :, :] = 2.0 * absolute_positions[:, 1, :, :] / (H - 1) - 1

    # Permute to match grid_sample's expected input format [B, H, W, 2]
    normalized_field = normalized_field.permute(0, 2, 3, 1)

    return normalized_field



def disp_warp(img, deform_field, padding_mode='border'):
    """Warping by displacement field for 2D medical image registration.
    Args:
        img: [B, C, H, W] input image to be warped.
        deform_field: [B, 2, H, W], displacement field.
        padding_mode: 'zeros', 'border', or 'reflection'
    Returns:
        warped_img: [B, C, H, W]
        valid_mask: [B, 1, H, W]
    """
    # Normalize coordinates of displacement field to [-1, 1]
    normalized_deform_field = normalize_coords(deform_field)
    # Perform the warping using the displacement field
    warped_img = F.grid_sample(img, normalized_deform_field, mode='bilinear', padding_mode=padding_mode)

    # Generate a valid mask to indicate where the sampling grid samples inside the original image
    mask = torch.ones_like(img)
    valid_mask = F.grid_sample(mask, normalized_deform_field, mode='bilinear', padding_mode='zeros')
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1

    return warped_img, valid_mask


