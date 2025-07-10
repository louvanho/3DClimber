import torch


def heatmap_to_image(
        coords: torch.Tensor, proc_side: int, stride: int, centered_stride: bool):
    # stride = FLAGS.stride_train if is_training else FLAGS.stride_test

    last_image_pixel = proc_side - 1
    last_receptive_center = last_image_pixel - (last_image_pixel % stride)
    coords_out = coords * last_receptive_center

    if centered_stride:
        coords_out = coords_out + stride // 2

    return coords_out


def heatmap_to_25d(
        coords: torch.Tensor, proc_side: int, stride: int, centered_stride: bool,
        box_size_m: float):
    coords2d = heatmap_to_image(coords[..., :2], proc_side, stride, centered_stride)
    return torch.cat([coords2d, coords[..., 2:] * box_size_m], dim=-1)


def heatmap_to_metric(
        coords: torch.Tensor, proc_side: int, stride: int, centered_stride: bool,
        box_size_m: float):
    coords2d = heatmap_to_image(
        coords[..., :2], proc_side, stride, centered_stride) * box_size_m / proc_side
    return torch.cat([coords2d, coords[..., 2:] * box_size_m], dim=-1)
