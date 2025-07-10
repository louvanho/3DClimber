from typing import Optional

import torch
import torchvision
import torchvision.ops
import torchvision.transforms.functional


class PersonDetector(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.input_size = 640
        self.model = torch.jit.load(model_path).half()
        self.person_class_id = '0'

    def forward(
            self, images: torch.Tensor, threshold: float = 0.2, nms_iou_threshold: float = 0.7,
            max_detections: int = 150, extrinsic_matrix: Optional[torch.Tensor] = None,
            world_up_vector: Optional[torch.Tensor] = None, flip_aug: bool = False,
            bothflip_aug: bool = False):
        device = images.device
        if extrinsic_matrix is None:
            extrinsic_matrix = torch.eye(4, device=device).unsqueeze(0)
        if len(extrinsic_matrix) == 1:
            extrinsic_matrix = torch.repeat_interleave(extrinsic_matrix, len(images), dim=0)
        if world_up_vector is None:
            world_up_vector = torch.tensor([0, -1, 0], device=device, dtype=torch.float32)

        images, x_factor, y_factor, half_pad_h_float, half_pad_w_float = resize_and_pad(
            images, self.input_size)

        cam_up_vector = matvec(extrinsic_matrix[:, :3, :3], world_up_vector)
        angle = torch.atan2(cam_up_vector[:, 1], cam_up_vector[:, 0])
        k = (torch.round(angle / (torch.pi / 2)) + 1).to(torch.int32) % 4
        images = batched_rot90(images, k)

        if bothflip_aug:
            boxes, scores = self.call_model_bothflip_aug(images)
        elif flip_aug:
            boxes, scores = self.call_model_flip_aug(images)
        else:
            boxes, scores = self.call_model(images)

        # Convert from cxcywh to xyxy (top-left-bottom-right)
        boxes = torch.stack([
            boxes[..., 0] - boxes[..., 2] / 2,
            boxes[..., 1] - boxes[..., 3] / 2,
            boxes[..., 0] + boxes[..., 2] / 2,
            boxes[..., 1] + boxes[..., 3] / 2], dim=-1)

        # Filter scores by person class id
        if self.person_class_id == '0':
            scores = scores[..., 0].to(device)
        else:
            class_ids = torch.tensor(
                [int(x) for x in self.person_class_id.split(',')], device=device)
            scores = scores[:, class_ids].sum(dim=-1)

        boxes, scores = nms(boxes, scores, threshold, nms_iou_threshold, max_detections)

        return [
            scale_boxes(
                boxes_, scores_, half_pad_w_float, half_pad_h_float, x_factor, y_factor, k_,
                self.input_size)
            for boxes_, scores_, k_ in zip(boxes, scores, k)]


    def call_model_flip_aug(self, images):
        device = images.device
        flipped = torch.flip(images, dims=[3])  # Horizontal flip
        net_input = torch.cat([images, flipped], dim=0)
        boxes, scores = self.call_model(net_input)
        padded_width = images.shape[3]
        boxes_normal, boxes_flipped = torch.chunk(boxes, 2, dim=0)
        # Horizontal backflip
        boxes_backflipped = torch.cat(
            [padded_width - boxes_flipped[..., :1], boxes_flipped[..., 1:]],
            dim=-1).to(device)
        boxes = torch.cat([boxes_normal, boxes_backflipped], dim=1)
        scores = torch.cat(torch.chunk(scores, 2, dim=0), dim=1)
        return boxes, scores

    def call_model_bothflip_aug(self, images):
        device=images.device
        flipped_horiz = torch.flip(images, dims=[3])  # Horizontal flip
        flipped_vert = torch.flip(images, dims=[2])  # Vertical flip
        net_input = torch.cat([images, flipped_horiz, flipped_vert], dim=0)
        boxes, scores = self.call_model(net_input)
        padded_width = images.shape[3]
        padded_height = images.shape[2]
        boxes_normal, boxes_flipped_horiz, boxes_flipped_vert = torch.chunk(boxes, 3, dim=0)
        # Horizontal backflip
        boxes_backflipped_horiz = torch.cat(
            [padded_width - boxes_flipped_horiz[..., :1], boxes_flipped_horiz[..., 1:]],
            dim=-1).to(device)
        # Vertical backflip
        boxes_backflipped_vert = torch.cat(
            [boxes_flipped_vert[..., :1], padded_height - boxes_flipped_vert[..., 1:2],
             boxes_flipped_vert[..., 2:]], dim=-1).to(device)
        boxes = torch.cat(
            [boxes_normal, boxes_backflipped_horiz, boxes_backflipped_vert],
            dim=1).to(device)
        scores = torch.cat(torch.chunk(scores, 3, dim=0), dim=1)
        return boxes, scores

    def call_model(self, images):
        preds = self.model(images.half())
        preds = torch.permute(preds, [0, 2, 1])  # [batch, n_boxes, 84]
        boxes = preds[..., :4]
        scores = preds[..., 4:]
        return boxes, scores


def nms(
        boxes: torch.Tensor, scores: torch.Tensor, threshold: float, nms_iou_threshold: float,
        max_detections: int):
    selected_boxes = []
    selected_scores = []
    for boxes_now, scores_now in zip(boxes, scores):
        is_above_threshold = scores_now > threshold
        boxes_now = boxes_now[is_above_threshold]
        scores_now = scores_now[is_above_threshold]
        nms_indices = torchvision.ops.nms(
            boxes_now, scores_now, nms_iou_threshold)[:max_detections]
        selected_boxes.append(boxes_now[nms_indices])
        selected_scores.append(scores_now[nms_indices])
    boxes = selected_boxes
    scores = selected_scores
    return boxes, scores


def batched_rot90(images, k):
    batch_size = images.size(0)
    rotated_images = torch.empty_like(images)
    for i in range(batch_size):
        rotated_images[i] = torch.rot90(images[i], k=k[i], dims=[1, 2])
    return rotated_images


def matvec(a, b):
    return (a @ b.unsqueeze(-1)).squeeze(-1)


def resize_and_pad(images: torch.Tensor, input_size: int):
    h = float(images.shape[2])
    w = float(images.shape[3])
    max_side = max(h, w)
    factor = float(input_size) / max_side
    target_w = int(factor * w)
    target_h = int(factor * h)
    y_factor = h / float(target_h)
    x_factor = w / float(target_w)
    pad_h = input_size - target_h
    pad_w = input_size - target_w
    half_pad_h = pad_h // 2
    half_pad_w = pad_w // 2
    half_pad_h_float = float(half_pad_h)
    half_pad_w_float = float(half_pad_w)
    images = (images.float() / 255) ** 2.2
    images = torchvision.transforms.functional.resize(
        images, (target_h, target_w), antialias=factor < 1)
    images = images ** (1 / 2.2)
    images = torch.nn.functional.pad(
        images, (half_pad_w, pad_w - half_pad_w, half_pad_h, pad_h - half_pad_h),
        value=0.5)
    return images, x_factor, y_factor, half_pad_h_float, half_pad_w_float


def scale_boxes(
        boxes: torch.Tensor, scores: torch.Tensor, half_pad_w_float: float, half_pad_h_float: float,
        x_factor: float, y_factor: float, k: torch.Tensor, input_size: int):
    midpoints = (boxes[:, :2] + boxes[:, 2:]) / 2
    midpoints = matvec(
        rotmat2d(k.to(torch.float32) * (torch.pi / 2)),
        midpoints - (input_size - 1) / 2) + (input_size - 1) / 2

    sizes = boxes[:, 2:] - boxes[:, :2]
    if k % 2 == 1:
        sizes = sizes[:, ::-1]

    boxes_ = torch.cat([midpoints - sizes / 2, sizes], dim=1)

    return torch.stack([
        (boxes_[:, 0] - half_pad_w_float) * x_factor,
        (boxes_[:, 1] - half_pad_h_float) * y_factor,
        (boxes_[:, 2]) * x_factor,
        (boxes_[:, 3]) * y_factor,
        scores
    ], dim=1)


def rotmat2d(angle: torch.Tensor):
    sin = torch.sin(angle)
    cos = torch.cos(angle)
    entries = [
        cos, -sin,
        sin, cos]
    result = torch.stack(entries, dim=-1)
    return torch.reshape(result, angle.shape + (2, 2))
