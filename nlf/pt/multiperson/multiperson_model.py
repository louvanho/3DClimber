import math
from typing import Dict, List, Optional

import cameralib
import imageio.v2 as imageio
import numpy as np
import smplfitter.pt as smplfitter
import torch
import torchvision.transforms.functional

import nlf.pt.backbones.efficientnet as effnet_pytorch
import nlf.pt.models.field as pt_field
import nlf.pt.models.nlf_model as pt_nlf_model
from nlf.paths import DATA_ROOT, PROJDIR
from nlf.pt import ptu, ptu3d
from nlf.pt.multiperson import person_detector, plausibility_check as plausib, warping
from nlf.pt.util import get_config
import simplepyutils as spu
import argparse


FLAGS = argparse.Namespace()

# Dummy value which will mean that the intrinsic_matrix are unknown
UNKNOWN_INTRINSIC_MATRIX = ((-1, -1, -1), (-1, -1, -1), (-1, -1, -1))
DEFAULT_EXTRINSIC_MATRIX = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
DEFAULT_DISTORTION = (0, 0, 0, 0, 0)
DEFAULT_WORLD_UP = (0, -1, 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-model-path', type=str)
    parser.add_argument('--output-model-path', type=str)
    parser.add_argument('--config-name', type=str, default='convert_from_tf_s')
    parser.parse_args(namespace=FLAGS)

    skeleton_infos = spu.load_pickle(f"{DATA_ROOT}/skeleton_conversion/skeleton_types_huge8.pkl")
    cano_joints = np.load(f'{PROJDIR}/canonical_joints/smpl.npy').astype(np.float32)
    image = imageio.imread(
        '/work/sarandi/data/example_256px.jpg')[np.newaxis].astype(np.float32) / 255
    intrinsics = cameralib.intrinsics_from_fov(
        30, [256, 256])[np.newaxis].astype(np.float32)
    cfg = get_config(FLAGS.config_name)

    with torch.device('cuda'):
        backbone_raw = getattr(effnet_pytorch, f'efficientnet_v2_{cfg.efficientnet_size}')()
        preproc_layer = effnet_pytorch.PreprocLayer()
        backbone = torch.nn.Sequential(preproc_layer, backbone_raw.features)
        weight_field = pt_field.build_field()
        model_pytorch = pt_nlf_model.NLFModel(backbone, weight_field)
        model_pytorch.load_state_dict(torch.load(FLAGS.input_model_path, weights_only=True))

    model_pytorch = model_pytorch.cuda().eval()
    model_pytorch.eval()
    with torch.inference_mode(), torch.device('cuda'), torch.amp.autocast('cuda'):
        result_pt = model_pytorch.predict_multi_same_canonicals(
            torch.from_numpy(image).permute(0, 3, 1, 2).half().cuda(),
            torch.from_numpy(intrinsics).cuda(),
            torch.from_numpy(cano_joints).cuda())

    model_pytorch.backbone.half()
    model_pytorch.heatmap_head.layer.half()
    detector = person_detector.PersonDetector(f'{DATA_ROOT}/yolov8x.torchscript')
    multimodel = MultipersonNLF(model_pytorch, detector, skeleton_infos).cuda().eval()
    multimodel = torch.jit.script(multimodel)
    torch.jit.save(multimodel, FLAGS.output_model_path)


class MultipersonNLF(torch.nn.Module):
    def __init__(self, crop_model, detector, skeleton_infos):
        super().__init__()

        self.crop_model = crop_model
        self.detector = detector

        body_models = ['smpl', 'smplx']
        vertex_subset = {
            k: np.load(f'{DATA_ROOT}/body_models/{k}/vertex_subset_1024.npz')['i_verts'] for k in
            body_models}
        cano_verts = {
            k: np.load(f'{PROJDIR}/canonical_verts/{k}.npy')[vertex_subset[k]] for k in body_models}
        cano_joints = {
            k: np.load(f'{PROJDIR}/canonical_joints/{k}.npy') for k in body_models}
        cano_all = {
            k: np.concatenate([cano_verts[k], cano_joints[k]], axis=0) for k in body_models}
        device = next(self.crop_model.parameters()).device

        self.cano_all = {
            k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in cano_all.items()}
        nothing = torch.zeros([], device=device, dtype=torch.float32)
        self.weights = {
            k: dict(
                w_tensor=nothing, b_tensor=nothing, w_tensor_flipped=nothing,
                b_tensor_flipped=nothing)
            for k in body_models}

        num_betas = 10
        self.fitter_smpl = smplfitter.SMPLFitter(
            smplfitter.SMPLBodyModel('smpl', num_betas=num_betas),
            vertex_subset=vertex_subset['smpl'], num_betas=num_betas)
        self.fitter_smplx = smplfitter.SMPLFitter(
            smplfitter.SMPLBodyModel('smplx', num_betas=num_betas),
            vertex_subset=vertex_subset['smplx'], num_betas=num_betas)

        self.per_skeleton_indices = {
            k: torch.tensor(v['indices'], dtype=torch.int32)
            for k, v in skeleton_infos.items()}
        self.per_skeleton_joint_names = {
            k: v['names'] for k, v in skeleton_infos.items()}
        self.per_skeleton_joint_edges = {
            k: torch.tensor(v['edges'], dtype=torch.int32)
            for k, v in skeleton_infos.items()}

        self.skeleton_joint_indices_table = {k: v['indices'] for k, v in skeleton_infos.items()}

    @torch.jit.export
    def detect_smpl_batched(
            self,
            images: torch.Tensor,
            intrinsic_matrix: Optional[torch.Tensor] = None,
            distortion_coeffs: Optional[torch.Tensor] = None,
            extrinsic_matrix: Optional[torch.Tensor] = None,
            world_up_vector: Optional[torch.Tensor] = None,
            default_fov_degrees: float = 55.0,
            internal_batch_size: int = 64,
            antialias_factor: int = 1,
            num_aug: int = 1, rot_aug_max_degrees: float = 25.0,
            detector_threshold: float = 0.3,
            detector_nms_iou_threshold: float = 0.7, max_detections: int = 150,
            suppress_implausible_poses: bool = True,
            beta_regularizer: float = 10.0,
            beta_regularizer2: float = 0.0,
            model_name: str = 'smpl'):
        boxes = self.detector(
            images=images, threshold=detector_threshold,
            nms_iou_threshold=detector_nms_iou_threshold, max_detections=max_detections,
            extrinsic_matrix=extrinsic_matrix, world_up_vector=world_up_vector)
        return self._estimate_smpl_batched(
            images, boxes, intrinsic_matrix, distortion_coeffs, extrinsic_matrix,
            world_up_vector, default_fov_degrees, internal_batch_size, antialias_factor, num_aug,
            rot_aug_max_degrees,
            suppress_implausible_poses, beta_regularizer, beta_regularizer2, model_name)

    @torch.jit.export
    def estimate_smpl_batched(
            self,
            images: torch.Tensor,
            boxes: List[torch.Tensor],
            intrinsic_matrix: Optional[torch.Tensor] = None,
            distortion_coeffs: Optional[torch.Tensor] = None,
            extrinsic_matrix: Optional[torch.Tensor] = None,
            world_up_vector: Optional[torch.Tensor] = None,
            default_fov_degrees: float = 55.0,
            internal_batch_size: int = 64,
            antialias_factor: int = 1,
            num_aug: int = 1,
            rot_aug_max_degrees: float = 25.0,
            beta_regularizer: float = 10.0,
            beta_regularizer2: float = 0.0,
            model_name: str = 'smpl'):
        return self._estimate_smpl_batched(
            images, boxes, intrinsic_matrix, distortion_coeffs, extrinsic_matrix,
            world_up_vector, default_fov_degrees, internal_batch_size, antialias_factor, num_aug,
            rot_aug_max_degrees,
            suppress_implausible_poses=False, beta_regularizer=beta_regularizer,
            beta_regularizer2=beta_regularizer2, model_name=model_name)

    def _estimate_smpl_batched(
            self,
            images: torch.Tensor,
            boxes: List[torch.Tensor],
            intrinsic_matrix: Optional[torch.Tensor] = None,
            distortion_coeffs: Optional[torch.Tensor] = None,
            extrinsic_matrix: Optional[torch.Tensor] = None,
            world_up_vector: Optional[torch.Tensor] = None,
            default_fov_degrees: float = 55.0,
            internal_batch_size: int = 64,
            antialias_factor: int = 1,
            num_aug: int = 1,
            rot_aug_max_degrees: float = 25.0,
            suppress_implausible_poses: bool = True,
            beta_regularizer: float = 10.0,
            beta_regularizer2: float = 0.0,
            model_name: str = 'smpl'):

        if self.weights[model_name]['w_tensor'].ndim == 0:
            self.weights[model_name] = self.get_weights_for_canonical_points(
                self.cano_all[model_name])

        result = self._estimate_poses_batched(
            images, boxes, self.weights[model_name], intrinsic_matrix, distortion_coeffs,
            extrinsic_matrix,
            world_up_vector, default_fov_degrees, internal_batch_size, antialias_factor, num_aug,
            rot_aug_max_degrees,
            suppress_implausible_poses=suppress_implausible_poses)
        boxes = result['boxes']
        n_pose_per_image_list = [len(b) for b in boxes]
        if sum(n_pose_per_image_list) == 0:
            return self._predict_empty_smpl(images, model_name)

        poses3d_flat = torch.cat(result['poses3d'], dim=0)
        mean_poses = torch.mean(poses3d_flat, dim=-2, keepdim=True)
        poses3d_flat = poses3d_flat - mean_poses
        poses2d_flat = torch.cat(result['poses2d'], dim=0)
        uncertainties_flat = torch.cat(result['uncertainties'], dim=0)

        fitter = self.fitter_smpl if model_name == 'smpl' else self.fitter_smplx
        vertices_flat, joints_flat = torch.split(
            poses3d_flat, [fitter.num_vertices, fitter.body_model.num_joints], dim=-2)
        vertex_uncertainties_flat, joint_uncertainties_flat = torch.split(
            uncertainties_flat, [fitter.num_vertices, fitter.body_model.num_joints],
            dim=-1)

        vertex_weights = vertex_uncertainties_flat ** -1.5
        vertex_weights = vertex_weights / torch.mean(vertex_weights, dim=-1, keepdim=True)
        joint_weights = joint_uncertainties_flat ** -1.5
        joint_weights = joint_weights / torch.mean(joint_weights, dim=-1, keepdim=True)

        fit = fitter.fit(
            vertices_flat / 1000, joints_flat / 1000, vertex_weights=vertex_weights,
            joint_weights=joint_weights, n_iter=3, beta_regularizer=beta_regularizer,
            beta_regularizer2=beta_regularizer2, final_adjust_rots=True,
            requested_keys=['pose_rotvecs', 'shape_betas', 'trans', 'joints', 'vertices'])
        result['pose'] = torch.split(fit['pose_rotvecs'], n_pose_per_image_list)
        result['betas'] = torch.split(fit['shape_betas'], n_pose_per_image_list)
        result['trans'] = torch.split(fit['trans'], n_pose_per_image_list)

        fit_vertices_flat = fit['vertices'] * 1000 + mean_poses
        fit_joints_flat = fit['joints'] * 1000 + mean_poses
        result['vertices3d'] = torch.split(fit_vertices_flat, n_pose_per_image_list)
        result['joints3d'] = torch.split(fit_joints_flat, n_pose_per_image_list)

        result['vertices2d'] = project_ragged_pytorch(
            images, [x for x in result['vertices3d']], extrinsic_matrix, intrinsic_matrix,
            distortion_coeffs,
            default_fov_degrees)
        result['joints2d'] = project_ragged_pytorch(
            images, [x for x in result['joints3d']], extrinsic_matrix, intrinsic_matrix,
            distortion_coeffs,
            default_fov_degrees)

        result['vertices3d_nonparam'] = torch.split(vertices_flat + mean_poses,
                                                    n_pose_per_image_list)
        result['joints3d_nonparam'] = torch.split(joints_flat + mean_poses, n_pose_per_image_list)
        vertices2d, joints2d = torch.split(
            poses2d_flat, [fitter.num_vertices, fitter.body_model.num_joints], dim=-2)

        result['vertices2d_nonparam'] = torch.split(vertices2d, n_pose_per_image_list)
        result['joints2d_nonparam'] = torch.split(joints2d, n_pose_per_image_list)

        result['vertex_uncertainties'] = torch.split(vertex_uncertainties_flat * 1000,
                                                     n_pose_per_image_list)
        result['joint_uncertainties'] = torch.split(joint_uncertainties_flat * 1000,
                                                    n_pose_per_image_list)

        del result['poses3d']
        del result['poses2d']
        del result['uncertainties']
        return result

    @torch.jit.export
    def detect_poses_batched(
            self,
            images: torch.Tensor,
            weights: Dict[str, torch.Tensor],
            intrinsic_matrix: Optional[torch.Tensor] = None,
            distortion_coeffs: Optional[torch.Tensor] = None,
            extrinsic_matrix: Optional[torch.Tensor] = None,
            world_up_vector: Optional[torch.Tensor] = None,
            default_fov_degrees: float = 55.0,
            internal_batch_size: int = 64,
            antialias_factor: int = 1,
            num_aug: int = 1,
            rot_aug_max_degrees: float = 25.0,
            detector_threshold: float = 0.3,
            detector_nms_iou_threshold: float = 0.7, max_detections: int = 150,
            suppress_implausible_poses: bool = True):

        boxes = self.detector(
            images=images, threshold=detector_threshold,
            nms_iou_threshold=detector_nms_iou_threshold, max_detections=max_detections,
            extrinsic_matrix=extrinsic_matrix, world_up_vector=world_up_vector)

        return self._estimate_poses_batched(
            images, boxes, weights, intrinsic_matrix, distortion_coeffs, extrinsic_matrix,
            world_up_vector,
            default_fov_degrees, internal_batch_size, antialias_factor, num_aug,
            rot_aug_max_degrees,
            suppress_implausible_poses)

    @torch.jit.export
    def estimate_poses_batched(
            self,
            images: torch.Tensor,
            boxes: List[torch.Tensor],
            weights: Dict[str, torch.Tensor],
            intrinsic_matrix: Optional[torch.Tensor] = None,
            distortion_coeffs: Optional[torch.Tensor] = None,
            extrinsic_matrix: Optional[torch.Tensor] = None,
            world_up_vector: Optional[torch.Tensor] = None,
            default_fov_degrees: float = 55.0,
            internal_batch_size: int = 64,
            antialias_factor: int = 1,
            num_aug: int = 5, rot_aug_max_degrees: float = 25.0):
        boxes = [torch.cat([b, torch.ones_like(b[..., :1])], dim=-1) for b in boxes]
        pred = self._estimate_poses_batched(
            images, boxes, weights, intrinsic_matrix, distortion_coeffs, extrinsic_matrix,
            world_up_vector,
            default_fov_degrees, internal_batch_size, antialias_factor, num_aug,
            rot_aug_max_degrees,
            suppress_implausible_poses=False)
        del pred['boxes']
        return pred

    def _estimate_poses_batched(
            self, images: torch.Tensor, boxes: List[torch.Tensor], weights: Dict[str, torch.Tensor],
            intrinsic_matrix: Optional[torch.Tensor], distortion_coeffs: Optional[torch.Tensor],
            extrinsic_matrix: Optional[torch.Tensor],
            world_up_vector: Optional[torch.Tensor], default_fov_degrees: float,
            internal_batch_size: int, antialias_factor: int, num_aug: int,
            rot_aug_max_degrees: float,
            suppress_implausible_poses: bool):
        if sum(len(b) for b in boxes) == 0:
            return self._predict_empty(images, weights)

        n_images = len(images)
        device = images.device
        # If one intrinsic matrix is given, repeat it for all images

        if intrinsic_matrix is None:
            intrinsic_matrix = ptu3d.intrinsic_matrix_from_field_of_view(
                default_fov_degrees, images.shape[2:4], device=device)

        if len(intrinsic_matrix) == 1:
            # If intrinsic_matrix is not given, fill it in based on field of view
            intrinsic_matrix = torch.repeat_interleave(intrinsic_matrix, n_images, dim=0)

        if distortion_coeffs is None:
            distortion_coeffs = torch.zeros((n_images, 5), device=device)
        # If one distortion coeff/extrinsic matrix is given, repeat it for all images
        if len(distortion_coeffs) == 1:
            distortion_coeffs = torch.repeat_interleave(distortion_coeffs, n_images, dim=0)

        if extrinsic_matrix is None:
            extrinsic_matrix = torch.eye(4, device=device).unsqueeze(0)
        if len(extrinsic_matrix) == 1:
            extrinsic_matrix = torch.repeat_interleave(extrinsic_matrix, n_images, dim=0)

        # Now repeat these camera params for each box
        n_box_per_image_list = [len(b) for b in boxes]
        n_box_per_image = torch.tensor([len(b) for b in boxes], device=device)

        intrinsic_matrix = torch.repeat_interleave(intrinsic_matrix, n_box_per_image, dim=0)
        distortion_coeffs = torch.repeat_interleave(distortion_coeffs, n_box_per_image, dim=0)

        # Up-vector in camera-space
        if world_up_vector is None:
            world_up_vector = torch.tensor([0, -1, 0], device=device, dtype=torch.float32)

        camspace_up = torch.einsum('c,bCc->bC', world_up_vector, extrinsic_matrix[..., :3, :3])
        camspace_up = torch.repeat_interleave(camspace_up, n_box_per_image, dim=0)

        # Set up the test-time augmentation parameters
        aug_gammas = ptu.linspace(0.6, 1.0, num_aug, dtype=torch.float32, device=device)

        aug_angle_range = rot_aug_max_degrees * (torch.pi / 180.0)
        aug_angles = ptu.linspace(-aug_angle_range, aug_angle_range, num_aug, dtype=torch.float32,
                                  device=device)

        if num_aug == 1:
            aug_scales = torch.tensor([1.0], device=device, dtype=torch.float32)
        else:
            aug_scales = torch.cat([
                ptu.linspace(0.8, 1.0, num_aug // 2, endpoint=False, dtype=torch.float32,
                             device=device),
                torch.linspace(1.0, 1.1, num_aug - num_aug // 2, dtype=torch.float32,
                               device=device)], dim=0)
        aug_should_flip = (torch.arange(0, num_aug, device=device) - num_aug // 2) % 2 != 0
        aug_flipmat = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32,
                                   device=device)
        aug_maybe_flipmat = torch.where(
            aug_should_flip[:, np.newaxis, np.newaxis], aug_flipmat, torch.eye(3, device=device))
        aug_rotmat = ptu3d.rotation_mat(-aug_angles, rot_axis='z')
        aug_rotflipmat = aug_maybe_flipmat @ aug_rotmat

        # crops_flat, poses3d_flat = self._predict_in_batches(
        poses3d_flat, uncert_flat = self._predict_in_batches(
            images, weights, intrinsic_matrix, distortion_coeffs, camspace_up, boxes,
            internal_batch_size,
            aug_should_flip, aug_rotflipmat, aug_gammas, aug_scales, antialias_factor)
        poses3d_flat = plausib.scale_align(poses3d_flat)
        mean = torch.mean(poses3d_flat, dim=(-3, -2), keepdim=True)
        poses3d_flat_submean = (poses3d_flat - mean).float()
        poses3d_flat_submean, final_weights = weighted_geometric_median(
            poses3d_flat_submean, uncert_flat ** -1.5, dim=-3, n_iter=10, eps=50.0)
        poses3d_flat = poses3d_flat_submean.double() + mean.squeeze(1)

        uncert_flat = weighted_mean(uncert_flat, final_weights, dim=-2)

        # Project the 3D poses to get the 2D poses
        poses2d_flat_normalized = ptu3d.to_homogeneous(
            warping.distort_points(ptu3d.project(poses3d_flat.float()), distortion_coeffs))
        poses2d_flat = torch.einsum(
            'bnk,bjk->bnj', poses2d_flat_normalized, intrinsic_matrix[:, :2, :])

        # Arrange the results back into ragged tensors
        poses3d = torch.split(poses3d_flat, n_box_per_image_list)
        poses2d = torch.split(poses2d_flat, n_box_per_image_list)
        uncert = torch.split(uncert_flat, n_box_per_image_list)

        if suppress_implausible_poses:
            # Filter the resulting poses for individual plausibility to reduce false positives
            boxes, poses3d, poses2d, uncert = self._filter_poses(boxes, poses3d, poses2d, uncert)

        n_box_per_image_list = [len(b) for b in boxes]

        if sum(n_box_per_image_list) == 0:
            return self._predict_empty(images, weights)

        n_box_per_image = torch.tensor(n_box_per_image_list, device=device)
        # Convert to world coordinates
        inv_extrinsic_matrix = torch.repeat_interleave(
            torch.linalg.inv(extrinsic_matrix.double()),
            n_box_per_image, dim=0)
        poses3d_flat = torch.einsum(
            'bnk,bjk->bnj', ptu3d.to_homogeneous(torch.cat(poses3d)),
            inv_extrinsic_matrix[:, :3, :])
        poses3d = torch.split(poses3d_flat.float(), n_box_per_image_list)

        result = dict(boxes=boxes, poses3d=poses3d, poses2d=poses2d, uncertainties=uncert)
        return result

    def _filter_poses(self, boxes: List[torch.Tensor], poses3d: List[torch.Tensor],
                      poses2d: List[torch.Tensor], uncert: List[torch.Tensor]):
        boxes_out = []
        poses3d_out = []
        poses2d_out = []
        uncert_out = []
        for boxes_, poses3d_, poses2d_, uncert_ in zip(boxes, poses3d, poses2d, uncert):
            is_uncert_low = plausib.is_uncertainty_low(uncert_)
            plausible_mask = torch.logical_and(
                is_uncert_low, plausib.is_pose_consistent_with_box(poses2d_, boxes_))
            nms_indices = plausib.pose_non_max_suppression(
                poses3d_, boxes_[..., 4] / torch.mean(uncert_, dim=-1), plausible_mask)
            boxes_out.append(boxes_[nms_indices])
            poses3d_out.append(poses3d_[nms_indices])
            poses2d_out.append(poses2d_[nms_indices])
            uncert_out.append(uncert_[nms_indices])
        return boxes_out, poses3d_out, poses2d_out, uncert_out

    def _predict_in_batches(
            self, images: torch.Tensor, weights: Dict[str, torch.Tensor],
            intrinsic_matrix: torch.Tensor, distortion_coeffs: torch.Tensor,
            camspace_up: torch.Tensor, boxes: List[torch.Tensor],
            internal_batch_size: int, aug_should_flip: torch.Tensor, aug_rotflipmat: torch.Tensor,
            aug_gammas: torch.Tensor, aug_scales: torch.Tensor,
            antialias_factor: int):

        num_aug = len(aug_gammas)
        boxes_per_batch = internal_batch_size // num_aug
        boxes_flat = torch.cat(boxes, dim=0)
        image_id_per_box = torch.repeat_interleave(
            torch.arange(len(boxes)), torch.tensor([len(b) for b in boxes]))

        # Gamma decoding for correct image rescaling later on
        images = (images.float() / 255) ** 2.2

        if boxes_per_batch == 0:
            # Run all as a single batch
            return self._predict_single_batch(
                images, weights, intrinsic_matrix, distortion_coeffs, camspace_up, boxes_flat,
                image_id_per_box, aug_rotflipmat, aug_should_flip, aug_scales, aug_gammas,
                antialias_factor)
        else:
            # Chunk the image crops into batches and predict them one by one
            n_total_boxes = len(boxes_flat)
            n_batches = int(math.ceil(n_total_boxes / boxes_per_batch))
            poses3d_batches = []
            uncert_batches = []
            # CROP
            # crop_batches = []
            for i in range(n_batches):
                batch_slice = slice(i * boxes_per_batch, (i + 1) * boxes_per_batch)
                # CROP
                poses3d, uncert = self._predict_single_batch(
                    images, weights, intrinsic_matrix[batch_slice], distortion_coeffs[batch_slice],
                    camspace_up[batch_slice], boxes_flat[batch_slice],
                    image_id_per_box[batch_slice], aug_rotflipmat, aug_should_flip, aug_scales,
                    aug_gammas, antialias_factor)
                poses3d_batches.append(poses3d)
                uncert_batches.append(uncert)
            return torch.cat(poses3d_batches, dim=0), torch.cat(uncert_batches, dim=0)

    def _predict_single_batch(
            self, images: torch.Tensor, weights: Dict[str, torch.Tensor],
            intrinsic_matrix: torch.Tensor, distortion_coeffs: torch.Tensor,
            camspace_up: torch.Tensor, boxes: torch.Tensor,
            image_ids: torch.Tensor,
            aug_rotflipmat: torch.Tensor, aug_should_flip: torch.Tensor, aug_scales: torch.Tensor,
            aug_gammas: torch.Tensor, antialias_factor: int):
        # Get crops and info about the transformation used to create them
        # Each has shape [num_aug, n_boxes, ...]
        crops, new_intrinsic_matrix, R = self._get_crops(
            images, intrinsic_matrix, distortion_coeffs, camspace_up, boxes, image_ids,
            aug_rotflipmat, aug_scales, aug_gammas, antialias_factor)

        # Flatten each and predict the pose with the crop model
        new_intrinsic_matrix_flat = torch.reshape(new_intrinsic_matrix, (-1, 3, 3))
        res = self.crop_model.input_resolution
        crops_flat = torch.reshape(crops, (-1, 3, res, res))

        n_cases = crops.shape[1]
        aug_should_flip_flat = torch.repeat_interleave(aug_should_flip, n_cases, dim=0)

        poses_flat, uncert_flat = self.crop_model.predict_multi_same_weights(
            crops_flat.half(), new_intrinsic_matrix_flat, weights, aug_should_flip_flat)
        poses_flat = poses_flat.double()
        n_joints = poses_flat.shape[-2]

        poses = torch.reshape(poses_flat, [-1, n_cases, n_joints, 3])
        uncert = torch.reshape(uncert_flat, [-1, n_cases, n_joints])
        poses_orig_camspace = poses @ R.double()

        # Transpose to [n_boxes, num_aug, ...]
        return poses_orig_camspace.transpose(0, 1), uncert.transpose(0, 1)
        # CROP
        # crops = torch.reshape(crops_flat, [num_aug, -1, 3, res, res])
        # return crops.transpose(0, 1), poses_orig_camspace.transpose(0, 1)

    def _get_crops(
            self, images: torch.Tensor, intrinsic_matrix: torch.Tensor,
            distortion_coeffs: torch.Tensor, camspace_up: torch.Tensor, boxes: torch.Tensor,
            image_ids: torch.Tensor,
            aug_rotflipmat: torch.Tensor, aug_scales: torch.Tensor, aug_gammas: torch.Tensor,
            antialias_factor: int):
        R_noaug, box_scales = self._get_new_rotation_and_scale(
            intrinsic_matrix, distortion_coeffs, camspace_up, boxes)

        device = images.device
        # How much we need to scale overall, taking scale augmentation into account
        # From here on, we introduce the dimension of augmentations
        crop_scales = aug_scales[:, np.newaxis] * box_scales[np.newaxis, :]
        # Build the new intrinsic matrix
        num_box = boxes.shape[0]
        num_aug = aug_gammas.shape[0]
        res = self.crop_model.input_resolution
        new_intrinsic_matrix = torch.cat([
            torch.cat([
                # Top-left of original intrinsic matrix gets scaled
                intrinsic_matrix[np.newaxis, :, :2, :2] * crop_scales[:, :, np.newaxis, np.newaxis],
                # Principal point is the middle of the new image size
                torch.full((num_aug, num_box, 2, 1), res / 2, dtype=torch.float32, device=device)],
                dim=3),
            torch.cat([
                # [0, 0, 1] as the last row of the intrinsic matrix:
                torch.zeros((num_aug, num_box, 1, 2), dtype=torch.float32, device=device),
                torch.ones((num_aug, num_box, 1, 1), dtype=torch.float32, device=device)], dim=3)],
            dim=2)
        R = aug_rotflipmat[:, np.newaxis] @ R_noaug
        new_invprojmat = torch.linalg.inv(new_intrinsic_matrix @ R)

        # If we perform antialiasing through output scaling, we render a larger image first and then
        # shrink it. So we scale the homography first.
        if antialias_factor > 1:
            scaling_mat = warping.corner_aligned_scale_mat(
                1 / antialias_factor)
            new_invprojmat = new_invprojmat @ scaling_mat.to(new_invprojmat.device)

        crops = 1 - warping.warp_images_with_pyramid(
            1 - images,
            intrinsic_matrix=torch.tile(intrinsic_matrix, [num_aug, 1, 1]),
            new_invprojmats=torch.reshape(new_invprojmat, [-1, 3, 3]),
            distortion_coeffs=torch.tile(distortion_coeffs, [num_aug, 1]),
            crop_scales=torch.reshape(crop_scales, [-1]) * antialias_factor,
            output_shape=(res * antialias_factor, res * antialias_factor),
            image_ids=torch.tile(image_ids, [num_aug]))
        crops = torch.clip(crops, 0, 1)

        # Downscale the result if we do antialiasing through output scaling
        if antialias_factor == 2:
            crops = torch.nn.functional.avg_pool2d(crops, 2, 2)
        elif antialias_factor == 4:
            crops = torch.nn.functional.avg_pool2d(crops, 4, 4)
        elif antialias_factor > 4:
            crops = torchvision.transforms.functional.resize(
                crops, (res, res), torchvision.transforms.functional.InterpolationMode.BILINEAR,
                antialias=True)
        crops = torch.reshape(crops, [num_aug, num_box, 3, res, res])
        # The division by 2.2 cancels the original gamma decoding from earlier
        crops **= torch.reshape(aug_gammas / 2.2, [-1, 1, 1, 1, 1])
        return crops, new_intrinsic_matrix, R

    def _get_new_rotation_and_scale(self, intrinsic_matrix, distortion_coeffs, camspace_up, boxes):
        # Transform five points on each box: the center and the midpoints of the four sides
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxpoints_homog = ptu3d.to_homogeneous(torch.stack([
            torch.stack([x + w / 2, y + h / 2], dim=1),  # center
            torch.stack([x + w / 2, y], dim=1),
            torch.stack([x + w, y + h / 2], dim=1),
            torch.stack([x + w / 2, y + h], dim=1),
            torch.stack([x, y + h / 2], dim=1)], dim=1))

        boxpoints_camspace = torch.einsum(
            'bpc,bCc->bpC', boxpoints_homog, torch.linalg.inv(intrinsic_matrix))
        boxpoints_camspace = ptu3d.to_homogeneous(
            warping.undistort_points(boxpoints_camspace[:, :, :2], distortion_coeffs))
        # Create a rotation matrix that will put the box center to the principal point
        # and apply the augmentation rotation and flip, to get the new coordinate frame
        box_center_camspace = boxpoints_camspace[:, 0]
        R_noaug = ptu3d.lookat_matrix(forward_vector=box_center_camspace, up_vector=camspace_up)

        # Transform the side midpoints of the box to the new coordinate frame
        sidepoints_camspace = boxpoints_camspace[:, 1:5]
        sidepoints_new = ptu3d.project(torch.einsum(
            'bpc,bCc->bpC', sidepoints_camspace, intrinsic_matrix @ R_noaug))

        # Measure the size of the reprojected boxes
        vertical_size = torch.linalg.norm(sidepoints_new[:, 0] - sidepoints_new[:, 2], dim=-1)
        horiz_size = torch.linalg.norm(sidepoints_new[:, 1] - sidepoints_new[:, 3], dim=-1)
        box_size_new = torch.maximum(vertical_size, horiz_size)

        # How much we need to scale (zoom) to have the boxes fill out the final crop
        box_scales = torch.tensor(self.crop_model.input_resolution,
                                  dtype=box_size_new.dtype) / box_size_new
        return R_noaug, box_scales

    def _predict_empty(self, image: torch.Tensor, weights: Dict[str, torch.Tensor]):
        device = image.device
        n_joints = weights['w_tensor'].shape[1]
        poses3d = torch.zeros((0, n_joints, 3), dtype=torch.float32, device=device)
        poses2d = torch.zeros((0, n_joints, 2), dtype=torch.float32, device=device)
        uncert = torch.zeros((0, n_joints), dtype=torch.float32, device=device)
        boxes = torch.zeros((0, 5), dtype=torch.float32, device=device)
        n_images = image.shape[0]

        result = dict(
            boxes=[boxes] * n_images,
            poses3d=[poses3d] * n_images,
            poses2d=[poses2d] * n_images,
            uncertainties=[uncert] * n_images)
        return result

    def _predict_empty_smpl(self, image: torch.Tensor, model_name: str):
        device = image.device
        fitter = self.fitter_smpl if model_name == 'smpl' else self.fitter_smplx
        n_joints = fitter.body_model.num_joints
        n_verts = fitter.num_vertices
        pose = torch.zeros((0, n_joints, 3), dtype=torch.float32, device=device)
        betas = torch.zeros((0, fitter.n_betas), dtype=torch.float32, device=device)
        trans = torch.zeros((0, 3), dtype=torch.float32, device=device)
        vertices3d = torch.zeros((0, n_verts, 3), dtype=torch.float32, device=device)
        joints3d = torch.zeros((0, n_joints, 3), dtype=torch.float32, device=device)
        vertices2d = torch.zeros((0, n_verts, 2), dtype=torch.float32, device=device)
        joints2d = torch.zeros((0, n_joints, 2), dtype=torch.float32, device=device)
        vertices3d_nonparam = torch.zeros((0, n_verts, 3), dtype=torch.float32, device=device)
        joints3d_nonparam = torch.zeros((0, n_joints, 3), dtype=torch.float32, device=device)
        vertices2d_nonparam = torch.zeros((0, n_verts, 2), dtype=torch.float32, device=device)
        joints2d_nonparam = torch.zeros((0, n_joints, 2), dtype=torch.float32, device=device)
        vertex_uncertainties = torch.zeros((0, n_verts), dtype=torch.float32, device=device)
        joint_uncertainties = torch.zeros((0, n_joints), dtype=torch.float32, device=device)
        n_images = image.shape[0]
        result = dict(
            pose=[pose] * n_images,
            betas=[betas] * n_images,
            trans=[trans] * n_images,
            vertices3d=[vertices3d] * n_images,
            joints3d=[joints3d] * n_images,
            vertices2d=[vertices2d] * n_images,
            joints2d=[joints2d] * n_images,
            vertices3d_nonparam=[vertices3d_nonparam] * n_images,
            joints3d_nonparam=[joints3d_nonparam] * n_images,
            vertices2d_nonparam=[vertices2d_nonparam] * n_images,
            joints2d_nonparam=[joints2d_nonparam] * n_images,
            vertex_uncertainties=[vertex_uncertainties] * n_images,
            joint_uncertainties=[joint_uncertainties] * n_images)
        return result

    @torch.jit.export
    def detect_poses(
            self,
            image: torch.Tensor,
            weights: Dict[str, torch.Tensor],
            intrinsic_matrix: Optional[torch.Tensor] = None,
            distortion_coeffs: Optional[torch.Tensor] = None,
            extrinsic_matrix: Optional[torch.Tensor] = None,
            world_up_vector: Optional[torch.Tensor] = None,
            default_fov_degrees: float = 55.0,
            internal_batch_size: int = 64,
            antialias_factor: int = 1,
            num_aug: int = 5, rot_aug_max_degrees: float = 25.0,
            detector_threshold: float = 0.3,
            detector_nms_iou_threshold: float = 0.7,
            max_detections: int = 150,
            suppress_implausible_poses: bool = True):

        images = image[np.newaxis]
        intrinsic_matrix = intrinsic_matrix[np.newaxis] if intrinsic_matrix is not None else None
        distortion_coeffs = distortion_coeffs[np.newaxis] if distortion_coeffs is not None else None
        extrinsic_matrix = extrinsic_matrix[np.newaxis] if extrinsic_matrix is not None else None

        result = self.detect_poses_batched(
            images, weights, intrinsic_matrix, distortion_coeffs, extrinsic_matrix, world_up_vector,
            default_fov_degrees, internal_batch_size, antialias_factor, num_aug,
            rot_aug_max_degrees,
            detector_threshold, detector_nms_iou_threshold, max_detections,
            suppress_implausible_poses)
        return {k: v[0] for k, v in result.items()}

    @torch.jit.export
    def estimate_poses(
            self,
            image: torch.Tensor,
            boxes: torch.Tensor,
            weights: Dict[str, torch.Tensor],
            intrinsic_matrix: Optional[torch.Tensor] = None,
            distortion_coeffs: Optional[torch.Tensor] = None,
            extrinsic_matrix: Optional[torch.Tensor] = None,
            world_up_vector: Optional[torch.Tensor] = None,
            default_fov_degrees: float = 55.0,
            internal_batch_size: int = 64,
            antialias_factor: int = 1,
            num_aug: int = 1, rot_aug_max_degrees: float = 25.0):

        images = image[np.newaxis]
        boxes = [boxes]
        intrinsic_matrix = intrinsic_matrix[np.newaxis] if intrinsic_matrix is not None else None
        distortion_coeffs = distortion_coeffs[np.newaxis] if distortion_coeffs is not None else None
        extrinsic_matrix = extrinsic_matrix[np.newaxis] if extrinsic_matrix is not None else None

        result = self.estimate_poses_batched(
            images, boxes, weights, intrinsic_matrix, distortion_coeffs, extrinsic_matrix,
            world_up_vector,
            default_fov_degrees, internal_batch_size, antialias_factor, num_aug,
            rot_aug_max_degrees)
        return {k: v[0] for k, v in result.items()}

    @torch.jit.export
    def get_weights_for_canonical_points(self, canonical_points: torch.Tensor):
        return self.crop_model.get_weights_for_canonical_points(canonical_points)


def project_ragged_pytorch(
        images: torch.Tensor, poses3d: List[torch.Tensor], extrinsic_matrix: Optional[torch.Tensor],
        intrinsic_matrix: Optional[torch.Tensor], distortion_coeffs: Optional[torch.Tensor],
        default_fov_degrees: float = 55.0):
    device = images.device
    n_box_per_image_list = [len(b) for b in poses3d]
    n_box_per_image = torch.tensor([len(b) for b in poses3d], device=device)

    n_images = images.shape[0]

    if intrinsic_matrix is None:
        intrinsic_matrix = ptu3d.intrinsic_matrix_from_field_of_view(
            default_fov_degrees, images.shape[2:4], device=device)

    # If one intrinsic matrix is given, repeat it for all images
    if intrinsic_matrix.shape[0] == 1:
        intrinsic_matrix = torch.repeat_interleave(intrinsic_matrix, n_images, dim=0)

    if distortion_coeffs is None:
        distortion_coeffs = torch.zeros((n_images, 5), device=device)

    # If one distortion coeff/extrinsic matrix is given, repeat it for all images
    if distortion_coeffs.shape[0] == 1:
        distortion_coeffs = torch.repeat_interleave(distortion_coeffs, n_images, dim=0)

    if extrinsic_matrix is None:
        extrinsic_matrix = torch.eye(4, device=device).unsqueeze(0)
    if extrinsic_matrix.shape[0] == 1:
        extrinsic_matrix = torch.repeat_interleave(extrinsic_matrix, n_images, dim=0)

    intrinsic_matrix_rep = torch.repeat_interleave(intrinsic_matrix, n_box_per_image, dim=0)
    distortion_coeffs_rep = torch.repeat_interleave(distortion_coeffs, n_box_per_image, dim=0)
    extrinsic_matrix_rep = torch.repeat_interleave(extrinsic_matrix, n_box_per_image, dim=0)

    poses3d_flat = torch.cat(poses3d, dim=0)
    poses3d_flat = torch.einsum(
        'bnk,bjk->bnj', ptu3d.to_homogeneous(poses3d_flat), extrinsic_matrix_rep[:, :3, :])

    poses2d_flat_normalized = ptu3d.to_homogeneous(
        warping.distort_points(ptu3d.project(poses3d_flat), distortion_coeffs_rep))
    poses2d_flat = torch.einsum(
        'bnk,bjk->bnj', poses2d_flat_normalized, intrinsic_matrix_rep[:, :2, :])
    poses2d = torch.split(poses2d_flat, n_box_per_image_list)
    return poses2d


def weighted_geometric_median(
        x: torch.Tensor, w: Optional[torch.Tensor], n_iter: int = 10, dim: int = -2,
        eps: float = 1e-1, keepdim: bool = False):
    if dim < 0:
        dim = len(x.shape) + dim

    if w is None:
        w = torch.ones_like(x[..., :1])
    else:
        w = w.unsqueeze(-1)

    new_weights = w
    y = weighted_mean(x, new_weights, dim=dim, keepdim=True)
    for _ in range(n_iter):
        dist = torch.norm(x - y, dim=-1, keepdim=True)
        new_weights = w / (dist + eps)
        y = weighted_mean(x, new_weights, dim=dim, keepdim=True)

    if not keepdim:
        y = y.squeeze(dim)

    return y, new_weights.squeeze(-1)


def weighted_mean(x: torch.Tensor, w: torch.Tensor, dim: int = -2, keepdim: bool = False):
    return (
            (x * w).sum(dim=dim, keepdim=keepdim) /
            w.sum(dim=dim, keepdim=keepdim))

if __name__ == '__main__':
    main()