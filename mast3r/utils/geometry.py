#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# visloc script with support for coarse to fine
# --------------------------------------------------------
import os
import numpy as np
import random
import torch
import torchvision.transforms as tvf
import argparse
from tqdm import tqdm
from PIL import Image
import math

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.utils.coarse_to_fine import select_pairs_of_crops, crop_slice
from mast3r.utils.collate import cat_collate, cat_collate_fn_map
from mast3r.utils.misc import mkdir_for
from mast3r.datasets.utils.cropping import crop_to_homography

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.inference import inference, loss_of_one_batch
from dust3r.utils.geometry import geotrf, colmap_to_opencv_intrinsics, opencv_to_colmap_intrinsics
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r_visloc.datasets import *
from dust3r_visloc.localization import run_pnp
from dust3r_visloc.evaluation import get_pose_error, aggregate_stats, export_results
from dust3r_visloc.datasets.utils import get_HW_resolution, rescale_points3d


@torch.no_grad()
def coarse_matching(query_view, map_view, model, device, pixel_tol, fast_nn_params):
    # prepare batch
    imgs = []
    for idx, img in enumerate([query_view['rgb_rescaled'], map_view['rgb_rescaled']]):
        imgs.append(dict(img=img.unsqueeze(0), true_shape=np.int32([img.shape[1:]]),
                         idx=idx, instance=str(idx)))
    output = inference([tuple(imgs)], model, device, batch_size=1, verbose=False)
    pred1, pred2 = output['pred1'], output['pred2']
    conf_list = [pred1['desc_conf'].squeeze(0).cpu().numpy(), pred2['desc_conf'].squeeze(0).cpu().numpy()]
    desc_list = [pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()]

    # find 2D-2D matches between the two images
    PQ, PM = desc_list[0], desc_list[1]
    if len(PQ) == 0 or len(PM) == 0:
        return [], [], [], []

    if pixel_tol == 0:
        matches_im_map, matches_im_query = fast_reciprocal_NNs(PM, PQ, subsample_or_initxy1=8, **fast_nn_params)
        HM, WM = map_view['rgb_rescaled'].shape[1:]
        HQ, WQ = query_view['rgb_rescaled'].shape[1:]
        # ignore small border around the edge
        valid_matches_map = (matches_im_map[:, 0] >= 3) & (matches_im_map[:, 0] < WM - 3) & (
            matches_im_map[:, 1] >= 3) & (matches_im_map[:, 1] < HM - 3)
        valid_matches_query = (matches_im_query[:, 0] >= 3) & (matches_im_query[:, 0] < WQ - 3) & (
            matches_im_query[:, 1] >= 3) & (matches_im_query[:, 1] < HQ - 3)
        valid_matches = valid_matches_map & valid_matches_query
        matches_im_map = matches_im_map[valid_matches]
        matches_im_query = matches_im_query[valid_matches]
        valid_pts3d = []
        matches_confs = []
    else:
        yM, xM = torch.where(map_view['valid_rescaled'])
        matches_im_map, matches_im_query = fast_reciprocal_NNs(PM, PQ, (xM, yM), pixel_tol=pixel_tol, **fast_nn_params)
        valid_pts3d = map_view['pts3d_rescaled'].cpu().numpy()[matches_im_map[:, 1], matches_im_map[:, 0]]
        matches_confs = np.minimum(
            conf_list[1][matches_im_map[:, 1], matches_im_map[:, 0]],
            conf_list[0][matches_im_query[:, 1], matches_im_query[:, 0]]
        )
    # from cv2 to colmap
    matches_im_query = matches_im_query.astype(np.float64)
    matches_im_map = matches_im_map.astype(np.float64)
    matches_im_query[:, 0] += 0.5
    matches_im_query[:, 1] += 0.5
    matches_im_map[:, 0] += 0.5
    matches_im_map[:, 1] += 0.5
    # rescale coordinates
    matches_im_query = geotrf(query_view['to_orig'], matches_im_query, norm=True)
    matches_im_map = geotrf(map_view['to_orig'], matches_im_map, norm=True)
    # from colmap back to cv2
    matches_im_query[:, 0] -= 0.5
    matches_im_query[:, 1] -= 0.5
    matches_im_map[:, 0] -= 0.5
    matches_im_map[:, 1] -= 0.5
    return valid_pts3d, matches_im_query, matches_im_map, matches_confs


@torch.no_grad()
def crops_inference(pairs, model, device, batch_size=48, verbose=True):
    assert len(pairs) == 2, "Error, data should be a tuple of dicts containing the batch of image pairs"
    # Forward a possibly big bunch of data, by blocks of batch_size
    B = pairs[0]['img'].shape[0]
    if B < batch_size:
        return loss_of_one_batch(pairs, model, None, device=device, symmetrize_batch=False)
    preds = []
    for ii in range(0, B, batch_size):
        sel = slice(ii, ii + min(B - ii, batch_size))
        temp_data = [{}, {}]
        for di in [0, 1]:
            temp_data[di] = {kk: pairs[di][kk][sel]
                             for kk in pairs[di].keys() if pairs[di][kk] is not None}  # copy chunk for forward
        preds.append(loss_of_one_batch(temp_data, model,
                                       None, device=device, symmetrize_batch=False))  # sequential forward
    # Merge all preds
    return cat_collate(preds, collate_fn_map=cat_collate_fn_map)


@torch.no_grad()
def fine_matching(query_views, map_views, model, device, max_batch_size, pixel_tol, fast_nn_params):
    assert pixel_tol > 0
    output = crops_inference([query_views, map_views],
                             model, device, batch_size=max_batch_size, verbose=False)
    pred1, pred2 = output['pred1'], output['pred2']
    descs1 = pred1['desc'].clone()
    descs2 = pred2['desc'].clone()
    confs1 = pred1['desc_conf'].clone()
    confs2 = pred2['desc_conf'].clone()

    # Compute matches
    valid_pts3d, matches_im_map, matches_im_query, matches_confs = [], [], [], []
    for ppi, (pp1, pp2, cc11, cc21) in enumerate(zip(descs1, descs2, confs1, confs2)):
        valid_ppi = map_views['valid'][ppi]
        pts3d_ppi = map_views['pts3d'][ppi].cpu().numpy()
        conf_list_ppi = [cc11.cpu().numpy(), cc21.cpu().numpy()]

        y_ppi, x_ppi = torch.where(valid_ppi)
        matches_im_map_ppi, matches_im_query_ppi = fast_reciprocal_NNs(pp2, pp1, (x_ppi, y_ppi),
                                                                       pixel_tol=pixel_tol, **fast_nn_params)

        valid_pts3d_ppi = pts3d_ppi[matches_im_map_ppi[:, 1], matches_im_map_ppi[:, 0]]
        matches_confs_ppi = np.minimum(
            conf_list_ppi[1][matches_im_map_ppi[:, 1], matches_im_map_ppi[:, 0]],
            conf_list_ppi[0][matches_im_query_ppi[:, 1], matches_im_query_ppi[:, 0]]
        )
        # inverse operation where we uncrop pixel coordinates
        matches_im_map_ppi = geotrf(map_views['to_orig'][ppi].cpu().numpy(), matches_im_map_ppi.copy(), norm=True)
        matches_im_query_ppi = geotrf(query_views['to_orig'][ppi].cpu().numpy(), matches_im_query_ppi.copy(), norm=True)

        matches_im_map.append(matches_im_map_ppi)
        matches_im_query.append(matches_im_query_ppi)
        valid_pts3d.append(valid_pts3d_ppi)
        matches_confs.append(matches_confs_ppi)

    if len(valid_pts3d) == 0:
        return [], [], [], []

    matches_im_map = np.concatenate(matches_im_map, axis=0)
    matches_im_query = np.concatenate(matches_im_query, axis=0)
    valid_pts3d = np.concatenate(valid_pts3d, axis=0)
    matches_confs = np.concatenate(matches_confs, axis=0)
    return valid_pts3d, matches_im_query, matches_im_map, matches_confs


def crop(img, mask, pts3d, crop, intrinsics=None):
    out_cropped_img = img.clone()
    if mask is not None:
        out_cropped_mask = mask.clone()
    else:
        out_cropped_mask = None
    if pts3d is not None:
        out_cropped_pts3d = pts3d.clone()
    else:
        out_cropped_pts3d = None
    to_orig = torch.eye(3, device=img.device)

    # If intrinsics available, crop and apply rectifying homography. Otherwise, just crop
    if intrinsics is not None:
        K_old = intrinsics
        imsize, K_new, R, H = crop_to_homography(K_old, crop)
        # apply homography to image
        H /= H[2, 2]
        homo8 = H.ravel().tolist()[:8]
        # From float tensor to uint8 PIL Image
        pilim = Image.fromarray((255 * (img + 1.) / 2).to(torch.uint8).numpy())
        pilout_cropped_img = pilim.transform(imsize, Image.Transform.PERSPECTIVE,
                                             homo8, resample=Image.Resampling.BICUBIC)

        # From uint8 PIL Image to float tensor
        out_cropped_img = 2. * torch.tensor(np.array(pilout_cropped_img)).to(img) / 255. - 1.
        if out_cropped_mask is not None:
            pilmask = Image.fromarray((255 * out_cropped_mask).to(torch.uint8).numpy())
            pilout_cropped_mask = pilmask.transform(
                imsize, Image.Transform.PERSPECTIVE, homo8, resample=Image.Resampling.NEAREST)
            out_cropped_mask = torch.from_numpy(np.array(pilout_cropped_mask) > 0).to(out_cropped_mask.dtype)
        if out_cropped_pts3d is not None:
            out_cropped_pts3d = out_cropped_pts3d.numpy()
            out_cropped_X = np.array(Image.fromarray(out_cropped_pts3d[:, :, 0]).transform(imsize,
                                                                                           Image.Transform.PERSPECTIVE,
                                                                                           homo8,
                                                                                           resample=Image.Resampling.NEAREST))
            out_cropped_Y = np.array(Image.fromarray(out_cropped_pts3d[:, :, 1]).transform(imsize,
                                                                                           Image.Transform.PERSPECTIVE,
                                                                                           homo8,
                                                                                           resample=Image.Resampling.NEAREST))
            out_cropped_Z = np.array(Image.fromarray(out_cropped_pts3d[:, :, 2]).transform(imsize,
                                                                                           Image.Transform.PERSPECTIVE,
                                                                                           homo8,
                                                                                           resample=Image.Resampling.NEAREST))

            out_cropped_pts3d = torch.from_numpy(np.stack([out_cropped_X, out_cropped_Y, out_cropped_Z], axis=-1))

        to_orig = torch.tensor(H, device=img.device)
    else:
        out_cropped_img = img[crop_slice(crop)]
        if out_cropped_mask is not None:
            out_cropped_mask = out_cropped_mask[crop_slice(crop)]
        if out_cropped_pts3d is not None:
            out_cropped_pts3d = out_cropped_pts3d[crop_slice(crop)]
        to_orig[:2, -1] = torch.tensor(crop[:2])

    return out_cropped_img, out_cropped_mask, out_cropped_pts3d, to_orig


def resize_image_to_max(max_image_size, rgb, K):
    W, H = rgb.size
    if max_image_size and max(W, H) > max_image_size:
        islandscape = (W >= H)
        if islandscape:
            WMax = max_image_size
            HMax = int(H * (WMax / W))
        else:
            HMax = max_image_size
            WMax = int(W * (HMax / H))
        resize_op = tvf.Compose([ImgNorm, tvf.Resize(size=[HMax, WMax])])
        rgb_tensor = resize_op(rgb).permute(1, 2, 0)
        to_orig_max = np.array([[W / WMax, 0, 0],
                                [0, H / HMax, 0],
                                [0, 0, 1]])
        to_resize_max = np.array([[WMax / W, 0, 0],
                                  [0, HMax / H, 0],
                                  [0, 0, 1]])

        # Generate new camera parameters
        new_K = opencv_to_colmap_intrinsics(K)
        new_K[0, :] *= WMax / W
        new_K[1, :] *= HMax / H
        new_K = colmap_to_opencv_intrinsics(new_K)
    else:
        rgb_tensor = ImgNorm(rgb).permute(1, 2, 0)
        to_orig_max = np.eye(3)
        to_resize_max = np.eye(3)
        HMax, WMax = H, W
        new_K = K
    return rgb_tensor, new_K, to_orig_max, to_resize_max, (HMax, WMax)
