import numpy as np
import cv2
import argparse
import json
import os
# import matplotlib.pyplot as plt
from numpy.typing import NDArray


import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.model import AsymmetricMASt3R

DEVICE = 'cuda'
MODEL_NAME = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
BORDER = 3

def get_mast3r_output(images, model):
    # Load model, run inference
    output = inference([images], model, DEVICE,
                       batch_size=1, verbose=False)
    
    # print(f"Output keys: {output.keys()}")
    # print(f"Output view1 keys: {output['view1'].keys()}")
    # print(f"Output view2 keys: {output['view2'].keys()}")
    # print(f"Output pred1 keys: {output['pred1'].keys()}")
    # print(f"Output pred2 keys: {output['pred2'].keys()}")

    # raw predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1 = pred1['desc'].squeeze(0).detach()
    desc2 = pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im1, matches_im2 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=DEVICE, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H1, W1 = view1['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= BORDER) & \
                        (matches_im1[:, 0] < int(W1) - BORDER) & \
                        (matches_im1[:, 1] >= BORDER) & \
                        (matches_im1[:, 1] < int(H1) - BORDER)

    H2, W2 = view2['true_shape'][0]
    valid_matches_im2 = (matches_im2[:, 0] >= BORDER) & \
                        (matches_im2[:, 0] < int(W2) - BORDER) & \
                        (matches_im2[:, 1] >= BORDER) & \
                        (matches_im2[:, 1] < int(H2) - BORDER)

    valid_matches = valid_matches_im1 & valid_matches_im2

    # matches are Nx2 image coordinates.
    matches_im1 = matches_im1[valid_matches]
    matches_im2 = matches_im2[valid_matches]

    # Convert the other outputs to numpy arrays
    pts3d_im1 = pred1['pts3d'].squeeze(0).detach().cpu().numpy()
    pts3d_im2 = pred2['pts3d_in_other_view'].squeeze(0).detach().cpu().numpy()

    conf_im1 = pred1['conf'].squeeze(0).detach().cpu().numpy()
    conf_im2 = pred2['conf'].squeeze(0).detach().cpu().numpy()

    desc_conf_im1 = pred1['desc_conf'].squeeze(0).detach().cpu().numpy()
    desc_conf_im2 = pred2['desc_conf'].squeeze(0).detach().cpu().numpy()

    return matches_im1, matches_im2, pts3d_im1, pts3d_im2, conf_im1, conf_im2, desc_conf_im1, desc_conf_im2

def scale_intrinsics(K: NDArray, prev_w: float, prev_h: float, inverse: bool) -> NDArray:
    """Scale the intrinsics matrix by a given factor .

    Args:
        K (NDArray): 3x3 intrinsics matrix
        scale (float): Scale factor

    Returns:
        NDArray: Scaled intrinsics matrix
    """
    assert K.shape == (3, 3), f"Expected (3, 3), but got {K.shape=}"

    scale_w = 512.0 / prev_w  # sizes of the images in the Mast3r dataset
    scale_h = 384.0 / prev_h  # sizes of the images in the Mast3r dataset

    if inverse:
        scale_w = 1.0 / scale_w
        scale_h = 1.0 / scale_h

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_w
    K_scaled[0, 2] *= scale_w
    # K_scaled[1, 1] *= scale_h
    # K_scaled[1, 2] *= scale_h
    K_scaled[1, 1] *= scale_w
    K_scaled[1, 2] *= scale_w

    return K_scaled

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgdir", type=str, required=True, help="Path to the image directory")
    parser.add_argument("--outdir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (default: cuda)")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Model name for MASt3R")
    parser.add_argument("--size", type=int, default=512, help="Size of the images (default: 512)")
    args = parser.parse_args()

    image_dir = args.imgdir
    filelist = []
    for image_name in sorted(os.listdir(image_dir)):
        if image_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_dir, image_name)
            filelist.append(image_path)

    # Load images
    images = load_images(filelist, size=args.size)
    model = AsymmetricMASt3R.from_pretrained(args.model_name).to(args.device)

    # Generate sequential pairs
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    optim_level = "refine"
    cache_dir = os.path.join(args.outdir, 'cache')

    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=0.07, niter1=500, lr2=0.014, niter2=200, device='cuda',
                                    opt_depth='depth' in optim_level, shared_intrinsics=False,
                                    matching_conf_thr=5)

    # Initialize a list to store the camera data
    camera_data = []
    # pose_data = []
    
    # Extract camera intrinsics and extrinsics
    # Assuming the first camera is the reference camera
    cam_poses = scene.get_im_poses()
    cam_intrinsics = scene.intrinsics
    for idx, (pose, intrinsic) in enumerate(zip(cam_poses, cam_intrinsics)):
        scaled_intrinsic = scale_intrinsics(intrinsic.detach().cpu().numpy(), prev_w=3840, prev_h=2160, inverse=True)
        
        # Extract rotation matrix and translation vector from the 4x4 pose matrix
        rotation_matrix = pose[:3, :3].detach().cpu().numpy()
        translation_vector = pose[:3, 3].detach().cpu().numpy()
        translation_vector *= 14.1

        # Convert rotation matrix to rotation vector
        rotation_vector, _ = cv2.Rodrigues(rotation_matrix)

        translation_vector = rotation_matrix.T @ translation_vector
        
        # invert x and z axis
        rotation_vector[0] = -rotation_vector[0]
        rotation_vector[2] = -rotation_vector[2]
        translation_vector[0] = -translation_vector[0]
        translation_vector[2] = -translation_vector[2]

        # Create a dictionary for the current camera
        camera_entry = {
            "name": f"c{idx+1:02d}",  # Camera name (e.g., c01, c02, ...)
            "size": [3840, 2160],  # Assuming the original image size
            "matrix": scaled_intrinsic.tolist(),  # Convert numpy array to list
            "distortions": [0.0, 0.0, 0.0, 0.0, 0.0],  # Assuming no distortions
            "rotation": rotation_vector.flatten().tolist(),  # Convert to list
            "translation": translation_vector.tolist()  # Convert to list
        }
        # Append the camera entry to the list
        camera_data.append(camera_entry)

        # # Create a dictionary for the current pose
        # pose_entry = {
        #     "name": f"c{idx+1:02d}",  # Camera name (e.g., c01, c02, ...)
        #     "pose": pose.tolist(),  # Flatten the pose matrix and convert to list
        # }
        # # Append the camera entry to the list
        # pose_data.append(pose_entry)




    # Save the camera data to a JSON file
    output_file = os.path.join(args.outdir, "camera_settings.json")
    with open(output_file, "w") as f:
        f.write("[")  # Start the JSON array
        for idx, entry in enumerate(camera_data):
            json_entry = json.dumps(entry)  # Compact JSON for each entry
            if idx < len(camera_data) - 1:
                f.write(f"{json_entry},\n ")  # Add a comma for all but the last entry
            else:
                f.write(f"{json_entry}")  # No comma for the last entry
        f.write("]")  # Close the JSON array
    print(f"Camera settings saved to {output_file}")

    # # Save the pose data to a JSON file
    # output_file = os.path.join(args.outdir, "pose_settings.json")
    # with open(output_file, "w") as f:
    #     f.write("[")
    #     # Start the JSON array
    #     for idx, entry in enumerate(pose_data):
    #         json_entry = json.dumps(entry)
    #         if idx < len(pose_data) - 1:
    #             f.write(f"{json_entry},\n ")
    #         else:
    #             f.write(f"{json_entry}")
    #     f.write("]")
    # print(f"Pose settings saved to {output_file}")

    # # Plot the camera poses in 3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot camera positions
    # ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], c='r', label='Camera Positions')

    # # Annotate each camera position with its index
    # for i, pos in enumerate(camera_positions):
    #     ax.text(pos[0], pos[1], pos[2], f"Cam {i+1}", color='black', fontsize=8)

    # # Optionally, plot orientation vectors (scaled for visualization)
    # for pos, ori in zip(camera_positions, camera_orientations):
    #     # Convert rotation vector to rotation matrix
    #     rotation_matrix, _ = cv2.Rodrigues(ori)
    #     # Define a direction vector (e.g., forward direction in camera space)
    #     direction_vector = np.array([0, 0, 1])  # Z-axis in camera space
    #     # Transform the direction vector to world space
    #     world_direction = rotation_matrix @ direction_vector
    #     # Plot the orientation as an arrow
    #     ax.quiver(pos[0], pos[1], pos[2], world_direction[0], world_direction[1], world_direction[2], color='b', length=0.1)

    # # Set labels and legend
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # plt.title("3D Camera Poses")
    # plt.show()


