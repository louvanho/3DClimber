import torch
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser 
import pickle
import json
from tqdm import tqdm
from body import HumanBody
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2


def get_loc_from_cube_idx(grid, pose3d_as_cube_idx):
    """
    Estimate 3d joint locations from cube index.

    Args:
        grid: a list of grids
        pose3d_as_cube_idx: a list of tuples (joint_idx, cube_idx)
    Returns:
        pose3d: 3d pose
    """
    njoints = len(pose3d_as_cube_idx)
    pose3d = torch.zeros(njoints, 3, device=grid[0].device)
    single_grid = len(grid) == 1
    for joint_idx, cube_idx in pose3d_as_cube_idx:
        gridid = 0 if single_grid else joint_idx
        if cube_idx == -1:  # Fallback to grid center
            pose3d[joint_idx] = grid[gridid].mean(dim=0)
        else:
            pose3d[joint_idx] = grid[gridid][cube_idx]
    return pose3d

def compute_grid(boxSize, boxCenter, nBins, device=None):
    grid1D = torch.linspace(-boxSize / 2, boxSize / 2, nBins, device=device)
    gridx, gridy, gridz = torch.meshgrid(
        grid1D + boxCenter[0],
        grid1D + boxCenter[1],
        grid1D + boxCenter[2],
    )
    gridx = gridx.contiguous().view(-1, 1)
    gridy = gridy.contiguous().view(-1, 1)
    gridz = gridz.contiguous().view(-1, 1)
    grid = torch.cat([gridx, gridy, gridz], dim=1)
    return grid


def pdist2(x, y):
    """
    Compute distance between each pair of row vectors in x and y

    Args:
        x: tensor of shape n*p
        y: tensor of shape m*p
    Returns:
        dist: tensor of shape n*m
    """
    p = x.shape[1]
    n = x.shape[0]
    m = y.shape[0]
    xtile = torch.cat([x] * m, dim=1).view(-1, p)
    ytile = torch.cat([y] * n, dim=0)
    dist = torch.pairwise_distance(xtile, ytile)
    return dist.view(n, m)


def compute_pairwise(skeleton, limb_length, grid, tolerance):

    pairwise = {}
    for node in skeleton:
        current = node['idx']
        children = node['children']
        for child in children:
            expect_length = limb_length[(current, child)]
            distance = pdist2(grid[current], grid[child]) + 1e-9
            pairwise[(current, child)] = (torch.abs(distance - expect_length) <
                                          tolerance).float()
    return pairwise

def project_pose(points_3d, camera):
    """
    Project 3D points onto the 2D image plane using camera parameters.

    Args:
        points_3d: Tensor of shape (N, 3), where N is the number of 3D points.
        camera: Dictionary containing camera parameters:
            - 'matrix': Intrinsic matrix (3x3)
            - 'rotation': Rotation vector (3,)
            - 'translation': Translation vector (3,)
            - 'distortions': Distortion coefficients (5,)

    Returns:
        points_2d: Tensor of shape (N, 2), the projected 2D points.
    """
    if not isinstance(points_3d, torch.Tensor):
        points_3d = torch.tensor(points_3d, dtype=torch.float32)
    device = points_3d.device

    # Extract camera parameters
    intrinsic_matrix = torch.tensor(camera['matrix'], dtype=torch.float32, device=device)  # (3, 3)
    rotation_vector = torch.tensor(camera['rotation'], dtype=torch.float32, device=device)  # (3,)
    translation_vector = torch.tensor(camera['translation'], dtype=torch.float32, device=device)/93.77886  # (3,)
    distortions = torch.tensor(camera['distortions'], dtype=torch.float32, device=device)  # (5,)

    # # Apply scaling
    # intrinsic_matrix[0, 0] /= 93  # fx
    # intrinsic_matrix[1, 1] /= 93  # fy
    # intrinsic_matrix[0, 2] /= 93  # cx
    # intrinsic_matrix[1, 2] /= 93  # cy
    # translation_vector /= 93  # translation
    

    # # Step 1: Convert rotation vector to rotation matrix
    # theta = torch.norm(rotation_vector)
    # if theta > 0:
    #     k = rotation_vector / theta
    #     K = torch.tensor([
    #         [0, -k[2], k[1]],
    #         [k[2], 0, -k[0]],
    #         [-k[1], k[0], 0]
    #     ], dtype=torch.float32, device=device)
    #     rotation_matrix = torch.eye(3, device=device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)
    # else:
    #     rotation_matrix = torch.eye(3, device=device)

    # # Step 2: Transform 3D points to the camera coordinate system
    # points_3d_camera = torch.matmul(points_3d, rotation_matrix.T) + translation_vector

    R_np , _ = cv2.Rodrigues(rotation_vector.cpu().numpy())
    R_np = torch.tensor(R_np, dtype=torch.float32, device=device)  # (3, 3)
    
    # print(f"View {view}: Rotation matrix =\n{R_np}")
    # print(f"View {view}: Translation vector = {translation_vector}")

    points_3d_camera = torch.matmul(points_3d, R_np.T) + translation_vector

    # Step 3: Apply intrinsic matrix to project onto the image plane
    points_2d_homogeneous = torch.matmul(points_3d_camera, intrinsic_matrix.T)  # (N, 3)
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]  # Normalize by depth

    # print(f"3D points (camera coordinates): {points_3d_camera}")
    # print(f"Intrinsic matrix: {intrinsic_matrix}")
    # print(f"Projected points (2D): {points_2d}")

    # # Step 4: Apply radial distortion (if needed)
    # if distortions.numel() > 0:
    #     k1, k2, p1, p2, k3 = distortions
    #     x = points_2d[:, 0]
    #     y = points_2d[:, 1]
    #     r2 = x**2 + y**2
    #     radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    #     x_distorted = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    #     y_distorted = y * radial + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
    #     points_2d = torch.stack([x_distorted, y_distorted], dim=1)

    return points_2d

def compute_unary_from_2d(grid, joint_2d_views, cameras):
    """
    Compute unary terms for a single joint based on its 2D locations in all views.

    Args:
        grid: 3D grid of possible joint locations (nbins^3, 3)
        joint_2d_views: List of 2D joint locations for this joint across all views (n_views, 2)
        cameras: Camera parameters for each view (intrinsics, extrinsics)
    Returns:
        unary: Unary term for each grid point (nbins^3,)
    """
    n_views = len(joint_2d_views)
    unary = torch.zeros(grid.shape[0], device=grid.device)

    for v in range(n_views):
        # Project 3D grid points to 2D using camera parameters
        projected_points = project_pose(grid, cameras[v])

        # print(f"View {v}: Projected points = {projected_points}")
        # print(f"View {v}: 2D joint locations = {joints_2d[v]}")
        # visualize_projection(projected_points, joint_2d_views, v)

        # Compute distance between projected points and 2D joint locations
        joint_2d_tensor = torch.tensor(joint_2d_views[v], dtype=torch.float32, device=grid.device)
        distances = torch.norm(projected_points - joint_2d_tensor, dim=1)
        unary += torch.exp(-distances) # dists sum +-3000

    # for v in range(n_views):
    #     projected_points = project_pose(grid, cameras[v])
    #     print(f"View {v}: Projected points range = {projected_points.min(dim=0)[0]} to {projected_points.max(dim=0)[0]}")
    #     print(f"View {v}: 2D joint locations = {joint_2d_views[v]}")
    #     is_within_range = (
    #         joint_2d_tensor[0] >= projected_points.min(dim=0)[0][0]
    #         and joint_2d_tensor[0] <= projected_points.max(dim=0)[0][0]
    #         and joint_2d_tensor[1] >= projected_points.min(dim=0)[0][1]
    #         and joint_2d_tensor[1] <= projected_points.max(dim=0)[0][1]
    #     )
    #     print(f"View {v}: 2D joint within projected grid range: {is_within_range}")

    # print(f"View {v}: Unary term = {unary}")

    return unary

def infer(unary, pairwise, body):
    """
    Args:
        unary: [list] unary terms of all joints
        pairwise: [list] pairwise terms of all edges
        body: tree structure human body
    Returns:
        pose3d_as_cube_idx: 3d pose as cube index
    """
    root_idx = 0

    skeleton = body.skeleton
    skeleton_sorted_by_level = body.skeleton_sorted_by_level

    states_of_all_joints = {}
    for node in skeleton_sorted_by_level:
        children_state = []
        u = unary[node['idx']].clone()
        if len(node['children']) == 0:
            energy = u
            children_state = [[-1]] * energy.numel()
        else:
            for child in node['children']:
                pw = pairwise[(node['idx'], child)]
                ce = states_of_all_joints[child]['Energy']
                ce = ce.expand_as(pw)
                pwce = torch.mul(pw, ce)
                max_v, max_i = torch.max(pwce, dim=1)
                u = torch.mul(u, max_v)
                children_state.append(max_i.detach().cpu().numpy())

            children_state = np.array(children_state).T

        res = {'Energy': u, 'State': children_state}
        states_of_all_joints[node['idx']] = res

    pose3d_as_cube_idx = []
    energy = states_of_all_joints[root_idx]['Energy'].detach().cpu().numpy()
    cube_idx = np.argmax(energy)
    pose3d_as_cube_idx.append([root_idx, cube_idx])

    queue = pose3d_as_cube_idx.copy()
    while queue:
        joint_idx, cube_idx = queue.pop(0)
        children_state = states_of_all_joints[joint_idx]['State']
        state = children_state[cube_idx]

        children_index = skeleton[joint_idx]['children']
        if -1 not in state:
            for joint_idx, cube_idx in zip(children_index, state):
                pose3d_as_cube_idx.append([joint_idx, cube_idx])
                queue.append([joint_idx, cube_idx])

    pose3d_as_cube_idx.sort()
    # print(f"Selected cube indices: {pose3d_as_cube_idx}")
    return pose3d_as_cube_idx

def recursive_infer(initpose, cameras, joints_2d, body, limb_length, grid_size, nbins, tolerance):
    """
    Perform a single refinement step for 3D pose estimation.

    Args:
        initpose: Current 3D pose estimate (n_joints, 3)
        cameras: Camera parameters for each view
        joints_2d: List of 2D joint locations for each view (n_views, n_joints, 2)
        skeleton: Skeleton structure (list of nodes with children)
        limb_length: Expected limb lengths (dict of (parent, child) -> length)
        grid_size: Size of the 3D grid (float)
        nbins: Number of bins in the grid (int)
        tolerance: Tolerance for limb length constraints (float)
        config: Configuration dictionary
    Returns:
        pose3d: Refined 3D joint locations (n_joints, 3)
    """
    device = initpose.device
    njoints = initpose.shape[0]

    # Step 1: Generate a new grid around the current pose
    grids = []
    for j in range(njoints):
        grids.append(compute_grid(grid_size, initpose[j], nbins, device=device))

    # Step 2: Compute unary terms for the new grid
    unary = []
    for j in range(njoints):
        unary.append(compute_unary_from_2d(grids[j], [view[j] for view in joints_2d], cameras))

    # Step 3: Compute pairwise terms for the new grid
    pairwise = compute_pairwise(body.skeleton, limb_length, grids, tolerance)

    # Step 4: Perform inference
    # body = {'skeleton': skeleton, 'skeleton_sorted_by_level': skeleton}  # Simplify body structure
    pose3d_cube = infer(unary, pairwise, body)

    # Step 5: Convert cube indices to 3D locations
    pose3d = get_loc_from_cube_idx(grids, pose3d_cube)

    return pose3d

def fuse_2d_to_3d(joints_2d, cameras, body, limb_length, grid_size, fst_nbins, rec_nbins, tolerance, rec_depth, initpose):
    """
    Fuse 2D joint locations into a single 3D pose using iterative refinement.

    Args:
        joints_2d: List of 2D joint locations for each view (n_views, n_joints, 2)
        cameras: Camera parameters for each view
        skeleton: Skeleton structure (list of nodes with children)
        limb_length: Expected limb lengths (dict of (parent, child) -> length)
        grid_size: Initial size of the 3D grid (float)
        nbins: Number of bins in the grid (int)
        tolerance: Tolerance for limb length constraints (float)
        rec_depth: Number of recursive refinement steps (int)
    Returns:
        pose3d: Refined 3D joint locations (n_joints, 3)
    """
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    njoints = len(joints_2d[0])  # Number of joints

    # Step 1: Generate the initial coarse grid
    grids = []
    for j in range(njoints):
        # initial_estimate = torch.tensor([0.0, 0.0, 0.0], device=device)  # Replace with your estimate
        initial_estimate = torch.tensor(initpose[j], device=device)
        grids.append(compute_grid(grid_size, initial_estimate, fst_nbins, device=device))
        # # Plot the initial grid around the initial estimate in 3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(grids[j][:, 0].cpu(), grids[j][:, 1].cpu(), grids[j][:, 2].cpu(), label=f'Joint {j}')
        # ax.scatter(initial_estimate[0].cpu(), initial_estimate[1].cpu(), initial_estimate[2].cpu(), color='red', label=f'Initial Estimate {j}')
        # ax.set_title('Initial Grids for Each Joint')
        # ax.set_xlabel('X-axis')
        # ax.set_ylabel('Y-axis')
        # ax.set_zlabel('Z-axis')
        # ax.legend()
        # plt.show()
    # for j in range(njoints):
    #     print(f"Joint {j}: Grid center = {grids[j].mean(dim=0)}, Initpose = {initpose[j]}")

    # Step 2: Compute unary terms for the initial grid
    unary = []
    for j in range(njoints):
        unary.append(compute_unary_from_2d(grids[j], [view[j] for view in joints_2d], cameras))

    # Step 3: Compute pairwise terms for the initial grid
    skeleton = body.skeleton
    pairwise = compute_pairwise(skeleton, limb_length, grids, tolerance)

    # Step 4: Perform initial inference
    # config = {'DATASET': {'ROOTIDX': 0}}  # Replace with your root joint index
    pose3d_cube = infer(unary, pairwise, body)

    # Step 5: Convert cube indices to 3D locations
    pose3d = get_loc_from_cube_idx(grids, pose3d_cube)
    # print("Initial 3D pose:", pose3d)
    # visualize_3D(pose3d.detach().cpu().numpy(), initpose, body)  # Visualize the initial 3D pose

    # Step 6: Iterative refinement using recursive_infer
    cur_grid_size = grid_size / fst_nbins
    for _ in range(rec_depth):
        # visualize_projection(project_pose(pose3d,cameras[0]), joints_2d[0], _)  # Visualize the projection for the first view
        pose3d = recursive_infer(
            pose3d, cameras, joints_2d, body, limb_length,
            cur_grid_size, rec_nbins, tolerance
        )
        cur_grid_size /= rec_nbins

    return pose3d

def compute_limb_length(body, pose):
    limb_length = {}
    skeleton = body.skeleton
    for node in skeleton:
        idx = node['idx']
        children = node['children']

        for child in children:
            length = np.linalg.norm(pose[idx] - pose[child])
            limb_length[(idx, child)] = length
    return limb_length

def visualize_projection(projected_points, joints_2d, view):
    plt.scatter(projected_points[:, 0].cpu(), projected_points[:, 1].cpu(), label='Projected Points')
    joints_2d_array = np.array(joints_2d)  # Convert to NumPy array
    if joints_2d_array.shape == (2,):
        plt.scatter(joints_2d_array[0], joints_2d_array[1], label='2D Joints')
    else:
        plt.scatter(joints_2d_array[:, 0], joints_2d_array[:, 1], label='2D Joints')
    plt.title(f'View {view} - Projected Points vs 2D Joints')
    plt.legend()
    plt.show()

def visualize_3D(pose3d, initpose, body):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the refined 3D pose
    ax.scatter(pose3d[:, 0], pose3d[:, 1], pose3d[:, 2], label='Refined Pose', color='blue')
    
    # Plot the initial 3D pose
    ax.scatter(initpose[:, 0], initpose[:, 1], initpose[:, 2], label='Initial Pose', color='red', alpha=0.6)
    
    # Draw lines between connected joints for the refined pose
    for node in body.skeleton:
        parent_idx = node['idx']
        for child_idx in node['children']:
            ax.plot(
                [pose3d[parent_idx, 0], pose3d[child_idx, 0]],
                [pose3d[parent_idx, 1], pose3d[child_idx, 1]],
                [pose3d[parent_idx, 2], pose3d[child_idx, 2]],
                color='blue'
            )
    
    # Draw lines between connected joints for the initial pose
    for node in body.skeleton:
        parent_idx = node['idx']
        for child_idx in node['children']:
            ax.plot(
                [initpose[parent_idx, 0], initpose[child_idx, 0]],
                [initpose[parent_idx, 1], initpose[child_idx, 1]],
                [initpose[parent_idx, 2], initpose[child_idx, 2]],
                color='red',
                alpha=0.6
            )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Pose Estimation')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--pose_pkl_list", type=str, default='')
    parser.add_argument("-c", "--camera_setting_file", type=str, default='setting1')
    parser.add_argument("-o", "--out_pkl", type=str, default='')

    args = parser.parse_args()

    pkls = args.pose_pkl_list.split(',')
    cam_nrs = []
    allData = []
    for pkl in pkls:
        cam_nr = pkl.split('/')[-2]
        cam_nrs.append(cam_nr)
        print("Camera number:", cam_nr)
        print("Read pose pkl:", pkl)
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
            allData.append(data)
            # joints_2d.append(data['allFrameHumans'][0][0]['j2d_smplx'])
            if cam_nr == 'c01':
                file = data
                # initpose = data['allFrameHumans'][0][0]['j3d_smplx']

    cameras = []
    cam_settings = args.camera_setting_file
    camera_file = "/home/vl10550y/Desktop/3DClimber/Datasets/aist_plusplus_final/cameras/" + cam_settings + ".json"
    with open(camera_file) as f:
        data = json.load(f)
        for i in cam_nrs:
            for cam in data:
                if cam['name'] == i:
                    cameras.append(cam)
                    # print("Camera parameters:", cam)
                    break
    # print("Camera parameters:", cameras)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    grid_size = 0.5  # Initial grid size 500 250
    fst_nbins = 16   # Initial number of bins in each dimension
    rec_nbins = 2    # Number of bins for recursive refinement
    tolerance = 0.08   # Tolerance for limb length constraints in mm 100 2
    rec_depth = 20    # Number of recursive refinement steps
    body = HumanBody()

    pbar = tqdm(total=len(file['allFrameHumans']), unit=' frames', dynamic_ncols=True, position=0, leave=True)

    for framenr, frame in enumerate(file['allFrameHumans']):
        joints_2d = []
        for view in range(len(allData)):
            joints_2d.append(allData[view]['allFrameHumans'][framenr][0]['j2d_smplx'])
            if view == 0:
                initpose = allData[view]['allFrameHumans'][framenr][0]['j3d_smplx']

        # Apply inverse translation and rotation for camera 1
        camera_1 = cameras[0]
        rotation_vector = torch.tensor(camera_1['rotation'], dtype=torch.float32, device=device)
        translation_vector = torch.tensor(camera_1['translation'], dtype=torch.float32, device=device)/93.77886

        # Convert rotation vector to rotation matrix
        R_np, _ = cv2.Rodrigues(rotation_vector.cpu().numpy())
        rotation_matrix = torch.tensor(R_np, dtype=torch.float32, device=device)

        # Apply inverse rotation and translation
        initpose = torch.matmul(torch.tensor(initpose, dtype=torch.float32, device=device) - translation_vector, rotation_matrix).detach().cpu().numpy()


        limb_length = compute_limb_length(body, initpose)
        # print("Limb length:", limb_length)


        # Fuse 2D poses into a 3D pose with recursive refinement
        pose3d = fuse_2d_to_3d(joints_2d, cameras, body, limb_length, grid_size, fst_nbins, rec_nbins, tolerance, rec_depth, initpose) 

        # for view in range(len(allData)):
        #     visualize_projection(project_pose(initpose, cameras[view]),joints_2d[view],view)
        #     visualize_projection(project_pose(pose3d, cameras[view]),joints_2d[view],view)
        # # Plot the 3D pose
        # visualize_3D(pose3d.detach().cpu().numpy(), initpose, body)    

        # Re-Apply the rotation and translation for camera 1 to the final result
        pose3d = torch.matmul(pose3d, rotation_matrix.T) + translation_vector

        pose3d = pose3d.detach().cpu().numpy()
        # print("Final 3D pose:", pose3d)

        # Save the data to the output file
        file['allFrameHumans'][framenr][0]['j3d_smplx'] = pose3d
        pbar.update(1)
    pbar.close()

    # Save the final output
    with open(args.out_pkl, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)