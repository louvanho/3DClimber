import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from body import HumanBody
from argparse import ArgumentParser
import pickle
import json
import cv2

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

def project_pose(points_3d, camera):
    """
    Project 3D points onto the 2D image plane using camera parameters.
    """
    intrinsic_matrix = torch.tensor(camera['matrix'], dtype=torch.float32, device=points_3d.device)
    rotation_vector = torch.tensor(camera['rotation'], dtype=torch.float32, device=points_3d.device)
    translation_vector = torch.tensor(camera['translation'], dtype=torch.float32, device=points_3d.device)/93.77886

    # Convert rotation vector to rotation matrix
    R_np, _ = cv2.Rodrigues(rotation_vector.cpu().numpy())
    rotation_matrix = torch.tensor(R_np, dtype=torch.float32, device=points_3d.device)

    # Transform 3D points to camera coordinates
    points_3d_camera = torch.matmul(points_3d, rotation_matrix) + translation_vector

    # Project to 2D
    points_2d_homogeneous = torch.matmul(points_3d_camera, intrinsic_matrix.T)
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]
    return points_2d

def optimize_pose(initpose, joints_2d, cameras, lr=0.01, iterations=100):
    """
    Optimize the 3D pose to match the 2D joint locations in all views.
    """
    pose3d = initpose.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([pose3d], lr=lr)

    for _ in range(iterations):
        optimizer.zero_grad()
        loss = 0
        for view, camera in enumerate(cameras):
            projected_points = project_pose(pose3d, camera)
            joints_2d_tensor = torch.tensor(joints_2d[view], dtype=torch.float32, device=pose3d.device)
            loss += torch.norm(projected_points - joints_2d_tensor, dim=1).mean()
        
            # visualize_projection(project_pose(torch.tensor(pose3d), camera),joints_2d[view],view)
        # print("Loss: ", loss.item())
        loss.backward()
        optimizer.step()

    return pose3d

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
        print('Loading: ', pkl)
        cam_nr = pkl[-7:-4]
        print('Camera nr: ', cam_nr)
        cam_nrs.append(cam_nr)
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
            allData.append(data)
            if cam_nr == 'c01':
                file = data

    cameras = []
    # camera_file = f"/home/vl10550y/Desktop/3DClimber/Datasets/aist_plusplus_final/cameras/{args.camera_setting_file}.json"
    camera_file = args.camera_setting_file
    with open(camera_file) as f:
        data = json.load(f)
        for i in cam_nrs:
            for cam in data:
                if cam['name'] == i:
                    cameras.append(cam)
                    break

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    body = HumanBody()

    pbar = tqdm(total=len(file['allFrameHumans']), unit=' frames', dynamic_ncols=True, position=0, leave=True)

    for framenr, frame in enumerate(file['allFrameHumans']):
        joints_2d = []
        for view in range(len(allData)):
            joints_2d.append(allData[view]['allFrameHumans'][framenr][0]['j2d_smplx'])
            if view == 0:
                initpose = torch.tensor(allData[view]['allFrameHumans'][framenr][0]['j3d_smplx'], device=device)

                
        # Apply inverse translation and rotation for camera 1
        camera_1 = cameras[0]
        rotation_vector = torch.tensor(camera_1['rotation'], dtype=torch.float32, device=device)
        translation_vector = torch.tensor(camera_1['translation'], dtype=torch.float32, device=device)/93.77886

        # Convert rotation vector to rotation matrix
        R_np, _ = cv2.Rodrigues(rotation_vector.cpu().numpy())
        rotation_matrix = torch.tensor(R_np, dtype=torch.float32, device=device)

        # Apply inverse rotation and translation
        initpose = torch.matmul(initpose - translation_vector, rotation_matrix.T)



        # Optimize the 3D pose
        pose3d = optimize_pose(initpose, joints_2d, cameras)

        # # Visualize the projections
        # for view in range(len(allData)):
        #     visualize_projection(project_pose(initpose, cameras[view]),joints_2d[view],view)
        #     visualize_projection(project_pose(torch.tensor(pose3d), cameras[view]),joints_2d[view],view)
        # visualize_3D(pose3d.detach().cpu().numpy(), initpose.detach().cpu().numpy(), body)

        # Re-Apply the rotation and translation for camera 1 to the final result
        pose3d = torch.matmul(pose3d, rotation_matrix) + translation_vector



        # Save the optimized pose
        pose3d = pose3d.detach().cpu().numpy()
        file['allFrameHumans'][framenr][0]['j3d_smplx'] = pose3d
        pbar.update(1)

    pbar.close()

    # Save the final output
    with open(args.out_pkl, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)