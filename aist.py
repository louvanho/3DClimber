import torch
import roma
import cv2
import pickle as pkl
import numpy as np
import pickle 
import os
import smplx
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

from tqdm import tqdm
from argparse import ArgumentParser
from smplfitter.pt.converter import SMPLConverter

from premiere.functionsCommon import projectPoints3dTo2d, keypointsBBox3D, keypointsBBox2D
from premiere.functionsSMPLX import updateHumanFromSMPLXForAIST

# Import the Rotation class for Euler angle conversion
from scipy.spatial.transform import Rotation as R

os.environ['DATA_ROOT'] = os.path.join(os.environ["MODELS_PATH"], 'smplfitter')


def set_axes_equal(ax):
    """
    Set equal scaling for all axes so that the coordinate axes are orthonormal.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_pkl", type=str, default='')
    parser.add_argument("--input_j3d_pkl", type=str, default='')
    parser.add_argument("--output_pkl", type=str, default='')
    parser.add_argument("--camera_position", type=str, default='')
    parser.add_argument("--camera_rotation", type=str, default='')

    args = parser.parse_args()

    input_pkl = args.input_pkl
    # input_j3d_pkl = args.input_j3d_pkl
    # output_pkl = args.output_pkl
    camera_position = args.camera_position
    camera_rotation = args.camera_rotation

    camera_position = camera_position.split(' ')
    camera_position = [float(i) for i in camera_position]
    camera_position = np.array(camera_position)

    # camera_rotation = camera_rotation.split(' ')
    # camera_rotation = [float(i) for i in camera_rotation]
    # camera_rotation = np.array(camera_rotation)

    # Parse camera rotation values from string
    camera_rotation_vals = [float(i) for i in camera_rotation.split(' ')]

    for (i, val) in enumerate(camera_rotation_vals):
        print ( val * 180.0 / math.pi )

    # Convert camera rotation:
    # If three numbers are provided, assume they are Euler angles (in radians, 'xyz' order).
    # If nine numbers are provided, reshape them into a 3x3 matrix.
    if len(camera_rotation_vals) == 3:
        r = R.from_euler('zyx', camera_rotation_vals)
        R_matrix = r.as_matrix()
        print("[INFO] Camera rotation converted from Euler angles to rotation matrix:\n", R_matrix)
    elif len(camera_rotation_vals) == 9:
        R_matrix = np.array(camera_rotation_vals).reshape(3, 3)
        print("[INFO] Camera rotation provided as a full rotation matrix.")
    else:
        print("[WARNING] Camera rotation does not have 3 or 9 elements; using identity matrix.")
        R_matrix = np.eye(3)


    print("Read AIST++ pkl:", input_pkl)
    with open(input_pkl, 'rb') as file:
        AISTPkl = pickle.load(file)
    print('[INFO] AIST++ pkl loaded')
    print(AISTPkl.keys())
    print(AISTPkl['smpl_scaling'])
    scaling = AISTPkl['smpl_scaling']

    camera_position /= scaling

    # print("Read AIST++ J3D pkl:", input_j3d_pkl)
    # with open(input_j3d_pkl, 'rb') as file:
    #     AISTPklJ3D = pickle.load(file)
    # print('[INFO] AIST++ J3D pkl loaded')
    # print(AISTPklJ3D.keys())

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # models_path = os.environ["MODELS_PATH"]

    # print('[INFO] Loading SMPL2SMPLX models')
    # smpl2smplx = SMPLConverter('smpl', 'neutral', 'smplx', 'neutral')
    # smpl2smplx.to(device)
    # smpl2smplx = torch.jit.script(smpl2smplx)  # optional: compile the converter for faster execution
    # print('[INFO] SMPL2SMPLX models loaded')

    # print('[INFO] Loading SMPLX model')
    # gender = 'neutral'
    # modelSMPLX = smplx.create(
    #     models_path, 'smplx',
    #     gender=gender,
    #     use_pca=False, flat_hand_mean=True,
    #     num_betas=10,
    #     ext='npz').cuda()
    # print('[INFO] SMPLX model loaded')

    # frames_count = len(AISTPkl['smpl_trans'])
        

    # -------------------------------------------------------------------------
    # Plot the camera properties in 3D using a perspective projection and show the view frustum
    # -------------------------------------------------------------------------
    fig = plt.figure()
    # If your version of Matplotlib supports it, you can enforce a perspective projection:
    ax = fig.add_subplot(111, projection='3d', proj_type='persp')
    
    # --- Plot World Coordinate Axes ---
    origin = np.array([0, 0, 0])
    axis_length = 1.0  # adjust as needed
    ax.quiver(origin[0], origin[1], origin[2], axis_length, 0, 0, color='r',
              arrow_length_ratio=0.1, label='World X')
    ax.quiver(origin[0], origin[1], origin[2], 0, axis_length, 0, color='g',
              arrow_length_ratio=0.1, label='World Y')
    ax.quiver(origin[0], origin[1], origin[2], 0, 0, axis_length, color='b',
              arrow_length_ratio=0.1, label='World Z')
    
    # --- Plot Camera Position ---
    ax.scatter(camera_position[0], camera_position[1], camera_position[2],
               color='k', s=50, label='Camera Position')
    
    # --- Plot Camera Orientation ---
    cam_axis_length = 1.0  # adjust for visibility
    cam_x = R_matrix[:, 0]
    cam_y = R_matrix[:, 1]
    cam_z = R_matrix[:, 2]
    
    ax.quiver(camera_position[0], camera_position[1], camera_position[2],
              cam_x[0], cam_x[1], cam_x[2],
              length=cam_axis_length, color='r', arrow_length_ratio=0.1,
              label='Camera X')
    ax.quiver(camera_position[0], camera_position[1], camera_position[2],
              cam_y[0], cam_y[1], cam_y[2],
              length=cam_axis_length, color='g', arrow_length_ratio=0.1,
              label='Camera Y')
    ax.quiver(camera_position[0], camera_position[1], camera_position[2],
              cam_z[0], cam_z[1], cam_z[2],
              length=cam_axis_length, color='b', arrow_length_ratio=0.1,
              label='Camera Z')
    
    # --- Draw the Camera Frustum ---
    # Define default frustum parameters
    fov = np.radians(72.2)  # field of view (in radians)
    aspect = 16/9         # aspect ratio
    near = 0.1            # near plane distance
    far = 1.0             # far plane distance

    # Compute dimensions of near and far planes
    h_near = near * np.tan(fov / 2)
    w_near = h_near * aspect

    h_far = far * np.tan(fov / 2)
    w_far = h_far * aspect

    # Define corners in the camera coordinate system (x: right, y: up, z: forward)
    near_top_left = np.array([-w_near,  h_near, near])
    near_top_right = np.array([ w_near,  h_near, near])
    near_bottom_right = np.array([ w_near, -h_near, near])
    near_bottom_left = np.array([-w_near, -h_near, near])

    far_top_left = np.array([-w_far,  h_far, far])
    far_top_right = np.array([ w_far,  h_far, far])
    far_bottom_right = np.array([ w_far, -h_far, far])
    far_bottom_left = np.array([-w_far, -h_far, far])

    # Function to transform a point from camera coordinates to world coordinates
    def transform_point(p):
        return camera_position + R_matrix @ p

    ntl = transform_point(near_top_left)
    ntr = transform_point(near_top_right)
    nbr = transform_point(near_bottom_right)
    nbl = transform_point(near_bottom_left)

    ftl = transform_point(far_top_left)
    ftr = transform_point(far_top_right)
    fbr = transform_point(far_bottom_right)
    fbl = transform_point(far_bottom_left)

    # Draw lines from the camera center to the corners of the near plane
    ax.plot([camera_position[0], ntl[0]], [camera_position[1], ntl[1]], [camera_position[2], ntl[2]], color='k')
    ax.plot([camera_position[0], ntr[0]], [camera_position[1], ntr[1]], [camera_position[2], ntr[2]], color='k')
    ax.plot([camera_position[0], nbr[0]], [camera_position[1], nbr[1]], [camera_position[2], nbr[2]], color='k')
    ax.plot([camera_position[0], nbl[0]], [camera_position[1], nbl[1]], [camera_position[2], nbl[2]], color='k')

    # Draw the rectangle for the near plane
    ax.plot([ntl[0], ntr[0]], [ntl[1], ntr[1]], [ntl[2], ntr[2]], color='k')
    ax.plot([ntr[0], nbr[0]], [ntr[1], nbr[1]], [ntr[2], nbr[2]], color='k')
    ax.plot([nbr[0], nbl[0]], [nbr[1], nbl[1]], [nbr[2], nbl[2]], color='k')
    ax.plot([nbl[0], ntl[0]], [nbl[1], ntl[1]], [nbl[2], ntl[2]], color='k')

    # Draw the rectangle for the far plane
    ax.plot([ftl[0], ftr[0]], [ftl[1], ftr[1]], [ftl[2], ftr[2]], color='k')
    ax.plot([ftr[0], fbr[0]], [ftr[1], fbr[1]], [ftr[2], fbr[2]], color='k')
    ax.plot([fbr[0], fbl[0]], [fbr[1], fbl[1]], [fbr[2], fbl[2]], color='k')
    ax.plot([fbl[0], ftl[0]], [fbl[1], ftl[1]], [fbl[2], ftl[2]], color='k')
    
    # --- Set Plot Labels, Title, and Limits ---
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Position, Orientation, and Perspective Projection')

    # Optionally, adjust the view limits to include all elements
    all_points = np.array([origin, camera_position, ntl, ntr, nbr, nbl, ftl, ftr, fbr, fbl])
    mins = all_points.min(axis=0) - 1
    maxs = all_points.max(axis=0) + 1
    ax.set_xlim([mins[0], maxs[0]])
    ax.set_ylim([mins[1], maxs[1]])
    ax.set_zlim([mins[2], maxs[2]])
    
    # Ensure the axes are equally scaled so the orthonormality is visually preserved
    set_axes_equal(ax)
    
    ax.legend()
    plt.show()
    
# python .\aist.py --input_pkl .\gBR_sBM_cAll_d04_mBR0_ch01.pkl --output_pkl .\result.pkl --input_j3d_pkl .\gBR_sBM_cAll_d04_mBR0_ch01_j3d.pkl --camera_position "-3.104077194098298 182.54559013217388 453.39325012127074" --camera_rotation "3.119905637132615 0.00793503727830112 -0.024953662352903517"
# python .\aist.py --input_pkl .\gBR_sBM_cAll_d04_mBR0_ch01.pkl --output_pkl .\result2.pkl --input_j3d_pkl .\gBR_sBM_cAll_d04_mBR0_ch01_j3d.pkl --camera_position "-0.8424770995171424 184.24424242336184 475.88675657738145" --camera_rotation "2.8843352355123546 0.010369166642977863 1.1906199150206112"