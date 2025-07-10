import pickle
import sys
import cv2
import os
import torch
import roma
import smplx
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from premiere.functionsSMPLX import updateHumanFromSMPLX
from premiere.functionsCommon import projectPoints3dTo2d

# -------------- WiLoR-specific imports (as in your code) --------------
from wilor.models import WiLoR, load_wilor
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils import recursive_to

# ----------------------------------------------------------------------
# 1) Helper functions
# ----------------------------------------------------------------------

def project_full_img(points, cam_trans, focal_length, img_res):
    """
    Projects 3D points (camera coords) into the 2D image coordinate system.
    """
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3)
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = camera_center[0]
    K[1,2] = camera_center[1]

    points = points + cam_trans
    points = points / points[..., -1:]  # perspective divide
    V_2d = (K @ points.T).T
    return V_2d[..., :-1]

def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    """
    Convert bounding-box-based camera parameters to full-image parameters.
    """
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2., img_h / 2.

    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam

def compute_bounding_box(selected_points, frame_h, frame_w , increaseSize=0.2):
    """
    Compute an expanded bounding box around the selected 2D points,
    clamped to the image boundaries.
    """
    x_coords = selected_points[:, 0]
    y_coords = selected_points[:, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    # Expand bounding box by 'increaseSize' (e.g. 20%)
    w = max_x - min_x
    h = max_y - min_y
    min_x -= increaseSize * w
    max_x += increaseSize * w
    min_y -= increaseSize * h
    max_y += increaseSize * h

    # Clamp
    min_x = max(0, min_x)
    max_x = min(frame_w, max_x)
    min_y = max(0, min_y)
    max_y = min(frame_h, max_y)

    return [min_x, min_y, max_x, max_y]

def rotmat_to_axis_angle(rotmats):
    """
    Convert rotation matrices (B, 3, 3) to axis-angle (B, 3).
    Uses 'roma' library for convenience.
    """
    # If input is (15, 3, 3), output is (15, 3)
    return roma.rotmat_to_rotvec(rotmats)

def extract_local_finger_pose(global_hand_rotmat, finger_rotmats):
    """
    Given a global/wrist rotation (in camera coords) and finger_rotmats (local or global),
    return local finger axis-angles (15 * 3 = 45) that SMPL-X expects.
    
    If 'finger_rotmats' are already local to the wrist, we can directly convert them.
    If they're global, we'd do: local = global_hand_rotmat^T @ finger_rotmats.
    Then convert local to axis-angle.

    For simplicity, assume 'finger_rotmats' is already local to the wrist.
    """
    # finger_rotmats: shape (15, 3, 3)
    # Convert to axis-angle per finger
    finger_axisangles = rotmat_to_axis_angle(finger_rotmats)  # (15, 3)
    # Flatten to (1, 45)
    finger_axisangles = finger_axisangles.view(1, -1)
    return finger_axisangles


# ----------------------------------------------------------------------
# 2) Main script
# ----------------------------------------------------------------------

if len(sys.argv) != 5:
    print("Usage: python injectHandsPkl.py <input_pkl> <video> <output_pkl> <display: 0 or 1>")
    sys.exit(1)

input_pkl_path = sys.argv[1]
video_path     = sys.argv[2]
output_pkl_path = sys.argv[3]
display       = (int(sys.argv[4]) == 1)

with open(input_pkl_path, 'rb') as f:
    dataPKL = pickle.load(f)

allFrameHumans = dataPKL['allFrameHumans']

video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print('[!] Erreur lors de l\'ouverture de la vidéo')
    sys.exit(1)

frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
video_width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(video.get(cv2.CAP_PROP_FPS))

# WiLoR model loading
models_path  = os.environ["MODELS_PATH"]
model_name   = os.path.join(models_path, 'wilor', 'wilor_final.ckpt')
model_config = os.path.join(models_path, 'wilor', 'model_config.yaml')

print('[INFO] Loading WiLoR model...')
modelWilor, modelWilor_cfg = load_wilor(checkpoint_path=model_name, cfg_path=model_config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelWilor = modelWilor.to(device)
modelWilor.eval()
print('[INFO] WiLoR model loaded')

print('[INFO] Loading SMPL-X model...')
gender = 'neutral'
modelSMPLX = smplx.create(
    models_path, 'smplx',
    gender=gender,
    use_pca=False, flat_hand_mean=True,
    num_betas=10,
    ext='npz').cuda()
print('[INFO] SMPL-X model loaded')

useExpression = True
if (dataPKL['model_type'] == "hmr2") or (dataPKL['model_type'] == "nlf"):
    useExpression = False

print('[+] Traitement de la vidéo...\n')
pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)

increaseSize = .2
count = 0

while video.isOpened():
    ret, frame = video.read()
    if not ret or count >= len(allFrameHumans):
        break

    humans = allFrameHumans[count]
    bboxes = []
    isRight = []

    for human in humans:
        points2D = human['j2d_smplx']

        # (Optional) global_orient of the body from SMPL-X (index 0 in rotvec), etc.
        # but we mainly care about the hand segments now.

        # Compute bounding box for left hand
        selected_points_left = np.vstack([points2D[25:40], points2D[20]])  # left hand + wrist
        bbox_left = compute_bounding_box(selected_points_left, frame.shape[0], frame.shape[1], increaseSize) 
        bboxes.append(bbox_left)
        isRight.append(False)

        # Compute bounding box for right hand
        selected_points_right = np.vstack([points2D[40:55], points2D[21]])  # right hand + wrist
        bbox_right = compute_bounding_box(selected_points_right, frame.shape[0], frame.shape[1], increaseSize) 
        bboxes.append(bbox_right)
        isRight.append(True)

        boxes = np.stack(bboxes)
        right = np.stack(isRight)

        dataset = ViTDetDataset(modelWilor_cfg, frame, boxes, right, rescale_factor=1.0)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

        allHandsPose = []  # will store the local finger axis-angles for each hand

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = modelWilor(batch)

            # Adjust sign for right-hand bounding boxes
            multiplier = (2 * batch['right'] - 1)
            pred_cam   = out['pred_cam']
            pred_cam[:,1] = multiplier * pred_cam[:,1]

            box_center = batch["box_center"].float()
            box_size   = batch["box_size"].float()
            img_size   = batch["img_size"].float()
            scaled_focal_length = (
                modelWilor_cfg.EXTRA.FOCAL_LENGTH / modelWilor_cfg.MODEL.IMAGE_SIZE 
                * img_size.max()
            )
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).detach().cpu().numpy()

            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Global-orient of the hand (3x3)
                hand_global_rotmat = out['pred_mano_params']['global_orient'][n]
                # Finger rotations (15 x 3 x 3) or (45, 3) depending on WiLoR config
                finger_rotmats = out['pred_mano_params']['hand_pose'][n]

                # Convert to local finger axis-angles
                local_finger_aa = extract_local_finger_pose(hand_global_rotmat, finger_rotmats)
                # shape: (1, 45)

                # Store in allHandsPose
                allHandsPose.append(local_finger_aa.detach().cpu().numpy())

                # If "display" is on, project & draw
                if display:
                    joints3d = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                    is_right = batch['right'][n].cpu().numpy()
                    # Flip x if right hand
                    joints3d[:,0] = (2*is_right - 1)*joints3d[:,0]
                    cam_t = pred_cam_t_full[n]
                    joints2d = project_full_img(
                        torch.tensor(joints3d), 
                        torch.tensor(cam_t), 
                        scaled_focal_length, 
                        img_size[n]
                    )
                    joints2d = joints2d.detach().cpu().numpy()

                    color = (0,255,0) if is_right else (255,0,0)
                    for p in joints2d:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 1, color, -1)

        # Now we place these finger angles back into the SMPL-X pose
        # 'pose' is the full axis-angle array for SMPL-X (55 x 3) or flattened (165).
        # Let's say we have [0..21 body joints, 25..39 left hand, 40..54 right hand]
        # For each human, we do:
        pose = human['rotvec']  # shape (55, 3) or similar

        # We expect allHandsPose = [leftHand, rightHand] if everything matched
        if len(allHandsPose) == 2:
            left_hand_pose  = allHandsPose[0][0]  # shape (45,)
            right_hand_pose = allHandsPose[1][0]  # shape (45,)

            # Indices in 'pose' for left hand typically 25..39
            # Indices in 'pose' for right hand typically 40..54
            # Insert left hand
            if useExpression:
                left_start = 23
            else:
                left_start = 25
            for i in range(15):
                # Each finger joint is 3 axis-angle components
                # left_hand_pose[i*3:(i+1)*3] is (3,) 
                pose[left_start + i] = -left_hand_pose[i*3:(i+1)*3]  # as in your code

            # Insert right hand
            if useExpression:
                right_start = 37
            else:
                right_start = 40
            for i in range(15):
                pose[right_start + i] = right_hand_pose[i*3:(i+1)*3]

    # Optional: show frame
    if display:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pbar.update(1)
    count += 1

pbar.close()
video.release()
if display:
    cv2.destroyAllWindows()

print ("Update SMPLX")
pbar = tqdm(total=len(allFrameHumans), unit=' frames', dynamic_ncols=True, position=0, leave=True)
for i in range(len(allFrameHumans)):
    for j in range(len(allFrameHumans[i])):
        allFrameHumans[i][j] = updateHumanFromSMPLX(allFrameHumans[i][j], modelSMPLX, useExpression)
        proj_2d = projectPoints3dTo2d( allFrameHumans[i][j]['j3d_smplx'], fov=dataPKL['fov_x_deg'], width=video_width, height=video_height )
        human['j2d_smplx'] = proj_2d
    pbar.update(1)
pbar.close()

# Finally, save the new dataPKL
with open(output_pkl_path, 'wb') as handle:
    pickle.dump(dataPKL, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f'[INFO] Hand poses injected and saved to {output_pkl_path}')
