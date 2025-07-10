import torch
import roma
import cv2
import pickle as pkl
import numpy as np
import pickle 
import os
import smplx
import math
import json

from tqdm import tqdm
from argparse import ArgumentParser
from smplfitter.pt.converter import SMPLConverter

from premiere.functionsCommon import projectPoints3dTo2d, keypointsBBox3D, keypointsBBox2D
from premiere.functionsSMPLX import updateHumanFromSMPLXForAIST

os.environ['DATA_ROOT'] = os.path.join(os.environ["MODELS_PATH"], 'smplfitter')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_pkl", type=str, default='')
    # parser.add_argument("--input_j3d_pkl", type=str, default='')
    parser.add_argument("--output_pkl", type=str, default='')
    parser.add_argument("--cam_settings", type=str, default='setting1')
    parser.add_argument("--cam_from", type=str, default='c02')
    parser.add_argument("--cam_to", type=str, default='c01')
    # parser.add_argument("--camera1_position", type=str, default='')
    # parser.add_argument("--camera1_rotation", type=str, default='')
    # parser.add_argument("--camera2_position", type=str, default='')
    # parser.add_argument("--camera2_rotation", type=str, default='')

    args = parser.parse_args()

    input_pkl = args.input_pkl
    output_pkl = args.output_pkl
    cam_settings = args.cam_settings
    cam_from = args.cam_from
    cam_to = args.cam_to

    # camera1_position = args.camera1_position
    # camera1_rotation = args.camera1_rotation
    # camera2_position = args.camera2_position
    # camera2_rotation = args.camera2_rotation

    # camera1_position = camera1_position.split(' ')
    # camera1_position = [float(i) for i in camera1_position]
    # camera1_position = np.array(camera1_position)

    # camera1_rotation = camera1_rotation.split(' ')
    # camera1_rotation = [float(i) for i in camera1_rotation]
    # camera1_rotation = np.array(camera1_rotation)

    # camera2_position = camera2_position.split(' ')
    # camera2_position = [float(i) for i in camera2_position]
    # camera2_position = np.array(camera2_position)

    # camera2_rotation = camera2_rotation.split(' ')
    # camera2_rotation = [float(i) for i in camera2_rotation]
    # camera2_rotation = np.array(camera2_rotation)

    camera_file= "/home/vl10550y/Desktop/3DClimber/Datasets/aist_plusplus_final/cameras/" + cam_settings + ".json"
    with open(camera_file) as f:
        data = json.load(f)
        for i in range(len(data)):
            if data[i]['name'] == cam_from:
                camera1_position = np.array(data[i]['translation'])
                camera1_rotation = np.array(data[i]['rotation'])
            if data[i]['name'] == cam_to:
                camera2_position = np.array(data[i]['translation'])
                camera2_rotation = np.array(data[i]['rotation'])
    
    print ("camera1_rotation",[math.degrees(camera1_rotation[i]) for i in range(3)])
    print ("camera2_rotation",[math.degrees(camera2_rotation[i]) for i in range(3)])

    print("Read input pkl:", input_pkl)
    with open(input_pkl, 'rb') as file:
        AISTPkl = pickle.load(file)
    print('[INFO] input pkl loaded')
    print(AISTPkl.keys())
    print(AISTPkl['allFrameHumans'][0][0].keys())
    # print(AISTPkl['smpl_scaling'])
    # scaling = AISTPkl['smpl_scaling']
    scaling = 93.77886
    camera1_position /= scaling
    camera2_position /= scaling

    print ("camera1_position", camera1_position)
    print ("camera2_position", camera2_position)

    # print("Read AIST++ J3D pkl:", input_j3d_pkl)
    # with open(input_j3d_pkl, 'rb') as file:
    #     AISTPklJ3D = pickle.load(file)
    # print('[INFO] AIST++ J3D pkl loaded')
    # print(AISTPklJ3D.keys())

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    models_path = os.environ["MODELS_PATH"]

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

    frames_count = len(AISTPkl['allFrameHumans'])
    # frames_count = len(AISTPkl['smpl_trans'])
    

    allFramesHumans = []

    pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)
    betas_tensor = torch.zeros(1, 10, dtype=torch.float32).to(device)

    for i in range(frames_count):
        allHumans = []

    #     camera = AISTPklJ3D['keypoints3d'][i][0]/scaling

    #     cam_tensor = torch.tensor(camera, dtype=torch.float32).reshape(1, 3).to(device)

    #     full_pose_rot_vec_tensor = torch.tensor(AISTPkl['smpl_poses'][i], dtype=torch.float32).reshape(1, 24, 3).to(device)        

    #     # Call the convert function with tensors
    #     smplx_out = smpl2smplx.convert(full_pose_rot_vec_tensor, betas_tensor, cam_tensor)
        
    #     human = {}
    #     human = updateHumanFromSMPLXForAIST(human, modelSMPLX, smplx_out['pose_rotvecs'], smplx_out['shape_betas'], smplx_out['trans'], 1.0, 1.0, 1.0, 0.0)
        human = AISTPkl['allFrameHumans'][i][0]

        R_np1 , _ = cv2.Rodrigues(camera1_rotation)
        R_np2 , _ = cv2.Rodrigues(camera2_rotation)
        # human['j3d_smplx'] = (R_np @ (human['j3d_smplx'] - pivot).T).T + pivot + camera_position


        human['j3d_smplx'] = (R_np2 @ np.linalg.inv(R_np1) @ (human['j3d_smplx'] - camera1_position).T).T + camera2_position

        global_rot_vec = human['rotvec'][0]
        global_rot_mat, _ = cv2.Rodrigues(global_rot_vec)
        corrected_global_rot_mat = R_np2 @ np.linalg.inv(R_np1) @ global_rot_mat
        corrected_global_rot_vec, _ = cv2.Rodrigues(corrected_global_rot_mat)
        human['rotvec'][0] = corrected_global_rot_vec.reshape(3)

    #     # Extract the global rotation vector (shape: [1, 3])
    #     global_rot_vec = smplx_out['pose_rotvecs'][:, :3]

    #     # Convert the tensor to numpy and squeeze to shape (3,)
    #     global_rot_vec_np = global_rot_vec.cpu().numpy().squeeze(0)

    #     # Convert the axis–angle vector to a rotation matrix using cv2.Rodrigues
    #     global_rot_mat, _ = cv2.Rodrigues(global_rot_vec_np)

    #     # Apply the correction by pre-multiplying with your rotation matrix
    #     corrected_global_rot_mat = R_np @ global_rot_mat

    #     # Convert the corrected rotation matrix back to an axis–angle vector
    #     corrected_global_rot_vec, _ = cv2.Rodrigues(corrected_global_rot_mat)

    #     # Update the SMPLX pose parameters with the corrected global rotation.
    #     # Make sure to reshape to (1, 3) if necessary.
    #     smplx_out['pose_rotvecs'][:, :3] = torch.tensor(corrected_global_rot_vec.reshape(1, 3), 
    #                                                     dtype=torch.float32, device=device)

    #     human['rotvec'] = smplx_out['pose_rotvecs'].detach().cpu().numpy().reshape(55, 3)
        
        points_2d = projectPoints3dTo2d ( human['j3d_smplx'], 72.2, 1920, 1080)
        human['j2d_smplx'] = points_2d
        # human['id']=0
        # human['score'] = 1.0
        # Compute the 3D bounding box from keypoints
        min_coords, max_coords = keypointsBBox3D(human['j3d_smplx'])
        human['bbox3d'] = [min_coords, max_coords]
        min_coords, max_coords = keypointsBBox2D(human['j2d_smplx'])
        human['bbox'] = [min_coords, max_coords]
        # human['trans'] = AISTPkl['smpl_trans'][i]/scaling
        human['transl'] = human['j3d_smplx'][15]
        allHumans.append(human)
        allFramesHumans.append(allHumans)
        pbar.update(1)
    pbar.close()

    AISTPkl['allFrameHumans'] = allFramesHumans
    with open(output_pkl, 'wb') as handle:
        pickle.dump(AISTPkl, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # allData = {}
    # allData['allFrameHumans'] = allFramesHumans
    # allData['model_name'] = 'AIST++'
    # allData['model_type'] = -1
    # allData['video'] = 'aistVideo'
    # allData['fov_x_deg'] = 72.2 
    # allData['video_width'] = 1920
    # allData['video_height'] = 1080
    # allData['video_fps'] = 60
    
    # with open(output_pkl, 'wb') as handle:
    #     pickle.dump(allData, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Exemples d'exécution :
# python .\convertAIST.py --input_pkl .\gBR_sBM_cAll_d04_mBR0_ch01.pkl --output_pkl .\result.pkl --input_j3d_pkl .\gBR_sBM_cAll_d04_mBR0_ch01_j3d.pkl --camera_position "-3.104077194098298 182.54559013217388 453.39325012127074" --camera_rotation "3.119905637132615 0.00793503727830112 -0.024953662352903517"
# python .\convertAIST.py --input_pkl .\gBR_sBM_cAll_d04_mBR0_ch01.pkl --output_pkl .\result2.pkl --input_j3d_pkl .\gBR_sBM_cAll_d04_mBR0_ch01_j3d.pkl --camera_position "-0.8424770995171424 184.24424242336184 475.88675657738145" --camera_rotation "2.8843352355123546 0.010369166642977863 1.1906199150206112"
