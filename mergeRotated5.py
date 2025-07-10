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
    parser.add_argument("--ref_pkl", type=str, default='')
    parser.add_argument("--extra1_pkl", type=str, default='')
    parser.add_argument("--extra2_pkl", type=str, default='')
    parser.add_argument("--extra3_pkl", type=str, default='')
    parser.add_argument("--extra4_pkl", type=str, default='')
    parser.add_argument("--output_pkl", type=str, default='')
    parser.add_argument("--cam_settings", type=str, default='setting1')
    parser.add_argument("--cam_ref", type=str, default='c01')
    parser.add_argument("--cam_extra1", type=str, default='c02')
    parser.add_argument("--cam_extra2", type=str, default='c08')
    parser.add_argument("--cam_extra3", type=str, default='c04')
    parser.add_argument("--cam_extra4", type=str, default='c06')
    parser.add_argument("--naive", type=bool, default=False)


    # parser.add_argument("--cameraref_rotation", type=str, default='')
    # parser.add_argument("--camera1_rotation", type=str, default='')
    # parser.add_argument("--camera2_rotation", type=str, default='')

    args = parser.parse_args()

    ref_pkl = args.ref_pkl
    extra1_pkl = args.extra1_pkl
    extra2_pkl = args.extra2_pkl
    extra3_pkl = args.extra3_pkl
    extra4_pkl = args.extra4_pkl
    output_pkl = args.output_pkl
    cam_settings = args.cam_settings
    cam_ref = args.cam_ref
    cam_x1 = args.cam_extra1
    cam_x2 = args.cam_extra2
    cam_x3 = args.cam_extra3
    cam_x4 = args.cam_extra4
    naive = args.naive

    # cameraref_rotation = args.cameraref_rotation
    # camera1_rotation = args.camera1_rotation
    # camera2_rotation = args.camera2_rotation

    # cameraref_rotation = cameraref_rotation.split(' ')
    # cameraref_rotation = [float(i) for i in cameraref_rotation]
    # cameraref_rotation = np.array(cameraref_rotation)

    # camera1_rotation = camera1_rotation.split(' ')
    # camera1_rotation = [float(i) for i in camera1_rotation]
    # camera1_rotation = np.array(camera1_rotation)

    # camera2_rotation = camera2_rotation.split(' ')
    # camera2_rotation = [float(i) for i in camera2_rotation]
    # camera2_rotation = np.array(camera2_rotation)

    camera_file= "/home/vl10550y/Desktop/3DClimber/Datasets/aist_plusplus_final/cameras/" + cam_settings + ".json"
    with open(camera_file) as f:
        data = json.load(f)
        for i in range(len(data)):
            if data[i]['name'] == cam_ref:
                cameraref_rotation = np.array(data[i]['rotation'])
            if data[i]['name'] == cam_x1:
                camera1_rotation = np.array(data[i]['rotation'])
            if data[i]['name'] == cam_x2:
                camera2_rotation = np.array(data[i]['rotation'])
            if data[i]['name'] == cam_x3:
                camera3_rotation = np.array(data[i]['rotation'])
            if data[i]['name'] == cam_x4:
                camera4_rotation = np.array(data[i]['rotation'])

    print ("cameraref_rotation",[math.degrees(cameraref_rotation[i]) for i in range(3)])
    print ("camera1_rotation",[math.degrees(camera1_rotation[i]) for i in range(3)])
    print ("camera2_rotation",[math.degrees(camera2_rotation[i]) for i in range(3)])
    print ("camera3_rotation",[math.degrees(camera3_rotation[i]) for i in range(3)])
    print ("camera4_rotation",[math.degrees(camera4_rotation[i]) for i in range(3)])

    print("Read reference pkl:", ref_pkl)
    with open(ref_pkl, 'rb') as file:
        AISTPkl = pickle.load(file)
    print('[INFO] reference pkl loaded')
    print(AISTPkl.keys())
    print(AISTPkl['allFrameHumans'][0][0].keys())

    print("Read extra1 pkl:", extra1_pkl)
    with open(extra1_pkl, 'rb') as file:
        AISTPklExtra1 = pickle.load(file)
    print('[INFO] extra1 pkl loaded')
    print(AISTPklExtra1.keys())
    print(AISTPklExtra1['allFrameHumans'][0][0].keys())

    print("Read extra2 pkl:", extra2_pkl)
    with open(extra2_pkl, 'rb') as file:
        AISTPklExtra2 = pickle.load(file)
    print('[INFO] extra2 pkl loaded')
    print(AISTPklExtra2.keys())
    print(AISTPklExtra2['allFrameHumans'][0][0].keys())

    print("Read extra3 pkl:", extra3_pkl)
    with open(extra3_pkl, 'rb') as file:
        AISTPklExtra3 = pickle.load(file)
    print('[INFO] extra3 pkl loaded')
    print(AISTPklExtra3.keys())
    print(AISTPklExtra3['allFrameHumans'][0][0].keys())

    print("Read extra4 pkl:", extra4_pkl)
    with open(extra4_pkl, 'rb') as file:
        AISTPklExtra4 = pickle.load(file)
    print('[INFO] extra4 pkl loaded')
    print(AISTPklExtra4.keys())
    print(AISTPklExtra4['allFrameHumans'][0][0].keys())

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    models_path = os.environ["MODELS_PATH"]

    frames_count = len(AISTPkl['allFrameHumans'])
    

    allFramesHumans = []

    pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)
    betas_tensor = torch.zeros(1, 10, dtype=torch.float32).to(device)

    coscam1xz = abs(np.cos(camera1_rotation[0]-cameraref_rotation[0]))
    coscam2xz = abs(np.cos(camera2_rotation[0]-cameraref_rotation[0]))
    coscam3xz = abs(np.cos(camera3_rotation[0]-cameraref_rotation[0]))
    coscam4xz = abs(np.cos(camera4_rotation[0]-cameraref_rotation[0]))
    sincam1xz = abs(np.sin(camera1_rotation[0]-cameraref_rotation[0]))
    sincam2xz = abs(np.sin(camera2_rotation[0]-cameraref_rotation[0]))
    sincam3xz = abs(np.sin(camera3_rotation[0]-cameraref_rotation[0]))
    sincam4xz = abs(np.sin(camera4_rotation[0]-cameraref_rotation[0]))
    coscam1yz = abs(np.cos(camera1_rotation[1]-cameraref_rotation[1]))
    coscam2yz = abs(np.cos(camera2_rotation[1]-cameraref_rotation[1]))
    coscam3yz = abs(np.cos(camera3_rotation[1]-cameraref_rotation[1]))
    coscam4yz = abs(np.cos(camera4_rotation[1]-cameraref_rotation[1]))
    sincam1yz = abs(np.sin(camera1_rotation[1]-cameraref_rotation[1]))
    sincam2yz = abs(np.sin(camera2_rotation[1]-cameraref_rotation[1]))
    sincam3yz = abs(np.sin(camera3_rotation[1]-cameraref_rotation[1]))
    sincam4yz = abs(np.sin(camera4_rotation[1]-cameraref_rotation[1]))

    for i in range(frames_count):
        allHumans = []
        human = AISTPkl['allFrameHumans'][i][0]
        joints = human['j3d_smplx']
        jointsExtra1 = AISTPklExtra1['allFrameHumans'][i][0]['j3d_smplx']
        jointsExtra2 = AISTPklExtra2['allFrameHumans'][i][0]['j3d_smplx']
        jointsExtra3 = AISTPklExtra3['allFrameHumans'][i][0]['j3d_smplx']
        jointsExtra4 = AISTPklExtra4['allFrameHumans'][i][0]['j3d_smplx']
        uncert = human['joint_uncertainties']
        uncertExtra1 = AISTPklExtra1['allFrameHumans'][i][0]['joint_uncertainties']
        uncertExtra2 = AISTPklExtra2['allFrameHumans'][i][0]['joint_uncertainties']
        uncertExtra3 = AISTPklExtra3['allFrameHumans'][i][0]['joint_uncertainties']
        uncertExtra4 = AISTPklExtra4['allFrameHumans'][i][0]['joint_uncertainties']

        for jointnum, joint in enumerate(joints):
            if naive:
                if jointnum < 55:
                    joint[0] = (joint[0]/uncert[jointnum] + jointsExtra1[jointnum][0]/uncertExtra1[jointnum] + jointsExtra2[jointnum][0]/uncertExtra2[jointnum] + jointsExtra3[jointnum][0]/uncertExtra3[jointnum] + jointsExtra4[jointnum][0]/uncertExtra4[jointnum])/(1/uncert[jointnum] + 1/uncertExtra1[jointnum] + 1/uncertExtra2[jointnum] + 1/uncertExtra3[jointnum] + 1/uncertExtra4[jointnum])
                    joint[1] = (joint[1]/uncert[jointnum] + jointsExtra1[jointnum][1]/uncertExtra1[jointnum] + jointsExtra2[jointnum][1]/uncertExtra2[jointnum] + jointsExtra3[jointnum][1]/uncertExtra3[jointnum] + jointsExtra4[jointnum][1]/uncertExtra4[jointnum])/(1/uncert[jointnum] + 1/uncertExtra1[jointnum] + 1/uncertExtra2[jointnum] + 1/uncertExtra3[jointnum] + 1/uncertExtra4[jointnum])
                    joint[2] = (joint[2]/uncert[jointnum] + jointsExtra1[jointnum][2]/uncertExtra1[jointnum] + jointsExtra2[jointnum][2]/uncertExtra2[jointnum] + jointsExtra3[jointnum][2]/uncertExtra3[jointnum] + jointsExtra4[jointnum][2]/uncertExtra4[jointnum])/(1/uncert[jointnum] + 1/uncertExtra1[jointnum] + 1/uncertExtra2[jointnum] + 1/uncertExtra3[jointnum] + 1/uncertExtra4[jointnum])
                else:
                    joint[0] = (joint[0] + jointsExtra1[jointnum][0] + jointsExtra2[jointnum][0] + jointsExtra3[jointnum][0] + jointsExtra4[jointnum][0])/5
                    joint[1] = (joint[1] + jointsExtra1[jointnum][1] + jointsExtra2[jointnum][1] + jointsExtra3[jointnum][1] + jointsExtra4[jointnum][1])/5
                    joint[2] = (joint[2] + jointsExtra1[jointnum][2] + jointsExtra2[jointnum][2] + jointsExtra3[jointnum][2] + jointsExtra4[jointnum][2])/5
            else:
                joint[0] = (joint[0] + sincam1xz*jointsExtra1[jointnum][0] + sincam2xz*jointsExtra2[jointnum][0] + sincam3xz*jointsExtra3[jointnum][0] + sincam4xz*jointsExtra4[jointnum][0])/(1+sincam1xz+sincam2xz+sincam3xz+sincam4xz)
                joint[1] = (joint[1] + sincam1yz*jointsExtra1[jointnum][1] + sincam2yz*jointsExtra2[jointnum][1] + sincam3yz*jointsExtra3[jointnum][1] + sincam4yz*jointsExtra4[jointnum][1])/(1+sincam1yz+sincam2yz+sincam3yz+sincam4yz)#-0.1225
                joint[2] = (coscam1xz*coscam1yz*jointsExtra1[jointnum][2] + coscam2xz*coscam2yz*jointsExtra2[jointnum][2] + coscam3xz*coscam3yz*jointsExtra3[jointnum][2] + coscam4xz*coscam4yz*jointsExtra4[jointnum][2])/(coscam1xz*coscam1yz+coscam2xz*coscam2yz+coscam3xz*coscam3yz+coscam4xz*coscam4yz)

        
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

