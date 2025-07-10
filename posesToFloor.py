import sys
import pickle
import numpy as np
import json
import math
import cv2

from scipy.spatial.transform import Rotation as R

# Description: This script is used to convert the poses from the camera frame to the floor frame.
def main():
    if len(sys.argv) != 4:
        print("Usage: python posesToFloor.py <input_pkl> <output_pkl> <video_json>")
        sys.exit(1)

    # Open the pkl file
    fileName  = sys.argv[1] 
    print ("read pkl file: ", fileName)
    file = open(fileName, 'rb')
    dataPKL = pickle.load(file) 
    file.close()

    allFrameHumans = dataPKL['allFrameHumans']

    # Open the video json file
    videoJSONFileName = sys.argv[3]
    print ("read video json file: ", videoJSONFileName)
    with open(videoJSONFileName, 'r') as json_file:
        videoJSONData = json.load(json_file)
    print("video JSON data:", videoJSONData)

    floor_angle_rad = videoJSONData["floor_angle_rad"]
    floor_angle_rad2 = videoJSONData["floor_angle_deg"]
    floor_angle_rad2 = math.radians(floor_angle_rad2)

    print("floor_angle_rad: ", floor_angle_rad)
    print("floor_angle_rad2: ", floor_angle_rad2)

    theta = -floor_angle_rad
    cos_ = math.cos(theta)
    sin_ = math.sin(theta)
    R_np = np.array([[1,    0,    0],
                    [0, cos_, -sin_],
                    [0, sin_,  cos_]])

    pivot = np.array([0, 0, 0])

    for humans in allFrameHumans:
        for human in humans:
            human['j3d_smplx'] = (R_np @ (human['j3d_smplx'] - pivot).T).T + pivot
            global_rot_vec_np = human['rotvec'][0]
            global_rot_mat, _ = cv2.Rodrigues(global_rot_vec_np)
            corrected_global_rot_mat = R_np @ global_rot_mat
            corrected_global_rot_vec, _ = cv2.Rodrigues(corrected_global_rot_mat)
            human['rotvec'][0] = corrected_global_rot_vec.reshape(1, 3)
            human['transl'] = human['j3d_smplx'][15]

    dataPKL['floor_angle_deg'] = 0
    dataPKL['floor_Zoffset'] = 0
 
    # Save the pkl file
    print ("write pkl file: ", sys.argv[2])
    with open(sys.argv[2], 'wb') as handle:
        pickle.dump(dataPKL, handle, protocol=pickle.HIGHEST_PROTOCOL) 

if __name__ == '__main__':
    main()
