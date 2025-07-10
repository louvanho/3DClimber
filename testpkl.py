import pickle 
from argparse import ArgumentParser
import sys
import os


if __name__ == "__main__":
    # PklName1 = sys.argv[1]
    # print ("Reading PKL: ", PklName1)
    # with open(PklName1, 'rb') as file:
    #     pkl = pickle.load(file)
    # print('[INFO] first pkl loaded')
    # print(pkl.keys())

    # PklName2 = sys.argv[2]
    # print ("Reading PKL: ", PklName2)
    # with open(PklName2, 'rb') as file:
    #     pkl2 = pickle.load(file)
    # print('[INFO] second pkl loaded')
    # print(pkl2.keys())

    # PklName3 = sys.argv[3]
    # print ("Reading PKL: ", PklName3)
    # with open(PklName3, 'rb') as file:
    #     pkl3 = pickle.load(file)
    # print('[INFO] third pkl loaded')
    # print(pkl3.keys())

    # print('shoulders:')
    # print(pkl['keypoints2d'][0][0][5:7,0:2])
    # print(pkl2['allFrameHumans'][0][0]['j2d_smplx'][16:18].round(0))
    # print(pkl3['allFrameHumans'][0][0]['j2d_smplx'][16:18].round(0))
    # print('elbows:')
    # print(pkl['keypoints2d'][0][0][7:9,0:2])
    # print(pkl2['allFrameHumans'][0][0]['j2d_smplx'][18:20].round(0))
    # print(pkl3['allFrameHumans'][0][0]['j2d_smplx'][18:20].round(0))
    # print('knees:')
    # print(pkl['keypoints2d'][0][0][13:15,0:2])
    # print(pkl2['allFrameHumans'][0][0]['j2d_smplx'][4:6].round(0))
    # print(pkl3['allFrameHumans'][0][0]['j2d_smplx'][4:6].round(0))




    for i in range(0, len(sys.argv)-1):
        inputPklName = sys.argv[i+1]
        print("Reading PKL: ", inputPklName)
        with open(inputPklName, 'rb') as file:
            pkl = pickle.load(file)
        print('[INFO] first pkl loaded')
        for frame in range(len(pkl['allFrameHumans'])):
            # Check if the frame is empty
            if not pkl['allFrameHumans'][frame]:
                pkl['allFrameHumans'][frame] = pkl['allFrameHumans'][frame-1]

        # Save the modified pickle back to the same file (or create a backup if you want)
        backup_name = inputPklName + ".bak"
        os.rename(inputPklName, backup_name)
        with open(inputPklName, 'wb') as file:
            pickle.dump(pkl, file)
        print(f"[INFO] Saved modified PKL to {inputPklName} (backup at {backup_name})")
            
        # print(pkl.keys())
        # print(pkl['allFrameHumans'][0][0].keys())
        # print(pkl['allFrameHumans'][2522])
        # print(pkl['allFrameHumans'][2523])
        # print(pkl['allFrameHumans'][2524])
        # print(pkl['allFrameHumans'][0][0]['j3d_smplx'].shape)
        # print(pkl['allFrameHumans'][0][0]['j3d_smplx'])
        # print(pkl['allFrameHumans'][0][0]['j2d_smplx'].shape)
        # print(pkl['allFrameHumans'][0][0]['j2d_smplx'])
        # print(pkl['video_width'])
        # print(pkl['video_height'])

        # # print(pkl['keypoints_number'])
        # print(pkl['allFrameHumans'][0][0]['j3d_smplx'])
        # print(pkl['allFrameHumans'][0][0]['id'])
        # # neg = 0
        # # for i in pkl['allFrameHumans']:
        # #     for j in i:
        # #         for k in j['joint_uncertainties']:
        # #             if k > neg:
        # #                 neg = k
        # # print(neg)
        # # print(pkl['allFrameHumans'][0][0]['joint_uncertainties'])
        # # print(pkl['allFrameHumans'][0][0]['rotvec'][0].dtype)
        # print(pkl['allFrameHumans'][0][0]['j3d_smplx'])

        # print(pkl['keypoints3d'][0]/93.77886)

    #     # print(pkl['smpl_poses'])
    # inputPklName1 = sys.argv[1]
    # inputPklName2 = sys.argv[2]
    # print ("Reading PKL: ", inputPklName1)
    # with open(inputPklName1, 'rb') as file:
    #     pkl1 = pickle.load(file)
    # print('[INFO] first pkl loaded')
    # print(pkl1.keys())
    # with open(inputPklName2, 'rb') as file:
    #     pkl2 = pickle.load(file)
    # print('[INFO] second pkl loaded')
    # print(pkl2.keys())
    # afh1 = pkl1['allFrameHumans']
    # afh2 = pkl2['allFrameHumans']
    # for i in range(len(pkl1)):
    #     print(pkl1[i][0]['j3d_smplx']-pkl2[i][0]['j3d_smplx'])
    
