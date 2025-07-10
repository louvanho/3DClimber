
import os 
os.environ["PYOPENGL_PLATFORM"] = "win32"
os.environ['EGL_DEVICE_ID'] = '0'
models_path = os.environ["MODELS_PATH"]

import math
import sys
from argparse import ArgumentParser
import random
import pickle as pkl
import numpy as np
from PIL import Image, ImageOps
import torch
from tqdm import tqdm
import time
import cv2

from multihmr.utils import normalize_rgb, get_focalLength_from_fieldOfView, demo_color as color, MEAN_PARAMS, CACHE_DIR_MULTIHMR, SMPLX_DIR
from multihmr.model import Model
from pathlib import Path
import pickle 
from premiere.functionsCommon import projectPoints3dTo2d

torch.cuda.empty_cache()
np.random.seed(seed=0)
random.seed(0)

def convert_image(img_pil, img_size, device=torch.device('cuda')):
    """ Open image at path, resize and pad """

    # Open and reshape
    #    img_pil = Image.open(img_path).convert('RGB')
    img_pil = ImageOps.contain(img_pil, (img_size,img_size)) # keep the same aspect ratio

    # Keep a copy for visualisations.
    img_pil_bis = ImageOps.pad(img_pil.copy(), size=(img_size,img_size), color=(255, 255, 255))
    img_pil = ImageOps.pad(img_pil, size=(img_size,img_size)) # pad with zero on the smallest side

    # Go to numpy 
    resize_img = np.asarray(img_pil)

    # Normalize and go to torch.
    resize_img = normalize_rgb(resize_img)
    x = torch.from_numpy(resize_img).unsqueeze(0).to(device)
    return x, img_pil_bis

def get_camera_parameters(img_size, fov=60, p_x=None, p_y=None, device=torch.device('cuda')):
    """ Given image size, fov and principal point coordinates, return K the camera parameter matrix"""
    K = torch.eye(3)
    # Get focal length.
    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0,0], K[1,1] = focal, focal

    # Set principal point
    if p_x is not None and p_y is not None:
            K[0,-1], K[1,-1] = p_x * img_size, p_y * img_size
    else:
            K[0,-1], K[1,-1] = img_size//2, img_size//2

    # Add batch dimension
    K = K.unsqueeze(0).to(device)
    return K

def load_model(model_name, device=torch.device('cuda')):
    """ Open a checkpoint, build Multi-HMR using saved arguments, load the model weigths. """
    # Model
    ckpt_path = os.path.join(models_path, 'multihmr', model_name+ '.pt')
    if not os.path.isfile(ckpt_path):
        assert "Multi-HMR model not present"

    # Load weights
    print("Loading model")
    print(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    print (ckpt['args'].img_size[0])

    # Get arguments saved in the checkpoint to rebuild the model
    kwargs = {}
    for k,v in vars(ckpt['args']).items():
            kwargs[k] = v

    # Build the model.
    kwargs['type'] = ckpt['args'].train_return_type
    kwargs['img_size'] = ckpt['args'].img_size[0]
    model = Model(**kwargs).to(device)

    # Load weights into model.
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print("Weights have been loaded")

    return model

def forward_model(model, input_image, camera_parameters,
                  det_thresh=0.3,
                  nms_kernel_size=1,
                 ):
        
    """ Make a forward pass on an input image and camera parameters. """
    
    # Forward the model.
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            humans = model(input_image, 
                           is_training=False, 
                           nms_kernel_size=int(nms_kernel_size),
                           det_thresh=det_thresh,
                           K=camera_parameters)
            #print (humans)

    return humans

def extractParameters ( W, H, M ):
    print (W,H,M)
    Oh =(M/2)*(W-H)/W
    factor = W/M
    return Oh, factor

def getCoordinates ( x, y, Oh, factor ):
    return (x*factor), ((y-Oh)*factor)


def keypointsBBox(keypoints, Oh, factor):
    # Compute min and max coordinates
    min_coords = np.min(keypoints, axis=0)
    max_coords = np.max(keypoints, axis=0)
    
    # Return the bounding box as [min_x, min_y, max_x, max_y]
    return min_coords, max_coords

def keypointsBBox3D(keypoints):
    min_coords = np.min(keypoints, axis=0)
    max_coords = np.max(keypoints, axis=0)
    return min_coords, max_coords

cap = None
model = None
allFrameHumans = []
lastFrame = -1
Oh = 0
factor = 1
total_frames = 0
display = False

if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("--model_name", type=str, default='multiHMR_896_L')
        parser.add_argument("--video", type=str, default='')
        parser.add_argument("--out_pkl", type=str, default='allFrame.pkl')
        parser.add_argument("--det_thresh", type=float, default=0.2)
        parser.add_argument("--nms_kernel_size", type=float, default=3)
        parser.add_argument("--fov", type=float, default=70)
        
        args = parser.parse_args()

        dict_args = vars(args)

        assert torch.cuda.is_available()

        # SMPL-X models
        smplx_fn = os.path.join(models_path, 'smplx', 'SMPLX_NEUTRAL.npz')
        if not os.path.isfile(smplx_fn):
            print(f"{smplx_fn} not found, please download SMPLX_NEUTRAL.npz file")
            print("To do so you need to create an account in https://smpl-x.is.tue.mpg.de")
            print("Then download 'SMPL-X-v1.1 (NPZ+PKL, 830MB) - Use thsi for SMPL-X Python codebase'")
            print(f"Extract the zip file and move SMPLX_NEUTRAL.npz to {smplx_fn}")
            print("Sorry for this incovenience but we do not have license for redustributing SMPLX model")
            assert NotImplementedError
        else:
             print('SMPLX found')
             
        # SMPL mean params download
        smpl_fn = os.path.join(models_path, 'smpl', 'smpl_mean_params.npz')
        if not os.path.isfile(smpl_fn):
            print('Download the SMPL mean params')
            assert NotImplementedError
        else:
            print('SMPL mean params found')
            
        # Loading
        print("Model: ",args.model_name)
        model = load_model(args.model_name)
        # Model name for saving results.
        model_name = os.path.basename(args.model_name)

        # Creating a VideoCapture object to read the video
        cap = cv2.VideoCapture(args.video)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) 
        
        Oh, factor = extractParameters ( video_width, video_height, model.img_size )
        
        pbar = tqdm(total=total_frames, unit=' frames', dynamic_ncols=True, position=0, leave=True)   
               
        while (cap.isOpened()):
            ret, frame = cap.read()
            if (ret):
                    img_size = model.img_size

                    converted = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    pil_im = Image.fromarray(converted)
                    x, img_pil_nopad = convert_image(pil_im, img_size)

                    allHumans = []
                    
                    # Get camera parameters
                    p_x, p_y = None, None
                    K = get_camera_parameters(model.img_size, fov=args.fov, p_x=p_x, p_y=p_y)
                    # print (K)
                    # Make model predictions
                    humans = forward_model(model, x, K,
                                            det_thresh=args.det_thresh,
                                            nms_kernel_size=args.nms_kernel_size)
                    # Convert tensors to regular variables
                    for h in humans:
                        human = {}
                        human['score']=h['scores'].cpu().numpy()
                        loc = h['loc'].cpu().numpy()
                        x1, y1 = getCoordinates ( loc[0], loc[1], Oh, factor)
                        human['loc']=np.array([x1, y1])
                        human['transl']=h['transl'].cpu().numpy()
                        human['transl_pelvis']=h['transl_pelvis'].cpu().numpy()
                        human['rotvec']=h['rotvec'].cpu().numpy().squeeze()
                        human['expression']=h['expression'].cpu().numpy()
                        human['shape']=h['shape'].cpu().numpy()
                        human['j3d_smplx']=h['j3d'].cpu().numpy()
                        proj_2d = projectPoints3dTo2d( human['j3d_smplx'], fov=args.fov, width=video_width, height=video_height )
                        human['j2d_smplx'] = proj_2d
                        if display:
                            color = (0,255,0)
                            for p in proj_2d:
                                cv2.circle(frame, (int(p[0]), int(p[1])), 1, color, -1)
                            cv2.imshow('frame', frame)
                            cv2.waitKey(1)
 
                        min_coords, max_coords = keypointsBBox3D(human['j3d_smplx'])
                        human['bbox3d']= [min_coords, max_coords]
                        min_coords, max_coords = keypointsBBox(human['j2d_smplx'], Oh, factor)
                        human['bbox']=[min_coords[0], min_coords[1], max_coords[0], max_coords[1]]
                        human['id']=-1
                        allHumans.append(human)
                    allFrameHumans.append(allHumans)
                    pbar.update(1)
            else:
                break

        pbar.close()

        allData = {}
        allData['allFrameHumans'] = allFrameHumans
        allData['model_name'] = model_name
        allData['model_type'] = 'multihmr'
        allData['model_size'] = model.img_size
        allData['video'] = args.video
        allData['det_thresh'] = args.det_thresh
        allData['nms_kernel_size'] = args.nms_kernel_size
        allData['fov_x_deg'] = args.fov
        allData['video_width'] = video_width
        allData['video_height'] = video_height
        allData['video_fps'] = video_fps 
        
        with open(args.out_pkl, 'wb') as handle:
            pickle.dump(allData, handle, protocol=pickle.HIGHEST_PROTOCOL) 

        if display:
            cv2.waitKey(0) 
            cv2.destroyAllWindows()

        print('end')

#python .\videoMultiHMR.py --video ..\videos\D0-talawa_technique_intro-Scene-015.mp4 --out_pkl D0-talawa_technique_intro-Scene-015_896L.pkl --model_name multiHMR_896_L --fov 70
#python .\videoMultiHMR.py --video "C:\Users\colantoni\Nextcloud\Tmp\Tracking\T1.MP4" --out_pkl pkl\T1-896L.pkl --model_name multiHMR_896_L --fov 80
#python .\videoMultiHMR.py --video "C:\Users\colantoni\Nextcloud\Tmp\Tracking\T3-2-1024.MP4" --out_pkl .\pkl\T3-2-1024-896L.pkl --fov 70
# python .\videoMultiHMR.py --video ..\videos\D0-talawa_technique_intro-Scene-015.mp4 --out_pkl pklmulihmr\D0-015.pkl --model_name multiHMR_896_L --fov 70