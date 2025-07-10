import os
import sys
import cv2
import av
import numpy as np
import torch
import json
import pickle
import time

from premiere.functionsMoge import initMoGeModel,  colorizeDepthImage
from premiere.functionsDepth import packDepthImage, computeMinMaxDepth
from tqdm import tqdm

models_path = os.environ["MODELS_PATH"]

if len(sys.argv) != 6:
    print("Usage: python videoProcessMoge.py <input_video> <moge_pkl> <output_video> <output_pkl> <fov>")
    sys.exit(1)

videoInName = sys.argv[1]
mogePklName = sys.argv[2]
videoOutName = sys.argv[3]
outputPklName = sys.argv[4]
fov_x_degrees = float(sys.argv[5])

print ("Reading PKL: ", mogePklName)
with open(mogePklName, 'rb') as file:
    mogePKL = pickle.load(file)
      
if (fov_x_degrees == 0):
    fov_x_degrees = 0
    for i in range(len(mogePKL)):
        fov_x_degrees += mogePKL[i]['fov_x_degrees']
    fov_x_degrees /= len(mogePKL)

print ("Fov_x: ",fov_x_degrees)

device_name ='cuda'
device, model = initMoGeModel(device_name)

video = cv2.VideoCapture(videoInName)
width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

if video.isOpened() == False:
	print('[!] error opening the video')

outputContainer = av.open(videoOutName, mode='w')
# Configurer les options du codec avec le multithreading
codec_options = {
    'lossless': '1',
    'preset': 'veryslow',
    'crf': '0',
    'threads': '12',
}
# Ajouter un flux vidéo de sortie avec le codec H.264
outputStream = outputContainer.add_stream('libx264rgb',options=codec_options)
outputStream.width = width
outputStream.height = height
outputStream.pix_fmt = 'rgb24'  # Format pixel standard pour libx264
outputStream.thread_type = 'AUTO'
# Verify stream is properly configured
if not outputStream.codec_context.is_open:
    outputStream.codec_context.open()

allFrameData = []

pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)
try:
    while video.isOpened():
        ret, frame = video.read()
        if ret == True:
            image_tensor = torch.tensor(frame / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
            output = model.infer(image_tensor, fov_x=fov_x_degrees)

            depth = output['depth'].cpu().numpy()
            mask = output['mask'].cpu().numpy()

            min_depth, max_depth = computeMinMaxDepth(depth, mask)
            # print (min_depth, max_depth)

            data = [ min_depth, max_depth ]
            allFrameData.append(data)

            # colorizeDepth = colorizeDepthImage (depth)
            # cv2.imshow("Depth Image", colorizeDepth)                    
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            colorImage = packDepthImage(depth, mask, min_depth, max_depth)

            # Reconvertir l'image OpenCV en un frame AV au format 'yuv444p'
            outframe = av.VideoFrame.from_ndarray(colorImage, format='rgb24')
            outframe = outframe.reformat(format='rgb24')
                # Encoder le frame
            packets = outputStream.encode(outframe)
            for packet in packets:
                outputContainer.mux(packet)
            pbar.update(1)
        else:
            break  
except KeyboardInterrupt:
    pass
pbar.close()

with open(outputPklName, 'wb') as handle:
    pickle.dump(allFrameData, handle, protocol=pickle.HIGHEST_PROTOCOL) 

# Encoder les frames restants
packets = outputStream.encode(None)
for packet in packets:
    outputContainer.mux(packet)
outputStream.close()
outputContainer.close()

video.release()

# python ./videoProcessMoge.py ../../Desktop/T3-2-1024.MP4 resultDepth-T3-2-1024.json resultDepth-T3-2-1024.mp4 resultDepth-T3-2-1024.pkl
# python .\videoProcessMoge.py ..\videos\D6-5min-1-Scene-008.mp4 .\depth-D6-008.json depth-D6-008.mp4 depth-D6-008.pkl
# python .\videoProcessMoge.py ..\videos\D0-talawa_technique_intro-Scene-015.mp4 .\depth-D0-015.json depth-D0-015.mp4 depth-D0-015.pkl