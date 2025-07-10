import sys
import re
import json
import os
import math
import subprocess

def calculate_horizontal_fov(camera_matrix, image_width):
    """
    Calculate horizontal field of view from camera matrix.
    
    Args:
        camera_matrix: 3x3 camera intrinsic matrix
        image_width: Width of the image in pixels
    
    Returns:
        Horizontal FOV in degrees
    """
    # Extract focal length (first element of the camera matrix)
    focal_length_x = camera_matrix[0][0]
    
    # Calculate FOV in radians
    fov_rad = 2 * math.atan(image_width / (2 * focal_length_x))
    
    # Convert to degrees
    fov_deg = fov_rad * 180 / math.pi
    
    return fov_deg

def readAllCamerasSettings ( aistFileDirectory, settingsFilenames):
    allSettings = {}
    for settingFilename in settingsFilenames:
        finalSettingFilename = settingFilename +".json"
        finalSettingFilename = os.path.join(aistFileDirectory, "cameras", finalSettingFilename)
        with open( finalSettingFilename, 'r', encoding='utf-8') as f:
            setting = json.load(f)
        allSettings[settingFilename] = setting
    return allSettings

def getVideoSettings( videoFilename, settingDances):
    basename = videoFilename[:-4]
    key = re.sub(r'_c\d{2}', '_cAll', basename)
    return settingDances[key]

if len(sys.argv) != 5:
    print("Usage: processAIST.py <aistFileList> <aistFileDirectory> <aistCameraMapping> <finalDirectory>")
    sys.exit(1)

aistFileList = sys.argv[1]
aistFileDirectory = sys.argv[2]
aistCameraMapping = sys.argv[3]
finalDirectory = sys.argv[4]

# Lire la liste complète des fichiers depuis aist.csv
with open(aistFileList, 'r', encoding='utf-8') as f:
    files = [line.strip() for line in f if line.strip()]

# Lire la liste complète des fichiers depuis aist.csv
with open(aistCameraMapping, 'r', encoding='utf-8') as f:
    filesMapping = [line.strip() for line in f if line.strip()]

#print(filesMapping)

settingDances = {}
settingCameras = {}

for line in filesMapping:
    # Supposons que la ligne est du type "nom_camera setting"
    parts = line.split()
    if len(parts) >= 2:
        cam = parts[0]
        setting = parts[1]
        settingDances[cam] = setting
        # Ajout dans settings2 : on regroupe par setting
        if setting in settingCameras:
            settingCameras[setting].append(cam)
        else:
            settingCameras[setting] = [cam]

print (settingDances)

settingsFilenames = list(settingCameras.keys())
allSettings = readAllCamerasSettings(aistFileDirectory, settingsFilenames)
for setting in settingsFilenames:
    # print(setting)
    # print(allSettings[setting])
    for camera in allSettings[setting]:
        # print ("Camera: ", camera)
        # print("Camera name: ", camera['name'] )
        # print("Camera intrinsec matrix: ", camera['matrix'] )
        # print("Camera resolution: ", camera['size'] )
        # print("Camera FOV: ", calculate_horizontal_fov(camera['matrix'], camera['size'][0]) )
        camera['fov'] = calculate_horizontal_fov(camera['matrix'], camera['size'][0])
    # print(allSettings[setting])

# Obtenir la liste de tous les fichiers qui contiennent '_c01'
files_with_c01 = [file for file in files if '_c01' in file]
#print(files_with_c01)
print (len(files_with_c01))

cameraFovs = {}
cameraRotation = {}
cameraTranslation = {}
for filesName in files_with_c01:
#     print(filesName)
#    print(allSettings[getVideoSettings(filesName, settingDances)])
    for i in range(len(allSettings[getVideoSettings(filesName, settingDances)])):
        if allSettings[getVideoSettings(filesName, settingDances)][i]['name'] == 'c01':
            cameraFovs[filesName] = allSettings[getVideoSettings(filesName, settingDances)][i]['fov']
            cameraRotation[filesName] = allSettings[getVideoSettings(filesName, settingDances)][i]['rotation']
            cameraTranslation[filesName] = allSettings[getVideoSettings(filesName, settingDances)][i]['translation']
            break

if not os.path.exists(finalDirectory):
    os.makedirs(finalDirectory)
    print(f"Created directory: {finalDirectory}")

for i in range(len(files_with_c01)):
    filename = files_with_c01[i]
    print(i, " - ",  filename)
    basename = filename[:-4]
    key = re.sub(r'_c\d{2}', '_cAll', basename)
    name = key+".pkl"

    input_pkl = os.path.join(aistFileDirectory, "motions", name)
    input_j3d_pkl = os.path.join(aistFileDirectory, "keypoints3d", name)
    print (input_pkl)
    output_pkl = os.path.join(finalDirectory, basename + ".pkl")
    command = [ "python", "convertAIST.py",
        "--input_pkl", input_pkl,
        "--input_j3d_pkl", input_j3d_pkl,
        "--output_pkl", output_pkl,
        "--camera_position", f"{cameraTranslation[filename][0]} {cameraTranslation[filename][1]} {cameraTranslation[filename][2]}",
        "--camera_rotation", f"{cameraRotation[filename][0]} {cameraRotation[filename][1]} {cameraRotation[filename][2]}",
    ]
    print("Running: ", command)
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Error running multiPersonProcessing for {filename}. Return code: {result.returncode}")
        sys.exit(1)

    # python .\processAIST.py .\aist.txt ..\aist++\ .\aist_camera_mapping.txt ../results/