
import os
import sys
import cv2
import pickle
import subprocess
import numpy as np
import shutil
import json

from argparse import ArgumentParser

def getMajorityValue(humansPerFrames):
    if humansPerFrames.size == 0:
        return None
    values, counts = np.unique(humansPerFrames.flatten(), return_counts=True)
    majority_value = values[np.argmax(counts)]
    return majority_value

def getHumanNumber(dataPKL):
    humansPerFrames = np.empty([len(dataPKL['allFrameHumans']), 1],dtype=int)
    for i in range(len(dataPKL['allFrameHumans'])):
        humansPerFrames[i] = len(dataPKL['allFrameHumans'][i])
    humanNumber = getMajorityValue(humansPerFrames)
    maxNbHumans = max(humansPerFrames)
    minNbHumans = min(humansPerFrames)
    return humanNumber, maxNbHumans, minNbHumans

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--directory", type=str, default=None, help="Directory to store the processed files")
    parser.add_argument("--video", type=str, default=None, help="Video file to process")
    parser.add_argument("--fov", type=float, default=0, help="Field of view for the 3D pose estimation")
    parser.add_argument("--cam_settings", type=str, default='setting1')
    parser.add_argument("--cam_nr", type=str, default='c01')
    parser.add_argument("--type", type=str, default="nlf", choices=["multihmr", "nlf"], help="Type of 3D pose estimation to use")
    parser.add_argument("--depthestimator", type=str, default="moge", choices=["vda", "moge"], help="Type of depth estimation to use")
    parser.add_argument("--rbfkernel", type=str, default="linear", choices=["linear", "multiquadric", "univariatespline"], help="RBF kernel to use for the 3D pose estimation filtering")
    parser.add_argument("--rbfsmooth", type=float, default=-1, help="Smoothness for the RBF kernel")
    parser.add_argument("--rbfepsilon", type=float, default=-1, help="Epsilon for the RBF kernel")
    parser.add_argument("--depthmode", type=str, default="pkl", choices=["pkl", "average", "head"], help="Depth mode to use for the 3D pose estimation")
    parser.add_argument("--step", type=int, default=0, help="Step to process (default: 0 for all steps)")
    parser.add_argument("--batchsize", type=int, default=25, help="Batch size for the nlf 3D pose estimation")
    parser.add_argument("--displaymode", action="store_true", help="Display mode activated if this flag is set")
    parser.add_argument("--handestimation", action="store_true", help="Inject hand estimation based on Wilor if this flag is set")
    parser.add_argument("--detectionthreshold", type=float, default=0.3,help="Threshold for detecting the human")
    parser.add_argument("--dispersionthreshold", type=float, default=.1, help="Threshold for human dispersion used for selecting the frame to start the segmentation/tracking")
    parser.add_argument("--stabilyzedepth", action="store_true", help="Depth map stabilyzation activated if this flag is set")
    parser.add_argument("--copyvideo", action="store_true", help="Copy the video if this flag is set")

    
    print ("\n############################################################")
    print ("# Arguments")
    print ("############################################################")
    args = parser.parse_args()
    print ("Type: ", args.type)
    print ("Directory: ", args.directory)
    print ("Video: ", args.video)
    print ("Fov: ", args.fov)
    print ("Depthestimator: ", args.depthestimator)
    print ("Rbfsmooth: ", args.rbfsmooth)
    print ("Rbfepsilon: ", args.rbfepsilon)
    print ("Rbfkernel: ", args.rbfkernel)
    print ("Depthmode: ", args.depthmode) 
    print ("Displaymode: ", args.displaymode)
    print ("Step: ", args.step)
    print ("Dispersionthreshold: ", args.dispersionthreshold)
    print ("Detectionthreshold: ", args.detectionthreshold)
    print ("Handestimation: ", args.handestimation)
    print ("Batchsize: ", args.batchsize)
    print ("Stabilyzedepth: ", args.stabilyzedepth)
    print ("Copyvideo: ", args.copyvideo)
    
    videoFileName = "\""+args.video+"\""

    if (args.depthmode == "average"):
        depthCenterMode = 0
        print("depthCenterMode: ", depthCenterMode)
    elif (args.depthmode == "head"):
        depthCenterMode = 1
    else:
        depthCenterMode = 2
    print("depthCenterMode: ", depthCenterMode)
    
    if args.displaymode:
        displayMode = 1
    else:
        displayMode = 0
    handEstimation = args.handestimation 
    rbfkernel = args.rbfkernel
    rbfsmooth = args.rbfsmooth
    rbfepsilon = args.rbfepsilon
    fov = args.fov
    cam_settings = args.cam_settings
    cam_nr = args.cam_nr
    dispersionthreshold = args.dispersionthreshold
    detectionThreshold = args.detectionthreshold
    type = args.type
    depthEstimator = args.depthestimator
    stabilyzedepth = args.stabilyzedepth
    copyVideo = args.copyvideo

    if args.directory is None:
        print("Please provide a directory")
        sys.exit(1)
    if args.video is None:
        print("Please provide a video")
        sys.exit(1)
    
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
        print(f"Created directory: {args.directory}")

    camera_file= "/home/vl10550y/Desktop/3DClimber/Datasets/aist_plusplus_final/cameras/" + cam_settings + ".json"
    with open(camera_file) as f:
        data = json.load(f)
        for i in range(len(data)):
            if data[i]['name'] == cam_nr:
                focal_x = np.array(data[i]['matrix'])[0][0]

    
    print ("\n############################################################")
    print ("# Video information")
    print ("############################################################")
    video = cv2.VideoCapture(args.video)
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video width: ", width)
    print("Video height: ", height)
    print("Video fps: ", fps)
    print("Video frames count: ", frames_count)
    video.release()
    
    if copyVideo:
        shutil.copyfile(args.video, os.path.join(args.directory, "video.mp4"))

    if rbfsmooth < 0:
        if (rbfkernel == "linear"):
            if fps > 100:
                rbfsmooth = 0.02
            elif fps > 60:
                rbfsmooth = 0.01
            else:
                rbfsmooth = 0.005
        elif (rbfkernel == "univariatespline"):
            if fps > 60:
                rbfsmooth = 0.5
            else:
                rbfsmooth = 0.25
        elif (rbfkernel == "multiquadric"):
            if fps > 60:
                rbfsmooth = 0.000025
            else:
                rbfsmooth = 0.00001
                
    if rbfepsilon < 0:
        if (rbfkernel == "multiquadric"):
            if fps > 60:
                rbfepsilon = 20
            else:
                rbfepsilon = 25
      
    print ("\n############################################################")
    print ("# Step 0: MoGe analysis")
    print ("############################################################")
    output_moge_pkl = os.path.join(args.directory, "moge.pkl")
    if args.step <= 0:
        print()
        output_moge_video = os.path.join(args.directory, "videoDepthAnalysis.mp4")
        command_videoAnalysisMoge = "python videoAnalysisMoge.py " + videoFileName + " -1 " + output_moge_video + " " + output_moge_pkl + " " + str(fov)
        print("Processing MoGe analysis...")
        print(command_videoAnalysisMoge)
        result = subprocess.run(command_videoAnalysisMoge, shell=True)
        if result.returncode != 0:
            print("\nError in MoGe analysis")
            sys.exit(1)

    print ("\n############################################################")
    print ("# Extract data from MoGe analysis")
    print ("############################################################")
    print()
    print ("Extracting data from MoGe analysis...")
    print ("Reading Pkl file: ", output_moge_pkl)
    with open(output_moge_pkl, 'rb') as file:
        mogePKL = pickle.load(file)
      
    fov_x_degrees = 0
    for i in range(len(mogePKL)):
        fov_x_degrees += mogePKL[i]['fov_x_degrees']
    fov_x_degrees /= len(mogePKL)
    print ("Estimated Fov_x: ",fov_x_degrees)
    
    angle = 0
    for i in range(len(mogePKL)):
        angle += mogePKL[i]['angle']
    angle /= len(mogePKL)
    print ("Angle: ",angle)

    
    # fov_x_degrees = 2*np.arctan(width/(2*focal_x)) * 180 / np.pi
    # print ("Used Fov_x: ",fov_x_degrees)

    if type == "nlf":
        print ("\n############################################################")
        print ("# Step 1: Extract 3D poses with NLF")
        print ("############################################################")
        output_type_pkl = os.path.join(args.directory, type+".pkl")
        if args.step <= 1:
            print()
            command_videoNLF = "python videoNLF.py --video " + videoFileName + " --out_pkl " + output_type_pkl + " --fov " + str(fov_x_degrees) + " --det_thresh " + str(detectionThreshold) + " --batchsize " + str(args.batchsize)
            print("Processing NLM poses estimation...") 
            print(command_videoNLF)
            result = subprocess.run(command_videoNLF, shell=True)
            if result.returncode != 0:
                print("\nError in NLF pose estimation")
                sys.exit(1)    
    else:
        print ("\n############################################################")
        print ("# Step 1: Extract 3D poses with MultiHMR")
        print ("############################################################")
        output_type_pkl = os.path.join(args.directory, type+".pkl")
        if args.step <= 1:
            print()
            command_videoMultiHMR = "python videoMultiHMR.py --video " + videoFileName + " --out_pkl " + output_type_pkl + " --fov " + str(fov_x_degrees)
            print("Processing multiHMR poses estimation...") 
            print(command_videoMultiHMR)
            result = subprocess.run(command_videoMultiHMR, shell=True)
            if result.returncode != 0:
                print("\nError in MultiHMR pose estimation")
                sys.exit(1)  

        
    print ("\n############################################################")
    print ("# Extract total human number")
    print ("############################################################")
    print()
    # Open the pkl file
    print ("Read pkl: ",output_type_pkl)
    file = open(output_type_pkl, 'rb')
    dataPKL = pickle.load(file) 
    file.close()
   
    print("Frames: ", len(dataPKL['allFrameHumans'])) 
    humanNumber, maxNbHumans, minNbHumans = getHumanNumber(dataPKL)
    print('humanNumber: ', humanNumber)
    print('maxNbHumans: ', maxNbHumans)
    print('minNbHumans: ', minNbHumans)    

    output_cleaned_pkl = os.path.join(args.directory, type+"-clean.pkl")
    threshold = 0.5
        
    print ("\n############################################################")
    print ("# Step 2: Clean poses")
    print ("############################################################")
    output_video_json = os.path.join(args.directory, "video.json")
    if args.step <= 2:
        print()
        command_cleanFramesPkl = "python cleanFramesPkl.py " + output_type_pkl + " " + output_cleaned_pkl + " " + str(humanNumber) + " " + str(threshold)
        print("Processing cleaned pkl...") 
        print(command_cleanFramesPkl)
        result = subprocess.run(command_cleanFramesPkl, shell=True)
        if result.returncode != 0:
            print("\nError in cleaning poses")
            sys.exit(1)
        print()
        output_video_json = os.path.join(args.directory, "video.json")
        command_computeCameraProperties = "python computeCameraProperties.py " + output_moge_pkl + " " + output_cleaned_pkl + " " + str(fov_x_degrees) + " " + output_video_json
        print("Processing camera properties...")
        print(command_computeCameraProperties)
        result = subprocess.run(command_computeCameraProperties, shell=True)
        if result.returncode != 0:
            print("\nError in camera properties")
            sys.exit(1)
        print()
        command_injectCameraPropertiesPkl = "python injectCameraPropertiesPkl.py " + output_video_json + " " + output_type_pkl + " " + output_type_pkl
        print("Inject camera properties in", output_type_pkl)
        print(command_injectCameraPropertiesPkl)
        result = subprocess.run(command_injectCameraPropertiesPkl, shell=True)
        if result.returncode != 0:
            print("\nError in camera properties")
            sys.exit(1)
        command_injectCameraPropertiesPkl = "python injectCameraPropertiesPkl.py " + output_video_json + " " + output_cleaned_pkl + " " + output_cleaned_pkl
        print("Inject camera properties in", output_cleaned_pkl)
        print(command_injectCameraPropertiesPkl)
        result = subprocess.run(command_injectCameraPropertiesPkl, shell=True)
        if result.returncode != 0:
            print("\nError in inject camera properties")
            sys.exit(1)

    threshold = 0.5
    if fps > 60:
        threshold = 0.4
    print ("\n############################################################")
    print ("# Step 3: 3D Tracking")
    print ("############################################################")
    output_tracking_pkl = os.path.join(args.directory, type+"-clean-track.pkl")
    if args.step <= 3:
        print()
        command_tracking3DPkl = "python tracking3DPkl.py " + output_cleaned_pkl + " " + output_tracking_pkl + " " + str(threshold)
        print("Processing tracking pkl...") 
        print(command_tracking3DPkl)
        result = subprocess.run(command_tracking3DPkl, shell=True)
        if result.returncode != 0:
            print("\nError in 3D tracking")
            sys.exit(1)
        
    print ("\n############################################################")
    print ("# Step 4: Add SAM2.1 tracking")
    print ("############################################################")
    trackMinSize = 30
    if fps < 50:
        trackMinSize = 5
    output_seg_pkl = os.path.join(args.directory, type+"-clean-track-seg.pkl")
    output_video_segmentation = os.path.join(args.directory, type+"-videoSegmentation.mp4")
    if args.step <= 4:
        print()
        command_fusionMultiHMRTracking = "python sam21MultiHMR.py " + output_tracking_pkl + " " + videoFileName + " " + str(humanNumber) + " " + str(trackMinSize) + " " + output_seg_pkl + " " + output_video_segmentation + " " + str(dispersionthreshold) + " " + str(displayMode)
        print("Processing fusion...") 
        print(command_fusionMultiHMRTracking)
        result = subprocess.run(command_fusionMultiHMRTracking, shell=True)
        if result.returncode != 0:
            print("\nError in SAM2.1 tracking")
            sys.exit(1)
           
    depthCenterModeStr = ""
    if depthCenterMode == 0:
        depthCenterModeStr = "average"
    elif depthCenterMode == 1:
        depthCenterModeStr = "head"

    if (depthCenterMode == 0) or (depthCenterMode == 1):  
        print ("\n############################################################")
        print ("# Step 5: Extract depth data")
        print ("############################################################")
        output_depth_video = os.path.join(args.directory, "videoDepth.mp4")
        output_depth_pkl = os.path.join(args.directory, "videoDepth.pkl")
        if args.step <= 5:
            print()
            if depthEstimator == "vda":
                command_videoProcessDepth = "python videoDepthAnything.py " + videoFileName + " " + output_depth_video + " " + output_depth_pkl + " vits"
            else:    
                command_videoProcessDepth = "python videoProcessMoge.py " + videoFileName + " " + output_moge_pkl + " " + output_depth_video + " " + output_depth_pkl + " " + str(fov_x_degrees)
            print("Processing depth data...") 
            print(command_videoProcessDepth)
            result = subprocess.run(command_videoProcessDepth, shell=True)
            if result.returncode != 0:
                print("\nError in depth data extraction")
                sys.exit(1)

        if stabilyzedepth:
            print ("\n############################################################")
            print ("# Step 6: Stabilyze depth map")
            print ("############################################################")
            output_depth_stabilyzed_video = os.path.join(args.directory, "videoDepthStabilyzed.mp4")
            output_depth_stabilyzed_pkl = os.path.join(args.directory, "videoDepthStabilyzed.pkl")
            if args.step <= 6:
                print()
                command_stabilyzeDepthMap = "python stabilyzeDepthMap.py " + output_depth_pkl + " " + output_depth_video + " " + output_video_segmentation + " " + output_depth_stabilyzed_pkl + " " + output_depth_stabilyzed_video + " " + str(displayMode)
                print("Processing depth map stabilyzation ...")
                print(command_stabilyzeDepthMap)
                result = subprocess.run(command_stabilyzeDepthMap, shell=True)
                if result.returncode != 0:
                    print("\nError in depth and segmentation merge")
                    sys.exit(1)
        
        print ("\n############################################################")
        print ("# Step 7: Merge depth, segmentation and pkl")
        print ("############################################################")
        if stabilyzedepth:
            output_depth_video = output_depth_stabilyzed_video
            output_depth_pkl = output_depth_stabilyzed_pkl
        output_depth_center_pkl = os.path.join(args.directory, type+"-depthCenter"+depthCenterModeStr+".pkl")
        if args.step <= 7:
            print()
            command_mergeDepthAndSeg = "python mergeDepthAndSeg.py " + output_seg_pkl + " " + output_depth_video + " " + output_video_segmentation + " " + output_depth_pkl + " " + output_depth_center_pkl + " multiperson " + str(depthCenterMode) + " " + str(displayMode)
            print("Processing merge depth and segmentation...")
            print(command_mergeDepthAndSeg)
            result = subprocess.run(command_mergeDepthAndSeg, shell=True)
            if result.returncode != 0:
                print("\nError in depth and segmentation merge")
                sys.exit(1)
        
    print ("\n############################################################")
    print ("# Step 8: Tracks fusion")
    print ("############################################################")
    output_fusion_pkl = os.path.join(args.directory, type+"-clean-track-seg-fusion.pkl")
    output_final_pkl = output_fusion_pkl
    trackMinSize = 30
    if fps < 50:
        trackMinSize = 10
    if args.step <= 8:
        print()
        command_tracksFusion = "python tracksFusion.py " + output_seg_pkl + " " + output_fusion_pkl + " 10"
        print("Processing track fusion...") 
        print(command_tracksFusion)
        result = subprocess.run(command_tracksFusion, shell=True)
        if result.returncode != 0:
            print("\nError in track fusion")
            sys.exit(1)

    if (depthCenterMode == 0) or (depthCenterMode == 1):  
        print ("\n############################################################")
        print ("# Step 9: Merge depth data in pkl")
        print ("############################################################")
        output_final_pkl = os.path.join(args.directory, type+"-clean-track-seg-fusion-"+depthCenterModeStr+".pkl")
        if args.step <= 9:
            print()
            command_mergeCenterInPkl = "python mergeCenterInPkl.py " + output_fusion_pkl + " " + output_depth_center_pkl + " " + output_final_pkl + " " + str(depthCenterMode)
            print("Processing merge center in pkl...") 
            print(command_mergeCenterInPkl)
            result = subprocess.run(command_mergeCenterInPkl, shell=True)
            if result.returncode != 0:
                print("\nError in merge center in pkl")
                sys.exit(1)
            output_fusion_pkl = output_final_pkl
            depthCenterModeStr = depthCenterModeStr + "-"

    depthCenterModeStr = ""
    if depthCenterMode == 0:
        depthCenterModeStr = "average-"
    elif depthCenterMode == 1:
        depthCenterModeStr = "head-"

    print ("\n############################################################")
    print ("# Step 10: Remove outlier in pkl")
    print ("############################################################")
    output_final_outlier_pkl = os.path.join(args.directory, type+"-clean-track-seg-fusion-"+depthCenterModeStr+"outlier.pkl")
    if args.step <= 10:
        print()
        command_removeoutlier = "python removeOutlier.py " + output_fusion_pkl + " " + output_final_outlier_pkl
        print("Processing Outlier removal...") 
        print(command_removeoutlier)
        result = subprocess.run(command_removeoutlier, shell=True)
        if result.returncode != 0:
            print("\nError in outlier removal")
            sys.exit(1)

    if handEstimation:
        print ("\n############################################################")
        print ("# Step 11: Inject hand estimation based on Wilor in pkl")
        print ("############################################################")
        previous_output_final_outlier_pkl = output_final_outlier_pkl
        output_final_outlier_pkl = os.path.join(args.directory, type+"-clean-track-seg-fusion-"+depthCenterModeStr+"outlier-handestimation.pkl")
        if args.step <= 11:
            print()
            command_injectHands = "python injectHandsPkl.py " + previous_output_final_outlier_pkl + " " + videoFileName + " " + output_final_outlier_pkl + " " + str(displayMode)
            print("Processing hand estimation...")
            print(command_injectHands)
            result = subprocess.run(command_injectHands, shell=True)
            if result.returncode != 0:
                print("\nError in hand estimation")
                sys.exit(1)

    print ("\n############################################################")
    print ("# Step 12: RBF and Filtering")
    print ("############################################################")
    if handEstimation:
        output_final_outlier_filtered_pkl = os.path.join(args.directory, type+"-clean-track-seg-fusion-"+depthCenterModeStr+"outlier-handestimation-filtered.pkl")
    else:
        output_final_outlier_filtered_pkl = os.path.join(args.directory, type+"-clean-track-seg-fusion-"+depthCenterModeStr+"outlier-filtered.pkl")
    if args.step <= 12:
        print()
        command_rbfFiltering = "python RBFFilterSMPLX.py " + output_final_outlier_pkl + " " + output_final_outlier_filtered_pkl + " " + rbfkernel + " " + str(rbfsmooth) + " " + str(rbfepsilon) 
        print("Processing RBF and filtering...") 
        print(command_rbfFiltering)
        result = subprocess.run(command_rbfFiltering, shell=True)
        if result.returncode != 0:
            print("\nError in RBF and filtering")
            sys.exit(1)

    print ("\n############################################################")
    print ("# Step 13: Copy final pkl files")
    print ("############################################################")
    print()
    output_destination_pkl = os.path.join(args.directory, type+"-final.pkl")
    output_destination_filtered_pkl = os.path.join(args.directory, type+"-final-filtered.pkl")

    print("Copying final pkl files...")
    print ("Final pkl")
    print("From: ",output_final_pkl)
    print("To: ",output_destination_pkl)
    shutil.copyfile(output_final_pkl, output_destination_pkl)
    print ("Final filtered pkl")
    print("From: ",output_final_outlier_filtered_pkl)
    print("To: ",output_destination_filtered_pkl)
    shutil.copyfile(output_final_outlier_filtered_pkl, output_destination_filtered_pkl)

    print ("\n############################################################")
    print ("# Step 14: Floor compensation")
    print ("############################################################")
    output_destination_floorc_pkl = os.path.join(args.directory, type+"-final-floorc.pkl")
    output_destination_floorc_filtered_pkl = os.path.join(args.directory, type+"-final-filtered-floorc.pkl")

    if args.step <= 14:
        print()
        command_floorCompensation = "python posesToFloor.py " + output_destination_pkl + " " + output_destination_floorc_pkl + " " + output_video_json 
        print("Processing floor compensation...")
        print(command_floorCompensation)
        result = subprocess.run(command_floorCompensation, shell=True)
        if result.returncode != 0:
            print("\nError in floor compensation on poses")
            sys.exit(1)
        command_floorCompensation = "python posesToFloor.py " + output_destination_filtered_pkl + " " + output_destination_floorc_filtered_pkl + " " + output_video_json 
        print("Processing floor compensation on filter poses...")
        print(command_floorCompensation)
        result = subprocess.run(command_floorCompensation, shell=True)
        if result.returncode != 0:
            print("\nError in floor compensation")
            sys.exit(1)


# python .\multiHMRPipeline.py --directory ../results/T3-2-1024 --video "F:\MyDrive\Tmp\Tracking\T3-2-1024.MP4" --step 6
# python .\multiHMRPipeline.py --video ..\videos\D0-talawa_technique_intro-Scene-003.mp4 --directory ..\results\D0-003\ --step 0