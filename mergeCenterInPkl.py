import pickle
import time
import sys
import cv2
import os
import numpy as np
import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLM
from sklearn.linear_model import HuberRegressor

if len(sys.argv) != 5:
    print("Usage: python mergeCenterInPkl.py <input_pkl> <center_pkl> <output_pkl> <center mode: 0 average, 1 head>")
    sys.exit(1)

# Define pathing & device usage
inputPklName = sys.argv[1]
centerPklName = sys.argv[2] 
outputPklName = sys.argv[3]
depthMode = int(sys.argv[4])

# Open the pkl file
print ("read input pkl: ", inputPklName)
file = open(inputPklName, 'rb')
dataPKL = pickle.load(file) 
file.close()

# Open the pkl file
print ("read center pkl: ", centerPklName)
file = open(centerPklName, 'rb')
centerPKL = pickle.load(file)
file.close()


# Calculate the maximum number of humans and the maximum id
maxHumans = 0
maxId = 0
maxIndex = -1

allFrameHumans = dataPKL['allFrameHumans']
for i in range(len(allFrameHumans)):
    currentHumans = len(allFrameHumans[i])
    if currentHumans > maxHumans:
        maxHumans = currentHumans
        maxIndex = i
    for j in range(len(allFrameHumans[i])):
        maxId = max(maxId, allFrameHumans[i][j]['id'])

print('maxHumans: ', maxHumans)
print('maxId: ', maxId)
print('maxIndex: ', maxIndex)

linearTransformationTracks = []

AllY1 = []
AllY2 = []
for i in range(maxId+1):
    AllY1.append([])
    AllY2.append([])

nbFrame = len(allFrameHumans)
print ("nbFrame: ",nbFrame)
print ("len(centerPKL): ",len(centerPKL))

for i in range(nbFrame):
    humans = allFrameHumans[i]
    posIn = -1
    for j in range(len(humans)):
        for k in range(len(centerPKL[i])):
            if (humans[j]['id'] == centerPKL[i][k][0]):
                posIn = k
                break
        if (posIn != -1) and (centerPKL[i][posIn][1] != -10):
            AllY2[humans[j]['id']].append(centerPKL[i][posIn][1])
            if (depthMode==0):
                AllY1[humans[j]['id']].append(humans[j]['transl'][2])
                delta = humans[j]['transl'][2] - centerPKL[i][posIn][1]
                humans[j]['transl_pelvis'][0][2] -= delta
                humans[j]['transl'][2] -= delta
                pelvis = humans[j]['j3d_smplx'][0][2]
                for l in range (len(humans[j]['j3d_smplx'])):
                    delta = humans[j]['j3d_smplx'][l][2] - pelvis
                    humans[j]['j3d_smplx'][l][2] = delta + centerPKL[i][posIn][1]
            else:
                AllY1[humans[j]['id']].append(humans[j]['j3d_smplx'][15][2])
                delta = humans[j]['j3d_smplx'][15][2] - centerPKL[i][posIn][1]
                humans[j]['transl_pelvis'][0][2] -= delta
                humans[j]['transl'][2] -= delta
                head = humans[j]['j3d_smplx'][15][2]
                for l in range (len(humans[j]['j3d_smplx'])):
                    delta = humans[j]['j3d_smplx'][l][2] - head
                    humans[j]['j3d_smplx'][l][2] = delta + centerPKL[i][posIn][1]

allY1 = []
allY2 = []

for i in range(maxId+1):
    if (len(AllY1[i])>0):
        print("id: ",i)
        Y1 = np.array(AllY1[i])
        Y2 = np.array(AllY2[i])
        print(f"Number of elements in Y1 for id {i}: {len(Y1)}")
        mask = ~np.isnan(Y2) & ~np.isinf(Y2)
        num_false_elements = np.sum(~mask)
        print(f"Number of False elements in mask for id {i}: {num_false_elements}")
 
        Y1_clean = Y1[mask]
        Y2_clean = Y2[mask]

        allY1.append(Y1_clean)
        allY2.append(Y2_clean)


computeLinearTransformation = True    
linearTransformation = [0, 1]

if len(allY1) != 0:
    allY1_clean = np.concatenate(allY1)
    allY2_clean = np.concatenate(allY2)
    allY2_clean = sm.add_constant(allY2_clean)      

    if len(allY1_clean) <=2:
        print(f"Empty allY1_clean inferior to 2")
        computeLinearTransformation = False    

    # Check if all elements in Y1_clean or Y2_clean are too close
    all_elements_tooclose = np.all(np.isclose(allY1_clean, allY1_clean[0], atol=1e-6)) or np.all(np.isclose(allY2_clean, allY2_clean[0], atol=1e-6))
    if all_elements_tooclose:
        print(f"All elements in allY1_clean or allY2_clean are too close ")
        computeLinearTransformation = False    
else:
    print(f"Empty allY1")
    computeLinearTransformation = False


if computeLinearTransformation:
    rlm = RLM(allY1_clean, allY2_clean)
    rlm_fit = rlm.fit()
    print(rlm_fit.params)
    linearTransformation = rlm_fit.params

for i in range(nbFrame):
    humans = allFrameHumans[i]
    posIn = -1
    for j in range(len(humans)):
        for k in range(len(centerPKL[i])):
            if (humans[j]['id'] == centerPKL[i][k][0]):
                posIn = k
                break
        if (posIn != -1) and (centerPKL[i][posIn][1] != -10):
            a,b = linearTransformation
            if (depthMode==0):
                delta = humans[j]['transl'][2] - (a+b*centerPKL[i][posIn][1])
                humans[j]['transl_pelvis'][0][2] -= delta
                humans[j]['transl'][2] -= delta
                pelvis = humans[j]['j3d_smplx'][0][2]
                for l in range (len(humans[j]['j3d_smplx'])):
                    delta = humans[j]['j3d_smplx'][l][2] - pelvis
                    humans[j]['j3d_smplx'][l][2] = delta + (a+b*centerPKL[i][posIn][1])
            else:
                delta = humans[j]['j3d_smplx'][15][2] - (a+b*centerPKL[i][posIn][1])
                humans[j]['transl_pelvis'][0][2] -= delta
                humans[j]['transl'][2] -= delta
                head = humans[j]['j3d_smplx'][15][2]
                for l in range (len(humans[j]['j3d_smplx'])):
                    delta = humans[j]['j3d_smplx'][l][2] - head
                    humans[j]['j3d_smplx'][l][2] = delta + (a+b*centerPKL[i][posIn][1])

# Save the pkl file
print ("save pkl: ", outputPklName)
file = open(outputPklName, 'wb')
pickle.dump(dataPKL, file)
file.close()

# python .\mergeCenterInPkl.py .\pkl\T3-2-1024-896L-Clean-Track-Seg-Fus.pkl .\center-T3-2-1024.pkl .\pkl\T3-2-1024-896L-Clean-Track-Seg-Fus-Cen.pkl
# python .\mergeCenterInPkl.py .\pkl\T3-2-1024-896L-Clean-Track-Seg-Fus.pkl .\center-T3-2.pkl .\pkl\T3-2-1024-896L-Clean-Track-Seg-Fus-Cen.pkl


