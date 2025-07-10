import pickle
import numpy as np
import sys

if len(sys.argv) < 3:
    print("Usage: mergeAISTPkl.py <firstPkl> <pkl1> <pkl2> ... <outputPkl>")

print (sys.argv)

firstPklName = sys.argv[1]
print ("Reading PKL: ", firstPklName)
with open(firstPklName, 'rb') as file:
    dataPKL = pickle.load(file)
allFrameHumans = dataPKL['allFrameHumans']
for f in allFrameHumans:
    f[0]['id'] = 0

all3DJoints = []
for i in range(len(allFrameHumans)):
    humans = allFrameHumans[i]
    if len(humans) == 0:
        continue
    for h in humans:
        h['transl_pelvis'][0][1] += 0.15
        h['transl'][1] += 0.15
        h['trans'][1] += 0.15
        h['j3d_smplx'][:, 1] += 0.15
        all3DJoints.append(h['j3d_smplx'])
min_y = 0
all3DJoints = np.array(all3DJoints)
y_coords = -all3DJoints[:, :, 1].flatten()
min_y = np.min(y_coords)
dataPKL['floor_Zoffset'] = float(min_y)
print (min_y)

minSize = len(allFrameHumans)

for i in range(1, len(sys.argv)-2):
    inputPklName = sys.argv[i+1]
    print ("Reading PKL: ", inputPklName)
    with open(inputPklName, 'rb') as file:
        dataPKL2 = pickle.load(file)
    allFrameHumans2 = dataPKL2['allFrameHumans']

    minSize = min(minSize, len(allFrameHumans2))

    for j in range(minSize):
        for k in range(len(allFrameHumans2[j])):
            allFrameHumans2[j][k]['id'] = i
            allFrameHumans[j].append(allFrameHumans2[j][k])

# Resize allFrameHumans to minSize
allFrameHumans = allFrameHumans[:minSize]
dataPKL['allFrameHumans'] = allFrameHumans

outputPklName = sys.argv[len(sys.argv)-1]
print ("Writing PKL: ", outputPklName)
with open(outputPklName, 'wb') as file:
    pickle.dump(dataPKL, file)

