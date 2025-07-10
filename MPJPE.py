import pickle
import numpy as np
import sys

if len(sys.argv) != 3:
    print("Usage: python MPJPE.py <GTPkl> <ValidatePkl>")

print(sys.argv)

GTPklName = sys.argv[1]
print("Reading PKL: ", GTPklName)
with open(GTPklName, 'rb') as file:
    dataPKL = pickle.load(file)
GTallFrameHumans = dataPKL['allFrameHumans']

ValPklName = sys.argv[2]
print("Reading PKL: ", ValPklName)
with open(ValPklName, 'rb') as file:
    dataPKL = pickle.load(file)
ValallFrameHumans = dataPKL['allFrameHumans']

minSize = min(len(GTallFrameHumans), len(ValallFrameHumans))

all3DJointsGT = []
all3DJointsVal = []
for i in range(minSize):
    GTHumans = GTallFrameHumans[i]
    ValHumans = ValallFrameHumans[i]
    if len(GTHumans) == 0 or len(ValHumans) == 0:
        continue
    for h in GTHumans:
        all3DJointsGT.append(h['j3d_smplx'])
    for h in ValHumans:
        all3DJointsVal.append(h['j3d_smplx'])

all3DJointsGT = np.array(all3DJointsGT)
all3DJointsVal = np.array(all3DJointsVal)
print(all3DJointsGT.shape, all3DJointsVal.shape)
assert all3DJointsGT.shape == all3DJointsVal.shape

# Adjust Val skeleton to match GT skeleton pelvis joint
for frame in range(all3DJointsVal.shape[0]):
    pelvis_offset = all3DJointsGT[frame, 0] - all3DJointsVal[frame, 0]
    all3DJointsVal[frame] += pelvis_offset

mpjpe = 0
for frame in range(all3DJointsGT.shape[0]):
    f_mpjpe = 0
    for joint in range(all3DJointsGT.shape[1]):
        diff = all3DJointsGT[frame][joint] - all3DJointsVal[frame][joint]
        diff = np.linalg.norm(diff, ord=2)
        f_mpjpe += diff
    f_mpjpe /= all3DJointsGT.shape[1]
    mpjpe += f_mpjpe
mpjpe /= all3DJointsGT.shape[0]
print("MPJPE: ", mpjpe)
print("MPJPE (mm): ", mpjpe * 1000)
