# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

import numpy as np


class HumanBody(object):

    def __init__(self):
        self.skeleton = self.get_skeleton()
        self.skeleton_sorted_by_level = self.sort_skeleton_by_level(
            self.skeleton)

    def get_skeleton(self):
        joint_names = [
            "pelvis",           # 0
            "left_hip",         # 1
            "right_hip",        # 2
            "spine1",           # 3
            "left_knee",        # 4
            "right_knee",       # 5
            "spine2",           # 6
            "left_ankle",       # 7
            "right_ankle",      # 8
            "spine3",           # 9
            "left_foot",        # 10
            "right_foot",       # 11
            "neck",             # 12
            "left_collar",      # 13
            "right_collar",     # 14
            "head",             # 15
            "left_shoulder",    # 16
            "right_shoulder",   # 17
            "left_elbow",       # 18
            "right_elbow",      # 19
            "left_wrist",       # 20
            "right_wrist",      # 21
            "jaw",              # 22
            "left_eye_smplhf",  # 23
            "right_eye_smplhf", # 24
            "left_index1",      # 25
            "left_index2",      # 26
            "left_index3",      # 27
            "left_middle1",     # 28
            "left_middle2",     # 29
            "left_middle3",     # 30
            "left_pinky1",      # 31
            "left_pinky2",      # 32
            "left_pinky3",      # 33
            "left_ring1",       # 34
            "left_ring2",       # 35
            "left_ring3",       # 36
            "left_thumb1",      # 37
            "left_thumb2",      # 38
            "left_thumb3",      # 39
            "right_index1",     # 40
            "right_index2",     # 41
            "right_index3",     # 42
            "right_middle1",    # 43
            "right_middle2",    # 44
            "right_middle3",    # 45
            "right_pinky1",     # 46
            "right_pinky2",     # 47
            "right_pinky3",     # 48
            "right_ring1",      # 49
            "right_ring2",      # 50
            "right_ring3",      # 51
            "right_thumb1",     # 52
            "right_thumb2",     # 53
            "right_thumb3",     # 54
            "nose",             # 55
            "right_eye",        # 56
            "left_eye",         # 57
            "right_ear",        # 58
            "left_ear",         # 59
            "left_big_toe",     # 60
            "left_small_toe",   # 61
            "left_heel",        # 62
            "right_big_toe",    # 63
            "right_small_toe",  # 64
            "right_heel",       # 65
            "left_thumb",       # 66
            "left_index",       # 67
            "left_middle",      # 68
            "left_ring",        # 69
            "left_pinky",       # 70
            "right_thumb",      # 71
            "right_index",      # 72
            "right_middle",     # 73
            "right_ring",       # 74
            "right_pinky",      # 75
            "right_eye_brow1",  # 76
            "right_eye_brow2",  # 77
            "right_eye_brow3",  # 78
            "right_eye_brow4",  # 79
            "right_eye_brow5",  # 80
            "left_eye_brow5",   # 81
            "left_eye_brow4",   # 82
            "left_eye_brow3",   # 83
            "left_eye_brow2",   # 84
            "left_eye_brow1",   # 85
            "nose1",            # 86
            "nose2",            # 87
            "nose3",            # 88
            "nose4",            # 89
            "right_nose_2",     # 90
            "right_nose_1",     # 91
            "nose_middle",      # 92
            "left_nose_1",      # 93
            "left_nose_2",      # 94
            "right_eye1",       # 95
            "right_eye2",       # 96
            "right_eye3",       # 97
            "right_eye4",       # 98
            "right_eye5",       # 99
            "right_eye6",       # 100
            "left_eye4",        # 101
            "left_eye3",        # 102
            "left_eye2",        # 103
            "left_eye1",        # 104
            "left_eye6",        # 105
            "left_eye5",        # 106
            "right_mouth_1",    # 107
            "right_mouth_2",    # 108
            "right_mouth_3",    # 109
            "mouth_top",        # 110
            "left_mouth_3",     # 111
            "left_mouth_2",     # 112
            "left_mouth_1",     # 113
            "left_mouth_5",     # 114
            "left_mouth_4",     # 115
            "mouth_bottom",     # 116
            "right_mouth_4",    # 117
            "right_mouth_5",    # 118
            "right_lip_1",      # 119
            "right_lip_2",      # 120
            "lip_top",          # 121
            "left_lip_2",       # 122
            "left_lip_1",       # 123
            "left_lip_3",       # 124
            "lip_bottom",       # 125
            "right_lip_3",      # 126
        ]
        children = [
            [1,2,3],            # pelvis
            [4],                # left_hip
            [5],                # right_hip
            [6],                # spine1
            [7],                # left_knee
            [8],                # right_knee
            [9],                # spine2
            [10,62],            # left_ankle
            [11,65],            # right_ankle
            [12,13,14],         # spine3
            [60,61],            # left_foot
            [63,64],            # right_foot
            [15],               # neck
            [16],               # left_collar
            [17],               # right_collar
            [22,23,24,55,58,59],# head
            [18],               # left_shoulder
            [19],               # right_shoulder
            [20],               # left_elbow
            [21],               # right_elbow
            [25,28,31,34,37],   # left_wrist
            [40,43,46,49,52],   # right_wrist
            [107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126],# jaw
            [57,81,82,83,84,85,101,102,103,104,105,106],# left_eye_smplhf
            [56,76,77,78,79,80,95,96,97,98,99,100],# right_eye_smplhf
            [26],               # left_index1
            [27],               # left_index2
            [67],               # left_index3
            [29],               # left_middle1
            [30],               # left_middle2
            [68],               # left_middle3
            [32],               # left_pinky1
            [33],               # left_pinky2
            [70],               # left_pinky3
            [35],               # left_ring1
            [36],               # left_ring2
            [69],               # left_ring3
            [38],               # left_thumb1
            [39],               # left_thumb2
            [66],               # left_thumb3
            [41],               # right_index1
            [42],               # right_index2
            [72],               # right_index3
            [44],               # right_middle1
            [45],               # right_middle2
            [73],               # right_middle3
            [47],               # right_pinky1
            [48],               # right_pinky2
            [75],               # right_pinky3
            [50],               # right_ring1
            [51],               # right_ring2
            [74],               # right_ring3
            [53],               # right_thumb1
            [54],               # right_thumb2
            [71],               # right_thumb3
            [86,87,88,89,90,91,92,93,94],# nose
            [],                 # right_eye
            [],                 # left_eye
            [],                 # right_ear
            [],                 # left_ear
            [],                 # left_big_toe
            [],                 # left_small_toe
            [],                 # left_heel
            [],                 # right_big_toe
            [],                 # right_small_toe
            [],                 # right_heel
            [],                 # left_thumb
            [],                 # left_index
            [],                 # left_middle
            [],                 # left_ring
            [],                 # left_pinky
            [],                 # right_thumb
            [],                 # right_index
            [],                 # right_middle
            [],                 # right_ring
            [],                 # right_pinky
            [],                 # right_eye_brow1
            [],                 # right_eye_brow2
            [],                 # right_eye_brow3
            [],                 # right_eye_brow4
            [],                 # right_eye_brow5
            [],                 # left_eye_brow5
            [],                 # left_eye_brow4
            [],                 # left_eye_brow3
            [],                 # left_eye_brow2
            [],                 # left_eye_brow1
            [],                 # nose1
            [],                 # nose2
            [],                 # nose3
            [],                 # nose4
            [],                 # right_nose_2
            [],                 # right_nose_1
            [],                 # nose_middle
            [],                 # left_nose_1
            [],                 # left_nose_2
            [],                 # right_eye1
            [],                 # right_eye2
            [],                 # right_eye3
            [],                 # right_eye4
            [],                 # right_eye5
            [],                 # right_eye6
            [],                 # left_eye4
            [],                 # left_eye3
            [],                 # left_eye2
            [],                 # left_eye1
            [],                 # left_eye6
            [],                 # left_eye5
            [],                 # right_mouth_1
            [],                 # right_mouth_2
            [],                 # right_mouth_3
            [],                 # mouth_top
            [],                 # left_mouth_3
            [],                 # left_mouth_2
            [],                 # left_mouth_1
            [],                 # left_mouth_5
            [],                 # left_mouth_4
            [],                 # mouth_bottom
            [],                 # right_mouth_4
            [],                 # right_mouth_5
            [],                 # right_lip_1
            [],                 # right_lip_2
            [],                 # lip_top
            [],                 # left_lip_2
            [],                 # left_lip_1
            [],                 # left_lip_3
            [],                 # lip_bottom
            [],                 # right_lip_3
        ]

        skeleton = []
        for i in range(len(joint_names)):
            skeleton.append({
                'idx': i,
                'name': joint_names[i],
                'children': children[i]
            })
        return skeleton

    def sort_skeleton_by_level(self, skeleton):
        njoints = len(skeleton)
        level = np.zeros(njoints)

        queue = [skeleton[0]]
        while queue:
            cur = queue[0]
            for child in cur['children']:
                skeleton[child]['parent'] = cur['idx']
                level[child] = level[cur['idx']] + 1
                queue.append(skeleton[child])
            del queue[0]

        desc_order = np.argsort(level)[::-1]
        sorted_skeleton = []
        for i in desc_order:
            skeleton[i]['level'] = level[i]
            sorted_skeleton.append(skeleton[i])
        return sorted_skeleton


if __name__ == '__main__':
    hb = HumanBody()
    print(hb.skeleton)
    print(hb.skeleton_sorted_by_level)
