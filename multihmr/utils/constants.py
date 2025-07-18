# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import os

SMPLX_DIR = os.environ["MODELS_PATH"]
MEAN_PARAMS = os.path.join(os.environ["MODELS_PATH"], 'smpl','smpl_mean_params.npz')
CACHE_DIR_MULTIHMR = 'models/multiHMR'

ANNOT_DIR = 'data'
BEDLAM_DIR = 'data/BEDLAM'
EHF_DIR = 'data/EHF'
THREEDPW_DIR = 'data/3DPW'

SMPLX2SMPL_REGRESSOR = 'models/smplx/smplx2smpl.pkl'