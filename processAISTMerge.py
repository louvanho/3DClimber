import sys
import re
import json
import os
import math
import subprocess

if len(sys.argv) != 5:
    print("Usage: processAISTMerge.py <aistFileList> <nlfDirectory> <gtdirectory> <outputDirectory>")
    sys.exit(1)

aistFileList = sys.argv[1]
NLFDirectory = sys.argv[2]
GTDirectory = sys.argv[3]
outputDirectory = sys.argv[4]

# Lire la liste complète des fichiers depuis aist.csv
with open(aistFileList, 'r', encoding='utf-8') as f:
    files = [line.strip() for line in f if line.strip()]

# Obtenir la liste de tous les fichiers qui contiennent '_c01'
files_with_c01 = [file for file in files if '_c01' in file]
#print(files_with_c01)
print (len(files_with_c01))

# Create output directory if missing
if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)
    print(f"Created directory: {outputDirectory}")


for i in range(len(files_with_c01)):
    filename = files_with_c01[i]
    print(i, " - ",  filename)
    basename = filename[:-4]
    
    GTPkl = os.path.join(GTDirectory, basename + ".pkl")
    NLFFinalPkl = os.path.join(NLFDirectory, basename, "nlf-final-floorc.pkl")
    NLFFinalFilteredPkl = os.path.join(NLFDirectory, basename, "nlf-final-filtered-floorc.pkl")
    # NLFFOVFinalPkl = os.path.join(NLFDirectory, basename+"_fov", "nlf-final-floorc.pkl")
    # NLFFOVFinalFilteredPkl = os.path.join(NLFDirectory, basename+"_fov", "nlf-final-filtered-floorc.pkl")
    outputPkl = os.path.join(outputDirectory, basename + ".pkl")
    
    command = ["python", "mergeAISTPkl.py", GTPkl, NLFFinalPkl, NLFFinalFilteredPkl, outputPkl ]
    # command = ["python", "mergeAISTPkl.py", GTPkl, NLFFinalPkl, NLFFinalFilteredPkl, NLFFOVFinalPkl, NLFFOVFinalFilteredPkl, outputPkl ]
    print("Running: ", command)
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Error running multiPersonProcessing for {filename}. Return code: {result.returncode}")
        sys.exit(1)

    # python .\processAIST.py .\aist.txt ..\aist++\ .\aist_camera_mapping.txt ../results/