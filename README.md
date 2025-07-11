# PREMIERE Multi Person Processing - AIST++ Processing
======

Installation
------------

```py
conda create -n premiere python=3.11
conda activate premiere
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::ffmpeg
pip install -r requirement.txt
```

Models
------------

```
wget https://www.couleur.org/premiere-files/models.zip
unzip to MODELS_DIR
then
export MODELS_PATH="MODELS_DIR"
```


Usage
------------

Multiview processing

```py
python multiPersonProcessingLouis_mast3r.py --videos path/to/center.mp4,path/to/left.mp4,path/to/right.mp4 --directory path/to/results/
```

For the 3D reconstruction, simply run GLOMAP. Don't forget to include a frame from the camera used for pose estimation named "ref.jpg".
Then convert the .bin files to .txt.

Fusion

```py
python rotatePLY.py --images_txt path/to/images.txt --ply_file path/to/input.ply --output path/to/input_transformed.ply
```

Visualization

```py
python appVisualization.py path/to/pose.pkl --ply_file path/to/input_transformed.ply
```
