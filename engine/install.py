import sys, os, json, hashlib, shutil
from runtime import ROOT, REPO, VENV, PYTHON, CODE_ROOT, run
ROOT.mkdir(parents=True,exist_ok=True)
run(sys.executable,'-m','pip','install','uv==0.6.17')
if not (VENV/'bin/python').is_file():
    run(sys.executable,'-m','uv','venv','--python','3.10',str(VENV))
PACKAGES = ['setuptools==69.5.1', 'wheel==0.43.0', 'packaging==24.2', 'numpy==1.26.4', 'Pillow==10.4.0', 'scipy==1.11.4', 'scikit-image==0.22.0', 'scikit-learn==1.4.2', 'matplotlib==3.8.4', 'pandas==2.2.2', 'PyYAML==6.0.2', 'addict==2.4.0', 'dill==0.3.8', 'face_alignment==1.3.4', 'huggingface-hub==0.26.5', 'transformers==4.44.2', 'safetensors==0.4.5', 'ninja==1.11.1.3', 'nvidia-ml-py==12.560.30', 'ftfy==6.2.3', 'gdown==5.2.0', 'torchmetrics==0.11.4', 'tqdm==4.66.5', 'requests==2.32.3', 'tensorboardX==2.6.2.2', 'tensorboard==2.17.1', 'clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33', 'opencv-python==4.10.0.84', 'numba==0.59.1', 'llvmlite==0.42.0', 'protobuf==4.25.3']
run(sys.executable,'-m','uv','pip','install','--python',PYTHON,
    'torch==2.2.2','torchvision==0.17.2',
    '--index-url','https://download.pytorch.org/whl/cu121')
if not shutil.which('g++'):
    run('apt-get','update','-qq')
    run('apt-get','install','-y','-qq','build-essential')
# Install the legacy build backend and a known CMake in the model environment.
run(sys.executable,'-m','uv','pip','install','--python',PYTHON,
    'setuptools==69.5.1','wheel==0.43.0','packaging==24.2','cmake==3.29.6')
run(sys.executable,'-m','uv','pip','install','--python',PYTHON,
    '--no-build-isolation-package','clip',*PACKAGES)
# Separating dlib makes compilation progress/failures visible in their own block.
print('Building/validating dlib 19.24.6 with bounded compilation jobs',flush=True)
run(sys.executable,'-m','uv','pip','install','--python',PYTHON,'dlib==19.24.6')
run(PYTHON,'-c','import clip, pkg_resources, cv2, dlib, torch; print("CLIP and dlib imports passed")')
import subprocess
freeze=subprocess.run([sys.executable,'-m','uv','pip','freeze','--python',PYTHON],
                      check=True,stdout=subprocess.PIPE,text=True).stdout
(CODE_ROOT/'requirements.resolved.txt').write_text(freeze)
print('Resolved dependency versions recorded.',flush=True)
