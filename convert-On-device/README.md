# On-Device-Sora

## create a virtual env and activate (conda as an example)
conda create -n convert python=3.10 (different to base OpenSora ENV)
conda activate convert

## install torch, torchvision and xformers
pip install -r requirements/requirements-cu121.txt

pip install -v .

## install coreml
## new version(beta) `pip install coremltools==8.0b1`
pip install coreml

## change numpy version to 1.26.4
pip install numpy==1.26.4
