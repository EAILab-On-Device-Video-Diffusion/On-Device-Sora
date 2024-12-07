## How to compose the converting Env

conda create -n convert python=3.10 (different to base OpenSora ENV)
conda activate convert

pip install -r requirements/requirements-convert.txt
pip install -v .
