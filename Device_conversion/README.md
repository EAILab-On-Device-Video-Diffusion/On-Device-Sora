## How to compose the converting Env

```
conda create -n convert python=3.10

conda activate convert

pip install -r requirements/requirements-convert.txt

pip install -v .
```


### 1. T5 Converting...
```
cd t5
python3 export-t5.py
```

### 2. STDiT Convering...
```
cd stdit3
python3 export-stdit3.py
```

### 3. VAE Converting...
When you run `export-vae-spatial.py`, There are some error that is `Fatal Python error: PyEval_SaveThread`.
To address this error, you should only run one code block for each VAE part. Comment out the rest.

```
cd vae

# for vae's temporal part
python3 export-vae-temporal.py

# for vae's spatial part
python3 export-vae-spatial.py
```
