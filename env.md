# Environment Setup

These commands target a Linux server with an NVIDIA GPU and a driver compatible with CUDA 11.8. Run them in order in a shell with Conda initialized.

## 1. Create and activate the environment

```bash
conda create -n taco python=3.10.12 pip -y
conda activate taco
```

## 2. Install PyTorch

Use the matching versions from the [official PyTorch installation guide](https://pytorch.org/get-started/previous-versions/#v201).

```bash
conda install pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.8 \
    -c pytorch -c nvidia -y
```

## 3. Install project dependencies

```bash
python -m pip install \
    "numpy<2" \
    "setuptools<70" \
    pyyaml==6.0 \
    dotmap==1.3.30 \
    decord==0.6.0 \
    torchnet==0.0.4 \
    tqdm==4.65.0 \
    termcolor==2.1.0 \
    ftfy==5.8 \
    regex==2022.7.9 \
    pandas==2.0.3
```

The NumPy and setuptools bounds support the legacy dependencies, including the bundled CLIP loader's `from pkg_resources import packaging` import.

Optional dependency:

```bash
python -m pip install einops
```

