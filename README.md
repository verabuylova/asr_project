# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a template for solving ASR task with PyTorch. This template branch is a part of the [HSE DLA course](https://github.com/markovka17/dla) ASR homework. Some parts of the code are missing (or do not follow the most optimal design choices...) and students are required to fill these parts themselves (as well as writing their own models, etc.).

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw1_asr).

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, download the files from [here](https://disk.yandex.ru/d/7MxFxgQEND61Ew) and locate them to the working repository. Then, run the following commands:

1. Train 30 epochs on train-clean-100.

```bash
python3 train.py -cn=deepspeech2
```

2. Train 20 epochs with augmentations on train-clean-360.

```bash
python3 train.py -cn=deepspeech2_augs
```

3. Train 20 epochs on train-other-500.

```bash
python3 train.py -cn=deepspeech2_other
```

3. Train 20 more epochs on train-other-500.

```bash
python3 train.py -cn=deepspeech2_other_round2
```

To run inference (evaluate the model or save predictions), download the files from [here](https://disk.yandex.ru/d/7MxFxgQEND61Ew) and locate them to the working repository. Then, run the following commands:

1. To run inference on clean set:

```bash
python3 inference.py -cn=inference_clean
```
2. To run inference on other set:

```bash
python3 inference.py -cn=inference_other
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
