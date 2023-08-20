# MamAria

## ARIA UFPB + UPENN

## About

## Installation

[![WSL2 Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-vscode)

Code tested on WSL2 Ubuntu 22.04.2 LTS

[![Python 3.10](https://img.shields.io/badge/python-3.10.12-blue.svg)](https://www.python.org/downloads/release/python-3106/)

```[python]
python3 -m venv .venv
source .venv/bin/activate
```

```[python]
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python3 -m pip install -r requirements.txt
```
