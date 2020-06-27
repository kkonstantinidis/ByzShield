# ByzShield
ByzShield's robust distributed ML framework implementation

This project builds on [DETOX] and implements our proposed ByzShield algorithm for robust distributed deep learning. Our placement involves three different techniques, namely MOLS, Ramanujan Case 1 & Ramanujan Case 2. It also includes three different types of attack on the DETOX framework.

# Requirements

We will be working with Python 2 for the local machine (to execute the bash scripts which configure the remote cluster and initiate training/testing) and with Python 3 for the remote cluster of PS/worker nodes (to execute the actual training/testing). We recommend using an Anaconda (tested with 2020.02) environment in both cases. Below, we have reported the exact version of each module that worked for us, however your mileage may vary.

## Prerequisites/Anaconda installation (both local and remote)
```sh
sudo apt-get update && sudo apt-get upgrade

# Find the latest Anaconda version from https://www.anaconda.com/products/individual (tested with 2020.02) and download
cd ~ && sudo apt-get install curl && curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

# Install Anaconda (press Enter multiple times until the license aggreement asks you to type 'yes' and press Enter)
bash Anaconda3-2020.02-Linux-x86_64.sh

# Press Enter to install in default location...
# Type 'yes' and press Enter to the prompt of the installer
# "Do you wish the installer to initialize Anaconda3 by running conda init?"...
# Apply the changes immediately so that you don't have to reboot/relogin
. .bashrc

# To disable each shell session having the base environment auto-activated
conda config --set auto_activate_base False
```

## Local machine Anaconda environment
For the local machine the tested dependencies are
| Module | Version |
| ------ | ------ |
| python | 2.7.18 |
| pip | 20.1.1 |
| setuptools | 44.1.0 |
| python-blosc | 1.7.0 |
| joblib | 0.13.2 |
| paramiko | 1.18.4 |
| boto3 | 1.12.39 |

The exact series of commands is
```sh
conda create -n detox_local_python2 python=2.7
conda activate detox_local_python2
conda install pip
python -m pip install --upgrade pip
pip install --upgrade setuptools
conda install -c conda-forge python-blosc
conda install -c anaconda joblib
conda install -c anaconda paramiko
conda install -c anaconda boto3
```

## Remote cluster Anaconda environment
For the remote machines (PS/workers) the tested dependencies are
| Module | Version |
| ------ | ------ |
| python | 3.7 |
| pip | 20.1.1 |
| setuptools | 47.1.1 |
| pytorch | 1.0.1 |
| torchvision | 0.2.2 |
| python-blosc | 1.7.0 |
| joblib | 0.15.1 |
| paramiko | 2.7.1 |
| boto3 | 1.9.66 |
| libgcc | 7.2.0 |
| pandas | 1.0.3 |
| scipy | 1.4.1 |
| mpi4py | 3.0.3 |
| hdmedians | 0.13 |

The exact series of commands is
```sh
conda create -n detox python=3.7
conda activate detox
conda install pip
python -m pip install --upgrade pip
pip install --upgrade setuptools
conda install pytorch==1.0.1 torchvision cpuonly -c pytorch
conda install -c anaconda python-blosc
conda install -c anaconda joblib
conda install -c anaconda paramiko
conda install -c anaconda boto3
conda install -c anaconda libgcc
conda install -c anaconda pandas
conda install -c anaconda scipy
conda install -c anaconda mpi4py

# Install hdmedians
sudo apt-get install gcc && sudo apt-get install git
git clone https://github.com/daleroberts/hdmedians.git
cd hdmedians
python setup.py install
```
[DETOX]: <https://github.com/hwang595/DETOX>

