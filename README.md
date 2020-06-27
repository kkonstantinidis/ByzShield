# ByzShield
ByzShield's robust distributed ML framework implementation

This project builds on [DETOX] and implements our proposed ByzShield algorithm for robust distributed deep learning. Our placement involves three different techniques, namely MOLS, Ramanujan Case 1 & Ramanujan Case 2. It also includes three different types of attack on the DETOX framework.

# Requirements

We will be working with Python 2 for the local machine (to execute the bash scripts which configure the remote cluster and initiate training/testing) and with Python 3 for the remote cluster of PS/worker nodes (to execute the actual training/testing). We recommend using an Anaconda (tested with 2020.02) environment in both cases. 

## Prereqisites/Anaconda installation (both local and remote)
```sh
# Find the latest version from https://www.anaconda.com/products/individual (tested with 2020.02) and download
cd ~ && sudo apt-get install curl && curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

# Install Anaconda (press Enter multiple times until the license aggreement asks you to type 'yes' and press Enter)
bash Anaconda3-2020.02-Linux-x86_64.sh

# Press Enter to install in default location...
# Type 'yes' and press Enter to the prompt of the installer "Do you wish the installer to initialize Anaconda3 by running conda init?"...
# Apply the changes immediately so that you don't have to reboot/relogin
. .bashrc

# To disable each shell session having the base environment auto-activated
conda config --set auto_activate_base False
```

For the local machine the tested dependencies are
| Module | Version |
| ------ | ------ |
| Python | 2.7.18 |
| pip | 20.1.1 |
| setuptools | 44.1.0 |
| python-blosc | 1.7.0 |
| joblib | 0.13.2 |
| paramiko | 1.18.4 |
| boto3 | 1.12.39 |

The exact series of commands is
```sh
cd dillinger
docker build -t joemccann/dillinger:${package.json.version} .
```

For the remote machines (PS/worker nodes) the tested dependencies are
| Module | Version |
| ------ | ------ |
| Python | 3.7 |
| pip | 20.1.1 |
| setuptools | 47.1.1 |
| python-blosc | 1.7.0 |
| joblib | 0.15.1 |
| paramiko | 2.7.1 |
| boto3 | 1.9.66 |

[DETOX]: <https://github.com/hwang595/DETOX>

