# ByzShield
ByzShield's robust distributed ML framework implementation

This project builds on [DETOX] and implements our proposed ByzShield algorithm for robust distributed deep learning. Our placement involves three different techniques, namely MOLS, Ramanujan Case 1 & Ramanujan Case 2. It also includes three different types of attack on the DETOX framework.

# Requirements

We will be working with Python 2 for the local machine (to execute the bash scripts which configure the remote cluster and initiate training/testing) and with Python 3 for the remote cluster of PS/worker nodes (to execute the actual training/testing). We recommend using an Anaconda (tested with 2020.02) environment in both cases. The local machine would typically be a Linux system (tested with Ubuntu). Below, we have reported the exact version of each module that worked for us, however your mileage may vary.

## AWS EC2 setup
The first steps we need to do before installing the required packages are
 - [Install] and [configure] AWS CLI on the local machine (tested with version 2.0.16).
 - [Launch] an AWS EC2 instance of AMI "Ubuntu Server 16.04 LTS (HVM), SSD Volume Type (64-bit (x86))". We will install the packages on this instance and we will use it as a basis to create PS/worker instances (see [AMI]). Most of the instance specs may be left to their default values but we strongly recommend a minimum 20GiB of storage and a security group with the following settings

| Type | Protocol | Port Range | Source |
| ------ | ------ | ------ | ------ |
| All traffic | All | 0-65535 | Anywhere | 

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
For the remote cluster (PS/workers) the tested dependencies are
| Module | Version |
| ------ | ------ |
| python | 3.7.7 |
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

# Job launching
We will now discuss how one can launch a cluster and train/test a model. In the sequel, we will use the notation `{x}` to denote a piece of a script that should be substituted with the value `x`. Some notation used in the paper that we will refer to is:
 - `K`: number of workers.
 - `q`: number of Byzantine workers.
 - `r`: replication.
 - `b`: batchsize.

## Training
The training algorithm should be run by the PS instance executing file `run_pytorch.sh`. The basic arguments of this script along with all possible values are the following:

| Argument                      | Values/description                                 |
| ----------------------------- | ---------------------------------------- |
| `n` | Total number of nodes (PS and workers), equal to K+1 in paper |
| `hostfile`      | Path to MPI hostfile that contains the private IPs of all nodes of the cluster. If ran on AWS this file will be `hosts_address`, discussed above. If ran locally this file should have the form `localhost:{K+1}` (can be a plain txt) |
| `lr` | Inital learning rate that will be use. |
| `momentum` | Value of momentum that will be use. |
| `network` | Types of deep neural nets, currently `LeNet`, `ResNet-18/32/50/110/152`, and `VGGs` are supported. |
| `dataset` | Datasets use for training. |
| `batch-size` | Batch size for optimization algorithms. |
| `comm-type` | A fake parameter, please always set it to be `Bcast`, which gives you logarithmic comm complexity. |
| `num-aggregate` | Number of gradients required for the PS to aggregate. |
| `mode` | Robust aggregation methods e.g. `bulyan`, `multi-krum`, `coord-median`, `signSGD` |
| `approach`  | This can be set to `baseline`, `draco-lite`(DETOX), and `maj_vote` |
| `epochs`    | The maximal number of epochs to train (somehow redundant).   |
| `err-mode`    | Byzantine attack to simulate can be set as `rev_grad` or `constant`   |
| `max-steps`    | total number if iterations to run.   |
| `eval-freq` | Frequency of iterations to evaluation the model. |
| `worker-fail` | Number of Byzantine nodes to simulate. |
| `lis-simulation` | Enable the ["A little is enough attack"](https://arxiv.org/pdf/1902.06156.pdf), note that if this is set to `simulate`, the `err-mode` won't work any more. |
|`train-dir`  | Directory to save model checkpoints for evaluation. |

[DETOX]: <https://github.com/hwang595/DETOX>
[Install]: <https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html>
[configure]: <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html>
[Launch]: <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/launching-instance.html>
[AMI]: <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html>
