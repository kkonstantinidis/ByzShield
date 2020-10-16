# ByzShield
ByzShield's robust distributed ML framework implementation.

This project builds on [DETOX] and implements our proposed ByzShield algorithm for robust distributed deep learning. Our placement involves three different techniques, namely MOLS, Ramanujan Case 1 & Ramanujan Case 2. It also includes three different types of attack on the DETOX framework.

# Requirements
We will be working with Python 2 for the local machine (to execute the bash scripts which configure the remote cluster and initiate training/testing) and with Python 3 for the remote cluster of PS/worker nodes (to execute the actual training/testing). We recommend using an Anaconda (tested with 2020.02) environment in both cases. The local machine would typically be a Linux system (tested with Ubuntu). Below, we have reported the exact version of each module that worked for us, however your mileage may vary.

## AWS EC2 setup
This project is intended to be launched on AWS EC2. It also supports local execution (for MNIST) which we won't discuss here but the procedure is very similar (email me if you need instructions for that).

The first steps we need to do before installing the required packages are
 - [Install] and [configure] AWS CLI on the local machine (tested with version 2.0.16).
 - [Launch] an AWS EC2 instance of AMI "Ubuntu Server 16.04 LTS (HVM), SSD Volume Type (64-bit (x86))". We will install the packages on this instance and we will use it as a basis to create PS/worker instances (see [AMI]). Most of the instance specs may be left to their default values but we will probably need a minimum 20GiB of storage and a security group with the following settings

| Type | Protocol | Port Range | Source |
| ------ | ------ | ------ | ------ |
| All traffic | All | 0-65535 | Anywhere |

Create a security group ([instructions][security_group_create]) with the above settings and give it a name, e.g., `byzshield_security_group`. We will also use it later.

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

# To disable each shell session having the base environment auto-activated (optional)
conda config --set auto_activate_base False
```

## Anaconda environments
The tested dependencies versions for the local/remote machines are given next:
| Module | Local | Remote |
| ------ | ------ | ------ |
| Python | 2.7.18 | 3.7.7 |
| pip | 20.1.1 | 20.1.1 |
| setuptools | 44.1.0 | 47.1.1 |
| python-blosc | 1.7.0 | 1.7.0 |
| joblib | 0.13.2 | 0.15.1 |
| paramiko | 1.18.4 | 2.7.1 |
| boto3 | 1.12.39 | 1.9.66 |
| pytorch | N/A | 1.0.1 |
| torchvision | N/A | 0.2.2 |
| libgcc | N/A | 7.2.0 |
| pandas | N/A | 1.0.3 |
| scipy | N/A | 1.4.1 |
| mpi4py | N/A | 3.0.3 |
| hdmedians | N/A | 0.13 |

The exact series of commands for the *local* machine is
```sh
conda create -n byzshield_local_python2 python=2.7
conda activate byzshield_local_python2
conda install pip
python -m pip install --upgrade pip
pip install --upgrade setuptools
conda install -y -c conda-forge python-blosc
conda install -y -c anaconda joblib
conda install -y -c anaconda paramiko
conda install -y -c anaconda boto3
```

The exact series of commands for the *remote* machine is
```sh
conda create -n byzshield python=3.7
conda activate byzshield
conda install pip
python -m pip install --upgrade pip
pip install --upgrade setuptools
conda install -y pytorch==1.0.1 torchvision cpuonly -c pytorch
conda install -y -c anaconda python-blosc
conda install -y -c anaconda joblib
conda install -y -c anaconda paramiko
conda install -y -c anaconda boto3
conda install -y -c anaconda libgcc
conda install -y -c anaconda pandas
conda install -y -c anaconda scipy
conda install -y -c anaconda mpi4py

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

## Remote AMI
Now that we have installed all needed dependencies on the remote EC2 instance, we need to make an AMI image of it so that we can quickly launch PS/worker instances out of it. For instructions see [here][AMI_create]. Make note of the created AMI ID, like `ami-xxxxxxxxxxxxxxxx`.

## AWS EFS
We will use Amazon Elastic File System (EFS) to share a folder with the trained model among the machines. Follow the [instructions][EFS_create] to create an EFS. We will probably need a security group with the settings discussed above for the instance (you can reuse the same one). Make note of the IP address of the EFS `xxx.xxx.xxx.xxx`.

## Cluster configuration
The script `pytorch_ec2.py` will launch the instances automatically. Before running it, you need to copy your AWS private key file `xxxxxx.pem` to the folder `./tools` and then edit the following part of the configuration (the rest of the parameters can be left unchanged and have been omitted here):
```python
cfg = Cfg({
    "key_name": "{Your AWS private key file without the .pem extenion}",                                         # string
    "n_workers" : {Number of workers (K)},                                                                       # integer
    "region" : "{Your AWS region, e.g., us-east-1}",                                                             # string
    "availability_zone" : "{Your AWS subnet, e.g., us-east-1c}",                                                 # string
    "master_type" : "{Instance type of PS, e.g., r3.xlarge}",                                                    # string
    "worker_type" : "{Instance type of workers, e.g., r3.xlarge}",                                               # string
    "image_id": "{AMI ID created before, like ami-xxxxxxxxxxxxxxxx}",                                            # string
    "spot_price" : "{Dollar amount per machine per hour at least max(price of PS, price of worker), e.g., 3.5}", # float
    "path_to_keyfile" : "{Your AWS private key file with the .pem extenion, like xxxxxx.pem}",                   # string
    "nfs_ip_address" : "{IP address of the EFS xxx.xxx.xxx.xxx}",                                                # string
    "nfs_mount_point" : "{Path to EFS folder named "shared", set to /home/ubuntu/shared}",                       # string
    "security_group": ["{Name of the AWS security group, mentioned before as byzshield_security_group}"],        # string
})
```
**Note**: The data sets will be downloaded from the PS to the EFS folder and then fetched from all machines from there. For this to work, the EFS folder needs to be named `shared` and located at the home directory `~`. Since `~` is same as `/home/ubuntu` for AWS Ubuntu instances, you need to set the EFS folder above to be `~/shared` or `/home/ubuntu/shared`.

**Note**: In above configuration, make sure that the chosen instance types `master_type` and `worker_type` are available in the selected AWS region (`region`) and availability zone (`availability_zone`).

Next, use the chmod command to make sure your private key file isn't publicly viewable:
```sh
chmod 400 {xxxxxx}.pem
```

Now, from the local machine and the Python 2 envirnoment `byzshield_local_python2`, launch replicas of the AMI (those will be the PS and worker instances) running the following:
```sh
conda activate byzshield_local_python2
python ./tools/pytorch_ec2.py launch
```

Wait for the command to finish so that all instances are ready. Then, run the following command to generate a file `hosts_address` with the private IPs of all instances. The first line of the file is the IP of the PS (this will also be printed on the terminal):
```sh
python ./tools/pytorch_ec2.py get_hosts
```

Use the private IP of the PS fetched above to do some SSH configuration and copy the project files to the PS:
```sh
bash ./tools/local_script.sh {private IP of the PS}
```

SSH into the PS, the rest of the commands will be executed there:
```sh
ssh -i  {Your AWS private key file with the .pem extenion} ubuntu@{private IP of the PS}
```

## Data set and worker preparation
On the PS, download, split and normalize the MNIST/Cifar10/SVHN/Cifar100 data sets:
```sh
conda activate byzshield && bash ./src/data_prepare.sh
```
**Note**: This requires `sudo` permissions to save data sets to the EFS folder. Since `sudo` uses a different path than your typical environment, you need to specify that you want to use the Anaconda Python 3 environment we created before rather than the system `python`. To do that make sure that `data_prepare.sh` points to that environment
```sh
sudo {Path to Anaconda "python" file, e.g., /home/ubuntu/anaconda3/envs/byzshield/bin/python} ./datasets/data_prepare.py
```

On the PS, run the `remote_script.sh` to configure the SSH and copy the project files and data sets to the workers:
```sh
bash ./tools/remote_script.sh
```

## Training
The training algorithm should be run by the PS instance executing file `run_pytorch.sh`. The basic arguments of this script along with all possible values are below. This is not an exhaustive list of all arguments but only the basic ones, the remaining can be left to their default values in `run_pytorch.sh`.

| Argument                      | Values/description                                 |
| ----------------------------- | ---------------------------------------- |
| `n` | Total number of nodes (PS and workers), equal to *K+1* in paper |
| `hostfile`      | Path to MPI hostfile that contains the private IPs of all nodes of the cluster. If ran on AWS this file will be `hosts_address`, discussed above. If ran locally this file can be a plain txt with content `localhost:{n+1}`. |
| `lr` | Inital learning rate. |
| `momentum` | Value of momentum. |
| `network` | Deep neural net: `LeNet`,`ResNet18`,`ResNet34`,`ResNet50`,`DenseNet`,`VGG11` or `VGG13`. |
| `dataset` | Data set: `MNIST`, `Cifar10`, `SVHN` or `Cifar100`. |
| `batch-size` | Batchsize: equal to b in ByzShield, equal to br/K in DETOX, equal to b/K in vanilla batch-SGD (baseline). |
| `mode` | Robust aggregation method: `coord-median`, `bulyan`, `multi-krum`, `sign-sgd` or `geometric_median` (only supported in baseline). |
| `approach` | Distributed learning scheme `baseline` (vanilla), `mols` (proposed MOLS), `rama_one` (proposed Ramanujan Case 1), `rama_two` (proposed Ramanujan Case 2), `draco-lite` (DETOX), `draco_lite_attack` (our attack on DETOX), `maj_vote`. |
| `eval-freq` | Frequency of iterations to backup trained model (for evaluation). |
| `err-mode` | Byzantine attack to simulate: `rev_grad` (reversed gradient) or `constant` (constant gradient) or `foe` ("Fall of Empires"), refer to `src/model_ops/util.py` for details. |
| `epochs` | Number of epochs to train. |
| `max-steps` | Total number of iterations (across all epochs). |
| `worker-fail` | Number of Byzantine workers, equal to *q* in paper. |
| `group-size` | Replication factor, equal to *r* in paper. |
| `lis-simulation` | Attack ["A Little Is Enough"](https://arxiv.org/pdf/1902.06156.pdf): `simulate` (enabled) or `no` (disabled), the `err-mode` will be disabled if ALIE attack is enabled. |
| `train-dir` | Directory to save model backups for evaluation (for AWS this should be the EFS folder). |
| `local-remote` | `local` (for local training) or `remote` (for training on AWS). |
| `rama-m` | Value of *m* (in paper), only needed for Ramanujan Case 2. |
| `detox-attack` | Our attack on DETOX (see `--approach`): `worst` (optimally attacks majority within groups), `benign` or `whole_group`.  |
| `byzantine-gen` | Type of byzantine set generation (`random` (random for each iteration) or `hard_coded` (fixed for all iterations and set in `util.py`)). This won't affect `draco_lite_attack`. |
| `gamma` | Learning rate decay (linear). |
| `lr-step` | Frequency of learning rate decay (measured in number of iterations). |

To initiate training, from the PS run:
```sh
bash ./src/run_pytorch.sh 1
```

## Testing
By convention, worker 1 will fetch the model from the shared EFS folder and evaluate it. To achieve this, from the PS, run (for the definitions of `q`, `lr` and `gamma` see above, they should match the values used for training):
```sh
bash ./src/evaluate_pytorch.sh 1 {q} {lr} {gamma}
```

The basic arguments of this script along with all possible values are below.
| Argument                      | Values/description                                 |
| ----------------------------- | ---------------------------------------- |
| `eval-batch-size` | Number of samples to iteratively evaluate model on. |
| `eval-freq` | Set to the same value as `eval-freq` used for training in `run_pytorch.sh` (any multiple of it will also work). |
| `network` | Deep neural net: `LeNet`,`ResNet18`,`ResNet34`,`ResNet50`,`DenseNet`,`VGG11` or `VGG13`. |
| `dataset` | Data set: `MNIST`, `Cifar10`, `SVHN` or `Cifar100`. |
| `model-dir` | Set to the same value as `train-dir` used for training in `run_pytorch.sh`. |


## Citation
If you use this code please cite our paper available at [arXiv](https://arxiv.org/abs/2010.04902). The BibTeX is:
```
@inproceedings{konstantinidis_ramamoorthy_byzshield,
title = {ByzShield: An Efficient and Robust System for Distributed Training},
author = {Konstantinos Konstantinidis and Aditya Ramamoorthy},
year = {2020},
month = {October},
url = {https://arxiv.org/abs/2010.04902}}
}
```


[DETOX]: <https://github.com/hwang595/DETOX>
[Install]: <https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html>
[configure]: <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html>
[Launch]: <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/launching-instance.html>
[AMI]: <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html>
[AMI_create]: <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/creating-an-ami-ebs.html>
[EFS_create]: <https://docs.aws.amazon.com/efs/latest/ug/gs-step-two-create-efs-resources.html>
[security_group_create]: <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/working-with-security-groups.html#creating-security-group>
