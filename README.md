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

## Anaconda environments
The tested dependencies versions for the local/remote machines are
| Module | Local | Remote |
| ------ | ------ | ------ |
| python | 2.7.18 | 3.7.7 |
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
conda install -c conda-forge python-blosc
conda install -c anaconda joblib
conda install -c anaconda paramiko
conda install -c anaconda boto3
```

The exact series of commands for the *remote* machine is
```sh
conda create -n byzshield python=3.7
conda activate byzshield
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

## Remote AMI
Now that we have installed all needed dependencies on the remote EC2 instance, we need to make an AMI image of it so that we can quickly launch PS/worker instances out of it. For instructions see [here][AMI_create]. Make note of the created AMI ID, like `ami-xxxxxxxxxxxxxxxx`.

## AWS EFS
We will use Amazon Elastic File System (EFS) to share a folder with the trained model among the machines. Follow the [EFS_create][instructions] to create an EFS. We will probably need a security group with the settings discussed above for the EFS too. Make note of the IP address of the EFS `xxx.xxx.xxx.xxx`.

## Cluster configuration
The script `pytorch_ec2.py` will la launch the instances automatically. Before running it, you should edit the following part:
```sh
cfg = Cfg({
    "name" : "Timeout",      # Unique name for this specific configuration
    "key_name": "virginiakey",          # ~ Necessary to ssh into created instances, WITHOUT .pem
    # Cluster topology
    "n_masters" : 1,                      # Should always be 1
    "n_workers" : 15,
    "num_replicas_to_aggregate" : "8", # deprecated, not necessary
    "method" : "spot",
    # Region speficiation
    "region" : "us-east-1",
    "availability_zone" : "us-east-1c",
    # Machine type - instance type configuration.
    "master_type" : "r3.xlarge",
    "worker_type" : "r3.xlarge",
    # please only use this AMI for pytorch
    "image_id": "ami-022c3bc433cd214b4",
    # Launch specifications
    "spot_price" : "0.5",                 # Has to be a string
    # SSH configuration
    "ssh_username" : "ubuntu",            # For sshing. E.G: ssh ssh_username@hostname
    "path_to_keyfile" : "virginiakey.pem", # ~ be careful with this path since the execution path depends on where you run the code from

    # NFS configuration
    # To set up these values, go to Services > Elastic File System > Create file system, and follow the directions.
    "nfs_ip_address" : "172.31.18.129",          # us-east-1c
    "nfs_mount_point" : "/home/ubuntu/shared",       # NFS base dir
    "base_out_dir" : "%(nfs_mount_point)s/%(name)s", # Master writes checkpoints to this directory. Outfiles are written to this directory.
    "setup_commands" :
    [
        "mkdir %(base_out_dir)s",
    ],
    # Command specification
    # Master pre commands are run only by the master
    "master_pre_commands" :
    [
        "cd my_mxnet",
        "git fetch && git reset --hard origin/master",
        "cd cifar10",
        "ls",
    ],
    # Pre commands are run on every machine before the actual training.
    "pre_commands" :
    [
        "cd my_mxnet",
        "git fetch && git reset --hard origin/master",
        "cd cifar10",
    ],
    # Model configuration
    "batch_size" : "4", # ~ never used
    "max_steps" : "2000", # ~ never used
    "initial_learning_rate" : ".001", # ~ never used
    "learning_rate_decay_factor" : ".9", # ~ never used
    "num_epochs_per_decay" : "1.0", # ~ never used
    # Train command specifies how the ps/workers execute tensorflow.
    # PS_HOSTS - special string replaced with actual list of ps hosts.
    # TASK_ID - special string replaced with actual task index.
    # JOB_NAME - special string replaced with actual job name.
    # WORKER_HOSTS - special string replaced with actual list of worker hosts
    # ROLE_ID - special string replaced with machine's identity (E.G: master, worker0, worker1, ps, etc)
    # %(...)s - Inserts self referential string value.
    "train_commands" :
    [
        "echo ========= Start ==========="
    ],
})
```

## Training
The training algorithm should be run by the PS instance executing file `run_pytorch.sh`. The basic arguments of this script along with all possible values are below. This is not an exhaustive list of all arguments but only the basic ones, the remaining can be left to their default values in `run_pytorch.sh`.

| Argument                      | Values/description                                 |
| ----------------------------- | ---------------------------------------- |
| `n` | Total number of nodes (PS and workers), equal to *K+1* in paper |
| `hostfile`      | Path to MPI hostfile that contains the private IPs of all nodes of the cluster. If ran on AWS this file will be `hosts_address`, discussed above. If ran locally this file can be a plain txt with content `localhost:{n+1}` |
| `lr` | Inital learning rate. |
| `momentum` | Value of momentum. |
| `network` | Deep neural net: `LeNet`,`ResNet18`,`ResNet34`,`ResNet50`,`DenseNet`,`VGG11` or `VGG13`. |
| `dataset` | Data set: `MNIST`, `Cifar10`, `SVHN` or `Cifar100`. |
| `batch-size` | Batchsize, equal to b in paper. |
| `mode` | Robust aggregation method: `coord-median`, `bulyan` or `multi-krum` |
| `approach` | Distributed learning scheme `baseline` (vanilla), `mols` (proposed MOLS), `rama_one` (proposed Ramanujan Case 1), `rama_two` (proposed Ramanujan Case 2), `draco-lite` (DETOX), `draco_lite_attack` (our attack on DETOX), `maj_vote` |
| `eval-freq` | Frequency of iterations to backup trained model (for evaluation). |
| `err-mode` | Byzantine attack to simulate: `rev_grad` (reversed gradient) or `constant` (constant gradient), refer to `src/model_ops/utily.py` for details |
| `epochs` | Number of epochs to train. |
| `max-steps` | Total number of iterations (across all epochs). |
| `worker-fail` | Number of Byzantine workers, equal to *q* in paper. |
| `group-size` | Replication factor, equal to *r* in paper. |
| `lis-simulation` | Attack ["A Little Is Enough"](https://arxiv.org/pdf/1902.06156.pdf): `simulate` (enabled) or `no` (disabled), the `err-mode` will be disabled if ALIE attack is enabled. |
| `train-dir` | Directory to save model backups for evaluation (for AWS this should be the EFS folder). |
| `local-remote` | `local` (for local training) or `remote` (for training on AWS). |
| `rama-m` | Value of *m* (in paper), only needed for Ramanujan Case 2. |

[DETOX]: <https://github.com/hwang595/DETOX>
[Install]: <https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html>
[configure]: <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html>
[Launch]: <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/launching-instance.html>
[AMI]: <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html>
[AMI_create]: <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/creating-an-ami-ebs.html>
[EFS_create]: <https://docs.aws.amazon.com/efs/latest/ug/gs-step-two-create-efs-resources.html>
