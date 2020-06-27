# ByzShield
ByzShield's robust distributed ML framework implementation

This project builds on [DETOX] and implements our proposed ByzShield algorithm for robust distributed deep learning. Our placement involves three different techniques, namely MOLS, Ramanujan Case 1 & Ramanujan Case 2. It also includes three different types of attack on the DETOX framework.

## Requirements

We will be working with Python 2 for the local machine (to execute the bash scripts which configure the remote cluster and initiate training/testing) and with Python 3 for the remote cluster of PS/worker nodes (to execute the actual training/testing). We recommend using an Anaconda (tested with v.2020.02) environment in both cases. 

# Prereqisites/Anaconda installation (both local and remote)
```sh
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install g++ && sudo apt-get install libopenmpi-dev && sudo apt-get install gfortran && sudo apt-get install make
# We will stick to Python 3 only. Make sure that Python 3 is installed and is the default instead of Python 2 (in case both are installed). Run 
python --version
# if it works, verify that it uses Python 3, you can enforce it by
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 10
# if it does not work, run the following command and repeat the above process
sudo apt install python-minimal
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

