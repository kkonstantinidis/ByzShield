# ByzShield
ByzShield's robust distributed ML framework implementation

This project builds on [DETOX] and implements our proposed ByzShield algorithm for robust distributed deep learning. Our placement involves three different techniques, namely MOLS, Ramanujan Case 1 & Ramanujan Case 2. It also includes three different types of attack on the DETOX framework.

### Requirements

We will be working with Python 2 for the local machine (to execute the bash scripts which configure the remote cluster and initiate training/testing) and with Python 3 for the PS/worker nodes (to execute the actual training/testing). We recommend using an Anaconda environment in both cases.

For the local machine the tested dependencies are
| Module | Version |
| ------ | ------ |
| Python | 2.7 |
| pip | 20.1.1 |
| setuptools | 47.1.1 |
| python-blosc | 1.7.0 |
| joblib | 0.15.1 |
| paramiko | 2.7.1 |
| boto3 | 1.9.66 |

[DETOX]: <https://github.com/hwang595/DETOX>

