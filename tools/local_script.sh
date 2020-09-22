# ~ copies code and some other files to PS
# $1: public IP of PS
KEY_PEM_DIR=virginiakey.pem
KEY_PEM_NAME=virginiakey.pem # ~ never used
PUB_IP_ADDR="$1"
PROJECT_INDEX="$2"
echo "Public address of master node: ${PUB_IP_ADDR}"

# ~ probably the hosts file for SSH warning suppression is useless
ssh -o "StrictHostKeyChecking no" ubuntu@${PUB_IP_ADDR}

scp -i ${KEY_PEM_DIR} ${KEY_PEM_DIR} ubuntu@${PUB_IP_ADDR}:~/.ssh
scp -i ${KEY_PEM_DIR} hosts hosts_address config ubuntu@${PUB_IP_ADDR}:~/
# scp -i ${KEY_PEM_DIR} -r "/home/kostas/Dropbox/Python Workspace/SGD/BYZSHIELD" ubuntu@${PUB_IP_ADDR}:~/
# ~ sync left-to-right only changed files and ignores __pycache__ and MNIST/CIFAR10/SVHN/CIFAR1000 data folders, either training or testing, need to escape spaces
rsync -avzh -e "ssh -i ${KEY_PEM_DIR}" ~/Dropbox/Python\ Workspace/SGD/BYZSHIELD/ ubuntu@${PUB_IP_ADDR}:~/BYZSHIELD${PROJECT_INDEX} --exclude '*data' --exclude '__pycache__'
# ssh -i ${KEY_PEM_DIR} ubuntu@${PUB_IP_ADDR} 'sudo apt-get update; cp BYZSHIELD/tools/remote_script.sh ~/'
ssh -i ${KEY_PEM_DIR} ubuntu@${PUB_IP_ADDR} 'sudo apt-get update'
