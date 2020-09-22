# ~ probably useless, does the same as kill_all_python() in pytorch_ec2.py
KEY_PEM_NAME=${HOME}/.ssh/virginiakey.pem

# ~ counts no. of lines in files
export DEEPLEARNING_WORKERS_COUNT=`wc -l < hosts`

for i in $(seq 2 $DEEPLEARNING_WORKERS_COUNT);
  do
  ssh -i ${KEY_PEM_NAME} deeplearning-worker${i} 'killall python'
 done