# ~ copies code and some other files from PS to workers
PROJECT_INDEX="$1"

# to be run by PS
KEY_PEM_NAME=virginiakey.pem
export DEEPLEARNING_WORKERS_COUNT=`wc -l < hosts`

# ~ run a command as the substitute user without starting an interactive shells
sudo bash -c "cat hosts >> /etc/hosts"
cp config ~/.ssh/

# ~ start an ssh-agent session, this enables single sign-on (SSO), i.e., the ssh-agent can use the keys to log into server without having the user type in a password or passphrase again
# -s: forces generation of Bourne shell (/bin/sh) commands on stdout. By default the shell is automatically detected.
cd ~/.ssh
eval `ssh-agent -s`

# ~ add ssh keys to the agent
ssh-add ${KEY_PEM_NAME}

# ~ -t: algorithm
# -b: key size
# -c: comment, changes the comment for a keyfile
ssh-keygen -t rsa -b 4096 -C "kostas@iastate.edu"

# ~ copies code to all workers
for i in $(seq 2 $DEEPLEARNING_WORKERS_COUNT);
  do
  # ~ test
  # ssh-keygen -f "/home/ubuntu/.ssh/known_hosts" -R deeplearning-worker${i}
  
  scp -i ${KEY_PEM_NAME} id_rsa.pub deeplearning-worker${i}:~/.ssh
  ssh -i ${KEY_PEM_NAME} deeplearning-worker${i} 'cd ~/.ssh; cat id_rsa.pub >> authorized_keys'
  
  # ~ test ONLY, use with caution to kill leftover Python processes at workers.
  # ssh -i ${KEY_PEM_NAME} deeplearning-worker${i} 'pkill -9 python'
  
  # scp -i ${KEY_PEM_NAME} -r ~/BYZSHIELD deeplearning-worker${i}:~/
  # ~ same as in "local_scipt.sh" but it does not exclude data folders
  # rsync -avzh -e "ssh -i ${KEY_PEM_NAME}" ~/BYZSHIELD ubuntu@deeplearning-worker${i}:~/ --exclude '__pycache__'
  rsync -avzh -e "ssh -i ${KEY_PEM_NAME}" ~/BYZSHIELD${PROJECT_INDEX} ubuntu@deeplearning-worker${i}:~/ --exclude '__pycache__'
  echo "Done writing public key to worker: deeplearning-worker${i}"
 done
