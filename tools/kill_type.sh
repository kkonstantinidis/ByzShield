# A simple script to terminate/stop all instances of a specific type after a timeout

# Instance type to stop (not all spot instances can be stopped)
# typeToStop="r3.large"

# Instance type to terminate
typeToKill="c3.4xlarge"

# Time to wait before terminating
sleepSec="180m"
# sleepSec="15"

sleep $sleepSec

# Stop
# aws ec2 stop-instances --instance-ids $(aws ec2 describe-instances --query "Reservations[*].Instances[*].InstanceId" --filters "Name=instance-type,Values="$typeToStop"" --output text)

sleep 5

# Terminate
aws ec2 terminate-instances --instance-ids $(aws ec2 describe-instances --query "Reservations[*].Instances[*].InstanceId" --filters "Name=instance-type,Values="$typeToKill"" --output text)


