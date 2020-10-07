# model-dir (for remote execution): AWS EFS folder since only rank-1 worker has the trained models

if [ "$#" -lt 4 ]; then
    echo "Illegal number of parameters!"
    echo "Uses: "
    echo "  (1): bash evaluate_pytorch.sh --project-index --q --lr --gamma"
    echo "  (2): bash evaluate_pytorch.sh --project-index --q --lr --gamma --last-step"
    exit 2
fi

PROJECT_INDEX="$1"
Q_VALUE="$2"
LR_VALUE="$3"
GAMMA_VALUE="$4"
LAST_STEP_VALUE="${5:-0}" # 5th parameter is optional, set to 0 if not given

# ~ for one-model subfolder
# python distributed_evaluator.py \
# --eval-batch-size=10000 \
# --eval-freq=60 \
# --network=LeNet \
# --dataset=MNIST \
# --model-dir=${HOME}/shared/


# ~ for tuning subfolder
python distributed_evaluator.py \
--eval-batch-size=10000 \
--eval-freq=52 \
--network=ResNet18 \
--dataset=Cifar10 \
--model-dir=${HOME}/shared/tune/BYZSHIELD${PROJECT_INDEX}/output_q_${Q_VALUE}_lr_${LR_VALUE}_gamma_${GAMMA_VALUE}/ \
--cur-step=0 \
--last-step=${LAST_STEP_VALUE}