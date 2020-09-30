# ~ this is the main file to do everything after EC2 setup, do not use pytorch_ec2.py for tasks that are done here, it has dead code
PROJECT_INDEX="$1"
n=16
group_size=3
bucket_size=1
batch_size=50
rama_m=5
eval_freq=40
max_steps=10
# lr_step=${eval_freq}
lr_step=15
checkpoint_step=0

approach=baseline
# approach=maj_vote
# approach=draco_lite
# approach=draco_lite_attack
# approach=mols
# approach=rama_one
# approach=rama_two

# err_mode=rev_grad
# err_mode=constant
err_mode=foe

# lis_simulation=simulate
lis_simulation=nope

mode=coord-median
# mode=bulyan
# mode=multi-krum
# mode=sign-sgd

# hostfile="tools/hosts_address"
hostfile="tools/hosts_address_local"

# local_remote=remote
local_remote=local

detox_attack=worst
# detox_attack=benign
# detox_attack=whole_group

# ~ test
# for tuning with varying q
tune_dir=${HOME}/shared/tune/BYZSHIELD${PROJECT_INDEX}
echo "Start parameter tuning ..."
for q in 3
do
    for lr in 0.05
    do
        for gamma in 0.96
        do
            START=$(date +%s.%N)

            echo "Trial running for q: ${q}"
            mkdir -p "${tune_dir}/output_q_${q}_lr_${lr}_gamma_${gamma}"
            mpirun -n ${n} --hostfile ../${hostfile} \
            python distributed_nn.py \
            --lr=${lr} \
            --momentum=0.9 \
            --network=LeNet \
            --dataset=MNIST \
            --batch-size=${batch_size} \
            --comm-type=Bcast \
            --mode=${mode} \
            --approach=${approach} \
            --eval-freq=${eval_freq} \
            --err-mode=${err_mode} \
            --adversarial=-100 \
            --epochs=50 \
            --max-steps=${max_steps} \
            --worker-fail=${q} \
            --group-size=${group_size} \
            --compress-grad=compress \
            --bucket-size=${bucket_size} \
            --checkpoint-step=${checkpoint_step} \
            --lis-simulation=${lis_simulation} \
            --train-dir="${tune_dir}/output_q_${q}_lr_${lr}_gamma_${gamma}/" \
            --local-remote=${local_remote} \
            --rama-m=${rama_m} \
            --byzantine-gen=hard_coded \
            --detox-attack=${detox_attack} \
            --gamma=${gamma} \
            --lr-step=${lr_step}

            END=$(date +%s.%N)
            DIFF=$(echo "$END - $START" | bc)
            echo "Total training time (sec): $DIFF"
        done
    done
done

# train_dir=${HOME}/shared/
# mpirun -n ${n} --hostfile ../${hostfile} \
# python distributed_nn.py \
# --lr=0.001 \
# --momentum=0.9 \
# --network=LeNet \
# --dataset=MNIST \
# --batch-size=${batch_size} \
# --comm-type=Bcast \
# --mode=coord-median \
# --approach=${approach} \
# --eval-freq=${eval_freq} \
# --err-mode=constant \
# --adversarial=-100 \
# --epochs=5 \
# --max-steps=${max_steps} \
# --worker-fail=${q} \
# --group-size=${group_size} \
# --compress-grad=compress \
# --bucket-size=${bucket_size} \
# --checkpoint-step=0 \
# --lis-simulation=simulate \
# --train-dir=${train_dir} \
# --local-remote=${local_remote} \
# --rama-m=${rama_m} \
# --byzantine-gen=hard_coded \
# --detox-attack=${detox_attack}


# for local DETOX
# mpirun -n 16 --hostfile "/home/kostas/Dropbox/Python Workspace/SGD/BYZSHIELD/tools/hosts_address_local" \
# python distributed_nn.py \
# --lr=0.001 \
# --momentum=0.9 \
# --network=LeNet \
# --dataset=MNIST \
# --batch-size=50 \
# --comm-type=Bcast \
# --mode=coord-median \
# --approach=draco_lite \
# --eval-freq=500 \
# --err-mode=constant \
# --adversarial=-100 \
# --epochs=5 \
# --max-steps=1200 \
# --worker-fail=3 \
# --group-size=3 \
# --compress-grad=compress \
# --bucket-size=5 \
# --checkpoint-step=0 \
# --lis-simulation=simulate \
# --train-dir=/home/kostas/shared/ \
# --local-remote=local \
# --rama-m=5 \
# --byzantine-gen=hard_coded \
# --detox-attack=worst


# for local BYZSHIELD
# mpirun -n 16 --hostfile "/home/kostas/Dropbox/Python Workspace/SGD/BYZSHIELD/tools/hosts_address_local" \
# python distributed_nn.py \
# --lr=0.001 \
# --momentum=0.9 \
# --network=LeNet \
# --dataset=MNIST \
# --batch-size=250 \
# --comm-type=Bcast \
# --mode=coord-median \
# --approach=rama_one \
# --eval-freq=60 \
# --err-mode=constant \
# --adversarial=-100 \
# --epochs=5 \
# --max-steps=1200 \
# --worker-fail=2 \
# --group-size=3 \
# --compress-grad=compress \
# --bucket-size=25 \
# --checkpoint-step=0 \
# --lis-simulation=simulate \
# --train-dir=/home/kostas/shared/ \
# --local-remote=local \
# --rama-m=5 \
# --byzantine-gen=hard_coded \
# --detox-attack=worst

