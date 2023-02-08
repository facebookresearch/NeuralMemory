#!/bin/bash

TIMESTAMP=$(date +%s)
model_type=$1
partition=$2

if [ "$partition" = "local" ]; then
    NGPUS=8
    NNODES=1
    BATCH_SZ=16
    PARTITION=local
else
    PARTITION=$partition
    NGPUS=4
    NNODES=1
    BATCH_SZ=32
    NUM_SEEDS=1
fi

########## USER VARIABLES ##########
WANDB_USERNAME=$USER
WANDB_PROJECT=neural_memory
WANDB_DIR="$(pwd)"/neural_memory_logs/
####################################

########### DATA PATHS #############
DATA_ROOT="$(pwd)"/data/data/
RESULTS_DIR="$(pwd)"/results/
####################################

############################## DATASETS TO RUN ####################################
DATASET="ex_100.world.active.all.all.ppn_1.cls_2.tlf.ts_100.sl_15.gpt.aws_50.V0"
###################################################################################

RUN_TYPE=large
SEED=0

args=" \
--run_name=$RUN_TYPE \
--verbosity=1 \
--wandb_entity=$WANDB_USERNAME \
--wandb_project=$WANDB_PROJECT \
--wandb_dir=$WANDB_DIR \
--plot \
--plot-dir=$RESULTS_DIR \
--seed=$SEED \
--tokenizer=gpt \
--large_train \
--overwrite-checkpoint \
--overwrite-logs \
--cosine_decay \
--text_loss \
"


if [[ $model_type = "structured" ]]; then
    args=$args" \
    --model-type=transformer \
    --context-type=triple_refobj_rel \
    --memid_loss \
    --triple_supervision \
    "
else
    args=$args" \
    --model-type=gpt \
    --context-type=text \
    "
fi


if [ "$partition" = "local" ]; then
    echo "local"
else
    echo "distributed"
    args=$args" --distributed"
fi


DATA_PATH="$DATA_ROOT$DATASET"

echo $DATA_PATH
JOB_NAME="${DATASET}.TIME_${TIMESTAMP}"
echo $JOB_NAME

args=$args" --simple-data-path=$DATA_PATH"


./training/slurm/launch.sh $JOB_NAME $PARTITION $NGPUS $NNODES $RUN_TYPE $NUM_SEEDS $args
