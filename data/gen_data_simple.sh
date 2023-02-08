#!/bin/bash

TIMESTAMP=$(date +%s)

partition=$1
SEED=1

if [ "$partition" = "local" ]; then
    echo "LOCAL"
    PARTITION=local
else
    PARTITION=$partition
    ngpus=1
    echo "SLURM"
fi

######## TESTING ALL #########
# CONFIG_FILE="world.active.all.all.txt"
# CONFIG_FILE="world.geometric_only.txt"
# CONFIG_FILE="world.property_only.txt"
# CONFIG_FILE="world.active.temporal_only.txt"
CONFIG_FILE=$2
NUM_EXAMPLES=$3
NUM_CLAUSES=2
ACTIVE_WORLD_STEPS=50
##################################


if [[ $PARTITION = "local" ]]; then
    echo running locally
    python ./data/build_memory_eqa_data.py --seed=$SEED \
                --num_examples=$NUM_EXAMPLES \
                --num_names=8  \
                --num_props=8 \
                --props_per_name=1 \
                --num_clauses=$NUM_CLAUSES \
                --save_directory=./data/data/ \
                --config_file=./data/config/$CONFIG_FILE \
                --max_output_tokens=1000 \
                --tokenizer=gpt \
                --data_name=V0 \
                --active_world_steps=$ACTIVE_WORLD_STEPS \
                --seed=$SEED \
                --split=train \
                --overwrite-data   
else

    echo running slurm
    array_iterations=$((array_iterations+2)) # for val/test set
    base_dir=/checkpoint/$USER/jobs/data/$name
    mkdir -p $base_dir
    pwd=`pwd`

    codedir=$base_dir/code/

    rsync -a --exclude=".*" --exclude="*.pyc" ./data/ $codedir
    rsync -a --exclude=".*" --exclude="*.pyc" ./transformemNN/db_dataloader/*.py $codedir
    rsync -a --exclude=".*" --exclude="*.pyc" ./transformemNN/utils/*.py $codedir
    rsync -a --exclude=".*" --exclude="*.pyc" ./transformemNN/interact.py $codedir
    rsync -a --exclude=".*" --exclude="*.pyc" ./fairo/agents/craftassist/tests/*.py $codedir
    rsync -a --exclude=".*" --exclude="*.pyc" ./fairo/droidlet/perception/craftassist/*.py $codedir
    cd $codedir
    echo $codedir
    export PYTHONPATH=$codedir:$PYTHONPATH

    

    name="DATA.${CONFIG_FILE}.${NUM_CLAUSES}.${ACTIVE_WORLD_STEPS}.${TIMESTAMP}"
    python slurm/submit_data.py --name $name --folder $base_dir/$name --partition $PARTITION --array_iterations $array_iterations --ngpus $ngpus --nodes 1 --slurm_array --args \
                "--num_examples=$NUM_EXAMPLES \
                --num_names=8  \
                --num_props=8 \
                --props_per_name=1 \
                --num_clauses=$NUM_CLAUSES \
                --save_directory=$pwd/data/data/ \
                --config_file=$codedir/config/$CONFIG_FILE \
                --max_output_tokens=1000 \
                --tokenizer=gpt \
                --no_val \
                --data_name=V0 \
                --active_world_steps=$ACTIVE_WORLD_STEPS \
                --seed=$SEED \
                --split=train"
fi