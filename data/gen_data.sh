#!/bin/bash

TIMESTAMP=$(date +%s)

partition=$1

if [ "$partition" = "local" ]; then
    echo "LOCAL"
    PARTITION=local
else
    echo "SLURM"
    # PARTITION=devlab
    # PARTITION=scavenge
    # PARTITION=learnfair
    PARTITION=learnlab
fi

SEED=0

######## TESTING FILTERS #########
CONFIG_FILE="world.property_only.txt"
num_examples=1000
array_iterations=100
NUM_CLAUSES=2
DATA_NAME=1M
ACTIVE_WORLD_STEPS=50

# CONFIG_FILE="world.tag.name.txt"
# num_examples=1000
# array_iterations=100
# NUM_CLAUSES=1
# DATA_NAME=100K
# ACTIVE_WORLD_STEPS=0
##################################


######## TESTING TEMPORAL #########

# CONFIG_FILE="world.active.all.all.txt"
# CONFIG_FILE="world.geometric_only.txt"
# CONFIG_FILE="world.property_only.txt"
# CONFIG_FILE="world.active.temporal_only.txt"
# num_examples=1000
# array_iterations=1000
# NUM_CLAUSES=2
# DATA_NAME=1M
# ACTIVE_WORLD_STEPS=50
##################################

# ######## TESTING OBJ TRACK #########
# CONFIG_FILE="obj_track.txt"
# num_examples=1000
# array_iterations=100
# NUM_CLAUSES=2
# DATA_NAME=1M
# ACTIVE_WORLD_STEPS=50
# ##################################

# ######## TESTING OBJ TRACK #########
# CONFIG_FILE="world.look_direction.txt"
# num_examples=200
# array_iterations=500
# NUM_CLAUSES=1
# DATA_NAME=50k
# ACTIVE_WORLD_STEPS=50
# ##################################



array_iterations=$((array_iterations+2)) # for val/test set

tokenizer=gpt

base_dir=/checkpoint/$USER/jobs/data/$name
mkdir -p $base_dir

ngpus=0

codedir=$base_dir/code/
rsync -a --exclude=".*" --exclude="*.pyc" /private/home/$USER/aszlam_droidlet_nsm/data/ $codedir
rsync -a --exclude=".*" --exclude="*.pyc" /private/home/$USER/aszlam_droidlet_nsm/training/db_dataloader/*.py $codedir
rsync -a --exclude=".*" --exclude="*.pyc" /private/home/$USER/aszlam_droidlet_nsm/training/utils/*.py $codedir
rsync -a --exclude=".*" --exclude="*.pyc" /private/home/$USER/aszlam_droidlet_nsm/training/interact.py $codedir
rsync -a --exclude=".*" --exclude="*.pyc" /private/home/$USER/aszlam_droidlet_nsm/fairo/agents/craftassist/tests/*.py $codedir
rsync -a --exclude=".*" --exclude="*.pyc" /private/home/$USER/aszlam_droidlet_nsm/fairo/droidlet/perception/craftassist/*.py $codedir
cd $codedir
echo $codedir

export PYTHONPATH=$codedir:$PYTHONPATH

if [[ $PARTITION = "local" ]]; then
    SEED=0
    name="DATA_GEN.${SEED}.${TIMESTAMP}"
    echo running locally $name
    python build_data.py --seed=$SEED \
                --num_examples 100 \
                --num_names 8  \
                --num_props 8 \
                --props_per_name 1 \
                --num_clauses $NUM_CLAUSES \
                --overwrite-data  \
                --save_directory /private/home/$USER/data_test/ \
                --config_file /private/home/$USER/aszlam_droidlet_nsm/data/config/$CONFIG_FILE \
                --max_output_tokens 1000 \
                --tokenizer $tokenizer \
                --no_val \
                --data_name 'debug' \
                --active_world_steps $ACTIVE_WORLD_STEPS \
                --seed $SEED \
                # --make_custom_val 
else
    echo running slurm
    name="DATA.${CONFIG_FILE}.${NUM_CLAUSES}.${ACTIVE_WORLD_STEPS}.${TIMESTAMP}"
    python slurm/submit_data.py --name $name --folder $base_dir/$name --partition $PARTITION --array_iterations $array_iterations --ngpus $ngpus --constraint "$constraint" --nodes 1 --slurm_array --args \
                "--seed=$SEED \
                --num_examples $num_examples \
                --num_names 8  \
                --num_props 8 \
                --props_per_name 1 \
                --num_clauses $NUM_CLAUSES \
                --overwrite-data  \
                --save_directory /private/home/$USER/data/ \
                --config_file /private/home/$USER/aszlam_droidlet_nsm/data/config/$CONFIG_FILE \
                --max_output_tokens 1000 \
                --no_val \
                --data_name $DATA_NAME \
                --seed $SEED \
                --active_world_steps $ACTIVE_WORLD_STEPS \
                --tokenizer $tokenizer"
fi;