#!/bin/bash

name=$1
partition=$2
ngpus=$3
nodes=$4
run_type=$5
num_seeds=$6
args=${@:7}

base_dir=/checkpoint/$USER/jobs/neural_memory/$name
mkdir -p $base_dir

# backup the code
codedir=$base_dir/code/
rsync -a --exclude=".*" --exclude="*.pyc" /private/home/$USER/aszlam_droidlet_nsm/training/ $codedir
rsync -a --exclude=".*" --exclude="*.pyc" /private/home/$USER/aszlam_droidlet_nsm/data/* $codedir

# log the git state
git log|head -6 > $base_dir/git.txt
echo -e "\n\n" >> $base_dir/git.txt
git status >> $base_dir/git.txt
echo -e "\n\n" >> $base_dir/git.txt
git diff >> $base_dir/git.txt

cd $codedir
host=$(hostname)
export PYTHONPATH=$codedir:$PYTHONPATH
if [[ "$partition" = "local" ]]; then
  echo running locally $name
  echo "python main.py "$args
  # python main.py $args --checkpoint $base_dir/model.pt --plot --plot-name $name
  python main.py $args
else
  python slurm/submit.py --name $name --folder $base_dir --partition $partition --ngpu $ngpus --constraint "$constraint" --nodes $nodes \
  --run_type $run_type --num_seeds $num_seeds --args "$args"
fi
