#!/usr/bin/env bash

# ${1}: sn list file
# ${2}: folder

while read p; do
    sbatch --ntasks=1 -D $SCENEMODELING_FOLDER/runs/${2} -J ${p}_smp -o $SCENEMODELING_FOLDER/runs/${2}/logs/log_${p}.txt -A ztf -L sps $SCENEMODELING_FOLDER/runs/${2}/batches/batch_${p}.sh
done <${1}
