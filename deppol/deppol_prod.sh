#!/usr/bin/env bash

# ${1}: sn list file
# ${2}: wd
# ${3}: folder

mkdir -p $SCENEMODELING_FOLDER/runs/${3}
mkdir -p $SCENEMODELING_FOLDER/runs/${3}/logs
mkdir -p $SCENEMODELING_FOLDER/runs/${3}/batches

#OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 deppol --ztfname=$p -j 1 --wd=${2} --lc-folder=\$SCENEMODELING_FOLDER/lc --func=make_catalog,mkcat2,makepsf,match_gaia,filter_seeing,filter_psfstars_count,reference_quadrant,astrometry_fit,astrometry_fit_plot,photometry_fit,photometry_fit_plot,smphot --lc-folder=/sps/ztf/data/storage/scenemodeling/lc --quadrant-workspace=/dev/shm/llacroix --rm-intermediates --scratch=/tmp/llacroix --astro-degree=5 --max-seeing=4. --discard-calibrated --astro-min-mag=-10."

#OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 deppol --ztfname=$p -j 40 --wd=${2} --lc-folder=\$SCENEMODELING_FOLDER/lc --func=make_catalog,mkcat2,makepsf,match_gaia,filter_seeing,filter_psfstars_count,reference_quadrant,astrometry_fit,astrometry_fit_plot,photometry_fit,photometry_fit_plot,smphot --lc-folder=/sps/ztf/data/storage/scenemodeling/lc --quadrant-workspace=/dev/shm/llacroix --rm-intermediates --scratch=/tmp/llacroix --astro-degree=5 --max-seeing=4. --discard-calibrated --astro-min-mag=-10."
#
while read p; do
    __script="#!/bin/sh
source ~/pyenv/bin/activate
export PYTHONPATH=\${PYTHONPATH}:~/phdev/tools
export PATH=\${PATH}:~/phdev/deppol
ulimit -n 4096
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 deppol --ztfname=$p -j 1 --wd=${2} --func=make_catalog,mkcat2,makepsf,filter_seeing,filter_psfstars_count,match_gaia,reference_quadrant,astrometry_fit,photometry_fit,smphot --lc-folder=/sps/ztf/data/storage/scenemodeling/lc --quadrant-workspace=/dev/shm/llacroix --rm-intermediates --scratch=/tmp/llacroix --astro-degree=5 --max-seeing=4. --discard-calibrated --astro-min-mag=-10."

    echo "$p"
    echo "$__script" > $SCENEMODELING_FOLDER/runs/${3}/batches/batch_${p}.sh
done <${1}
