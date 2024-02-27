# Dependency package installation

## C++ project: poloka suit

use waf builder version without binding

## python packages

## With pip install

## With git clone / local pip install

# Set environment at CCIN2P3

```
#! /bin/bash


module load Compilers/gcc/10.2.0

source /sps/ztf/users/colley/install/miniconda3/etc/profile.d/conda.sh

conda activate /sps/ztf/users/colley/env/ztf3_ll

cd /sps/ztf/users/colley/poloka_waf/src


export PKG_CONFIG_PATH=$PWD/local/lib/pkgconfig

export PATH=$PATH:$PWD/local/bin:$PWD/phdev/deppol

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/local/lib:$PWD/local/lib64

export PYTHONPATH=$PWD/phdev/tools:$PYTHONPATH

#
export SCENE_PATH=/sps/ztf/data/storage/scenemodeling/


export ZTFDATA=/sps/ztf/data

export TOADSCARDS=/sps/ztf/users/colley/poloka_waf/src/poloka-core/datacards

# donnee internal release lie table SN, pos, Spectre
export ZTFCOSMOIDR=/sps/data/datarelease/ztfcosmoidr/dr2

```
