#!/bin/bash
# usage: ./x.sh

clear

# settings
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# setup (cython) 
python3 components_metrics_setup.py build_ext --inplace

# execute python scripts
python3 function_main.py

printf "SCRIPTS EXECUTED SUCCESSFULLY\n"

