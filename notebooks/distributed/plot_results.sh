#!/bin/bash

ENROOT_SLURM_HOOK=0 enroot start \
    --mount /capstor:/capstor \
    --mount /iopsstor:/iopsstor \
    --mount "$(pwd)":/workspace \
    /capstor/store/cscs/cscs/jupyter/pytorch/pytorch-course-tf-25.05-py3.sqsh \
    bash -c "python collect_results.py && python plot_results.py"
