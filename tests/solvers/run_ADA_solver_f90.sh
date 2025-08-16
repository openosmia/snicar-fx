#!/bin/bash
# gfortran fortran_CRTM_ADA_solver.f90 -o run_ADA
for w in $(seq 2 0.5 5); do
    for t_od in $(seq 0.5 0.5 2); do
        ./run_ADA ${w} ${t_od}
    done
done
