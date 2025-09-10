#!/bin/bash
# gfortran fortran_CRTM_ADA_solver.f90 -o run_ADA
for w in $(seq 0.2 0.1 0.6); do
    for t_od in $(seq 5 20 200); do
        ./run_ADA ${w} ${t_od}
    done
done


